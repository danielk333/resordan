import logging
from dataclasses import asdict, dataclass
import numpy as np
from resordan.data.events import EventDataset, EventsDataset
from resordan.data.gmf import GMFDataset

logger = logging.getLogger(__name__)


@dataclass
class DetectorConfig:
    """Config for clustering algorithm

    Parameters
    ----------
    segment_split_time, optional
        Minimum time (in seconds) to consider SNR peaks as beloning to different objects.
        Default = 5
    snr_db_threshold, optional
        Threshold (in dB) for detecting peaks in SNR. Default = 15
    loss_weights, optional
        Loss function weights. Default = (1e-3, 1e-3)
    loss_threshold, optional
        Loss function threshold. Default = 10
    min_n_samples, optional
        Minimum number of samples in an event. Default = 5
    """

    segment_split_time: float = 5
    snr_db_threshold: float = 15
    loss_weights: tuple[float, float] = (1e-3, 1e-3)
    loss_threshold: float = 10
    min_n_samples: int = 5


def _detect_in_window(
    t: np.ndarray,
    r: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    cfg: DetectorConfig
) -> bool:

    """Run detection for a window of data."""
    if t.size < cfg.min_n_samples:
        logger.warning(f"Not enough samples for polynomial fit ({t.size}).")
        return False

    # TODO: Check shapes

    # Fit polynomials
    r_poly = np.polynomial.Polynomial.fit(t, r, 2)
    v_poly = np.polynomial.Polynomial.fit(t, v, 1)
    a_poly = np.polynomial.Polynomial.fit(t, a, 0)
    cr0, cr1, cr2 = r_poly.coef
    cv0, cv1 = v_poly.coef
    ca0 = a_poly.coef[0]

    # Compute loss from polynomial coefficients
    coef_loss_range_rate = np.abs(cr1 - cv0)
    coef_loss_acceleration = np.abs(2 * cr2 - cv1) + np.abs(2 * cr2 - ca0) + np.abs(cv1 - ca0)
    coef_loss = (
        cfg.loss_weights[0] * coef_loss_range_rate + cfg.loss_weights[1] * coef_loss_acceleration
    )

    logger.info(
        "Polynomial coefficient losses: "
        f"range_rate={coef_loss_range_rate}, "
        f"acceleration={coef_loss_acceleration}, "
        f"weighted_total={coef_loss}."
    )
    # Threshold loss (determine if this is a detected object)
    detected = coef_loss < cfg.loss_threshold
    return detected


def _slice_if_not_none_or_nan(arr, idx):
    if arr is None:
        return None
    if np.isscalar(arr) and np.isnan(arr):
        return np.nan
    return arr[idx]


def _create_event_dataset_for_detection(gmf_dataset, idx, event_number):
    t = gmf_dataset.t[idx]
    epoch = t[0]
    t -= epoch

    event = EventDataset(
        event_number=event_number,
        event_duration=t.max() - t.min(),
        epoch=epoch,
        t=t,
        idx=idx,
        range=gmf_dataset.range_peak[idx],
        range_rate=gmf_dataset.range_rate_peak[idx],
        acceleration=gmf_dataset.acceleration_peak[idx],
        snr=gmf_dataset.snr[idx],
        tx_power=gmf_dataset.tx_power[idx],
        pointing=_slice_if_not_none_or_nan(gmf_dataset.pointing, idx),
    )
    return event


def run_detection(gmf_dataset, window_inds, cfg):
            # Run detection
            detected = _detect_in_window(
                t=gmf_dataset.t[window_inds],
                r=gmf_dataset.range_peak[window_inds],
                v=gmf_dataset.range_rate_peak[window_inds],
                a=gmf_dataset.acceleration_peak[window_inds],
                cfg=cfg,
            )
            return detected


def snr_peaks_detection(gmf_dataset: GMFDataset, **kwargs) -> EventsDataset:
    """Run clustering by detecting SNR peaks.

    Parameters
    ----------
    data
        Input dataset (GMF output)
    **kwargs
        Parameters for detection algorithm. See `DetectorConfig`.

    Returns
    -------
        EventsDataset representing detected objects.
    """
    logger.info(f"Running detection. Number of timesteps = {gmf_dataset.t.size}.")

    cfg = DetectorConfig(**kwargs)
    logger.info(f"Detector config: {cfg}.")

    # Threshold SNR to get detection candidates
    inds = np.where(gmf_dataset.snr > cfg.snr_db_threshold)[0]

    if inds.size < 4:
        logger.warning("No detections found.")
        return EventsDataset(meta=gmf_dataset.meta, detector_config=asdict(cfg), events=[])
    else:
        logger.info(f"Found {inds.size} timesteps with SNR > {cfg.snr_db_threshold} dB.")

    # Split detection candidates into windows
    times = gmf_dataset.t[inds]
    time_deltas = times[1:] - times[:-1]
    window_starts = np.where(time_deltas > cfg.segment_split_time)[0] + 1
    windows = np.split(inds, window_starts)

    # Run detection in each window
    events = []
    event_number = 0
    for window_inds in windows:
        rr=gmf_dataset.range_peak[window_inds]
        range_deltas = rr[1:] - rr[:-1]
        # Finding if there are different range in the detections
        window_rr = np.where(np.abs(range_deltas) > 2000)[0] + 1

        # When there are several ranges
        if len(window_rr) >= 1:
            ss = np.round(gmf_dataset.range_peak[window_inds]/1e3)
            hist, bin_edges = np.histogram(ss, bins = int((np.max(ss)-np.min(ss))/3))
            idxh = np.argwhere(hist>2)
            histmean = np.mean([bin_edges[idxh],bin_edges[idxh+1]], axis = 0)
            histdif = np.abs(gmf_dataset.range_peak[window_inds] - histmean*1e3)

            # confirming the timing of the detection is in sequence
            medianv = [np.median(np.where(histdif[ii,:] < 2000)) for ii in range(len(histdif[:,0]))]

            # checking the time in the subdetections. Time separation not greather than the given time
            for ii in np.argsort(medianv):
                idxr = np.where(histdif[ii,:] < 2000)
                tt_idxr = gmf_dataset.t[window_inds[idxr]]
                diff_tt = tt_idxr[1:] - tt_idxr[:-1]
                window_tt = np.where(diff_tt > cfg.segment_split_time)[0] + 1

                if len(window_tt) == 0:
                    detected = run_detection(gmf_dataset, window_inds[idxr], cfg)

                    if detected:
                        print('A',event_number)
                        events.append(
                            _create_event_dataset_for_detection(gmf_dataset, window_inds[idxr], event_number)
                        )
                        event_number += 1
                else:
                    subwindows = np.split(window_inds[idxr], window_tt)

                    for sub_windows in subwindows:
                        detected = run_detection(gmf_dataset, sub_windows, cfg)

                        if detected:
                            print('B',event_number)
                            events.append(
                                _create_event_dataset_for_detection(gmf_dataset, sub_windows, event_number)
                            )
                            event_number += 1
    
        # When there is a single range
        else:
            detected = run_detection(gmf_dataset, window_inds, cfg)
        
            if detected:
                print('C',event_number)
                # Ensure that window indices are linearly increasing.
                # This ensures constant sampling period in the time series of the detected event.
                window_inds = np.arange(window_inds[0], window_inds[-1] + 1)
                # Create EventDataset for detected event.
                events.append(
                    _create_event_dataset_for_detection(gmf_dataset, window_inds, event_number)
                )
                event_number += 1

    logger.info(f"Detected {len(events)} targets.")
    return EventsDataset(meta=gmf_dataset.meta, detector_config=asdict(cfg), events=events)


def event_detection(src, **params):

    """
    does snr_peaks_detection per directory (to limit memory usage)

    Parameters
    ----------
    src: str
        path to GMF product folder, or subfolder within GMF product

    Returns
    -------
        EventsDataset representing detected objects, or None        
    """

    if not src.is_dir():
        raise Exception(f"src is not directory {src}")

    def process(subdir):
        gmf_files = [f for f in subdir.iterdir() if f.is_file() and f.suffix == ".h5"]
        if not gmf_files: 
            return None, None
        gmf_dataset = GMFDataset.from_files(list(sorted(gmf_files)))
        return [snr_peaks_detection(gmf_dataset, **params), gmf_dataset]

    def results(dirs):
        """run process across dirs and include results if not None"""
        results = []
        gmf_dataset = []
        for d in dirs:
            res, gmf_dataset = process(d)
            if res is not None:
                results.append(res)
        return results, gmf_dataset

    # first, assume that src is a subfolder in a GMF product
    # if it is not the result will be empty
    # then assume that it is a GMF product containing subfolders
    ed_list, gmf_dataset = results([src])
    if not ed_list:
        ed_list, gmf_dataset = results([d for d in sorted(src.iterdir()) if d.is_dir()])

    # check if any data was found
    if not ed_list:
        return None

    # event numbering
    event_counter = 0
    for ed in ed_list:
        for event in ed.events:
            event.event_number = event_counter
            event_counter += 1

    # return events dataset
    meta = ed_list[0].meta
    detector_config = ed_list[0].detector_config
    elist = []
    for ed in ed_list: 
        elist.extend(ed.events)
    
    return EventsDataset(meta=meta, detector_config=detector_config, events=elist), gmf_dataset
