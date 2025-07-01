import logging
from dataclasses import asdict, dataclass
import numpy as np
from scipy import stats
from resordan.data.events import EventDataset, EventsDataset
from resordan.data.gmf import GMFDataset

logger = logging.getLogger(__name__)


@dataclass
class DetectorConfig:
    """Config for clustering algorithm

    Parameters
    ----------
    segment_split_time, optional
        Minimum time (in seconds) to consider SNR peaks as belonging to different objects.
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
    # TODO: using stats to better the selection of events?
    #r_poly, r_stats = np.polynomial.Polynomial.fit(t, r, 2, full=True)
    #v_poly, v_stats = np.polynomial.Polynomial.fit(t, v, 1, full=True)
    #a_poly, a_stats = np.polynomial.Polynomial.fit(t, a, 0, full=True)

    # Fit polynomials: assuming the object has constant acceleration
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


def save_detected(gmf_dataset, window_inds, event_number, events):
    events.append(
                _create_event_dataset_for_detection(gmf_dataset, window_inds, event_number)
                )
    event_number += 1

    return event_number


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
    time_delta = times[1:] - times[:-1]
    window_start = np.where(time_delta > cfg.segment_split_time)[0] + 1
    windows = np.split(inds, window_start)

    # set the segment_split_time to half its value
    split_time = cfg.segment_split_time/(stats.mode(time_delta))[0]

    # Run detection in each window
    events = []
    event_number = 0
    for window_inds in windows:
        if (len(window_inds) < cfg.min_n_samples):
            continue

        if run_detection(gmf_dataset, window_inds, cfg):
            event_number = save_detected(gmf_dataset, window_inds, event_number, events)
            continue

        # Check for different ranges in one event
        r_round=np.round(gmf_dataset.range_peak[window_inds]/1e3)
        bin_range = np.arange(np.nanmin(r_round),np.nanmax(r_round)+2)
        hist, r_edge = np.histogram(r_round, bins=bin_range)
        idxh = np.where(hist>0)[0]

        idx_delta = idxh[1:] - idxh[:-1]
        idx_win = np.where(np.abs(idx_delta) > 3)[0] + 1 # change
        idx_win = np.split(idxh, idx_win)
        
        to_sortime = []
        to_idx = []
        for idxw in idx_win:
            if r_edge[idxw[0]] < 350 or np.sum(hist[idxw]) < cfg.min_n_samples:
                continue
            
            # taking each group at a time
            new_r_win = gmf_dataset.range_peak[window_inds]
            min_range = (np.nanmin(r_edge[idxw]) - 2)*1e3
            max_range = (np.nanmax(r_edge[idxw]) + 2)*1e3
            select_r = np.where((new_r_win >= min_range) & (new_r_win <= max_range))[0]
            select_r_dif = select_r[1:] - select_r[:-1]
            select_win = np.where(np.abs(select_r_dif) > split_time)[0] + 1
            select_win = np.split(select_r, select_win)

            for select_windown in select_win:
                if len(select_windown) < cfg.min_n_samples:
                    continue

                ridx = np.argmax(gmf_dataset.snr[select_windown])
                to_sortime.append(gmf_dataset.t[select_windown[ridx]])
                to_idx.append(window_inds[select_windown])

        for idxs in np.argsort(to_sortime):
            if run_detection(gmf_dataset, to_idx[idxs], cfg):
                event_number = save_detected(gmf_dataset, to_idx[idxs], event_number, events)
            
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
