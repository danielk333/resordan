import logging
from dataclasses import dataclass

import numpy as np

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


@dataclass
class _DetectorResult:
    coef_loss: float = np.nan
    coef_loss_range_rate: float = np.nan
    coef_loss_acceleration: float = np.nan
    detected: bool = False


def _detect_in_window(
    t: np.ndarray,
    r: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    cfg: DetectorConfig,
) -> _DetectorResult:
    """Run detection for a window of data."""
    result = _DetectorResult()
    if t.size < cfg.min_n_samples:
        logger.warning(f"Not enough samples for polynomial fit ({t.size}).")
        return result

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

    # Set result fields
    result.coef_loss_range_rate = coef_loss_range_rate
    result.coef_loss_acceleration = coef_loss_acceleration
    result.coef_loss = coef_loss
    # Threshold loss
    result.detected = coef_loss < cfg.loss_threshold

    return result


def _prepare_data(data, cfg):
    # TODO: Update based on data format
    keys = ["t", "range_peak", "range_rate_peak", "acceleration_peak", "snr"]
    return [data.get(k) for k in keys]


def snr_peaks_detection(data, **kwargs):
    """Run clustering by detecting SNR peaks.

    Parameters
    ----------
    data
        Input dataset (GMF output)
    **kwargs
        Parameters for detection algorithm. See `DetectorConfig`.

    Returns
    -------
        Nested list of event indices.
    """
    logger.info(f"Running detection. Number of timesteps = {data['t'].size}.")

    cfg = DetectorConfig(**kwargs)
    logger.info(f"Detector config: {cfg}.")

    # Load data
    t, r, v, a, snr = _prepare_data(data, cfg)

    # Threshold SNR to get detection candidates
    inds = np.where(snr > cfg.snr_db_threshold)[0]

    if inds.size < 4:
        logger.warning("No detections found.")
        return []
    else:
        logger.info(f"Found {inds.size} timesteps with SNR > {cfg.snr_db_threshold} dB.")

    # Split detection candidates into windows
    times = t[inds]
    time_deltas = times[1:] - times[:-1]
    window_starts = np.where(time_deltas > cfg.segment_split_time)[0] + 1
    windows = np.split(inds, window_starts)

    # Run detection in each window
    detected_inds = []
    for window_inds in windows:
        result = _detect_in_window(
            t[window_inds], r[window_inds], v[window_inds], a[window_inds], cfg
        )
        if result.detected:
            detected_inds.append(window_inds)

    logger.info(f"Detected {len(detected_inds)} targets.")
    return detected_inds
