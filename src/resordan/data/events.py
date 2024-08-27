from typing import Any, Optional

import numpy as np

from .base import BaseDataset, dataset
from .gmf import GMFMetadata


@dataset
class EventDataset(BaseDataset):
    # Identification
    event_number: int
    target_id: Optional[int] = None
    # Time series
    epoch: float
    t: np.ndarray
    range: np.ndarray
    range_rate: np.ndarray
    acceleration: np.ndarray
    snr: np.ndarray
    tx_power: np.ndarray
    idx: np.ndarray
    pointing: Optional[np.ndarray | float] = None
    # Time series standard deviation
    sd_range: Optional[np.ndarray] = None
    sd_range_rate: Optional[np.ndarray] = None
    sd_acceleration: Optional[np.ndarray] = None
    sd_snr: Optional[np.ndarray] = None
    # Detection
    effective_diameter: Optional[float] = None
    rcs_lower_bound: Optional[float] = None
    event_duration: Optional[float] = None
    multiple_targets: bool = False


@dataset
class EventsDataset(BaseDataset):
    # Metadata
    meta: GMFMetadata
    # Detector config
    detector_config: dict[str, Any]
    # Events
    events: list[EventDataset]
