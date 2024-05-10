from typing import Optional

import numpy as np

from .base import BaseDataset, dataset
from .gmf import GMFMetadata


@dataset
class EventDataset(BaseDataset):
    # Identification
    event_number: int
    target_id: Optional[int] = None
    # Time-series
    idx: np.ndarray
    t: np.ndarray
    range: np.ndarray
    range_rate: np.ndarray
    acceleration: np.ndarray
    snr: np.ndarray
    # Detection
    effective_diameter: Optional[float] = None
    rcs_lower_bound: Optional[float] = None
    event_duration: Optional[float] = None
    multiple_targets: bool = False


@dataset
class EventsDataset(BaseDataset):
    # Metadata
    meta: GMFMetadata
    events: list[EventDataset]
