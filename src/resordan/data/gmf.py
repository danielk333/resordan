from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

from .base import BaseDataset, dataset

logger = logging.getLogger(__name__)


################################################################################
# GMF Dataset definitions
################################################################################


@dataset
class GMFMetadata(BaseDataset):
    ranges: np.ndarray
    range_rates: np.ndarray
    accelerations: np.ndarray
    range_gates: np.ndarray
    processing: dict[str, Any]
    experiment: dict[str, Any]


@dataset
class GMFDataset(BaseDataset):
    meta: GMFMetadata
    gmf: Optional[np.ndarray] = None
    gmf_optimized_peak: Optional[np.ndarray] = None
    gmf_zero_frequency: Optional[np.ndarray] = None
    range_rate_index: Optional[np.ndarray] = None
    acceleration_index: Optional[np.ndarray] = None
    nf_vec: Optional[np.ndarray] = None
    nf_range: Optional[np.ndarray] = None
    range_peak: Optional[np.ndarray] = None
    range_rate_peak: Optional[np.ndarray] = None
    acceleration_peak: Optional[np.ndarray] = None
    gmf_optimized: Optional[np.ndarray] = None
    gmf_peak: Optional[np.ndarray] = None
    tx_power: Optional[np.ndarray] = None
    t: Optional[np.ndarray] = None
    snr: Optional[np.ndarray] = None
    pointing: Optional[np.ndarray | float] = None

    @classmethod
    def from_files(cls, *args, **kwargs) -> GMFDataset:
        """Create dataset from a list of files. See `load_gmf` for description and parameters."""
        data, meta = load_gmf(*args, **kwargs)
        meta_dataset = GMFMetadata(**meta)
        dataset = cls(meta=meta_dataset, **data)
        return dataset


################################################################################
# Loaders for GMF data
################################################################################


DEFAULT_MATS = {
    "gmf",
    "gmf_optimized_peak",
    "gmf_zero_frequency",
    "range_rate_index",
    "acceleration_index",
    "nf_vec",
    "pointing",
}
DEFAULT_VECS = {
    "range_peak",
    "range_rate_peak",
    "acceleration_peak",
    "gmf_optimized",
    "gmf_peak",
    "tx_power",
    "t",
}
DERIVED = {
    "t",
    "nf_vec",
}
OPTIONAL = {
    "gmf_optimized_peak",
    "gmf_optimized",
    "pointing",
}


def _load_key_into_dict(hf, data, key):
    try:
        data[key] = hf[key][()]
    except KeyError as err:
        if key not in OPTIONAL:
            raise err


def _load_data_from_file(gmf_path, keys):
    data = {}
    with h5py.File(gmf_path, "r") as hf:
        for key in keys:
            _load_key_into_dict(hf, data, key)

        epoch_unix = hf["epoch_unix"].astype(np.int64)[()]
        _t_conv = (hf["processing"].attrs["n_ipp"][()] * hf["experiment"].attrs["ipp"][()]) * 1e-6

    n_cohints = data["gmf"].shape[0]
    data["t"] = (np.arange(n_cohints) + 1) * _t_conv + epoch_unix
    nf_vec = np.nanmedian(data["gmf_zero_frequency"], axis=0)
    nf_vec = nf_vec.reshape((1, nf_vec.size))
    data["nf_vec"] = nf_vec

    return data


def _load_meta_from_file(gmf_path):
    with h5py.File(gmf_path, "r") as hf:
        meta = {
            "ranges": hf["ranges"][()],
            "range_rates": hf["range_rates"][()],
            "accelerations": hf["accelerations"][()],
            "range_gates": np.arange(
                hf["processing"].attrs["min_range_gate"],
                hf["processing"].attrs["min_range_gate"],
            ),
        }
        meta["processing"] = {key: val for key, val in hf["processing"].attrs.items()}
        meta["experiment"] = {key: val for key, val in hf["experiment"].attrs.items()}
        for key, val in hf.attrs.items():
            meta[key] = val

    return meta


def _dict_concatenate(data):
    out = {}
    for key in data[0].keys():
        out[key] = np.concatenate([d[key] for d in data], axis=0)
    return out


def _convert(_data, monostatic=True, flip_sign=False):
    if monostatic:
        _data *= 0.5
    if flip_sign:
        _data *= -1
    return _data


def compute_snr(gmf_values, noise_floor, range_gates=None, dB=False):
    """Convert GMF value to SNR based on range dependant noise floor
    (assumes GMF has dimensions [?,range] unless range_gates is given)

    Function copied from hardtarget repo.
    """
    if range_gates is None:
        snr = (np.sqrt(gmf_values) - np.sqrt(noise_floor[None, :])) ** 2 / noise_floor[None, :]
    else:
        inds = np.logical_and(range_gates >= 0, range_gates < len(noise_floor))
        snr = np.full_like(gmf_values, np.nan)
        snr[inds] = (
            np.sqrt(gmf_values[inds]) - np.sqrt(noise_floor[range_gates[inds]])
        ) ** 2 / noise_floor[range_gates[inds]]
    if dB:
        return 10 * np.log10(snr)
    else:
        return snr


def load_gmf(
    files: list[str | Path],
    mats: set | None = None,
    vecs: set | None = None,
    monostatic: bool = True,
    flip_range_rate_sign: bool = True,
    flip_acceleration_sign: bool = True,
) -> tuple[dict, dict]:
    """Load data from list of GMF output files

    Parameters
    ----------
    files
        List of GMF output files (.h5)
    mats, optional
        Optional set of matrix keys to load, by default None. `None` = load all matrix keys.
    vecs, optional
        Optional set of vector keys to load, by default None. `None` = load all vector keys.
    monostatic, optional
        Data is monostatic? By default True
    flip_range_rate_sign, optional
        Flip range_rate sign? By default True
    flip_acceleration_sign, optional
        Flip acceleration sign? By default True

    Returns
    -------
        Tuple of `(data, meta)`, where data and meta contain data and metadata, respectively.
    """

    assert files, "List of GMF files is empty."

    logger.info(f"Loading GMF data from {len(files)} file(s).")

    # Figure out which keys to load
    mats = DEFAULT_MATS if mats is None else set(mats)
    vecs = DEFAULT_VECS if vecs is None else set(vecs)
    load_keys = (mats | vecs) - DERIVED

    # Load metadata from first file
    meta = _load_meta_from_file(files[0])

    # Load data from file list
    data = [_load_data_from_file(gmf_path, load_keys) for gmf_path in files]
    data = _dict_concatenate(data)

    # Compute 'nf_range'
    data["nf_range"] = np.nanmedian(data["nf_vec"], axis=0)

    # Compute SNR
    if "gmf" in data:
        r_inds = np.argmax(data["gmf"], axis=1)
        coh_inds = np.arange(data["gmf"].shape[0])
        snr = compute_snr(data["gmf"], data["nf_range"])
        snr = snr[coh_inds, r_inds]
        data["snr"] = 10 * np.log10(snr)
    else:
        logger.warning("Key 'gmf' not in dataset. Skipping SNR computations.")

    # Convert data
    keys_to_convert = [
        "range_peak",
        "range_rate_peak",
        "acceleration_peak",
    ]
    flip_signs = [
        False,
        flip_range_rate_sign,
        flip_acceleration_sign,
    ]
    for key, flip_sign in zip(keys_to_convert, flip_signs, strict=True):
        if key in data:
            data[key] = _convert(data[key], monostatic=monostatic, flip_sign=flip_sign)

    return data, meta
