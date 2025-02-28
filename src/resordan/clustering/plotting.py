import numpy as np
from scipy import constants
from datetime import datetime, UTC
import pandas as pd

from resordan.data.gmf import compute_snr


def _convert(data, km=True, monostatic=True):
    """Assume data is [m] and two-way range"""
    _data = data.copy()
    if km:
        _data *= 0.001
    if monostatic:
        _data *= 0.5
    return _data


def to_relative_range_gate(ranges, meta):
    sample_rate = meta.experiment["sample_rate"]
    il0_r_samp = (ranges / constants.c) * sample_rate + meta.experiment[
        "T_tx_start_samp"
    ]
    return il0_r_samp.astype(np.int64) - meta.processing["il0_min_range_gate"]


def plot_peaks(axes, data, detected_inds, monostatic=True):
    coh_inds = np.arange(data.gmf.shape[0])
    optimized = data.gmf_optimized_peak is not None

    meta = data.meta
    min_acc = meta.processing["min_acceleration"]
    max_acc = meta.processing["max_acceleration"]

    snr = compute_snr(data.gmf, data.nf_range)
    if optimized:
        range_gates = to_relative_range_gate(data.gmf_optimized_peak[:, 0], meta)
        snr_opt = compute_snr(
            data.gmf_optimized, data.nf_range, range_gates=range_gates
        )

    # TODO: use a interpolation of nf-range to determine the SNR of the optimized results
    # snr = compute_snr(data.gmf_optimized, data.nf_range)
    r_inds = np.argmax(snr, axis=1)
    snr = snr[coh_inds, r_inds]
    snrdb = 10 * np.log10(snr)

    # inds = snrdb > snr_dB_limit
    inds = np.isin(np.arange(snrdb.shape[0]), detected_inds)
    not_inds = np.logical_not(inds)

    _inds0_sty = dict(marker="x", alpha=0.5, ls="none", color="r")
    _not0_inds_sty = dict(marker="x", alpha=0.5, ls="none", color="b")
    _inds_sty = dict(marker=".", ls="none", color="r")
    _not_inds_sty = dict(marker=".", ls="none", color="b")

    t = data.t
    nt = [datetime.fromtimestamp(ti, UTC) for ti in t]
    t = np.array(nt)

    r = _convert(data.range_peak, monostatic=monostatic)
    v = _convert(data.range_rate_peak, monostatic=monostatic)
    a = _convert(data.acceleration_peak, monostatic=monostatic, km=False)

    r = data.range_peak/1e3
    v = data.range_rate_peak/1e3
    a = data.acceleration_peak

    axes[0, 0].plot(t[not_inds], r[not_inds], **_not_inds_sty)
    axes[0, 0].plot(t[inds], r[inds], **_inds_sty)
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("range [km]")

    axes[0, 1].plot(t[not_inds], v[not_inds], **_not_inds_sty)
    axes[0, 1].plot(t[inds], v[inds], **_inds_sty)
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("range rate [km/s]")

    axes[1, 0].plot(t[not_inds], a[not_inds], **_not_inds_sty)
    axes[1, 0].plot(t[inds], a[inds], **_inds_sty)
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("acceleration [m/s/s]")
    axes[1, 0].set_ylim([min_acc, max_acc])
    axes[1, 0].tick_params(axis='x', labelrotation=45)

    axes[1, 1].plot(t[not_inds], snrdb[not_inds], **_not_inds_sty)
    axes[1, 1].plot(t[inds], snrdb[inds], **_inds_sty)
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("SNR [dB]")
    axes[1, 1].tick_params(axis='x', labelrotation=45)

    if optimized:
        r0 = _convert(data.gmf_optimized_peak[:, 0], monostatic=monostatic)
        v0 = _convert(data.gmf_optimized_peak[:, 1], monostatic=monostatic)
        a0 = _convert(data.gmf_optimized_peak[:, 2], monostatic=monostatic, km=False)
        axes[0, 0].plot(t[inds], r0[inds], **_inds0_sty)
        axes[0, 1].plot(t[inds], v0[inds], **_inds0_sty)
        axes[1, 0].plot(t[inds], a0[inds], **_inds0_sty)
        axes[1, 1].plot(t[inds], np.sqrt(snr_opt[inds]), **_inds0_sty)
        axes[1, 1].plot(t[not_inds], np.sqrt(snr_opt[not_inds]), **_not0_inds_sty)

    return axes, None
