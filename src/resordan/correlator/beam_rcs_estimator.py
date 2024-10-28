#!/usr/bin/env python

from pathlib import Path
import pickle
import argparse
import scipy
import similaritymeasures
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from tqdm import tqdm
import datetime as dt

import h5py

import sorts
import pyant
import resordan.correlator.update_tle as utle
from resordan.correlator.discos_cat import get_discos_objects

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    class COMM_WORLD:
        rank = 0
        size = 1

        def Barrier(self):
            pass

    comm = COMM_WORLD()


###################################################################
# UTIL
###################################################################

def offaxis_weighting(angles):
    """
    Weight for offaxis angle, normalized for this set of angles
    """

    if angles.size == 0:
        return np.ones_like(angles)
    w = angles.copy()
    w += 1 - np.min(w)
    w = 1.0/w
    w = w/np.sum(w)
    return w


def matching_function(data, SNR_sim, off_angles, min_snr, verbose=False):
    """
    Matching between the simulated and measured SNR time series
    """

    # Filter measurnments
    _SNR_sim = SNR_sim.copy()
    _SNR_sim[_SNR_sim <= 0] = np.nan
    sndb = data.snr
    snsdb = np.log10(_SNR_sim)*10
    if np.all(np.isnan(_SNR_sim)):
        match = np.nan
        meta = [np.nan, np.nan]
        mae = np.nan
        return match, meta, mae

    max_sndb_x = np.nanmax(data.snr)
    max_sndb_y = np.log10(np.nanmax(_SNR_sim))*10

    # ratio of peak signal and limit should be the same
    # as the ratio of the simulated peak and the simulated signal
    # use this to calculate the limit used in simulation
    snrdb_lim_rel = max_sndb_x - min_snr
    sns_lim = max_sndb_y - snrdb_lim_rel
    # the normalized limit is, since the normalizations put peaks to 1
    snrdb_lim_norm = -snrdb_lim_rel

    off_weight = offaxis_weighting(off_angles)

    sn = 10**(data.snr/10)
    xsn = np.full(sn.shape, np.nan, dtype=sn.dtype)
    idx_x = sn > 0
    if np.sum(idx_x) > 1:
        xsn[idx_x] = np.log10(sn[idx_x]/np.nanmax(sn[idx_x]))
    idx_x = np.logical_not(np.isnan(xsn))

    sns = _SNR_sim
    ysn = np.full(sns.shape, np.nan, dtype=sn.dtype)
    idx_y = sns > 0
    if np.sum(idx_y) > 1:
        ysn[idx_y] = np.log10(sns[idx_y]/np.nanmax(sns[idx_y]))
    idx_y = np.logical_not(np.isnan(ysn))

    idx = np.logical_and(idx_x, idx_y)
    tot = np.sum(idx)

    dhit_idx = np.logical_and.reduce([
        sndb > min_snr,
        snsdb > sns_lim,
        idx,
    ])
    dcut_idx = np.logical_and.reduce([
        np.logical_or(sndb <= min_snr, np.logical_not(idx_x)),
        snsdb > sns_lim,
        idx_y,
    ])
    dmiss_idx = np.logical_and.reduce([
        sndb > min_snr,
        np.logical_or(snsdb <= sns_lim, np.logical_not(idx_y)),
        idx_x,
    ])

    dhit = np.nan
    dcut = np.nan
    dmiss = np.nan

    if tot > 1:
        if np.any(dhit_idx):
            dhit = np.sum(((xsn[dhit_idx] - ysn[dhit_idx])*off_weight[dhit_idx])**2)
        if np.any(dcut_idx):
            dcut = np.sum(((ysn[dcut_idx] - snrdb_lim_norm)*off_weight[dcut_idx])**2)
        if np.any(dmiss_idx):
            dmiss = np.sum((xsn[dmiss_idx]*off_weight[dmiss_idx])**2)

    match = dhit
    if not np.isnan(dcut):
        match += dcut
    if not np.isnan(dmiss):
        match += dmiss
    match_0 = match
    match = np.sqrt(match/len(SNR_sim))
    meta = [dcut/len(SNR_sim), dmiss/len(SNR_sim)]

    timearray = data.t
    xxx = np.array([timearray[idx], xsn[idx]]).T
    yyy = np.array([timearray[idx], ysn[idx]]).T
    rowy, coly = np.shape(yyy)

    if rowy < 4:
        mae = float('NaN')
    else:
        mae = similaritymeasures.mae(xxx[:], yyy[:])

    if verbose:
        print('idx_x: ', idx_x)
        print('idx_y: ', idx_y)
        print('idx: ', idx)
        print('dhit_idx: ', dhit_idx)
        print('dcut_idx: ', dcut_idx)
        print('dmiss_idx: ', dmiss_idx)
        print('tot: ', tot)
        print('dhit: ', dhit)
        print('dcut: ', dcut)
        print('dmiss: ', dmiss)
        print('match_0: ', match_0)
        print('match: ', match)
        print('meta: ', meta)

    if np.isnan(dhit):
        match = float('NaN')
    else:
        match = match

    return match, meta, mae


def snr2rcs(
        gain_tx,
        gain_rx,
        wavelength,
        power_tx,
        range_tx_m,
        range_rx_m,
        snr,
        bandwidth=10,
        rx_noise_temp=150.0,
        radar_albedo=1.0):

    """
    Use the radar equation to compute RCS values from SNR data
    """

    rx_noise = scipy.constants.k*rx_noise_temp*bandwidth
    power = snr*rx_noise/radar_albedo

    rcs_dividend = 64.0 * (np.pi**3.0) * (range_rx_m**2.0*range_tx_m**2.0) * power
    rcs_divisor = power_tx*(gain_tx*gain_rx)*(wavelength**2.0)
    return rcs_dividend / rcs_divisor


def generate_prediction(data, t, obj, radar, min_gain):
    """
    Simulate the diamer and SNR values for the poiniting direction of the radar beam
    Compute the RCS values
    """

    states_ecef = obj.get_state(t)  # /sorts/space_object.py & /sorts/propagator/pysgp4.py
    ecef_r = states_ecef[:3, :] - radar.tx[0].ecef[:, None]

    local_pos = sorts.frames.ecef_to_enu(
        radar.tx[0].lat, 
        radar.tx[0].lon, 
        radar.tx[0].alt, 
        ecef_r,
        degrees=True,
    )

    pth = local_pos/np.linalg.norm(local_pos, axis=0)
    G_pth = radar.tx[0].beam.gain(pth)  # $ENV/site-packages/pyant/beams/eiscat_uhf.py
    G_pth = np.squeeze(G_pth)
    G_pth_db = 10*np.log10(G_pth)

    antilogsnr = 10**(data.snr/10)

    rcsvalue = snr2rcs(
        G_pth,
        G_pth,
        radar.tx[0].wavelength,
        radar.tx[0].power,
        data.range,  # in meters, 1way
        data.range,  # in meters, 1way
        antilogsnr,  # Not in dB
        bandwidth=radar.tx[0].coh_int_bandwidth,
        rx_noise_temp=radar.rx[0].noise,
        radar_albedo=1.0,
    )

    diam = sorts.signals.hard_target_diameter(
        G_pth,
        G_pth,
        radar.tx[0].wavelength,
        radar.tx[0].power,
        data.range,  # in meters, 1way
        data.range,  # in meters, 1way
        antilogsnr,  # Not in dB
        bandwidth=radar.tx[0].coh_int_bandwidth,
        rx_noise_temp=radar.rx[0].noise,
        radar_albedo=1.0,
    )

    SNR_sim = sorts.signals.hard_target_snr(
        G_pth,
        G_pth,
        radar.tx[0].wavelength,
        radar.tx[0].power,
        data.range,  # in meters, 1way
        data.range,  # in meters, 1way
        diameter=1,
        bandwidth=radar.tx[0].coh_int_bandwidth,
        rx_noise_temp=radar.rx[0].noise,
        radar_albedo=1.0,
    )

    low_gain_inds = G_pth_db < min_gain
    diam[low_gain_inds] = np.nan
    SNR_sim[low_gain_inds] = 0
    rcsvalue[low_gain_inds] = np.nan

    return SNR_sim, G_pth, diam, low_gain_inds, ecef_r, pth, rcsvalue


def updated_tle(line1, line2, data, radar):
    """
    Update TLE information of objects in catalogue
    """

    line1 = (line1.astype(str))[0]
    line2 = (line2.astype(str))[0]

    r = (data.range) * 2  # in m, 2way approach used in Daniel's program
    v = (data.range_rate) * 2  # in m/s, 2way approach used in Daniel's program

    times = Time(data.epoch + data.t, format="unix", scale="utc")  # in unix
    rx_ecef, tx_ecef = utle.load_radar_ecef(radar)

    new_tle = utle.update_tle(line1, line2, times, r, v, rx_ecef, tx_ecef)

    return new_tle


def plot_estimator_results(
        data, 
        norad, 
        radar, 
        t_jitter, 
        matches, 
        bmae, 
        pmae, 
        SNR_sim, 
        snridmax, 
        ecef_r, 
        r, 
        diam, 
        pth, 
        fileformat, 
        results_folder):
    """
    Plot parameters that resulted from RCS_ESTIMATOR
    """

    alog = 10**(data.snr/10)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(pth[0, :], pth[1, :], '-w')
    pyant.plotting.gain_heatmap(radar.tx[0].beam, min_elevation=85.0, ax=ax)
    fig.savefig(results_folder / f'correlated_pass_pth_gain.{fileformat}')
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    axes[0, 0].plot(t_jitter, matches, '.k')
    axes[0, 0].plot(t_jitter, pmae, 'k')
    if bmae == bmae:
        axes[0, 0].plot(t_jitter[bmae], pmae[bmae], 'or')
    axes[0, 0].set_xlabel('Jitter [s]')
    axes[0, 0].set_ylabel('Distance [1]')
    axes[0, 1].plot(data.t, np.log10(diam*1e2))
    axes[0, 1].set_ylabel('Diameter [log10(cm)]')
    interval = 0.2
    d_peak = diam[snridmax]*1e2
    if not (np.isnan(d_peak) or np.isinf(d_peak)):
        logd_peak = np.log10(d_peak)
        axes[0, 1].set_ylim(logd_peak*(1 - interval), logd_peak*(1 + interval))
    axes[1, 0].plot(data.t, np.linalg.norm(ecef_r, axis=0)/1e3)
    axes[1, 0].plot(data.t, data.range/1e3)
    axes[1, 0].plot(data.t[snridmax], r/1e3, 'or')
    axes[1, 0].set_ylabel('range [km]')
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 1].plot(data.t, 10*np.log10(SNR_sim/np.max(SNR_sim)), label='Estimated')
    axes[1, 1].plot(data.t, 10*np.log10(alog/np.nanmax(alog)), 'x', label='Measured')
    axes[1, 1].plot(data.t, 10*np.log10(SNR_sim/np.max(SNR_sim)), '+b')
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Normalized SNR [dB]')
    title_date = results_folder.stem.split('_')
    mdate = Time(data.epoch, format='unix', scale='utc').iso
    fig.suptitle(f'{title_date[0].upper()} - {mdate}: NORAD-ID = {norad}')
    fig.savefig(results_folder / f'correlated_pass_snr_match.{fileformat}')
    plt.close(fig)


def save_estimator_results(
        discos_map, 
        norad, 
        SNR_sim, 
        diam_sim, 
        G_pth, 
        pth, 
        rcs_data, 
        data, 
        results_folder):
    """
    Save measured and simulated time series for each object with 
    their respective geometries and shapres retrieved from DISCOS
    """

    # discos object
    _catid = norad
    discos_tup = discos_map[_catid]
    (
        nameo, satno, oClass, mission, mass, 
        shape, width, height, depth, diam, 
        span, xsMax, xsMin, xsAvg
    ) = discos_tup

    summary_data = dict(
                SNR_sim = SNR_sim,
                diam_sim = diam_sim,
                gain_sim = G_pth,
                pth_sim = pth,
                # measurement_id = meas_id,
                # object_id = obj_id,
                # offset_angle = offset_angle,
                catid = satno,
                rcs_data = rcs_data,
                snr_data = data.snr,
                timearray = data.t,
                range = data.range,
                range_rate = data.range_rate,
                nameo = nameo,
                objectClass = oClass,
                mission = mission,
                mass = mass,
                shape = shape,
                width = width,
                height = height,
                depth = depth,
                diameter = diam,
                span = span,
                xSectMax = xsMax,
                xSectMin = xsMin,
                xSectAvg = xsAvg,
    )


    with open(results_folder / 'correlated_snr_prediction.pickle', 'wb') as fh:
        pickle.dump(summary_data, fh)


###################################################################
# RCS ESTIMATE
###################################################################

def rcs_estimator(
        radarid, 
        catalog,
        correlation_events, 
        correlation_data,
        output,
        token,
        jitter_width=1.5,
        min_gain=10.0,
        min_snr=5.0,
        fileformat='png',
        verbose=False):
    """
    Compute Radar Cross Section and estinate SNR and diameter from measured SNR 

    Params
    ------

    radarid: (str)
        identifier for radarsystem 
    catalog: (str)
        path to TLE file
    correlation_events: (str)
        path to directory with pickled detection files
    correlation_data: (str)
        path to directory with correlations
    token: (str)
        access token for Discos service
    output: (str)
        path to output directory

    """

    radar = getattr(sorts.radars, radarid)

    t_jitter = np.arange(
        -jitter_width, 
        jitter_width, 
        0.1, 
        dtype=np.float64,
    )

    output_pth = Path(output).resolve()
    output_pth.mkdir(exist_ok=True, parents=True)
    path2correvents = Path(correlation_events).resolve()
    path2corrdata = Path(correlation_data).resolve()
    tle_pth = Path(catalog).resolve()

    corr_events = list(sorted(path2correvents.rglob('*.pkl')))

    pop = sorts.population.tle_catalog(tle_pth, cartesian=False)
    pop.unique()

    for corr_event in corr_events:
        corrdata_filename = corr_event.name.split('.')[0] + '.h5'
        if verbose:
            print(path2corrdata / corrdata_filename)
            print(corr_event)

        with h5py.File(path2corrdata / corrdata_filename, 'r') as ds:
            indecies = ds['matched_object_index'][()][0, :]
            measurnment_id = ds['observation_index'][()]
            time_id = ds['matched_object_time'][()]
            selected = ds['correlated'][()]
            indecies = indecies[measurnment_id]  # sort according to detection
            time_id = time_id[measurnment_id]
            selected = selected[measurnment_id]
            measurnment_id = np.arange(len(measurnment_id))

        with open(str(corr_event), 'rb') as fip:
            h_det = pickle.load(fip)

        # Create a list of catids
        norads = []
        for events in h_det.events:
            data = events
            eepoch = events.epoch
            snrs = events.snr
            snridmax = np.nanargmax(snrs)
            ts = eepoch + events.t
            t = ts[snridmax]  # time at the max SNR  
            select_id = np.where(time_id == t)
            select_id = select_id[0]
            obj_id = indecies[select_id][0]
            norads.append(pop.data['oid'][obj_id])

        norads = np.array(norads)
        catids = norads[selected]  # succesful correlations. Distance < thrreshold
        number = np.ceil(len(catids)/100)

        discos_map = {}
        for i in range(int(number)):
            if i == (number - 1):
                dictm = get_discos_objects(catids[i*100:len(catids)], token)
            else:
                dictm = get_discos_objects(catids[i*100:i*100+100], token)
            discos_map.update(dictm)

        pbar = tqdm(
            total=len(catids),
            position=0,
            desc='Predicting events',
        )

        for jj, events in enumerate(h_det.events):
            if selected[jj]:
                data = events
                eepoch = events.epoch
                snrs = events.snr
                snridmax = np.nanargmax(snrs)
                ts = eepoch + events.t
                rs = events.range
                t = ts[snridmax]  # time at the max SNR
                r = rs[snridmax]  # in m, one way
                az = events.pointing[0][0]  # assuming that all values are the same
                ele = events.pointing[0][1]

                select_id = np.where(time_id == t)
                select_id = select_id[0]

                meas_id = measurnment_id[select_id][0]
                obj_id = indecies[select_id][0]
                toutc = dt.datetime.fromtimestamp(eepoch, dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
                ename = radarid + '_' + toutc

                obj = pop.get_object(obj_id)
                norad = pop.data['oid'][obj_id]

                if verbose:
                    print('Correlated TLE object for :')
                    print(f' - measurement_id: {meas_id}')
                    print(f' - object_id: {obj_id}')
                    print(f' - CAT ID: {norad}')
                    print(f' - event: {ename}')

                if norad < 81000:
                    results_folder = output_pth / ename
                    results_folder.mkdir(exist_ok=True)

                    radar.tx[0].beam.sph_point(azimuth = az, elevation = ele)

                    # ==================
                    # TLE update # The update does not produce simulations
                    # ==================
                    # line1 = pop.data[np.where(pop.data['oid'] == norad)]['line1']
                    # line2 = pop.data[np.where(pop.data['oid'] == norad)]['line2']
                    # print(line1)
                    # print(line2)
                    # new_tle = updated_tle(line1,line2,data,radarid) # corr_event or data
                    # print(new_tle)
                    # new_pop = sorts.population.tle_catalog(
                    #     [(new_tle[0],new_tle[1])], cartesian=False
                    # )
                    # obj = new_pop.get_object(0)
                    # ==================

                    obj.out_frame = 'ITRS'

                    t = Time(t, format='unix', scale='utc')
                    delta_t = (t - obj.epoch).sec  # unix time - mjd. format. Use astropy
                    t_vec = data.t - data.t[snridmax]

                    matches = np.empty_like(t_jitter)
                    pdatas = [None]*len(t_jitter)
                    pmetas = [None]*len(t_jitter)
                    pmae = [None]*len(t_jitter)

                    for tind in range(len(t_jitter)):
                        _t = t_vec + delta_t + t_jitter[tind]
                        pdatas[tind] = generate_prediction(data, _t, obj, radar, min_gain)
                        SNR_sim, G_pth, diam, LG_ind, ecef_r, pth, rcs_data = pdatas[tind]

                        nrows, ncols = np.shape(pth)
                        pths_off_angle = pyant.coordinates.vector_angle(
                            np.repeat(radar.tx[0].beam.pointing, ncols, axis=1),
                            pth,
                            degrees=True
                        )

                        matches[tind], pmetas[tind], pmae[tind] = matching_function( 
                                data, SNR_sim, pths_off_angle, min_snr)

                    # best_match = np.nanargmin(matches)
                    if np.isnan(pmae).all():
                        print(f'All NaN in event: {ename}')
                    else:
                        best_mae = np.nanargmin(pmae)
                        if verbose:
                            print(best_mae)

                        SNR_sim, G_pth, diam, LG_ind, ecef_r, pth, rcs_data = pdatas[best_mae]

                        # Currently, the lines commented below are not used
                        # offset_angle = pyant.coordinates.vector_angle(pth[:, snridmax],
                        #                radar.tx[0].beam.pointing,
                        #                degrees=True
                        # )

                        plot_estimator_results(
                            data, norad, radar, t_jitter, matches,
                            best_mae, pmae, SNR_sim, snridmax, ecef_r, r, diam, pth,
                            fileformat, results_folder
                        )

                        save_estimator_results(
                            discos_map, norad, SNR_sim, diam, G_pth, pth, 
                            rcs_data, data, results_folder
                        )

                else:
                    print('At the time of the TLE (date), the object was to be assigned')
            pbar.update(1)
        pbar.close()


###################################################################
# CLI
###################################################################

def main(input_args=None):
    parser = argparse.ArgumentParser(description='Estimate RCS \
        values for target using correlation and TLE')

    parser.add_argument('radar', type=str, help='The observing radar system')
    parser.add_argument('catalog', type=str, help='TLE catalog path')
    parser.add_argument('correlation_events', type=str, help='Path to \
        correlated events')
    parser.add_argument('correlation_data', type=str, help='Path to \
        correlation data')
    parser.add_argument('output', type=str, help='Results output location')

    parser.add_argument(
        '--event',
        default='',
        help='To pick a specific event name.',
    )
    parser.add_argument(
        '--min-gain',
        default=10.0,
        metavar='MIN',
        type=float, 
        help='The minimum amount of dB one way gain at which to evalute \
        SNR and RCS',
    )
    parser.add_argument(
        '--min-snr',
        default=5.0,
        metavar='MIN',
        type=float, 
        help='The minimum amount of measured SNR dB at which to evalute \
        SNR and RCS',
    )
    parser.add_argument(
        '--jitter-width',
        default=1.5,  # it was 5.0
        metavar='JITTER',
        type=float, 
        help='The number of seconds to jitter back and forth',
    )
    parser.add_argument(
        '-v',
        action='store_true',
        help='Verbose output',
    )
    parser.add_argument(
        '-f', '--format',
        default='png',
        help='Plot format',
    )

    credentials = 'discos_credentials.txt'
    proc = subprocess.Popen("sed -n '1p' "+credentials, stdout=subprocess.PIPE, shell=True)
    token = proc.stdout.read()
    token = token.strip().decode("utf-8")

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    # rcs estimate
    rcs_estimator(
        args.radar,
        args.catalog,
        args.correlation_events,
        args.correlation_data,
        args.output,
        token,
        jitter_width=args.jitter_width,
        min_gain=args.min_gain,
        min_snr=args.min_snr,
        fileformat=args.format,
        verbose=args.v
    )


###################################################################
# MAIN
###################################################################

if __name__ == '__main__':
    main()
