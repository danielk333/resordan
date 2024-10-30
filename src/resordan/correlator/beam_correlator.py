#!/usr/bin/env python

import argparse
import pathlib
import pickle
import numpy as np
import h5py
from astropy.time import Time
import sorts

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats as st

"""
Correlation analysis between radar measurements and a catalogue of know objetcs.
This analysis attemps to attribute measurements to objects from a given catalogue.
This annalysis use a bi-modal distributin approach to select correlated objects.

Example:
"python beam_correlator.py eiscat_uhf \t #
    ~/path/to/spacetrack.tles \t # catalogue. Data 24h prior the experiment 
    ~/path/to/leo_events.h5 \t # input
    ~/path/to/correlated_data.h5 -c # output"

"mpirun -n 6 ./beam_correlator.py eiscat_esr ~/data/{spacetrack.tles,leo.h5,correlation.h5} -c"
"""

# dtype for residuals
res_t = np.dtype([
                    ('dr', np.float64),
                    ('dv', np.float64),
                    ('metric', np.float64),
                    ('jitter_index', np.float64)
                ])

t_jitter = np.linspace(-5, 5, num=11)

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None


def default_propagation_handling(obj, t, t_measurement_indices, measurements):
    """
    Get states for time t
    """
    states = obj.get_state(t)
    return states


def jitter_propagation_handling(obj, t, t_measurement_indices, measurements):
    """
    Get states for t_jitter
    """
    t_get = t[:, None] + t_jitter[None, :]

    ret_shape = t_get.shape
    t_get.shape = (t_get.size, )

    states = obj.get_state(t_get)
    states.shape = (6, ) + ret_shape

    return states


def vector_diff_metric(t, r, v, r_ref, v_ref, **kwargs):
    """
    eq 1 of Kastinen et al. 2023, Acta Atronautica
    Return a vector of residuals for range and range rate
    """
    r_std = kwargs.get('r_std', None)
    v_std = kwargs.get('v_std', None)

    index_tuple = (slice(None), ) + tuple(None for x in range(len(r_ref.shape) - 1))

    base_shape = r_ref.shape
    ret = np.empty(base_shape, dtype=res_t)

    ret['dr'] = r_ref - r[index_tuple]
    ret['dv'] = v_ref - v[index_tuple]
    if r_std is not None:
        ret['dr'] /= r_std[index_tuple]
        ret['dv'] /= v_std[index_tuple]

    ret['metric'] = np.hypot(
        ret['dr']/kwargs['dr_scale'],
        ret['dv']/kwargs['dv_scale'],
    )

    # Reduce jitter if it exists
    if len(base_shape) > 1:
        ret.shape = (base_shape[0], np.prod(base_shape[1:]))
        inds = np.argmin(ret['metric'], axis=1)
        ret = ret[np.arange(base_shape[0]), inds]
        ret['jitter_index'] = inds
    else:
        ret['jitter_index'] = np.nan

    return ret


def save_correlated_data(out_pth,
                         indices,
                         metric, 
                         correlated_data, 
                         measurements, 
                         select, 
                         not_select,
                         meta=None,
                         save_states=False):

    print(f'Saving correlation data to {out_pth}')
    out_pth.parent.mkdir(exist_ok=True, parents=True)
    with h5py.File(out_pth, 'w') as ds:
        match_index = np.arange(indices.shape[0])
        observation_index = np.arange(indices.shape[1])

        # Create global attributes for dataset
        if meta is not None:
            for key in meta:
                ds.attrs[key] = meta[key]

        ds['object_index'] = np.array(list(correlated_data.keys()))
        ds_obj_ind = ds['object_index']
        ds_obj_ind.make_scale('object_index')
        ds_obj_ind.attrs['long_name'] = 'Object index in the used population'

        ds['correlated'] = select  # this
        ds_selected = ds['correlated']
        ds_selected.make_scale('correlated')
        ds_selected.attrs['long_name'] = 'correlated objects of the population, select'

        ds['uncorrelated'] = not_select
        ds_not_selected = ds['uncorrelated']
        ds_not_selected.make_scale('uncorrelated')
        ds_not_selected.attrs['long_name'] = 'uncorrelated objects of the population, no selected'

        cartesian_pos = ds.create_dataset(
            "cartesian_pos_axis",
            data=[s.encode() for s in ["x", "y", "z"]],
        )
        cartesian_pos.make_scale('cartesian_pos_axis')
        cartesian_pos.attrs['long_name'] = 'Cartesian position axis names'

        cartesian_vel = ds.create_dataset(
            "cartesian_vel_axis",
            data=[s.encode() for s in ["vx", "vy", "vz"]],
        )
        cartesian_vel.make_scale('cartesian_vel_axis')
        cartesian_vel.attrs['long_name'] = 'Cartesian velocity axis names'

        ds['maching_rank'] = match_index
        ds_mch_ind = ds['maching_rank']
        ds_mch_ind.make_scale('maching_rank')
        mrln = 'Matching rank numbering from best, lowest, to worst, hightest rank'
        ds_mch_ind.attrs['long_name'] = mrln

        ds['observation_index'] = observation_index  # this
        ds_obs_ind = ds['observation_index']
        ds_obs_ind.make_scale('observation_index')
        ds_obs_ind.attrs['long_name'] = 'Observation index in the input radar data'

        def _create_ia_var(base, name, long_name, data, scales, units=None):
            base[name] = data.copy()
            var = base[name]
            for ind in range(len(scales)):
                var.dims[ind].attach_scale(scales[ind])
            var.attrs['long_name'] = long_name
            if units is not None:
                var.attrs['units'] = units

        scales = [ds_mch_ind, ds_obs_ind]

        _create_ia_var(
            ds, 
            'matched_object_index', 
            'Index of the correlated object',  
            # this
            indices, 
            scales
        )
        _create_ia_var(
            ds, 
            'matched_object_metric', 
            'Correlation metric for correlated object',
            metric, scales
        )
        _create_ia_var(
            ds, 
            'matched_object_time', 
            'Time of correlation', 
            # this
            measurements[0]['times'].unix, 
            [ds_obs_ind], 
            units='unix'
        )

        # We currently only supply one dat dict to the correlator
        msi = 0  # measurement set index

        def stacker(x, key):
            return np.stack([val[msi][key] for val in x.values()], axis=0)

        scales = [ds_obj_ind, ds_obs_ind]
        _create_ia_var(
            ds, 
            'simulated_range', 
            'Simulated range',
            stacker(correlated_data, 'r_ref'), 
            scales, 
            units='m'
        )
        _create_ia_var(
            ds, 
            'simulated_range_rate', 
            'Simulated range rate',
            stacker(correlated_data, 'v_ref'), 
            scales, 
            units='m/s'
        )
        _create_ia_var(
            ds, 
            'simulated_correlation_metric',
            'Calculated metric for simulated ITRS state',
            stacker(correlated_data, 'match'), 
            scales
        )

        if save_states:

            # TODO - this probably could be done once, and then the array could be 
            # copied if two independent indstancs are needed?
            def as_array(correlated_data):
                values = [val[msi]['states'][:3, ...].T for val in correlated_data.values()]
                return np.stack(values, axis=0)

            _create_ia_var(
                ds, 
                'simulated_position', 
                'Simulated ITRS positions', 
                as_array(correlated_data),
                scales + [cartesian_pos],
                units='m',
            )
            _create_ia_var(
                ds, 
                'simulated_velocity', 
                'Simulated ITRS velocities', 
                as_array(correlated_data),
                scales + [cartesian_vel],
                units='m/s',
            )


def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    """
    Computes a bimodal distribution
    """
    return st.norm.pdf(x, mu1, sigma1)*np.abs(A1) + st.norm.pdf(x, mu2, sigma2)*np.abs(A2)


def draw_ellipse(x_size, y_size, ax, res=100, style='-r', log=False):

    th = np.linspace(0, np.pi*2, res)
    ex = np.cos(th)*x_size
    ey = np.sin(th)*y_size

    if log:
        ex = np.log10(np.abs(ex))
        ey = np.log10(np.abs(ey))

    ax.plot(ex, ey, style)
    return ax


def plot_analysis(
        name,
        metric,
        epoch,
        xp,
        yp,
        scale_x,
        scale_y, 
        threshold, 
        threshold_,
        threshold_est, 
        log10_elip_dst,
        r,
        t, 
        v, 
        select, 
        not_select, 
        out_path, 
        iformat,
        num, 
        bins,
        params):

    out_path.mkdir(exist_ok=True, parents=True)
    # Plotting 1
    fig, axes = plt.subplots(1, 2, figsize=(15, 15))
    title_str = f'{name.upper().replace("_", " ")} - {epoch.datetime.date()}'
    for ax in axes:
        ax.plot(xp, yp, '.b')
        ax.plot([0, 0], [yp.min(), yp.max()], '-r')
        ax.plot([xp.min(), xp.max()], [0, 0], '-r')
        draw_ellipse(scale_x*threshold_, scale_y*threshold_, ax)
        ax.set_xlabel('Range residuals [km]')
        ax.set_ylabel('Range-rate residuals [km/s]')
        ax.set_title(title_str)
    axes[1].set_xlim([-scale_x*threshold_, scale_x*threshold_])
    axes[1].set_ylim([-scale_y*threshold_, scale_y*threshold_])
    fig.savefig(out_path / f'{name}_residuals.{iformat}')
    plt.close(fig)

    # Plotting 2
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.hist(log10_elip_dst, num)
    if threshold is not None:
        ax.plot(
            [np.log10(threshold), np.log10(threshold)],
            ax.get_ylim(), '-g', label='Input threshold')
    if threshold_est is not None:
        ax.plot(
            [np.log10(threshold_est), np.log10(threshold_est)],
            ax.get_ylim(), '--g', label='Estimated threshold')
        ax.plot(
            np.linspace(bins[0], bins[-1], 1000),
            bimodal(np.linspace(bins[0], bins[-1], 1000), *params),
            '-r', label='Fit')
    ax.set_xlabel('Distance function [log10(1)]')
    ax.set_ylabel('Frequency [1]')
    ax.set_title(title_str)
    ax.legend()
    fig.savefig(out_path / f'{name}_ellipse_distance.{iformat}')
    plt.close(fig)

    # Plotting 3
    fig, axes = plt.subplots(2, 1, figsize=(15, 15))  # r starts here
    ax = axes[0]
    ax.plot(t[select]/3600.0, r[select], '.r', label='Correlated')
    ax.plot(t[not_select]/3600.0, r[not_select], '.b', label='Uncorrelated')
    ax.legend()
    ax.set_ylabel('Range [km]')
    ax.set_xlabel('Time [h]')
    ax.set_title(title_str)
    ax = axes[1]
    ax.plot(t[select]/3600.0, v[select], '.r')  # v starts here
    ax.plot(t[not_select]/3600.0, v[not_select], '.b')
    ax.set_ylabel('Range-rate [km/s]')
    ax.set_xlabel('Time [h]')
    fig.savefig(out_path / f'{name}_rv_t_correlations.{iformat}')
    plt.close(fig)

    # Plotting 4
    fig, ax = plt.subplots(1, 1, figsize=(15, 15)) 
    ax.plot(r[select], v[select], '.r', label='Correlated')
    ax.plot(r[not_select], v[not_select], '.b', label='Uncorrelated')
    ax.legend()
    ax.set_xlabel('Range [km]')
    ax.set_ylabel('Range-rate [km/s]')
    ax.set_title(title_str)
    fig.savefig(out_path / f'{name}_rv_correlations.{iformat}')
    plt.close(fig)

    # Plotting 5
    if 'jitter_index' in metric.dtype.names:
        ji = metric['jitter_index']
        if not np.all(np.isnan(ji)):
            fig, ax = plt.subplots(1, 1, figsize=(15, 15))
            ax.hist([ji.flatten()[select], ji.flatten()[not_select]], stacked=True)
            ax.set_xlabel('Jitter index [1]')
            ax.set_ylabel('Frequency [1]')
            ax.set_title(title_str)
            fig.savefig(out_path / f'{name}_jitter.{iformat}')
            plt.close(fig)


def population_analysis(
        name, 
        metric, 
        measurements, 
        out_path, 
        threshold=None,
        range_rate_scaling=None, 
        range_scaling=None, 
        format=None):

    """
    Analyse the two distinct populations generated by the correlation

    ---------------
    Treshold = 'Treshold in elliptical distance to choose correlations'
    format = 'saved figure format'
    """
    # setting basic parameters
    if range_scaling is None:
        scale_x = 1.0
    else:
        scale_x = range_scaling
    if range_rate_scaling is None:
        scale_y = 0.2
    else:
        scale_y = range_rate_scaling
    if threshold is not None:
        threshold *= 1e-3
    if format is None:
        format = 'png'

    # Loading metric data
    x = metric['dr']*1e-3
    y = metric['dv']*1e-3
    m = metric['metric']*1e-3
    inds = np.logical_not(np.logical_or(np.isnan(x), np.isnan(y)))
    xp = x[inds]
    yp = y[inds]

    log10_elip_dst = np.log10(m[np.logical_not(np.isnan(m))])
    num = int(np.round(np.sqrt(len(log10_elip_dst))))

    if num < 10:
        if threshold is None:
            print('treshold cannot be automatically determined: defaulting to 1')
            threshold = 1.0
        threshold_est = None
    else:
        count, bins = np.histogram(log10_elip_dst, num)
        bin_centers = (bins[1:] + bins[:-1])*0.5

        start = (
            0, 
            np.std(log10_elip_dst)*0.5, 
            np.max(count)*0.5, 
            np.max(bin_centers), 
            np.std(log10_elip_dst)*0.5, 
            np.max(count)*0.5
        )
        params, cov = curve_fit(bimodal, bin_centers, count, start)

        # this is intersection and is better estimate i think
        log_threshold_sample = np.linspace(params[0], params[3], 1000)
        log_threshold_intersection = np.argmin(bimodal(log_threshold_sample, *params))
        threshold_est = 10**log_threshold_sample[log_threshold_intersection]
        # threshold_est = 10**((params[0] + params[3])*0.5)

        print(f'ESTIMATED threshold: {threshold_est*1e3}')

    # Loading clustering data
    r = measurements[0]['r']/2
    v = measurements[0]['v']/2
    t = measurements[0]['t']
    epoch = measurements[0]['epoch']
    t = t - t.min()

    # selection
    if threshold is None:
        threshold_ = threshold_est
    else:
        threshold_ = threshold

    select = np.logical_and(
        m < threshold_,
        np.logical_not(np.isnan(m)),
    ).flatten()
    not_select = np.logical_not(select)

    if num < 10:
        plot_analysis(
            name, metric, epoch, xp, yp, scale_x, scale_y, threshold,
            threshold_, threshold_est, log10_elip_dst, r, t, v, select, 
            not_select, out_path, format, None, None, None
        )
    else:
        plot_analysis(
            name, metric, epoch, xp, yp, scale_x, scale_y, threshold,
            threshold_, threshold_est, log10_elip_dst, r, t, v, select,
            not_select, out_path, format, num, bins, params
        )

    return select, not_select


def radar_sd_correlator(
        radarid,
        catalog,
        clustered_events,
        output,
        clobber=False,
        stdev=False,
        jitter=False,
        savestates=False,
        rangeratescaling=0.2,
        rangescaling=1.0,
        targetepoch=None):

    """
    Compute Radar Cross Section and estinate SNR and diameter from measured SNR 

    Params
    ------

    radarid: (str)
        identifier for radarsystem 
    catalog: (str)
        path to TLE file
    clustered_events: (str)
        path to directory with pickled detection files
    output: (str)
        path to output directory

    """

    if comm is not None:
        print('MPI detected')
        if comm.size > 1:
            print('Using MPI...')

    s_dr = rangeratescaling
    s_r = rangescaling

    radar = getattr(sorts.radars, radarid)

    tle_pth = pathlib.Path(catalog).resolve()
    output_pth = pathlib.Path(output).resolve()
    input_pth = pathlib.Path(clustered_events).resolve()

    if stdev:
        meta_vars = ['r_std', 'v_std']
    else:
        meta_vars = []

    if jitter:
        propagation_handling = jitter_propagation_handling
    else:
        propagation_handling = default_propagation_handling

    meta_vars.append('dr_scale')
    meta_vars.append('dv_scale')

    input_files = input_pth.glob('*.pkl')
    input_files = sorted(input_files)

    for in_file in input_files:
        print('Loading monostatic measurements')
        h5_file = (in_file.name.split('.'))[0] + '.h5'
        output_file = output_pth / h5_file
        measurements = []
        with open(str(in_file), 'rb') as filer:
            h_det = pickle.load(filer)
            t = []  # unix time
            r = []  # meters
            v = []  # meters/s
            for event in h_det.events:
                epoch = event.epoch
                snr = event.snr
                snrid = np.argmax(snr)
                t.append(epoch + event.t[snrid])
                r.append(event.range[snrid])
                v.append(event.range_rate[snrid])

            t = np.array(t)
            r = np.array(r)
            v = np.array(v)

            inds = np.argsort(t)
            t = t[inds]
            r = r[inds]
            v = v[inds]

            times = Time(t, format='unix', scale='utc')
            epoch = times[0]
            t = (times - epoch).sec

            # #######################
            # This is the interface layer between the input
            # data type and the correlator
            # #######################
            dat = {
                'r': r*2,  # two way
                'v': v*2,  # two way
                't': t,
                'times': times,
                'epoch': epoch,
                'tx': radar.tx[0],
                'rx': radar.rx[0],
                'measurement_num': len(t),
                'dr_scale': s_r,
                'dv_scale': s_dr,
            }

            if stdev:
                dat['r_std'] = h_det['r_std'][()]
                dat['v_std'] = h_det['v_std'][()]

            measurements.append(dat)

        print('Loading TLE population')
        pop = sorts.population.tle_catalog(tle_pth, cartesian=False)
        if targetepoch is not None:
            targetepoch = Time(targetepoch, format='iso', scale='utc').mjd

        # choosing TLEs close to the observation window
        pop.unique(target_epoch=targetepoch)

        # correlate requires output in ECEF
        pop.out_frame = 'ITRS'
        print(f'Population size: {len(pop)}')
        print('Correlating measurements and population')

        if output_file.is_file() and not clobber:
            print('Loading correlation data from cache')
            with h5py.File(output_file, "r") as hf:
                metric = hf['matched_object_metric']
                indices = hf['matched_object_index']
                cdat = hf['object_index']
                metric = metric[:]
                indices = indices[:]
                cdat = cdat[:]
        else:

            if comm is not None:
                comm.barrier()

            MPI = comm is not None and comm.size > 1

            # Correlate all objects in the catalogue
            indices, metric, correlated_data = sorts.correlate(
                measurements = measurements,
                population = pop,
                n_closest = 1,  # Number of closest matches to save
                meta_variables = meta_vars,
                metric = vector_diff_metric,
                sorting_function = lambda x: np.argsort(x['metric'], axis=0),
                metric_dtype = res_t,
                metric_reduce = None,
                scalar_metric = False,
                propagation_handling = propagation_handling,
                MPI = MPI,
                save_states = savestates,
            )

            # Bi-modal distribution approach to select correlated objects
            select, not_select = population_analysis( 
                radarid, 
                metric, 
                measurements, 
                output_pth,
                threshold=None, 
                range_rate_scaling=None, 
                range_scaling=None, 
                format=None)

            # Save data
            if comm is None or comm.rank == 0:
                save_correlated_data(
                    output_file,
                    indices,
                    metric,
                    correlated_data,
                    measurements,
                    select,
                    not_select,
                    meta = dict(
                        radar_name = radarid,
                        tx_lat = radar.tx[0].lat,
                        tx_lon = radar.tx[0].lon,
                        tx_alt = radar.tx[0].alt,
                        rx_lat = radar.rx[0].lat,
                        rx_lon = radar.rx[0].lon,
                        rx_alt = radar.rx[0].alt,
                        range_scaling = s_r,
                        range_rate_scaling = s_dr,
                    ),
                    save_states = savestates,
                )

        if comm is None or comm.rank == 0:
            print('Individual measurement match metric:')
            for mind, (ind, dst) in enumerate(zip(indices.T, metric.T)):
                print(f'measurement = {mind}')
                for res in range(len(ind)):
                    msg = (
                        f'-- result rank = {res}',
                        f' | object ind = {ind[res]}',
                        f' | metric = {dst[res]}',
                        f' | obj = {pop["oid"][ind[res]]}'
                    )
                    print(msg)


def main(input_args=None):

    parser = argparse.ArgumentParser(description='Calculate TLE catalog correlation for a beampark')
    parser.add_argument(
        'radar', 
        type=str, 
        help='The observing radar system')
    parser.add_argument(
        'catalog', 
        type=str, 
        help='TLE catalog path')
    parser.add_argument(
        'input', 
        type=str, 
        help='Observation data folder/file')
    parser.add_argument(
        'output', 
        type=str, 
        help='Results output location')
    parser.add_argument(
        '-c', '--clobber', 
        action='store_true', 
        help='Override output location if it exists')
    parser.add_argument(
        '--stdev', 
        action='store_true', 
        help='Use measurement errors')
    parser.add_argument(
        '--jitter', 
        action='store_true', 
        help='Use time jitter')
    parser.add_argument(
        '--range-rate-scaling', 
        default=0.2, type=float, 
        help='Scaling used on range rate in the sorting function of the correlator')
    parser.add_argument(
        '--range-scaling', 
        default=1.0, type=float, 
        help='Scaling used on range in the sorting function of the correlator')
    parser.add_argument(
        '--save-states', 
        action='store_true', 
        help='Save simulated states')
    parser.add_argument(
        '--target-epoch', 
        type=str, default=None, 
        help='When filtering unique TLEs use this target epoch [ISO]')

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    radar_sd_correlator(
        args.radar,
        args.catalog,
        args.input,
        args.output,
        stdev=args.stdev,
        jitter=args.jitter,
        savestates=args.save_states,
        clobber=args.clobber,
        rangeratescaling= args.range_rate_scaling,
        rangescaling=args.range_scaling,
        targetepoch=args.target_epoch
    )


if __name__ == '__main__':
    main()
