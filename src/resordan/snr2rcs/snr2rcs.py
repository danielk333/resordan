
import argparse
from pathlib import Path
import sys
import h5py
import re
import datetime as dt
import tempfile
import shutil
import subprocess
from resordan.clustering import algorithm
from resordan.data.gmf import GMFDataset
from resordan.data.events import EventsDataset
from resordan.correlator.beam_rcs_predict import main_predict as rcs_predict
from resordan.correlator.space_track_download import fetch_tle
#import spacetrack

ISO_FMT = '%Y-%m-%dT%H:%M:%S'
TMP = Path("/cluster/work/users/inar/usecase/tmp")


def str_to_bool(value):
    """convert string to bool"""
    if value.lower() in ['true', '1', 'yes']:
        return True
    elif value.lower() in ['false', '0', 'no']:
        return False
    else:
        raise ValueError(f"Cannot convert {value} to boolean")


def get_value(config, section, option):
    """read value from config, support None values"""
    value = config.get(section, option, fallback=None)
    return None if value.strip() == '' else value


CLUSTER_PARAM_DEFAULTS = dict(
    loss_weights=(1e-3, 1e-3),
    segment_split_time=1.5,
    snr_db_threshold=20,
    loss_threshold=10,
)

CORRELATE_PARAM_DEFAULTS = dict(
    std = False,
    jitter = False,
    range_rate_scaling = 0.2,
    range_scaling = 0.1,
    save_states = False,
    target_epoch = None
)

PREDICT_PARAM_DEFAULTS = dict(
    min_gain=10.0,
    min_snr=5.0,
    jitter_width=1.5,
    format='png'
)


def snr2rcs(gmf, cfg, verbose=False, clobber=False):

    ###########################################
    # CREDENTIALS
    ###########################################

    if not cfg.has_option('SPACETRACK', 'user'):
        raise Exception("Missing Credentials: SPACETRACK user")
    if not cfg.has_option('SPACETRACK', 'passwd'):
        raise Exception("Missing Credentials: SPACETRACK passwd")
    if not cfg.has_option('DISCOS', 'token'):
        raise Exception("Missing Credentials: DISCOS token")

    st_user = cfg.get('SPACETRACK', 'user')
    st_passwd = cfg.get('SPACETRACK', 'passwd')
    discos_token = cfg.get('DISCOS', 'token')

    ###########################################
    # SETUP
    ###########################################
    gmf = Path(gmf)
    if not gmf.is_dir():
        raise Exception(f"GMF product is not directory: {gmf}")

    # make temporary folder
    #temp_dir = Path(tempfile.mkdtemp())
    tmp_dir = TMP
    events_file = tmp_dir / "events.pkl"
    tle_file = tmp_dir / "tle.txt"
    correlations_file = tmp_dir / "events.h5"
    rcs_dir = tmp_dir / "rcs"

    ###########################################
    # CLUSTERING
    ###########################################
    
    if not events_file.exists():
        gmf_files = list(sorted([file for file in gmf.rglob('*.h5') if file.is_file()]))
        gmf_dataset = GMFDataset.from_files(gmf_files)

        CLUSTER_PARAMS = {**CLUSTER_PARAM_DEFAULTS}
        for key in CLUSTER_PARAM_DEFAULTS:
            if cfg.has_option('CLUSTER', key):
                CLUSTER_PARAMS[key] = eval(get_value(cfg, 'CLUSTER', key))
        
        if verbose:
            print('CLUSTERING:')
        events_dataset = algorithm.snr_peaks_detection(gmf_dataset, **CLUSTER_PARAMS)
        EventsDataset.to_pickle(events_dataset, events_file)
        if verbose:
            print(f"{len(events_dataset.events)} detections")

    ###########################################
    # SPACETRACK TLE DOWNLOAD
    ###########################################

    if not tle_file.exists():

        # get timestamp for start of gmf product
        with h5py.File(gmf_files[0], "r") as f:        
            epoch = float(f['epoch_unix'][()])
            epoch_dt = dt.datetime.utcfromtimestamp(epoch)

        if verbose:
            print("TLE DOWNLOAD:")
        lines = fetch_tle(epoch_dt, st_user, st_passwd)

        if verbose:
            print(f"{len(lines)} lines")
        # write to file
        with open(tle_file, 'w') as file:
            for line in lines:
                file.write(line + "\n")

    ###########################################
    # CORRELATE
    ###########################################

    if not correlations_file.exists():

        CORRELATE_PARAMS = {**CORRELATE_PARAM_DEFAULTS}
        for key in CORRELATE_PARAM_DEFAULTS:
            if not cfg.has_option('CORRELATE', key):
                continue
            if key in ['jitter', 'std', 'save_states']:
                CORRELATE_PARAMS[key] = str_to_bool(get_value(cfg, 'CORRELATE', key))
            else:
                CORRELATE_PARAMS[key] = get_value(cfg, 'CORRELATE', key)

        args = [
            "rcorrelate", "eiscat_uhf",
            str(tle_file),
            str(events_file),
            str(correlations_file),
        ]
        if CORRELATE_PARAMS['std']:
            args.append("--std")
        if CORRELATE_PARAMS['jitter']:
            args.append("--jitter")
        if CORRELATE_PARAMS['save_states']:
            args.append("--save-states")
        if clobber:
            args.append("--clobber")
        args.extend(['--range-rate-scaling', str(CORRELATE_PARAMS['range_rate_scaling'])])
        args.extend(['--range-scaling', str(CORRELATE_PARAMS['range_scaling'])])

        if verbose:
            print("CORRELATE:")
    
        proc = subprocess.Popen(args, stdout=sys.stdout, stderr=sys.stderr, text=True)
        # Wait for the process to complete and get the output
        stdout, stderr = proc.communicate()

    ###########################################
    # PREDICT
    ###########################################

    PREDICT_PARAMS = {**PREDICT_PARAM_DEFAULTS}
    for key in PREDICT_PARAM_DEFAULTS:
        if not cfg.has_option('PREDICT', key):
            continue
        if key in ['min_gain', 'min_snr', 'jitter_width']:
            PREDICT_PARAMS[key] = cfg.getfloat('PREDICT', key)
        if key in ['format']:
            PREDICT_PARAMS[key] = get_value(cfg, 'PREDICT', key)


    print(PREDICT_PARAMS)
    args = argparse.Namespace(
        radar='eiscat_uhf',
        catalog=str(tle_file),
        correlation_events=str(events_file.parent),
        correlation_data=str(correlations_file.parent),
        output=str(rcs_dir),
        min_gain= PREDICT_PARAMS['min_gain'],
        min_snr= PREDICT_PARAMS['min_snr'],
        jitter_width= PREDICT_PARAMS['jitter_width'],
        v=verbose,
        format=PREDICT_PARAMS['format']
    )
    print("PREDICT:")
    print(args)
    rcs_predict(args, discos_token)




    # find gmf files
 
    #shutil.rmtree(tmp_dir)
