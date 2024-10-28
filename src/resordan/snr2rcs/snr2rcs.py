from pathlib import Path
import h5py
import datetime as dt
import tempfile
import shutil
from resordan.clustering import algorithm
from resordan.data.gmf import GMFDataset
from resordan.data.events import EventsDataset
from resordan.correlator.beam_rcs_estimator import rcs_estimator
from resordan.correlator.space_track_download import fetch_tle
from resordan.correlator.beam_correlator import radar_sd_correlator

ISO_FMT = '%Y-%m-%dT%H:%M:%S'

###############################################################
# CONFIG
###############################################################


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
    stdev = False,
    jitter = False,
    range_rate_scaling = 0.2,
    range_scaling = 1.0,
    save_states = False,
    target_epoch = None
)

PREDICT_PARAM_DEFAULTS = dict(
    min_gain=10.0,
    min_snr=5.0,
    jitter_width=1.5,
    format='png'
)


###############################################################
# SNR2RCS
###############################################################

def snr2rcs(src, cfg, dst, tmp=None, verbose=False, clobber=False, cleanup=False):

    """
    For a given GMF product, does clustering, correlation and rcs prediction

    Manages files in temporary directory, unless

    Output RCS results

    Params
    ------
        src: (str)
            path to GMF product
        cfg: (configparser.ConfigParser)
            config object with credentials and processing parameters
        dst: (str)
            Path to directory with RCS results
        verbose: (bool)
            If true, print to screen
        clobber: (bool)
            If products already exists overwrite if clobber is True, else reuse
        tmp: (str) (optional)
            Path to temporary directory. Automatically created if not specified.
        cleanup: (bool)
            If true, cleanup temporary directory with intermedia results on termination

    Results
    -------
        (str) 
            path to RCS directory

    """

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
    src = Path(src)
    if not src.is_dir():
        raise Exception(f"GMF product is not directory: {src}")

    dst = Path(dst)
    if not dst.exists():
        raise Exception(f"Out directory does not exists {dst}")

    # make temporary folder
    if tmp is None:
        tmp = tempfile.mkdtemp()
    tmp = Path(tmp)
    if not tmp.is_dir():
        raise Exception(f"tmp is not a directory: {tmp}")        

    tle_file = tmp / "tle.txt"
    events_file = tmp / "events.pkl"
    correlations_dir = tmp

    ###########################################
    # CLUSTERING
    ###########################################

    if verbose:
        print('CLUSTERING:')

    gmf_files = list(sorted([file for file in src.rglob('*.h5') if file.is_file()]))
    if not events_file.exists() or clobber:
        #gmf_files = list(sorted([file for file in src.rglob('*.h5') if file.is_file()]))
        gmf_dataset = GMFDataset.from_files(gmf_files)

        CLUSTER_PARAMS = {**CLUSTER_PARAM_DEFAULTS}
        for key in CLUSTER_PARAM_DEFAULTS:
            if cfg.has_option('CLUSTER', key):
                CLUSTER_PARAMS[key] = eval(get_value(cfg, 'CLUSTER', key))
        
        events_dataset = algorithm.snr_peaks_detection(gmf_dataset, **CLUSTER_PARAMS)
        EventsDataset.to_pickle(events_dataset, events_file)
        if verbose:
            print(f"{len(events_dataset.events)} detections")

    ###########################################
    # SPACETRACK TLE DOWNLOAD
    ###########################################
    if verbose:
        print("TLE DOWNLOAD:")

    if not tle_file.exists() or clobber:

        # get timestamp for start of gmf product
        with h5py.File(gmf_files[0], "r") as f:        
            epoch = float(f['epoch_unix'][()])
            epoch_dt = dt.datetime.fromtimestamp(epoch, dt.UTC)

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

    if verbose:
        print("CORRELATE:")

    CORRELATE_PARAMS = {**CORRELATE_PARAM_DEFAULTS}
    for key in CORRELATE_PARAM_DEFAULTS:
        if not cfg.has_option('CORRELATE', key):
            continue
        if key in ['jitter', 'stdev', 'save_states']:
            CORRELATE_PARAMS[key] = str_to_bool(get_value(cfg, 'CORRELATE', key))
        elif key in ['range_scaling', 'range_rate_scaling']:
            CORRELATE_PARAMS[key] = cfg.getfloat('CORRELATE', key)
        else:
            CORRELATE_PARAMS[key] = get_value(cfg, 'CORRELATE', key)

    radar_sd_correlator(
        "eiscat_uhf",
        str(tle_file),
        str(events_file.parent),
        str(correlations_dir),
        stdev=CORRELATE_PARAMS['stdev'],
        jitter=CORRELATE_PARAMS['jitter'],
        savestates=CORRELATE_PARAMS['save_states'],
        clobber=clobber,
        rangeratescaling=CORRELATE_PARAMS['range_rate_scaling'],
        rangescaling=CORRELATE_PARAMS['range_scaling'],
        targetepoch=CORRELATE_PARAMS['target_epoch']
    )

    ###########################################
    # PREDICT
    ###########################################
    if verbose:
        print("PREDICT:")

    PREDICT_PARAMS = {**PREDICT_PARAM_DEFAULTS}
    for key in PREDICT_PARAM_DEFAULTS:
        if not cfg.has_option('PREDICT', key):
            continue
        if key in ['min_gain', 'min_snr', 'jitter_width']:
            PREDICT_PARAMS[key] = cfg.getfloat('PREDICT', key)
        if key in ['format']:
            PREDICT_PARAMS[key] = get_value(cfg, 'PREDICT', key)

    # rcs estimate
    rcs_estimator(
        'eiscat_uhf',
        str(tle_file),
        str(events_file.parent),
        str(correlations_dir),
        str(dst),
        discos_token,
        jitter_width=PREDICT_PARAMS['jitter_width'],
        min_gain=PREDICT_PARAMS['min_gain'],
        min_snr=PREDICT_PARAMS['min_snr'],
        fileformat=PREDICT_PARAMS['format'],
        verbose=verbose
    )

    ###########################################
    # CLEANUP
    ###########################################
    if cleanup:
        shutil.rmtree(str(tmp))

    return str(dst)