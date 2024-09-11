
import argparse
from pathlib import Path
import h5py
import re
import datetime as dt

ISO_FMT = '%Y-%m-%dT%H:%M:%S'

def all_gmf_h5_files(gmf_folder):
    """generate all files matching 'yyyy-mm-ddThh-00-00/gmf-*.h5'"""
    top = Path(gmf_folder)
    dir_pattern = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}-00-00')
    subdirs = [d for d in top.iterdir() if d.is_dir() and dir_pattern.match(d.name)]
    file_pattern = re.compile(r'^gmf-.*\.h5$')
    files = []
    for subdir in subdirs:
        files += [f for f in subdir.iterdir() if f.is_file and file_pattern.match(f.name)]
    return files



def snr2rcs(gmf, cfg):

    # get timestamp for start of gmf product
    files = sorted(all_gmf_h5_files(gmf))
    with h5py.File(files[0], "r") as f:        
        epoch = float(f['epoch_unix'][()])
        epoch_dt = dt.datetime.utcfromtimestamp(epoch)

    # make temporary directory

    # run tle


    # run clustering




