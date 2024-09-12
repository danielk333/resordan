# Correlator and RCS

This submodule implements the 'beam_rcs_predict.py' function, which estimate RCS values and other parameters for targets (objects). It uses the time series of the targets, together with the results from 'beam_correlator.py' and 'space_track_download.py'. For each object, the physical characteristics are added using 'get_discos_cat.py'. At the same time, for each object, the best match between simulated and direved time series is selected.


## Brief overview

1. **space_track_download**: Download tle snapshot from space-track from start time to end time. This end data should be before EISCAT GMF starting time.
1. **save_cluster_data**: Getting clustering targets and saving them into hourly files.
1. **beam_correlator**: correlation analysis between radar measurements and a catalogue of know objetcs. The analysis uses TLE data from space-track. Each object is propagated in time using the SGP4 propagator. The shortest distance difference between the propagated and simulated range and range-rate are computed. Two distint populations of didstances will result. The analysis will continue with the population encapsulating the shortest ditances.
1. **beam_rcs_predict**: Estimate RCS values and other parameters for targets using correlation and TLE. For each object, it creates an iteration in time of +-1.5 seconds. The best match is selected from this iteration.
 

## Pre requirements
    - python
    - Java/8
    - Maven
    - JCC
    - astropy
    - spacetrack
    - similaritymeasures
    - sorts
    - orekit
    - create credentials to access space-track data (https://www.space-track.org/auth/login)
        - a file (e.g., st_credentials.txt) should contain two lines. Line 1: username; line 2: password
    - create credentials to access discos data (https://discosweb.esoc.esa.int/tokens)
        - a file (e.g. discos_credentials.txt) should contain an access token. This token will be use in beam_rcs_predict.py


## Example usage

```
# Download TLEs for the day before the GMF start time data

python space_track_download.py 2021-11-22T00:00:00 2021-11-23T08:00:00 \
    /Users/licr/Documents/Data/sdebris/bp/tle/st20211123.tle \
    --c /Users/licr/Documents/Data/sdebris/script_passes/st_credentials.txt

# Getting clustering data from GMF
# Two ways:
#1. using CLI
rcluster path/to/gmf_product path/to/store

#2. using the python command
python save_cluster_data.py \
    -gmf /Users/licr/Documents/Data/sdebris/bp/input/GMF/new/leo_bpark_2.1u_NO-20211123-UHF/ \
    -sf /Users/licr/Documents/Data/sdebris/bp/clus_event/new/uhf/ \
    -p True
    
# Getting the correlated data

python beam_correlator.py eiscat_uhf \
    /Users/licr/Documents/Data/sdebris/bp/tle/st20211123.tle \
    /Users/licr/Documents/Data/sdebris/bp/clus_event/uhf/20211123/ce_uhf_20211123T10.pkl \
    /Users/licr/Documents/Data/sdebris/bp/cor_data/uhf/20211123/ce_uhf_20211123T10.h5
   
# Getting RCS and other parameters

python beam_rcs_predict.py eiscat_uhf \
    /Users/licr/Documents/Data/sdebris/bp/tle/st20211123.tle \
    /Users/licr/Documents/Data/sdebris/bp/clus_event/uhf/20211123/ \
    /Users/licr/Documents/Data/sdebris/bp/cor_event/uhf/20211123/ \
    /Users/licr/Documents/Data/sdebris/bp/output/test/rcs_uhf_20211123 \
    --min-gain 10.0 --min-snr 15.0
```

## Data format


