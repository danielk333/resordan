# Correlator and rcs_predict

This submodule implements the correlator and rcs_predict functions to estimate RCS values for a given target (object) using its corresponding Two-Line Element (TLE) info.

## Correlator overview

The correlation function uses radar measurements together with a given catalogue of TLEs. This function attributes measurements to objects from the given catalogue. Thus, it removes objects that are not the focus of the intended analysis.

The function uses as inputs radar measurement and TLEs from space-track.org obtained 24h prior of the radar measurement starting time. When there are multiple TLEs for the same object, the closest to the observation window is choosed. Each of the objects of the catalogue is propagated using the SGP4 propagator for the duration of the radar measurement, and its simulated range and range rate time series are obtained. A single value of the measured range and range rate associated to the time of the maximum signal to noise ratio (SNR) are also fetched. With these values, the weighted total residual is computed following eq 1 of Kastinen et al. (2023), Acta Astronautica, (https://www.sciencedirect.com/science/article/pii/S0094576522005586). This equation results in a distance parameter. The shortest distance and the object associated to that shortest distance is choosen.


## rcs_predict overview

The rcs_predict function uses radar measurements, the correlation results, and a given catalogue of TLEs to estimate RCS value, simulated SNR, simulated diameter, satellite ID, and object's shape from DISCOS for targets.

Similarly as in the correlation function, rcs_predict uses TLEs from space-track.org obtained 24h prior of the radar measurement starting time. When there are multiple TLEs for the same object, the closest to the observation window is choosed. The function uses the standard radar equation, takes into account the azimuth and elevation of the antenna beam pointing direction, as well as an itteration in time around the maximum SNR observation. The function assumes that the peak SNR detection of the object is made at the maximum of the antenna gain. The received and transmitted gain patterns are focused for the path of the target. The function also get the geometries from a given object from DISOS (https://discosweb.esoc.esa.int/objects)


## Other functions

1. **space_track_download**: Download tle snapshot from space-track from start time to end time. This end data should be before EISCAT GMF starting time.
2. **save_cluster_data**: Getting clustering targets and saving them into hourly files.
 

## Note

    To get accesss to space-track and discos database, get credentials to access:
    - space-track (https://www.space-track.org/auth/login)
    - discos (https://discosweb.esoc.esa.int/tokens)


## Example usage

```
# Download TLEs for the period 24h before the GMF data start time
python space_track_download.py YYYY-MM-DDThh:mm:dd YYYY-MM-DDThh:mm:dd \
    /path/to/store/space-track.tle \
    /path/to/st_credentials.txt

# Getting clustering data from GMF
rcluster path/to/gmf_product \
    /path/to/store/cluster
    
# Getting the correlated data
python beam_correlator.py radar \
    /path/to/space-track.tle \
    /path/to/cluster
    /path/to/store/correlation
   
# Getting RCS and other parameters
python beam_rcs_predict.py radar \
    /path/to/space-track.tle \
    /path/to/cluster
    /path/to/correlation
    /path/to/store/RCS_parameters
    --min-gain 10.0 --min-snr 15.0
```

## Data format


