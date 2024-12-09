#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:48:36 2024

@author: stia
"""

import numpy as np
from scipy.stats import skew, kurtosis

# Define 1/2-order moment
def half_order_moment(data):
    
    return np.nanmean(np.sqrt(data))

# Compute test statistics (feature vector for regression or clustering)
def compute_feature_vector(rcs, # radar cross-section data
                           nan_ignore=False, # RCS data NaN policy
                           nan_to_zero=False): # Feature vector NaN policy
    # Handle NaNs in RCS data
    if nan_ignore:
        rcs = [x for x in rcs if not np.isnan(x)]
    
    # Allocate feature vector
    n_features = 9 # number of features
    fvec = np.zeros((n_features,), dtype=float) # feature vector
    
    # RCS on logarithmic scale
    logrcs = np.log(rcs)
    
    # Compute feature vector
    fvec[0] = np.nanmean(rcs) # mean
    fvec[1] = np.nanvar(rcs) # variance
    fvec[2] = skew(rcs, bias=False, nan_policy='omit') # skewness
    fvec[3] = kurtosis(rcs, bias=False, nan_policy='omit') # kurtosis
    fvec[4] = np.nanmean(logrcs) # log-mean
    fvec[5] = np.nanmean(logrcs) # log-variance
    fvec[6] = skew(logrcs, bias=False, nan_policy='omit') # log-skewness
    fvec[7] = kurtosis(logrcs, bias=False, nan_policy='omit') # log-kurtosis
    fvec[8] = half_order_moment(rcs) # 1/2-order moment
    #fvec[8] = half_order_moment(logrcs) # log-1/2-order moment
    
    # Handle NaN in feature vector
    if nan_to_zero:
        fvec[np.isnan(fvec)] = 0 # Replace NaN values with zeros
        fvec[fvec<1e-6] = 0 # Replace small values with zeros
    
    return fvec

# Name tags of test statistics / features
def stat_names():
    
    statnames = ["mean", "variance", "skewness", "kurtosis", 
                 "log-mean", "log-variance", "log-skewness", "log-kurtosis",
                 "half-order moment"]
    
    return statnames
