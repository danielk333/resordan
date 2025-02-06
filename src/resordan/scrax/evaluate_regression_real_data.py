#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:12:42 2024

@author: stia
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import pickle
from resordan.scrax.read_eiscat_data import read_data_per_pass
from resordan.scrax.features import compute_feature_vector

# Input data
data_dir = Path(__file__).parent.parent.parent.parent / "tests/data/rcs_test_0904"
#model_dir = Path(__file__).parent / "models/spyderenv"
model_dir = Path(__file__).parent / "models/venv"
ext = "pickle"
model_file = "size_predict_n10" + os.extsep + ext
#model_file = "size_predict_n100" + os.extsep + ext
model_path = os.path.join(model_dir, model_file)

#
# Read campaign data
#

pass_list = read_data_per_pass(data_dir)
rcs_data = []
xSectMax = []
xSectMin = []
nameo = []
satno = []
span = []

for this_pass in pass_list:
    rcs_data.append(this_pass["rcs_data"])
    xSectMax.append(this_pass["xSectMax"])
    xSectMin.append(this_pass["xSectMin"])
    nameo.append(this_pass["nameo"])
    satno.append(this_pass["satno"])
    span.append(this_pass["span"])

#%%

# Plot campaign RCS data

sat_list = np.unique(satno).tolist()

# For all satellites
for sat in sat_list:
    # Make a figure
    plt.figure(sat)
    # Identify all pass indices for this satellite
    pass_idx = [idx for idx in range(len(satno)) if satno[idx]==sat]
    # For all passes
    for this_pass in pass_idx:
        # Plot the RCS data
        plt.semilogy(rcs_data[this_pass])
        plt.title(nameo[this_pass]) # satellite name
        plt.ylabel('RCS')
    plt.legend(pass_idx)

#%%

#
# Test regression model on campaign RCS data
#

# Load regression model
with open(model_path, "rb") as f:
    model = pickle.load(f) # scikit-learn regression model

size = []
for rcs in rcs_data:
    fvec = compute_feature_vector(rcs, nan_ignore=True, nan_to_zero=True)
    fvec = fvec.reshape(1, fvec.size)
    size.append(model.predict(fvec).item())

#%%

#
# Compare with ground truth
#

# Estimated maximum dimension
max_dim_est = np.asarray(size)

# Maximum cross-section from DISCOS
max_dim = np.sqrt(np.asarray(xSectMax))

plt.figure(1)
plt.scatter(max_dim, max_dim_est)
plt.xlabel('Span [DISCOS]')
plt.ylabel("Estimated max. dim.")
plt.title("Estimate per pass (n=10)")

#%%

# Retry size estimation with data from all passes stacked
size = [] # Estimated size
mcs = [] # Maximum cross-section from DISCOS
max_dim = [] # Maximum dimension (span) from DISCOS
filter_outliers = True

# For all satellites
for sat in sat_list:
    # Collect RCS data
    rcs = []
    # Identify all pass indices for this satellite
    pass_idx = [idx for idx in range(len(satno)) if satno[idx]==sat]
    # Maximum cross-section from DISCOS database
    mcs.append(xSectMax[pass_idx[0]])
    # Maximum dimension from DISCOS database
    max_dim.append(span[pass_idx[0]])
    # For all passes
    for this_pass in pass_idx:
        # Concatenate RCS data
        rcs += rcs_data[this_pass].tolist()
    if filter_outliers:
        # Filter outliers
        rcs = [x for x in rcs if not np.isnan(x)] # remove NaNs
        qtl10pct = np.quantile(rcs, 0.1) # 10% quantile
        qtl90pct = np.quantile(rcs, 0.9) # 90% quantile
        filt = [k for k in range(len(rcs)) if rcs[k]>qtl10pct and rcs[k]<qtl90pct]
        rcs = [rcs[k] for k in filt] # remove outliers
    # Compute feature vector from stacked data
    fvec = compute_feature_vector(rcs, nan_ignore=True, nan_to_zero=True)
    fvec = fvec.reshape(1, fvec.size)
    # Estimate maximum dimension
    size.append(model.predict(fvec).item())

# Estimated maximum dimension
max_dim_est = np.asarray(size)
# Maximum cross-section
mcs = np.sqrt(np.asarray(mcs))
max_dim = np.asarray(max_dim)

plt.figure(2)
plt.scatter(max_dim, max_dim_est)
plt.xlabel('Span [DISCOS]')
plt.ylabel("Estimated max. dim.")
plt.title("Estimate per object (n=10)")

