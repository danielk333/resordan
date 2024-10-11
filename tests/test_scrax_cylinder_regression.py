# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:09:21 2024

@author: stia
"""

import numpy as np
from numpy.random import uniform
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
#from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.manifold import TSNE
from DielectricMaterial import DielectricMaterial
from SpaceDebris import DebrisMeasurement, DebrisObject

# Define measurement setup
fc = 2.24e8 # EISCAT VHF radar frequency
fc = 9.30e8 # EISCAT UHF radar frequency
c = 2.99792458e8 # speed of light (in vacuum)
polangle = np.pi
vacuum = DielectricMaterial(1., 0.) # propagation medium: vacuum
eiscat = DebrisMeasurement(frequency=fc, 
                           pol_angle=polangle, 
                           medium=vacuum,
                           name='EISCAT radar')

#%%

"""
Experiment: 
    Regression of object dimensions radius and height from
    simulated RCS samples of a closed cylinder
"""

n_stat = 10000 # Number of test statistic samples
n_rcs = [10, 100, 1000] # Number of RCS samples per test statistic
#n_rcs = [8, 16, 32, 64, 128, 256, 512, 1024] # Number of RCS samples per test statistic
L = len(n_rcs)
n_var = 9 # Number of regression variables (test statistics)
sampling_method = 'fibonacci' # Fibonacci sphere sampling

# Object dimension limits
rmin = 0.01 # minimum radius [m]
rmax = 10.0 # maximum radius [m]
hmin = 0.01 # minimum height [m]
hmax = 10.0 # maximum height [m]
Nsmp = 1000 # number of logarithmic samples of radius and height

# Random generate radius and height on logarithmic scale
logr = uniform(np.log(rmin), np.log(rmax), n_stat)
r = np.exp(logr) # radius
logh = uniform(np.log(hmin), np.log(hmax), n_stat)
h = np.exp(logh) # height
d = np.sqrt(np.power(h,2) + np.power(2*r,2)) # diagonal

# Allocate regressor vector array
regvec = np.zeros((n_stat,n_var,L),dtype=float)

#%%

def histdiffrange(data):
    
    n = len(data)
    nbins = int(np.ceil(np.sqrt(n)))
    hist, bin_edges = np.histogram(data, bins=nbins, density=True)
    hdiff = hist[1:nbins] - hist[0:nbins-1]
    
    return max(hdiff) - min(hdiff)

def histdiffvar(data):
    
    n = len(data)
    nbins = int(np.ceil(np.sqrt(n)))
    hist, bin_edges = np.histogram(data, bins=nbins, density=True)
    hdiff = hist[1:nbins] - hist[0:nbins-1]
    
    return np.var(hdiff)

def histdiffskew(data):
    
    n = len(data)
    nbins = int(np.ceil(np.sqrt(n)))
    hist, bin_edges = np.histogram(data, bins=nbins, density=True)
    hdiff = hist[1:nbins] - hist[0:nbins-1]
    
    return skew(hdiff)

def halfordermoment(data):
    
    return np.mean(np.sqrt(data))    

# Generate training dataset
for idx in range(L): # loop RCS sample sizes
    ns = n_rcs[idx] # No. RCS samples per test statistic
    print("Generating test statistics for RCS samples of length", ns, " ...")
    for idy in range(n_stat): # loop RCS samples
        # Generate RCS sample
        CC = DebrisObject(radius=r[idy], height=h[idy], objtype='ClosedCylinder')
        rcs = CC.measure(eiscat, ns, sampling_method)
        logrcs = np.log(rcs)
        regvec[idy,0,idx] = np.mean(rcs) # mean
        regvec[idy,1,idx] = np.var(rcs) # variance
        regvec[idy,2,idx] = skew(rcs, bias=False) # skewness (G1)
        regvec[idy,3,idx] = kurtosis(rcs, bias=False) # kurtosis
        regvec[idy,4,idx] = np.mean(logrcs) # log-mean
        regvec[idy,5,idx] = np.var(logrcs) # log-variance
        regvec[idy,6,idx] = skew(logrcs, bias=False) # log-skewness
        regvec[idy,7,idx] = kurtosis(logrcs, bias=False) # log-kurtosis
        regvec[idy,8,idx] = halfordermoment(rcs) # half-order moment
        #regvec[idy,8,idx] = histdiffrange(logrcs)
        #regvec[idy,9,idx] = histdiffvar(logrcs)
        #regvec[idy,10,idx] = histdiffskew(logrcs)

# Name tags of test statistics
statnames = ["mean", "variance", "skewness", "kurtosis", 
             "log-mean", "log-variance", "log-skewness", "log-kurtosis",
             "half-order moment"]

#%%
#
# Train random forest regression
#

# Define hyperparameters
rfr_params = {
    "n_estimators": 100,
    "max_depth": 4,
    "min_samples_split": 5,
    "warm_start": False,
    "oob_score": True,
    "random_state": 42
}

hgbr_params = {
    "max_iter": 400,
    "max_depth": 8,
    "warm_start": False,
    "random_state": 42
}

gbr_params = {
    "n_estimators": 100,
    "max_depth": 4,
    "min_samples_split": 5,
    "warm_start": False,
    "random_state": 42
}

svr_params = {
    "kernel": 'rbf', # kernel function
    "gamma": 0.05, # 'auto', # kernel width
    "C": 8.0 # regularisation parameter
}

algorithm = "HGBR"

# Build regression model
if algorithm == "RGR": # Random forest regression
    reg = RandomForestRegressor(**rfr_params)
elif algorithm == "HGBR": # Histogram gradient boosting regression
    reg = HistGradientBoostingRegressor(**hgbr_params)
elif algorithm == "GBR": # Gradient boosting regression
    reg = GradientBoostingRegressor(**gbr_params)
elif algorithm == "SVR": # Support vector regression
    reg = SVR(**svr_params)

# Allocate performance metrics
mse = np.zeros((L,1)) # mean squared error
mae = np.zeros((L,1)) # mean absolute error
mape = np.zeros((L,1)) # mean absolute percentage error

dimension = "d"
if dimension=="r":
    y = r # Use r as regressand
elif dimension=="h":
    y = h # Use h as regressand
elif dimension=="d":
    y = d # Use d as regressand

for idx in range(L):
    print("RCS sample size: ", n_rcs[idx])
    
    # Prepare datasets
    print("Preparing dataset ...")
    X = np.squeeze(regvec[:,:,idx])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, train_size=0.80, random_state=42)
    
    # Fit regression model
    print("Fitting regression model ...")
    reg.fit(X_train, y_train)

    # Make predictions
    print("Making predictions ...")
    y_pred = reg.predict(X_test)
    
    # Measure performance
    mse[idx] = mean_squared_error(y_test, y_pred, squared=False)
    mae[idx] = mean_absolute_error(y_test, y_pred)
    mape[idx] = mean_absolute_percentage_error(y_test, y_pred)
    print("MSE:", mse[idx], "MAE:", mae[idx], "MAPE:", mape[idx])

    if algorithm=="RGR" or algorithm=="GBR":
        # Obtain feature importance
        feature_importance = reg.feature_importances_

        # Sort features according to importance
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0])

        # Plot feature importances
        plt.figure(idx)
        plt.barh(pos, feature_importance[sorted_idx], align="center")
        plt.yticks(pos, np.array(statnames)[sorted_idx])
        plt.title("Feature importance (MDI)")
        plt.xlabel("Mean decrease in impurity")

# Plot performance metrics
fig3 = plt.figure(3)
fig3, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.semilogx(np.array(n_rcs), mse)
ax1.set_title("MSE("+dimension+")")
ax1.set_xlabel("No. data points")
ax2.semilogx(np.array(n_rcs), mae)
ax2.set_title("MAE("+dimension+")")
ax2.set_xlabel("No. data points")
ax3.semilogx(np.array(n_rcs), mape)
ax3.set_title("MAPE("+dimension+")")
ax3.set_xlabel("No. data points")

#%%

#
# t-SNE dimensionality reduction and visualisation
#

# Build 2d t-SNE model
nc = 2
ppl = 30.0
lr = 'auto'
init = 'pca'
tsne = TSNE(n_components=nc, perplexity=ppl, learning_rate=lr, init=init)

# Compute radius to height ratio
ratio = r/h
rtmin = min(ratio)
rtmax = max(ratio)

# Allocate colour array
col = np.zeros((n_stat, 3), dtype=float)
col[:,0] = (np.log(r) - np.log(rmin)) / (np.log(rmax) - np.log(rmin)) # red = radius
col[:,1] = (np.log(h) - np.log(hmin)) / (np.log(hmax) - np.log(hmin)) # green = height
col[:,2] = (ratio - rtmin) / (rtmax - rtmin) # blue = r/h ratio

for idx in range(L):
    # Import data
    X = np.squeeze(regvec[:,:,idx])
    # Fit t-SNE model
    X_tsne = tsne.fit_transform(X)
    # Plot t-SNE space
    plt.figure(idx+4)
    plt.scatter(X_tsne[:,0], X_tsne[:,1], c=col)
    plt.show()

#%%

# Build 3d t-SNE model
nc = 3
tsne = TSNE(n_components=nc, perplexity=ppl, learning_rate=lr, init=init)

# Compute radius to height ratio
ratio = r/h
rtmin = min(ratio)
rtmax = max(ratio)

for idx in range(L):
    # Import data
    X = np.squeeze(regvec[:,:,idx])
    # Fit t-SNE model
    X_tsne = tsne.fit_transform(X)
    # Plot t-SNE space
    fig = plt.figure(idx+4)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X_tsne[:,0], X_tsne[:,1], X_tsne[:,2], c=col)
    plt.show()

