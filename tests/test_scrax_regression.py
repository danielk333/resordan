# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:09:21 2024

@author: stia
"""

import numpy as np
from numpy.random import uniform
from scipy.stats import skew, kurtosis
from scipy.io import savemat
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.manifold import TSNE
from DielectricMaterial import DielectricMaterial
from SpaceDebris import DebrisMeasurement, DebrisObject

# Define measurement setup
#fc = 2.24e8 # EISCAT VHF radar frequency
#radar_name = 'EISCAT VHF radar'
fc = 9.30e8 # EISCAT UHF radar frequency
radar_name = 'EISCAT UHF radar'
c = 2.99792458e8 # speed of light (in vacuum)
polangle = np.pi / 4 # polarisation angle
vacuum = DielectricMaterial(1., 0.) # propagation medium: vacuum
eiscat = DebrisMeasurement(frequency=fc, 
                           pol_angle=polangle, 
                           medium=vacuum,
                           name=radar_name)

#%%

"""
Experiment: 
    Regression of object maximum dimension from simulated RCS 
    samples of a variety of canonical geometric objects
"""

n_stat = 10000 # No. test statistic samples per geometric object
n_obj = 5 # No. geometric objects
n_rcs = [10, 100, 1000] # Number of RCS samples per test statistic
#n_rcs = [8, 16, 32, 64, 128, 256, 512, 1024] # Number of RCS samples per test statistic
L = len(n_rcs)
n_var = 9 # Number of regression variables (test statistics)
sampling_method = 'fibonacci' # Fibonacci sphere sampling of viewing angles

# Object dimension limits
rmin = 0.01 # minimum radius [m]
rmax = 10.0 # maximum radius [m]
hmin = 0.01 # minimum height [m]
hmax = 10.0 # maximum height [m]
lmin = 0.50 # minimum length [m]
lmax = 10.0 # maximum length [m]
l2rmin = 100 # minimum r/l ratio
l2rmax = 1000 # maximum r/l ratio
#Nsmp = 1000 # number of logarithmic samples of radius and height

# Allocate regressor vector array
regvec = np.zeros((n_stat*n_obj, n_var, L), dtype=float)
y = np.zeros(n_stat*n_obj, dtype=float)

data_generated = False

#%%

# Define 1/2-order moment
def halfordermoment(data):
    
    return np.mean(np.sqrt(data))

# Name tags of test statistics
statnames = ["mean", "variance", "skewness", "kurtosis", 
             "log-mean", "log-variance", "log-skewness", "log-kurtosis",
             "half-order moment"]

# Compute test statistics (regression features)
def teststatistics(rcs):

    # Allocate feature vector
    featvec = np.zeros((n_var,), dtype=float)
    
    # RCS on logarithmic scale
    logrcs = np.log(rcs)
    logrcs[np.isinf(logrcs)] = np.nan

    # Compute feature vector
    featvec[0] = np.mean(rcs) # mean
    featvec[1] = np.var(rcs) # variance
    featvec[2] = skew(rcs, bias=False) # skewness (G1)
    featvec[3] = kurtosis(rcs, bias=False) # kurtosis
    featvec[4] = np.nanmean(logrcs) # log-mean
    featvec[5] = np.nanvar(logrcs) # log-variance
    featvec[6] = skew(logrcs, bias=False, nan_policy='omit') # log-skewness
    featvec[7] = kurtosis(logrcs, bias=False, nan_policy='omit') # log-kurtosis
    featvec[8] = halfordermoment(rcs) # half-order moment
    
    return featvec

#%%

#
# Cylinder
#

# Random generate radius and height on logarithmic scale
logr = uniform(np.log(rmin), np.log(rmax), n_stat)
r = np.exp(logr) # radius
logh = uniform(np.log(hmin), np.log(hmax), n_stat)
h = np.exp(logh) # height
#logl = uniform(np.log(lmin), np.log(lmax), n_stat)
#l = np.exp(logl) # length
#l2r = uniform(l2rmin, l2rmax, n_stat)
d = np.sqrt(np.power(h,2) + np.power(2*r,2)) # diagonal

# Generate training dataset
print("Generating test RCS samples for a cylinder.")
for idx in range(L): # loop RCS sample sizes
    ns = n_rcs[idx] # No. RCS samples per test statistic
    print("Generating test statistics for RCS samples of length", ns, " ...")
    for idy in range(n_stat): # loop RCS samples
        # Generate RCS sample
        CC = DebrisObject(radius=r[idy], height=h[idy], objtype='ClosedCylinder')
        rcs = CC.measure(eiscat, ns, sampling_method)
        # logrcs = np.log(rcs)
        # logrcs[np.isinf(logrcs)] = np.nan
        regvec[idy,:,idx] = teststatistics(rcs)
        # regvec[idy,0,idx] = np.mean(rcs) # mean
        # regvec[idy,1,idx] = np.var(rcs) # variance
        # regvec[idy,2,idx] = skew(rcs, bias=False) # skewness (G1)
        # regvec[idy,3,idx] = kurtosis(rcs, bias=False) # kurtosis
        # regvec[idy,4,idx] = np.nanmean(logrcs) # log-mean
        # regvec[idy,5,idx] = np.nanvar(logrcs) # log-variance
        # regvec[idy,6,idx] = skew(logrcs, bias=False, nan_policy='omit') # log-skewness
        # regvec[idy,7,idx] = kurtosis(logrcs, bias=False, nan_policy='omit') # log-kurtosis
        # regvec[idy,8,idx] = halfordermoment(rcs) # half-order moment

y[0:n_stat] = d # regressand vector

#%%

#
# Circular plate
#

# Random generate radius on logarithmic scale
logr = uniform(np.log(rmin), np.log(rmax), n_stat)
r = np.exp(logr) # radius
d = 2 * r # diagonal

# Generate training dataset
print("Simulating RCS samples for a circular plate.")
for idx in range(L): # loop RCS sample sizes
    ns = n_rcs[idx] # No. RCS samples per test statistic
    print("Generating test statistics for RCS samples of length", ns, " ...")
    for idy in range(n_stat): # loop RCS samples
        # Generate RCS sample
        CP = DebrisObject(radius=r[idy], objtype='CircularPlate')
        rcs = CP.measure(eiscat, ns, sampling_method)
        # Compute test statistics (feature vector)
        idz = idy + n_stat
        regvec[idz,:,idx] = teststatistics(rcs)

y[n_stat:2*n_stat] = d # regressand vector

#%%

#
# Square plate
#

# Random generate edge length on logarithmic scale
logr = uniform(np.log(rmin), np.log(rmax), n_stat)
r = np.exp(logr) # radius
d = np.sqrt(2) * r # diagonal

# Generate training dataset
print("Simulating RCS samples for a square plate.")
for idx in range(L): # loop RCS sample sizes
    ns = n_rcs[idx] # No. RCS samples per test statistic
    print("Generating test statistics for RCS samples of length", ns, " ...")
    for idy in range(n_stat): # loop RCS samples
        # Generate RCS sample
        SP = DebrisObject(length=r[idy], objtype='SquarePlate')
        rcs = SP.measure(eiscat, ns, sampling_method)
        # Compute test statistics (feature vector)
        idz = idy + n_stat * 2
        regvec[idz,:,idx] = teststatistics(rcs)

y[2*n_stat:3*n_stat] = d # regressand vector

#%%

#
# Wire
#

# Random generate radius on logarithmic scale
logl = uniform(np.log(lmin), np.log(lmax), n_stat)
l = np.exp(logl) # length
l2r = uniform(l2rmin, l2rmax, n_stat) # length/radius ratio
r = l / l2r # radius
#d = l # diagonal
d = np.sqrt(np.power(l,2) + np.power(2*r,2)) # diagonal

# Generate training dataset
print("Simulating test RCS samples for a wire.")
for idx in range(L): # loop RCS sample sizes
    ns = n_rcs[idx] # No. RCS samples per test statistic
    print("Generating test statistics for RCS samples of length", ns, " ...")
    for idy in range(n_stat): # loop RCS samples
        # Generate RCS sample
        Wire = DebrisObject(radius=r[idy], length=l[idy], objtype='Wire')
        rcs = Wire.measure(eiscat, ns, sampling_method)
        # Compute test statistics (feature vector)
        idz = idy + n_stat * 3
        regvec[idz,:,idx] = teststatistics(rcs)

y[3*n_stat:4*n_stat] = d # regressand vector

#%%

regdata = {"regressor":regvec, "regressand":y}
savemat('regvec4.mat', regdata)

#%%

#
# Sphere
#

# Random generate radius and height on logarithmic scale
logr = uniform(np.log(rmin), np.log(rmax), n_stat)
r = np.exp(logr) # radius
d = 2 * r # diagonal
PEC = DielectricMaterial(1e8,0,1e-8,0) # Perfect electric conductor

# Generate training dataset
print("Simulating test RCS samples for a sphere.")
for idx in range(L): # loop RCS sample sizes
    ns = n_rcs[idx] # No. RCS samples per test statistic
    print("Generating test statistics for RCS samples of length", ns, " ...")
    for idy in range(n_stat): # loop RCS samples
        # Generate RCS sample
        Sph = DebrisObject(radius=r[idy], material=PEC, objtype='Sphere')
        rcs = Sph.measure(eiscat, ns, sampling_method)
        # Compute test statistics (feature vector)
        idz = idy + n_stat * 4
        regvec[idz,:,idx] = teststatistics(rcs)
        if not np.mod(idy,100):
            print(idx, idy)
            print(regvec[idz,:,idx])

y[4*n_stat:5*n_stat] = d # regressand vector

#%%

# Replace NaN values with zeros
regvec[np.isnan(regvec)] = 0

# Replace small values with zeros
small = 1e-6
regvec[regvec<small] = 0

regdata = {"regressor":regvec, "regressand":y}
savemat('regvec.mat', regdata)

data_generated = True

#%%
#
# Load saved regression data 
#

if not data_generated:
    regdata = loadmat('regvec.mat')
    regvec = regdata.get("regressor")
    y = regdata.get("regressand")

y = np.reshape(y, y.size)

#%%
#
# Train regression algorithm
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
    "max_iter": 400, # default=100
    "max_leaf_nodes": 41,# default=31
    "max_depth": 12, # default=None
    "min_samples_leaf": 10, # default=20
    "l2_regularization": 0.0, # default=0.0
    "warm_start": False,
    "random_state": 42
}

gbr_params = {
    "n_estimators": 100,
    "max_depth": 4, # default=3
    "min_samples_split": 5, # default=2
    "max_leaf_nodes": None, # default=None
    "warm_start": False,
    "random_state": 42
}

svr_params = {
    "kernel": 'rbf', # kernel function
    "gamma": 0.05, # 'auto', # kernel width
    "C": 8.0 # regularisation parameter
}

algorithm = "HGBR"
dimension = "d"

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

#for idx in range(1):
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

# Test models trained on separate objects

# Allocate performance metrics
mse = np.zeros((L,n_obj)) # mean squared error
mae = np.zeros((L,n_obj)) # mean absolute error
mape = np.zeros((L,n_obj)) # mean absolute percentage error

# Test regression with different sample sizes (no. measurements per object)
for idx in range(L):
    print("RCS sample size: ", n_rcs[idx])
    
    # Test regression for different canonical objects
    for idy in range(n_obj):
        # Prepare datasets
        print("Preparing dataset ...")
        obj_data = np.arange(n_stat) + idy * n_stat
        X = np.squeeze(regvec[:,:,idx])
        X_train, X_test, y_train, y_test = train_test_split(
            X[obj_data,:], y[obj_data], test_size=0.20, train_size=0.80, 
            random_state=42)
    
        # Fit regression model
        print("Fitting regression model ...")
        reg.fit(X_train, y_train)
        
        # Make predictions
        print("Making predictions ...")
        y_pred = reg.predict(X_test)
        
        # Measure performance
        mse[idx, idy] = mean_squared_error(y_test, y_pred, squared=False)
        mae[idx, idy] = mean_absolute_error(y_test, y_pred)
        mape[idx, idy] = mean_absolute_percentage_error(y_test, y_pred)
        print("MSE:", mse[idx, idy], "MAE:", mae[idx, idy], "MAPE:", mape[idx, idy])
        
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
            
#%%
# Plot performance metrics
fig4 = plt.figure(4)
fig4, ax4 = plt.subplots(5,1)
ax4.semilogx(np.array(n_rcs), mse[:,0])
ax4.semilogx(np.array(n_rcs), mse[:,1])
ax4.semilogx(np.array(n_rcs), mse[:,2])
ax4.semilogx(np.array(n_rcs), mse[:,3])
ax4.semilogx(np.array(n_rcs), mse[:,4])
ax4.set_xlabel("No. data points")
        

#%%

# Test sequential clustering and regression

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

