# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 00:32:51 2024

@author: stia
"""

import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
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

ns = 10000 # No. samples
sampling_method = 'fibonacci' # Fibonacci sphere sampling

"""
Experiment: 
    Compute closed cylinder statistics for varying r but constant r/h
    Check if statistics depend on r/h or both r and h    
"""

# Cylinder properties
r = 2.**np.linspace(0, 4, 5, endpoint=True) # radius
h = 2**np.linspace(0, 4, 5, endpoint=True) # height
name = "Closed cylinder"
#PEC = DielectricMaterial(1.e8, 0., 1.e-8, 0.)

Nr = r.size
Nh = h.size

# Summary statistics of RCS samples
mu = np.zeros((Nr,Nh))
med = np.zeros((Nr,Nh))
var = np.zeros((Nr,Nh))
std = np.zeros((Nr,Nh))
g1 = np.zeros((Nr,Nh))
G1 = np.zeros((Nr,Nh))
kur = np.zeros((Nr,Nh))
logmu = np.zeros((Nr,Nh))
logmed = np.zeros((Nr,Nh))
logvar = np.zeros((Nr,Nh))
logstd = np.zeros((Nr,Nh))
logg1 = np.zeros((Nr,Nh))
logG1 = np.zeros((Nr,Nh))
logkur = np.zeros((Nr,Nh))
rcssample = np.zeros((Nr,Nh,ns))

for idx in range(Nr):
    for idy in range(Nh):
        CC = DebrisObject(radius=r[idx], height=h[idy], objtype='ClosedCylinder')
        rcs = CC.measure(eiscat, ns, sampling_method)
        logrcs = np.log(rcs)
        mu[idx,idy] = np.mean(rcs)
        med[idx,idy] = np.median(rcs)
        std[idx,idy] = np.std(rcs)
        var[idx,idy] = np.var(rcs)
        g1[idx,idy] = skew(rcs, bias=True)
        G1[idx,idy] = skew(rcs, bias=False)
        kur[idx,idy] = kurtosis(rcs, bias=False)
        logmu[idx,idy] = np.mean(logrcs)
        logmed[idx,idy] = np.median(logrcs)
        logstd[idx,idy] = np.std(logrcs)
        logvar[idx,idy] = np.var(logrcs)
        logg1[idx,idy] = skew(logrcs, bias=True)
        logG1[idx,idy] = skew(logrcs, bias=False)
        logkur[idx,idy] = kurtosis(logrcs, bias=False)
        rcssample[idx,idy,:] = rcs

rstr = str(r).replace('[',' ').replace(']',' ').split()
hstr = str(h).replace('[',' ').replace(']',' ').split()

#%%

fig1a = plt.figure()
plt.imshow(mu)
plt.title('Mean')
plt.ylabel('Radius r')
plt.xlabel('Height h')
plt.colorbar()
plt.xticks(np.arange(Nr),rstr)
plt.yticks(np.arange(Nh),hstr)
plt.show()

fig1b = plt.figure()
plt.imshow(var)
plt.title('Variance')
plt.ylabel('Radius r')
plt.xlabel('Height h')
plt.colorbar()
plt.xticks(np.arange(Nr),rstr)
plt.yticks(np.arange(Nh),hstr)
plt.show()

#%%

fig1c = plt.figure()
plt.imshow(logG1)
plt.title('log-skewness')
plt.ylabel('Radius r')
plt.xlabel('Height h')
plt.colorbar()
plt.xticks(np.arange(Nr),rstr)
plt.yticks(np.arange(Nh),hstr)
plt.show()

fig1d = plt.figure()
plt.imshow(logkur)
plt.title('log-kurtosis')
plt.ylabel('Radius r')
plt.xlabel('Height h')
plt.colorbar()
plt.xticks(np.arange(Nr),rstr)
plt.yticks(np.arange(Nh),hstr)
plt.show()

#%%

fig2a = plt.figure()
plt.scatter(G1.flatten(), kur.flatten(), marker='o')
plt.title('Skewness-kurtosis diagram')
plt.xlabel('Skewness')
plt.ylabel('Kurtosis')
plt.show()

fig2b = plt.figure()
plt.scatter(logG1.flatten(), logkur.flatten(), marker='o')
plt.title('log-skewness - log-kurtosis diagram')
plt.xlabel('log-skewness')
plt.ylabel('log-kurtosis')
plt.show()

#%%

fig2c = plt.figure()
plt.scatter(mu.flatten(), var.flatten(), marker='o')
plt.title('Mean-variance diagram')
plt.xlabel('Mean')
plt.ylabel('Variance')
plt.show()

fig2d = plt.figure()
plt.scatter(logmu.flatten(), logvar.flatten(), marker='o')
plt.title('log-mean - log-variance diagram')
plt.xlabel('log-mean')
plt.ylabel('log-variance')
plt.show()

#%%

fig5, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, ncols=1, sharex=True)
nbins = 100
rcsdbsample = 10*np.log10(rcssample)
# 
ax1.hist(rcsdbsample[0,0], bins=nbins, density=True, histtype='step')
#ax1.legend('r='+rstr[0]+' h='+hstr[0], loc='upper left')
ax2.hist(rcsdbsample[1,0], bins=nbins, density=True, histtype='step')
#ax2.legend('r='+rstr[0]+' h='+hstr[1], loc='upper left')
ax3.hist(rcsdbsample[2,0], bins=nbins, density=True, histtype='step')
#ax3.legend('r='+rstr[0]+' h='+hstr[2], loc='upper left')
ax4.hist(rcsdbsample[3,0], bins=nbins, density=True, histtype='step')
#ax4.legend('r='+rstr[0]+' h='+hstr[3], loc='upper left')
ax5.hist(rcsdbsample[4,0], bins=nbins, density=True, histtype='step')
#ax5.legend('r='+rstr[0]+' h='+hstr[4], loc='upper left')
ax1.set_xlim([-80,50])
ax2.set_xlim([-80,50])
ax3.set_xlim([-80,50])
ax4.set_xlim([-80,50])
ax5.set_xlim([-80,50])

#%%

fig6, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, ncols=1, sharex=True)
nbins = 100
rcsdbsample = 10*np.log10(rcssample)
ax1.hist(rcsdbsample[0,1], bins=nbins, density=True, histtype='step')
#ax1.legend('r='+rstr[0]+' h='+hstr[0], loc='upper left')
ax2.hist(rcsdbsample[1,1], bins=nbins, density=True, histtype='step')
#ax2.legend('r='+rstr[0]+' h='+hstr[1], loc='upper left')
ax3.hist(rcsdbsample[2,1], bins=nbins, density=True, histtype='step')
#ax3.legend('r='+rstr[0]+' h='+hstr[2], loc='upper left')
ax4.hist(rcsdbsample[3,1], bins=nbins, density=True, histtype='step')
#ax4.legend('r='+rstr[0]+' h='+hstr[3], loc='upper left')
ax5.hist(rcsdbsample[4,1], bins=nbins, density=True, histtype='step')
#ax5.legend('r='+rstr[0]+' h='+hstr[4], loc='upper left')
ax1.set_xlim([-80,50])
ax2.set_xlim([-80,50])
ax3.set_xlim([-80,50])
ax4.set_xlim([-80,50])
ax5.set_xlim([-80,50])

#%%
fig7, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, ncols=1, sharex=True)
nbins = 100
rcsdbsample = 10*np.log10(rcssample)
ax1.hist(rcsdbsample[0,2], bins=nbins, density=True, histtype='step')
#ax1.legend('r='+rstr[0]+' h='+hstr[0], loc='upper left')
ax2.hist(rcsdbsample[1,2], bins=nbins, density=True, histtype='step')
#ax2.legend('r='+rstr[0]+' h='+hstr[1], loc='upper left')
ax3.hist(rcsdbsample[2,2], bins=nbins, density=True, histtype='step')
#ax3.legend('r='+rstr[0]+' h='+hstr[2], loc='upper left')
ax4.hist(rcsdbsample[3,2], bins=nbins, density=True, histtype='step')
#ax4.legend('r='+rstr[0]+' h='+hstr[3], loc='upper left')
ax5.hist(rcsdbsample[4,2], bins=nbins, density=True, histtype='step')
#ax5.legend('r='+rstr[0]+' h='+hstr[4], loc='upper left')
ax1.set_xlim([-80,50])
ax2.set_xlim([-80,50])
ax3.set_xlim([-80,50])
ax4.set_xlim([-80,50])
ax5.set_xlim([-80,50])


#%%
fig8, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, ncols=1, sharex=True)
nbins = 100
rcsdbsample = 10*np.log10(rcssample)
ax1.hist(rcsdbsample[0,3], bins=nbins, density=True, histtype='step')
#ax1.legend('r='+rstr[0+' h='+hstr[0], loc='upper left')
ax2.hist(rcsdbsample[1,3], bins=nbins, density=True, histtype='step')
#ax2.legend('r='+rstr[0]+' h='+hstr[1], loc='upper left')
ax3.hist(rcsdbsample[2,3], bins=nbins, density=True, histtype='step')
#ax3.legend('r='+rstr[0]+' h='+hstr[2], loc='upper left')
ax4.hist(rcsdbsample[3,3], bins=nbins, density=True, histtype='step')
#ax4.legend('r='+rstr[0]+' h='+hstr[3], loc='upper left')
ax5.hist(rcsdbsample[4,3], bins=nbins, density=True, histtype='step')
#ax5.legend('r='+rstr[0]+' h='+hstr[4], loc='upper left')
ax1.set_xlim([-80,50])
ax2.set_xlim([-80,50])
ax3.set_xlim([-80,50])
ax4.set_xlim([-80,50])
ax5.set_xlim([-80,50])

#%%
fig9, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, ncols=1, sharex=True)
nbins = 100
rcsdbsample = 10*np.log10(rcssample)
ax1.hist(rcsdbsample[0,4], bins=nbins, density=True, histtype='step')
#ax1.legend('r='+rstr[0]+' h='+hstr[0], loc='upper left')
ax2.hist(rcsdbsample[1,4], bins=nbins, density=True, histtype='step')
#ax2.legend('r='+rstr[0]+' h='+hstr[1], loc='upper left')
ax3.hist(rcsdbsample[2,4], bins=nbins, density=True, histtype='step')
#ax3.legend('r='+rstr[0]+' h='+hstr[2], loc='upper left')
ax4.hist(rcsdbsample[3,4], bins=nbins, density=True, histtype='step')
#ax4.legend('r='+rstr[0]+' h='+hstr[3], loc='upper left')
ax5.hist(rcsdbsample[4,4], bins=nbins, density=True, histtype='step')
#ax5.legend('r='+rstr[0]+' h='+hstr[4], loc='upper left')
ax1.set_xlim([-80,50])
ax2.set_xlim([-80,50])
ax3.set_xlim([-80,50])
ax4.set_xlim([-80,50])
ax5.set_xlim([-80,50])
