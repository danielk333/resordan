# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:56:28 2023

@author: stia
"""

import numpy as np
from scipy.special import j1 as besselj1
from scipy.special import struve
import matplotlib.pyplot as plt

def getOpenCylinderRCS(radius, # radius [m]
                       height, # Cylinder height [m]
                       frequency, # Wave frequency [Hz]
                       azimuth, # Azimuth angle [-pi<rad<pi]
                       elevation): # Elevation angle [-pi/2<=rad<=pi/2]
    '''
        Calculates the radar cross-section for a circular cylinder defined by 
        its radii and height, the wave frequency and elevation angle. Note 
        that the cylinder is closed and has end plates.
    '''
    eps = 1e-6 # small number
    phir = np.asarray(azimuth) # azimuth angle [rad]
    thetar = (elevation + np.pi/2) # aspect angle [rad]
    c = 2.99792458e8 # speed of light in vacuum [m/s]
    lamda = c / frequency # wavelength
    k = 2 * np.pi / lamda
    r = radius
    h = height

    # Declare RCS array
    Naz = np.asarray(phir).size
    Nel = np.asarray(thetar).size
    if Naz == Nel:
        rcs = np.zeros(Naz)
    else:
        rcs = np.zeros((Naz,Nel))

    if Naz == 1:
        phir = np.reshape(phir,(1))
    if Nel == 1:
        thetar = np.reshape(thetar,(1))

    for idy in range(Nel):
        theta = thetar[idy]
        sth = np.sin(theta)
        cth = np.cos(theta)
        arg = 2 * k * r * sth
        A = besselj1(arg)
        B = 2 / np.pi - struve(1, arg)
        # When theta -> pi/2, this ratio converges to (kh)**2
        ratio = np.where(abs(cth) > eps, \
                         (np.sin(k*h*cth)/cth)**2, (k*h)**2)
        if Naz == Nel:
            rcs[idy] = np.pi * r**2 * sth**2 * ratio * (A**2 + B**2)
        else:
            for idx in range(Naz):
                rcs[idx,idy] = np.pi * r**2 * sth**2 * ratio * (A**2 + B**2)
    
    return rcs

#
# Unit test
#
# Has been compared to the rcscylinder function in MATLAB's Radar Toolbox, 
# EM simulations and empirical data. The current RCS model captures the 
# broadside main lobe and the sidelobes at low elevation angles much more 
# accurately, but fails to model the true high sidelobes at high elevation 
# angles that originate from the end plates.
#

if __name__=="__main__":
    
    vhf = 2.24e8 # EISCAT VHF radar frequency [Hz]
    uhf = 9.30e8 # EISCAT UHF radar frequency [Hz]
    fc = uhf
    r = 1.0
    h = 1.0

    #el = np.asarray([-90,-60,-45,-30,0,30,45,60,90])
    el = np.deg2rad(np.asarray(list(range(-90,90))))

    rcs = getOpenCylinderRCS(r,h,fc,0,el)
    rcs_dB = 10*np.log10(rcs+1e-5)
    plt.plot(el.flatten(),rcs_dB.flatten())

"""
    h = 1.0
    rcs = getOpenCylinderRCS(r,h,fc,0,el)
    rcs_dB = 10*np.log10(rcs+1e-5)
    plt.plot(el.flatten(),rcs_dB.flatten(),'k-')
    
    h = 2.0
    rcs = getOpenCylinderRCS(r,h,fc,0,el)
    rcs_dB = 10*np.log10(rcs+1e-5)
    plt.plot(el.flatten(),rcs_dB.flatten(),'r-')

    h = 4.0
    rcs = getOpenCylinderRCS(r,h,fc,0,el)
    rcs_dB = 10*np.log10(rcs+1e-5)
    plt.plot(el.flatten(),rcs_dB.flatten(),'k--')

    h = 8.0
    rcs = getOpenCylinderRCS(r,h,fc,0,el)
    rcs_dB = 10*np.log10(rcs+1e-5)
    plt.plot(el.flatten(),rcs_dB.flatten(),'r--')

    h = 16.0
    rcs = getClosedCylinderRCS(r,h,fc,0,el)
    rcs_dB = 10*np.log10(rcs+1e-5)
    plt.plot(el.flatten(),rcs_dB.flatten(),'g:')
"""