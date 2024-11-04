# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 23:44:50 2023

@author: stia
"""

import numpy as np
from matplotlib import pyplot as plt

def getWireRCS(length, # wire length [m]
               radius, # wire radius [m]
               frequency, # wave frequency [Hz]
               polangle, # polarisation angle [rad]
               azimuth, # azimuth angle [-pi < rad < pi] 
               elevation): # elevation angle [-pi/2 < rad < pi/2]
    '''
        Calculates the radar cross-section for a perfectly conducting,
        nonresonant wire that is many wavelengths long but only a fraction  
        of a wavelength in diameter by use of Chu's formula. 
        The RCS is defined by the wire's length and radius, the wave 
        frequency, the angle between the wire and the direction of 
        incidence, and the angle between the polarization direction and 
        the plane defined by the wire and the direction of incidence
    '''
    thetar = np.asarray(elevation + np.pi/2.) # aspect angle [rad]
    phir = np.asarray(polangle) # polarisation angle [rad]
    c = 2.99792458e8 # speed of light in vacuum [m/s]
    lamda = c / frequency # wavelength (lambda)
    r = radius # kr << 1
    L = length / 2 # formula uses half-length, kL >> 1
    #k = 2 * np.pi / lamda
    #kL = k * L
    gamma = np.exp(.5772) # constant in Chu's formula
    eps = 1e-6 # small number

    # Declare RCS array
    Nth = np.asarray(thetar).size
    Nph = np.asarray(phir).size
    if Nth == Nph:
        rcs = np.zeros(Nth)
    else:
        rcs = np.zeros((Nth,Nph))
    if Nth == 1:
        thetar = np.reshape(thetar,(1))
    if Nph == 1:
        phir = np.reshape(phir,(1))

    if Nth == Nph:
        for idx in range(Nth):
            # Compute RCS
            tr = thetar[idx]
            pr = phir[idx]
            u = 2 * np.pi * L / lamda * np.cos(tr)
            v = lamda / (gamma * np.pi * r * np.sin(tr) + eps)
            rcs[idx] = np.pi * L**2 * np.sin(tr)**2 \
                * np.sinc(u / np.pi)**2 * np.cos(pr)**4 \
                    / ((np.pi / 2)**2 + np.log(v)**2)
        #rcs = rcs.flatten()
    else:
        for idx in range(Nth):
            for idy in range(Nph):
                # Compute RCS
                tr = thetar[idx]
                pr = phir[idy]
                u = 2 * np.pi * L / lamda * np.cos(tr)
                v = lamda / (gamma * np.pi * r * np.sin(tr) + eps)
                rcs[idx,idy] = np.pi * L**2 * np.sin(tr)**2 \
                    * np.sinc(u / np.pi)**2 * np.cos(pr)**4 \
                        / ((np.pi / 2)**2 + np.log(v)**2)
    
    return rcs

#
# Unit test
#

if __name__=="__main__":
    
    vhf = 2.24e8 # EISCAT VHF radar frequency [Hz]
    uhf = 9.30e8 # EISCAT UHF radar frequency [Hz]
    fc = uhf
    L = 1.
    r = 5.25e-3
    c = 2.99792458e8
    az = []
    el = np.deg2rad(60.)
    el = np.deg2rad(np.asarray(list(range(-90,90))))
    pol = np.pi/4 # 45.
    
    eps = 1e-6
    rcs = getWireRCS(L,r,fc,pol,[],el)
    rcs_dB = 10*np.log10(rcs+eps)
    
    #print(rcs,rcs_dB)
    plt.plot(el.flatten(),rcs_dB.flatten())
    
