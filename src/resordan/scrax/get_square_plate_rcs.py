# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 23:20:16 2023

@author: stia
"""

import numpy as np
from matplotlib import pyplot as plt

def getSquarePlateRCS(side, # side length [m]
                      frequency, # Wave frequency [Hz]
                      azimuth, # Azimuth angle [-pi<rad<pi]
                      elevation): # Elevation angle [-pi/2<=rad<=pi/2]
    '''
        Calculates the radar cross-section for a square plate defined by 
        its side length, the wave frequency, the azimuth angle and the
        elevation angle.
    '''
    phir = np.asfarray(azimuth) # azimuth angle [rad]
    thetar = elevation # elevation angle [rad]
    c = 2.99792458e8 # speed of light in vacuum [m/s]
    lamda = c / frequency # wavelength
    k = 2 * np.pi / lamda
    a = side
    ka = k * a

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

    #for idx in range(Naz):
    for idy in range(Nel):
#        if (thetar[idy] == -np.pi/2.) or (thetar[idy] == np.pi/2.):
#            # Normal incidence case: elevation = 0 or 180 degrees
#            rcs[idx,idy] = 4. * np.pi**3 * a**4 / lamda**2
#        else:
        # General non-normal incidence case
        u = ka * np.sin(thetar[idy])
        rcs_idy = 4 * np.pi * a**4 / lamda**2 * np.sinc(u / np.pi)**2
        if Naz == Nel:
            rcs[idy] = rcs_idy
        else:
            for idx in range(Naz):
                rcs[idx,idy] = rcs_idy
    
    return rcs

#
# Unit test
#

#
# Has NOT been tested against any reference functions.
# MATLAB's Radar Toolbox does not simulate the RCS of square plates
#

if __name__=="__main__":
    
    vhf = 2.24e8 # EISCAT VHF radar frequency [Hz]
    uhf = 9.30e8 # EISCAT UHF radar frequency [Hz]
    fc = uhf
    a = 0.225
    c = 2.99792458e8
    el = np.deg2rad(np.asfarray(list(range(-90,90))))
    
    eps = 1e-6
    rcs = getSquarePlateRCS(a,fc,0,el)
    rcs_dB = 10*np.log10(rcs+eps)
    
    #print(rcs)
    plt.plot(el.flatten(),rcs_dB.flatten())
    
