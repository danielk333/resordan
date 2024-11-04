# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 14:55:11 2023

@author: stia
"""

import numpy as np
from scipy.special import jve as besselj
from matplotlib import pyplot as plt


def getCircularPlateRCS(radius,  # radius [m]
                        frequency,  # Wave frequency [Hz]
                        azimuth,  # Azimuth angle [-pi<rad<pi]
                        elevation):  # Elevation angle [-pi/2<=rad<=pi/2]
    '''
        Calculates the radar cross-section for a circular plate defined by 
        its radius, the wave frequency and the elevation angle. The RCS is
        independent of azimuth angle, which is therefore redundant.
    '''
    phir = np.asfarray(azimuth)  # azimuth angle [rad]
    thetar = elevation  # elevation angle [rad]
    c = 2.99792458e8  # speed of light in vacuum [m/s]
    lamda = c / frequency  # wavelength
    k = 2 * np.pi / lamda
    r = radius
    kr = k * r

    # Declare RCS array
    Naz = np.asarray(phir).size
    Nel = np.asarray(thetar).size

    if Naz == Nel:
        rcs = np.zeros(Naz)
    else:
        rcs = np.zeros((Naz, Nel))

    if Naz == 1:
        phir = np.reshape(phir, (1))
    if Nel == 1:
        thetar = np.reshape(thetar, (1))

    for idy in range(Nel):
        theta = thetar[idy]
        if (theta == -np.pi/2) or (theta == np.pi/2):
            # Normal incidence case: elevation = 0 or 180 degrees
            rcs_idy = 4. * np.pi**3 * r**4 / lamda**2
            if Naz == Nel:
                rcs[idy] = rcs_idy
            else:
                for idx in range(Naz):
                    rcs[idx, idy] = rcs_idy
        else:
            # General non-normal incidence case:
            # Using expression from Crispin and Maffett (1965), rewritten
            # with cos(theta) instead of sin(theta) and theta as the 
            # elevation angle to avoid dividing by tan(theta), 
            # which causes problems at theta = 0
            u = 2 * kr * np.cos(theta)
            J1 = besselj(1, u)  # 1st-order Bessel function of the 1st kind
            rcs_idy = np.pi * (r * J1 * np.tan(theta))**2
            if Naz == Nel:
                rcs[idy] = rcs_idy
            else:
                for idx in range(Naz):
                    rcs[idx, idy] = rcs_idy

    return rcs

#
# Unit test
#

#
# Has been tested and verified against the rcsdisc function
# in MATLAB's Radar Toolbox
#


if __name__ == "__main__":

    vhf = 2.24e8  # EISCAT VHF radar frequency [Hz]
    uhf = 9.30e8  # EISCAT UHF radar frequency [Hz]
    fc = uhf
    r = 0.225  # radius [m]
    c = 2.99792458e8  # speed of light (in vacuum)
    h = 1.0  # height [m]
    # el = np.deg2rad(np.asfarray([-90,-60,-45,-30,0,30,45,60,90]))
    el = np.deg2rad(np.asfarray(list(range(-90, 90))))  # elevation angles
    # el = np.deg2rad(np.asfarray([45,46,47,48,49,50])  #,51,52,53,54,55]))
    eps = 1e-6

    rcs = getCircularPlateRCS(r, fc, 0, el)
    rcs_dB = 10*np.log10(rcs+eps)

    # print(rcs)
    plt.plot(el.flatten(), rcs_dB.flatten())
