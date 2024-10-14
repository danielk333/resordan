# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:56:28 2023

@author: stia
"""

import numpy as np
import matplotlib.pyplot as plt

def getSimplifiedOpenCylinderRCS(radius1, # radius/1st semiaxis of circular/elliptical base [m]
                                 radius2, # None or 2nd semiaxis of elliptical base [m]
                                 height, # Cylinder height [m]
                                 frequency, # Wave frequency [Hz]
                                 azimuth, # Azimuth angle [-pi<rad<pi]
                                 elevation): # Elevation angle [-pi/2<=rad<=pi/2]
    '''
        Calculates the radar cross-section for a circular cylinder defined by 
        its radii and height, the wave frequency and elevation angle. Note 
        that the cylinder is open: it has no bases or end pieces.
    '''
    eps = 1e-5 # small number
    phir = np.asfarray(azimuth) # azimuth angle [rad]
    thetar = elevation + np.pi/2 # aspect angle [rad]
    c = 2.99792458e8 # speed of light in vacuum [m/s]
    lamda = c / frequency # wavelength

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

    # Treat circular and elliptic case separately
    if (radius2 == None) or (radius1 == radius2):
        # Circular cylinder
        r = radius1 # radius
        for idx in range(Naz):
            for idy in range(Nel):
                theta = thetar[idy]
                if abs(abs(theta) - np.pi/2) > eps:
                    # Compute RCS with general formula
                    rcs[idx,idy] = (lamda * r * np.sin(theta) / \
                                    (8. * np.pi * np.cos(theta)**2))
                else:
                    # Compute RCS for broadside specular case: theta = pi/2
                    h2 = height * height
                    rcs[idx,idy] = 2 * np.pi * h2 * r / lamda
    else:
        # Elliptic cylinder
        r12 = radius1 * radius1 # 1st semiaxis squared
        r22 = radius2 * radius2 # 2nd semiaxis squared
        h2 = height * height # height squared

        for idx in range(Naz):
            for idy in range(Nel):
                theta = thetar[idy]
                phi = phir[idx]
                if abs(abs(theta) - np.pi/2) > eps:
                    # Compute RCS with general formula

                    rcs[idx,idy] = lamda * r12 * r22 * \
                        np.sin(theta) * np.sign(theta) / \
                            (8. * np.pi * np.cos(theta)**2 * \
                             (r12 * np.cos(phi)**2 + \
                              r22 * np.sin(phi)**2)**1.5)
                else:
                    # Compute RCS for broadside specular case
                    thetar = np.pi / 2
                    rcs[idx,idy] = 2 * np.pi * h2 * r12 * r22 / \
                            (lamda * (r12 * np.cos(phi)**2 + \
                                      r22 * np.sin(phi)**2)**1.5)
    
    return np.squeeze(rcs)

#
# Unit test
#
# Has been tested and verified against the rcscylinder function in Matlab's
# Radar Toolbox. It reproduces the Matlab function, but does not match 
# results from EM simulations and empirical data. The function gives a sharp
# and overestimated peak at broadside (zero elevation). It also misses nulls 
# and sidelobes patterns. It does not model the high sidelobes at high 
# elevation angles that would result from the end plates of a closed cylinder.
#

if __name__=="__main__":
    
    vhf = 2.24e8 # EISCAT VHF radar frequency [Hz]
    uhf = 9.30e8 # EISCAT UHF radar frequency [Hz]
    fc = uhf
    r = 1.0#0.125
    r1 = r#0.1
    r2 = r1#0.2
    h = 1.0
    el = np.deg2rad(np.asfarray([-90,-60,-45,-30,0,30,45,60,90]))
    rcs = getSimplifiedOpenCylinderRCS(r1,r2,h,fc,0,el)
    rcs_dB = 10*np.log10(rcs+1e-5)
    
    #print(rcs,rcs_dB)
    
    el = np.deg2rad(np.asfarray(list(range(-90,90))))
    rcs = getSimplifiedOpenCylinderRCS(r1,r2,h,fc,0,el)
    rcs_dB = 10*np.log10(rcs+1e-5)
    
    #print(rcs,rcs_dB)
    plt.plot(el.flatten(),rcs_dB.flatten())
    
