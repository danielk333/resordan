# -*- coding: utf-8 -*-

import numpy as np
import math
import scipy.special as ss
import mpmath as mp
import bessel
import refsystem
from DielectricMaterial import DielectricMaterial as DM



def getNMax(radius, sphere, background, frequency):
    '''
        determines the appropriate number of mie terms to evaluate
        Based on the wiscombe 1980 recommendation (which was deternined through 
        convergence behavior bessel functions at high orders that were
        calculated recursivelly).

        Designed to work for single-layered (monolithic) sphere. 
    '''
    #check that frequency input is correct  
    if (type(frequency) == int or type(frequency) == float):
        frequency = np.array([frequency])
    if (type(frequency) == list or type(frequency) == np.ndarray):
        frequency = np.array(frequency).flatten()
        M = len(frequency)
    else:
        print("wrong data type for frequency (in getNMax)")
    
    
    k_m = DM.getWaveNumber(background, frequency)
    x = abs(k_m * radius)
    #print(x)

    N_m = DM.getComplexRefractiveIndex(background, frequency)
    m = DM.getComplexRefractiveIndex(sphere, frequency) / N_m #relative refractive index

    N_max = np.ones((M,))
    for k in range(0,M):
        if (x[k] < 0.02):
            print("WARNING: it is better to use Rayleigh Scattering models for low frequencies.")
            print("\tNo less than 3 Mie series terms will be used in this calculation")
            #this comes from Wiscombe 1980: for size parameter = 0.02 or less, the number of terms
            #recommended will be 3 or less. 
            N_stop = 3
        elif (0.02 <= x[k] and x[k] <= 8):
            N_stop = x[k] + 4.*x[k]**(1/3) + 1
        elif (8 < x[k] and x[k] < 4200):
            N_stop = x[k] + 4.05*x[k]**(1/3) + 2
        elif (4200 <= x[k] and x[k] <= 20000):
            N_stop = x[k] + 4.*x[k]**(1/3) + 2
        else:
            print("WARNING: it is better to use Physical Optics models for high frequencies.")
            N_stop = 20000 + 4.*20000**(1/3) + 2
        
        #this is the KZHU original nmax formula (adapted for single sphere)
        #it recommends 100's of terms for real metals
        N_max[k] = max(N_stop, abs(m[k] * x[k]) )+15

        #this is the Wiscombe-only implementation, seems to be accurate enough
        #N_max[k] = N_stop
        
    return math.ceil(max(N_max))


#
# Compute radar cross section (RCS) of sphere
#
def rcssphere(r,   # sphere radius [m]
              c,   # wave speed [m/s]
              fc): # wave frequency [Hz]
#              az,  # azimuth angle [rad]
#              el): # elevation angle [rad]
# RCS of sphere is independent of azimuth and elevation angles

    # Format input/output variables
    nr = np.asarray(r).size
    nf = np.asarray(fc).size
    if nr > 1:
        r = np.reshape(r, (nr,1)) # Shape as column vector
    if nf > 1:
        fc = np.reshape(fc, (1,nf)) # Shape as row vector
    rcs = np.zeros((nr, nf)) # Declare output variable
    rcs_db = rcs
    
    # Prepare wavenumber radius product
    k = 2 * np.pi * fc / c # wavenumber
    kr = np.atleast_1d(k * r) # wavenumber radius product
    
    eps = 1e-5 # error tolerance in Mie series computation
    
    for idx, kr in enumerate(kr):

        # Initialisation
        this_rcs = 0.0 + 0.0j # initialising RCS
        f1 = 0.0 + 1.0j
        f2 = 1.0 + 0.0j
        m = 1.0
        n = 0.0
        q = -1.0; # alternating sign in Mie series
        inc = 1.0e5 + 1.0e5j # increment term

        # Mie series computation
        while abs(inc) > eps:
            q = -q
            n += 1
            m += 2
            inc = (2*n-1) * f2 / kr - f1
            f1 = f2
            f2  = inc
            inc = q*m / (f2 * (kr*f1 - n*f2))
            this_rcs = this_rcs + inc
            print(this_rcs)

        rcs[idx,:] = np.abs(this_rcs)
        rcs_db[idx,:] = 10. * np.log10(rcs[idx,:])

    rcs = np.squeeze(rcs)
    rcs_db = np.squeeze(rcs_db)

    print(rcs)
    print(rcs_db)

    return rcs, rcs_db

# Unit test rcssphere

r = 0.20 # radius [m]
c = 3.0e8 # light speed [m/s]
fc = 4.5e9 # frequency [Hz]

rcs = rcssphere(r, c, fc)
