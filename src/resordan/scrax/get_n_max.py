# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 11:42:53 2023

@author: stia
"""

"""
This code is based on the MATLAB "Sphere scattering" authored by Kevin Zhu. 
The original license notice is reproduced below. Redistribution of this 
library with and without modifications is permitted as long as 

  (1) the copyright notice (below) is included,
  (2) the authors of the work are cited as follows: 
        G. Kevin Zhu (2021). Sphere scattering, MATLAB Central File Exchange.
            (https://www.mathworks.com/matlabcentral/fileexchange/31119-sphere-scattering)
        I. Shofman, D. Marek, S. Sharma, P. Triverio, Python Sphere RCS,
            (https://github.com/modelics/Sphere-RCS/)

----------
From Matlab File Exchange, Sphere scattering. 
(https://www.mathworks.com/matlabcentral/fileexchange/31119-sphere-scattering)

Copyright (c) 2011, Kevin Zhu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution
* Neither the name of  nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import math
from resordan.scrax.dielectric_material import DielectricMaterial as DM

def getNMax(radius, sphere, background, frequency):
    '''
        Determines the appropriate number of Mie terms to evaluate
        Based on the Wiscombe 1980 recommendation (which was determined 
        through the convergence behaviour of bessel functions at high orders 
        that were calculated recursively).

        Designed to work for single-layered (monolithic) sphere. 
    '''
    #check that frequency input is correct  
    if (type(frequency) == int or type(frequency) == float):
        frequency = np.array([frequency])
        M = np.asarray(frequency).size
    if (type(frequency) == list or type(frequency) == np.ndarray):
        frequency = np.array(frequency).flatten()
        M = len(frequency)
    else:
        print("wrong data type for frequency (in getNMax)")
        M = np.asarray(frequency).size
    
    
    k_m = DM.getWaveNumber(background, frequency)
    x = abs(k_m * radius)

    N_m = DM.getComplexRefractiveIndex(background, frequency)
    m = DM.getComplexRefractiveIndex(sphere, frequency) / N_m #relative refractive index

    N_max = np.ones((M,))

    for k in range(0,M):
        if M == 1:
            kr = x
        else:
            kr = x[k]
        if (kr < 0.02):
            print("WARNING: it is better to use Rayleigh Scattering models for low frequencies.")
            print("\tNo less than 3 Mie series terms will be used in this calculation")
            #this comes from Wiscombe 1980: for size parameter = 0.02 or less, the number of terms
            #recommended will be 3 or less. 
            N_stop = 3
        elif (0.02 <= kr and kr <= 8):
            N_stop = kr + 4.*kr**(1/3) + 1
        elif (8 < kr and kr < 4200):
            N_stop = kr + 4.05*kr**(1/3) + 2
        elif (4200 <= kr and kr <= 20000):
            N_stop = kr + 4.*kr**(1/3) + 2
        else:
            print("WARNING: it is better to use Physical Optics models for high frequencies.")
            N_stop = 20000 + 4.*20000**(1/3) + 2
        
        #this is the KZHU original nmax formula (adapted for single sphere)
        #it recommends 100's of terms for real metals
        if M == 1:
            N_max = max(N_stop, abs(m * kr) )+15
        else:
            N_max[k] = max(N_stop, abs(m[k] * x[k]) )+15

        #this is the Wiscombe-only implementation, seems to be accurate enough
        #N_max[k] = N_stop
        
    if M > 1:
        N_max = max(N_max)
    
    return math.ceil(N_max)
