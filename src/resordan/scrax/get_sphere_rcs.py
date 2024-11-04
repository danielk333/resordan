# -*- coding: utf-8 -*-

"""
Created on Thu Aug 10 11:48:58 2023

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
import matplotlib.pyplot as plt
from resordan.scrax.dielectric_material import DielectricMaterial as DM
from resordan.scrax.get_dielectric_sphere_field_under_plane_wave import getDielectricSphereFieldUnderPlaneWave
from resordan.scrax.ref_system import norm

def getSphereRCS(radius, # sphere radius
                 background_material, # DielectricMaterial class object (propagation medium)
                 sphere_material, # DielectricMaterial class object (sphere)
                 ratio, # ratio = radius * frequency / c (light speed)
                 sensor_location): # Sensor location wrt sphere [x,y,z]
    '''
        Calculates the RCS frequency for a sphere defined by 'radius' and 
        'sphere_material' located at the origin. The incident wavelengths are 
        defined using argument 'ratio', where ratio is radius / wavelength.
        The incident plane wave is polarised in the +x direction and 
        propagating in the +z direction.
    '''
    lamda = radius / ratio # = c / frequency
    c = 2.99792458 # speed of light (in propagation medium)
    frequency = background_material.getPhaseVelocity(c / lamda) / lamda
    frequency = np.asarray(frequency, dtype=float, order='C')

    [E_r, E_theta, E_phi, H_r, H_theta, H_phi] = \
        getDielectricSphereFieldUnderPlaneWave(radius,
                                               sphere_material,
                                               background_material,
                                               sensor_location,
                                               frequency)
    E = (np.stack((E_r,E_theta,E_phi), axis=0))
    RCS = 4*np.pi* ( norm(sensor_location)**2 ) \
        * np.sum( (E * np.conj(E)) , 0)
    RCS = abs(RCS)

    return RCS

#
# Unit test
#
# Has been tested and verified against the rcssphere function
# in Matlab's Radar Toolbox
#
# Has been extended to handle multiple frequencies
#

if __name__=="__main__":
    
    c = 2.99792458e8 # speed of light in vacuum [m/s]
    
    # Define radius
    #r = np.zeros(2)
    #r[0] = 0.10 # sphere radius
    #r[1] = r[0] / 2
    r = 0.10 # sphere radius
    
    # Define frequencies
    vhf = 2.24e8 # EISCAT VHF radar frequency [Hz]
    uhf = 9.30e8 # EISCAT UHF radar frequency [Hz]
    fc = uhf
    #fc = np.zeros(2)
    #fc[0] = 224.0e6 # EISCAT radar frequency [Hz]
    #fc[1] = 2*fc[0]
    ratio = r * fc / c
    
    # Define sphere material and propagation medium
    PEC = DM(1e8,0,1e-8,0)
    vacuum = DM(1,0)
    
    # Define sensor location
    sensor_location = [0,0,-1e7]
    
    # Compute RCS
    RCS = getSphereRCS(r, vacuum, PEC, ratio, sensor_location)
    print(RCS,10*np.log10(RCS))
    
    plot_rcs_vs_scale = False

    if plot_rcs_vs_scale:
        # Plot RCS as function of radius to wavelength ratio
        #el = np.asarray(list(range(-90,90)))
        ratio = 5*np.logspace(-2.0, 0.0, num=200)
        rcs, fc = getSphereRCS(r, vacuum, PEC, ratio, sensor_location)
        rcs_dB = 10*np.log10(rcs+1e-5)

        #print(rcs,rcs_dB)
        plt.rcParams['text.usetex'] = True
        fig, ax = plt.subplots()
        ax.semilogx(ratio.flatten(),rcs_dB.flatten())
        ax.set_xlabel(r'Radius-to-wavelength ratio')
        ax.set_ylabel('RCS [dB]')
        ax.set_title(r'RCS as function of $R/\lambda$')
    
