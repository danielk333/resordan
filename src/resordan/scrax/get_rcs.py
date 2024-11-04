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
from resordan.scrax.dielectric_material import DielectricMaterial as DM
from resordan.scrax.get_dielectric_sphere_field_under_plane_wave import getDielectricSphereFieldUnderPlaneWave
from resordan.scrax.ref_system import norm

def getSphereRCS(radius, ratio, background_material, sphere_material, sensor_location):
    '''
        Calculates the RCS frequency for a sphere defined by 'radius' and 
        'sphere_material' located at the origin. The incident wavelengths are 
        defined using argument 'ratio', where ratio is radius / wavelength.
    '''
    wavelength = radius / ratio
    frequency = background_material.getPhaseVelocity(3e8 / wavelength)  / wavelength

    [E_r, E_theta, E_phi, H_r, H_theta, H_phi] = \
        getDielectricSphereFieldUnderPlaneWave(radius, sphere_material, background_material, sensor_location, frequency)
    E = (np.stack((E_r,E_theta,E_phi), axis=0))
    RCS = 4*np.pi* ( norm(sensor_location)**2 ) * np.sum( (E * np.conj(E)) , 0)

    return (frequency, RCS)

#
# Unit test
#

c = 299792458 # speed of light in vacuum [m/s]
radius = 0.10 # sphere radius
fc = 224.0e6 # EISCAT radar frequency [Hz]
ratio = radius * fc / c
PEC = DM(1e8,0,1e-8,0)
vacuum = DM(1,0)
sensor_location = [0,0,-1e6]
[fc, RCS] = getSphereRCS(radius, ratio, vacuum, PEC, sensor_location)
