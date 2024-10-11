# -*- coding: utf-8 -*-

"""
Created on Thu Aug 10 11:36:40 2023

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

"""
Class: DielectricMaterial

Defines a medium by its electric permittivity, electric conductivity, and 
magnetic permeability as:

DielectricMaterial(eps_r, sigma_e=0, mu_r=1, sigma_m=0, name=None)

Example I: A silicon nanoparticle with a conductivity of 10 S/cm can be 
defined as:

doped_silicon = DielectricMaterial(11.7, 10, 1)

Example II: A perfect electric conductor can be defined by setting electric 
permettivity to a large value and magnetic permeability to a small value such 
that their product equals 1, for example:
    
PEC = DielectricMaterial(1e8,0,1e-8,0)
"""

class DielectricMaterial:
    ''' Translation of KZHU Dielectric material Class
        G. Kevin Zhu (2021). Sphere scattering MATLAB Central File Exchange
        (https://www.mathworks.com/matlabcentral/fileexchange/31119-sphere-scattering)
    '''
    

    def __init__(self, epsilon_r, sigma_e, mu_r=1, sigma_m=0, name=""):
        '''
            epsilon_r:  dielectric permittivity of material in steady state
            sigma_e:    electronic conductivity of material
            mu_r:       relative magnetic permeability in steady state 
                        (assumed to be 1)
            sigma_m:    magnetic conductivity
                        (assumed to be 0)
        '''
        self.epsilon_r = epsilon_r
        self.sigma_e = sigma_e
        self.mu_r = mu_r
        self.sigma_m = sigma_m
        self.eps_0 = 8.8541878176e-12 # vacuum permittivity
        self.mu_0 = 4*np.pi*(1e-7) # vacuum permeability
        self.c_0 = 2.997924580003452e+08 # speed of light
        self.name = name

        if self.getComplexRefractiveIndex(1) == 1:
            self.name = "vacuum"
        
        if (self.epsilon_r > 1e5 and self.mu_r < 1e-5 and \
            self.epsilon_r * self.mu_r == 1):
            self.name = "PEC"


    def convertToAbsoptionDepth(self, distance, frequency):
        z = self.getAbsorptionDepth(frequency)
        z = distance/z
        return z

    def convertToLength(self, f, x):
        '''
            f    frequency, type np.array
            x    length (skin depth)
        '''
        z = x*self.getAbsorptionDepth(f)
        return z
    
    def getAbsorptionDepth(self, frequency):
        '''
            x = getAbsorptionDepth(this, frequency) calculates the absorption
            depth from the wave number. The absorption depth is always
            positive regardless the choice of the time-harmonic factor.
            f    frequency, type np.array
        '''
        k = self.getWaveNumber(frequency)
        x = abs(np.real(1j*k))
        return x

    def getComplexPermeability(self, frequency):
        ''' computes the relative complex permeability.
            Input:
                frequency    Nx1 vector (Hz) type np.array

            Note: will fail for frequency = 0
        '''
        mu_r = self.mu_r + self.sigma_m / (1j*2*np.pi*frequency * self.mu_0)
        return mu_r
    
    def getComplexPermittivity(self, frequency):
        ''' computes the relative complex permittivity.
            Input:
                frequency    Nx1 vector (Hz) type np.array

            Note: will fail for frequency = 0
        '''
        epsilon_r = self.epsilon_r + self.sigma_e / (1j*2*np.pi*frequency * self.eps_0)
        return epsilon_r
    
    def getComplexRefractiveIndex(self, frequency):
        eps_r = self.getComplexPermittivity(frequency)
        mu_r  = self.getComplexPermeability(frequency)
        ref_idx =  np.sqrt(eps_r*mu_r)
        return ref_idx
    
    def getGroupVelocity(self, frequency):
        '''
            v_g = getGroupVelocity(this, frequency) evalutes the group velocity
            by numerically differentiating the angular frequency with respect to
            the wave number.
        ''' 
        pass
    
    def getIntrinsicImpedance(self, frequency):
        
        eta_0 = np.sqrt(self.mu_0 / self.eps_0)
        eta = eta_0 * \
              np.sqrt( self.getComplexPermeability(frequency) / self.getComplexPermittivity(frequency) )
        return eta
    
    def getMagneticPermeability(self):
        ''' returns the object's magnetic permeability
        '''
        return (self.mu_r, self.sigma_m)
    
    def getDielectricPermeability(self):
        ''' returns the object's dielectric permittivity
        '''
        return (self.epsilon_r, self.sigma_e)

    def getPhaseVelocity(self, frequency):
        omega = 2*np.pi*frequency;
        k     = self.getWaveNumber(frequency)
        v_p   = omega/np.real(k);
        return v_p
    
    def getWavelength(self, frequency):
        k = self.getWaveNumber(frequency);
        wavelength = 2*np.pi/np.real(k);
        return wavelength
    
    def getWaveNumber(self, frequency):
        '''
            f    frequency, type np.array
        '''
        permittivity = self.getComplexPermittivity( frequency);
        permeability = self.getComplexPermeability( frequency);
        k = 2*np.pi*(frequency/self.c_0)*np.sqrt((permittivity*permeability));
        return k
    
    def getSkinDepth(self, frequency):
        omega = 2*np.pi*frequency;
        epsilon = self.epsilon_r * self.eps_0
        sigma = self.sigma_e
        mu = self.mu_r * self.mu_0

        skin_depth = np.sqrt( (2/(sigma*mu)) / omega)   *   \
                     np.sqrt( np.sqrt( 1 + ( (epsilon/sigma)*omega )^2   ) + (epsilon/sigma)*omega ) 
        return skin_depth
