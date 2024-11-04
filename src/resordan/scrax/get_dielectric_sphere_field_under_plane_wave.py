# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 12:00:05 2023

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
from resordan.scrax.ref_system import cartToSph, getLegendre
from resordan.scrax.bessel import ricBesselJ, ricBesselJDerivative, \
    ricBesselH, ricBesselHDerivative
from resordan.scrax.get_n_max import getNMax

def getDielectricSphereFieldUnderPlaneWave(radius, 
                                           sphere, 
                                           background, 
                                           sensor_location, 
                                           frequency):
    '''
        Calculate the field as a plane wave is scattered by a dielectric
        sphere centered at the origin. The incident plane wave is
        polarized in the +x direction propagating in the +z
        direction.

        Inputs:
        radius             Scalar to denote the radius of the sphere (m)
        sphere             Object of DielectricMaterial
        background         Object of DielectricMaterial
        sensor_location    3x1 vector (m)
        frequency          Nx1 vector (Hz)

        Outputs:
        E_r, E_phi, E_theta     Nx1 vector (V/m)
        H_r, H_phi, H_theta     Nx1 vector (A/m)           
       
    '''
    # Convert sensor location (3D coordinates) to:
    # distance (r), elevation angle (theta) and azimuth angle (phi)
    [r, theta, phi] = cartToSph(sensor_location[0],sensor_location[1],sensor_location[2])

    # Number of Mie terms to evaluate based on Wiscombe recommendation
    N = getNMax(radius, sphere, background, frequency)
    nu = np.arange(1,N+1,1)
    
    # Need to convert frequency into numpy array
    if (type(frequency) == int or type(frequency) == float):
        frequency = [frequency]
        M = np.asarray(frequency).size
    if (type(frequency) == list or type(frequency) == np.ndarray):
        frequency = np.array(frequency)
        frequency = frequency.flatten() 
        M = len(frequency)
    else:
        #print("wrong data type for frequency (in getDielectricSphereFieldUnderPlaneWave)")
        M = np.asarray(frequency).size

    EPS_0   = 8.8541878176*1e-12 # vacuum permittivity
    MU_0    = 4*np.pi*1e-7 # vacuum permeability

    #all these values are of class 'numpy.ndarray'
    eta_m   = DM.getIntrinsicImpedance(background, frequency)
    k_m     = DM.getWaveNumber(background, frequency)
    mu_m    = DM.getComplexPermeability(background, frequency)*MU_0
    eps_m   = DM.getComplexPermittivity(background, frequency)*EPS_0
    
    #eta_s   = DM.getIntrinsicImpedance(sphere, frequency)
    k_s     = DM.getWaveNumber(sphere, frequency)
    mu_s    = DM.getComplexPermeability(sphere, frequency)*MU_0
    eps_s   = DM.getComplexPermittivity(sphere, frequency)*EPS_0
 
    a_n = np.ones((np.asarray(frequency).size, len(nu)), np.complex128)
    for c in range(0, len(nu)):
        n = nu[c]
        a_n[:,c] = (1j ** (-1*n)) * (2*n + 1) / (n * (n+1) )

    #range iterates through same numbers
    #math n not modified, index n adjusted for python zero-indexing
    aux0 = np.zeros((len(nu), 1), np.complex128); 
    aux0[0] = -1;
    aux0[1] = -3*np.cos(theta)
    for n in range(2, N):
        aux0[n] = (2*n+1)/n*np.cos(theta)*aux0[n-1] - (n+1)/n*aux0[n-2]
    
    aux1 = np.zeros((len(nu), 1), np.complex128); 
    aux1[0] = np.cos(theta);
    for n in range(2, N+1):
        aux1[n-1] = (n+1)*aux0[n-2] -n*np.cos(theta)*aux0[n-1]
    
    aux0 = np.matmul(np.ones((np.asarray(frequency).size,1)),
                     np.reshape(aux0, (1, len(aux0))))
    aux1 = np.matmul(np.ones((np.asarray(frequency).size,1)),
                     np.reshape(aux1, (1, len(aux1))))

    nr = np.asarray(radius).size
    
    #for idx in range(1,nr):
    for idx in range(nr):
        
        if nr == 1:
            R = radius
        else:
            R = radius[idx]
        
        if r <= R:
            print("Error: Sensor cannot be located within sphere.")
            break
        
        A = np.matmul(np.reshape(np.sqrt(mu_s*eps_m), (M,1)), 
                      np.ones((1,N)))
        B = np.transpose(ricBesselJ(nu,k_m*R))
        C = np.transpose(ricBesselJDerivative(nu,k_s*R))
        D = np.matmul(np.reshape(np.sqrt(mu_m*eps_s), (M,1)), 
                      np.ones((1,N)))
        E = np.transpose(ricBesselJ(nu,k_s*R))
        F = np.transpose(ricBesselJDerivative(nu,k_m*R))
        num = A*B*C - D*E*F

        G = np.transpose(ricBesselHDerivative(nu, k_m*R, 2,1))
        H = np.transpose(ricBesselH(nu,k_m*R,2));
        den = D*E*G - A*H*C;
        
        b_n = (num/den)*a_n
        ####calculating b_n series
        num = A*E*F - D*B*C        
        den = D*H*C - A*E*G
        c_n = (num/den)*a_n
        
        #cleaning the b_n, c_n matrices to remove inf, nan 
        # that come after values become very small
        for i in range(1, M):
            num_zeros = 0
            for j in range(0,N):
                if (abs( b_n[i,j]) < 1e-300 ):
                    num_zeros += 1
                if (num_zeros > 4):
                    b_n[i, j:] = 0
                    num_zeros = 0
                    break
        for i in range(1, M):
            num_zeros = 0
            for j in range(0,N):
                if (abs( c_n[i,j]) < 1e-300 ):
                    num_zeros += 1
                if (num_zeros > 4):
                    b_n[i, j:] = 0
                    num_zeros = 0
                    break
        
        x = k_m*r
        alpha00 = np.transpose(ricBesselHDerivative(nu,x,2,2));   
        alpha01 = np.transpose(ricBesselH(nu,x,2));
        alpha10 = np.array(getLegendre(nu,1, np.cos(theta)))
        alpha11 = np.transpose(np.matmul(np.reshape(alpha10, (N,1)), 
                                         np.ones((1,M)) ) )
        alpha = (alpha00 + alpha01) * alpha11

        E_r = -1j * np.cos(phi) * np.sum((b_n * alpha), 1)
        H_r = -1j * np.sin(phi) * np.sum((c_n * alpha), 1) / eta_m

        alpha = np.transpose(ricBesselHDerivative(nu,x,2)) * aux1
        beta = np.transpose(ricBesselH(nu,x,2)) * aux0
        summation1 = 1j*b_n*alpha - c_n*beta
        summation2 = 1j*c_n*alpha - b_n*beta
        E_theta = (np.cos(phi)/ x) * np.sum(summation1,1)
        H_theta = (np.sin(phi)/x ) * np.sum(summation2,1) / eta_m
        
        alpha = np.transpose(ricBesselHDerivative(nu,x,2)) * aux0
        beta = np.transpose(ricBesselH(nu,x,2)) * aux1;
        summation1 = 1j*b_n*alpha - c_n*beta
        summation2 = 1j*c_n*alpha - b_n*beta
        E_phi = (np.sin(phi)/x) * np.sum(summation1,1)
        H_phi = (-1* np.cos(phi)/x) * np.sum(summation2,1) / eta_m
    
    return [E_r, E_theta, E_phi, H_r, H_theta, H_phi]
