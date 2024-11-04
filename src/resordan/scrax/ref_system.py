# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 11:40:14 2023

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
import scipy.special as ss

def cartToSph(x,y,z):
    '''
        [r, theta, phi] = cartToSph(x, y, z) converts the cartesian
        coordinate system to the spherical coordinate system according to
        the following definition:
        r       distance from the origin to the point in the interval 
                [0, \infty)
        theta   elevation angle measured between the positive z-axis and
                the vector in the interval [0, pi]
        phi     azimuth angle measured between the positive x-axis and
                the vector in the interval [0, 2*pi)
    '''

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y , x)

    return [r, theta, phi]

def sphToCart(r,theta,phi):
    '''
        [x,y,z] = sphToCart(r,theta,phi)
        for converting from spherical to cartesian coordinates
        r is radius, 
        theta is angle of elevation (0 at positive z axis, pi at negative z axis)
        phi is angle of azimuth (0 at positive x axis, increase counterclockwise)
    '''
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    return [x,y,z]

def getLegendre(n,m,x):
    '''
        Returns an array dimensions len(N) by 1 with the 
        value of the m-th degree term of the n-th order 
        associated legendre polynomial evaluated at x. 

        Inputs: 
            n: a sequence of integers  
            m: a single integer, for now.
            x: the argument to legenre polynomial
        Output: 
            P
    '''
    P = []
    for i in range(0,len(n)):
        # use scipy.special to computePmn(x)
        #into an m+1 x n+1 array for every value
        #0...m, 0...n.
        a,b = ss.lpmn(m,n[i],x)
        #select the value at the m,n of interest
        P.append(a[m,n[i]])
    return P

def norm(sensor_location):
    '''
        return the pythagorean distance from the sensor location
        to the origin. 
    '''
    x = sensor_location[0]
    y = sensor_location[1]
    z = sensor_location[2]
    return np.sqrt(x**2 + y**2 + z**2)
