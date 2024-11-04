# -*- coding: utf-8 -*-
import numpy as np
import mpmath as mp

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

#
# Bessel functions
#

# Set precision to 50 decimal places (default: 15).
# This will be applied only to evaluate besselj, bessely functions
mp.dps = 50


def ricBesselJ(nu, x, scale = 1):
    '''
        Implementation of Riccati-Bessel function
        Inputs: 
            nu: Order of Riccati-Bessel function. Column vector of integers 
            from the interval [1,N]
            x: rpw vector of M complex-valued arguments, M = len(frequency)
            scale: equals 1 by default, scales the output of bessel function 
                    by e**(-abs(imag(x))). if zero, does not scale. 
                    (not recommended)
        Output: 
            sqrt(pi*x/2)* J(nu+1/2, x)
            returned as an N by M array for each N values of nu and each M values of x
        Notes:    
            scale factors will cancel out in the end, and minimize numerical issues due to
            arguments with very large complex arguments. 
    '''
    M = np.asarray(x).size
    N = np.asarray(nu).size
    if N == 1:
        nu = np.reshape(nu, (1,))
    a = np.zeros((N, M), np.complex128)

    if (scale != 0 and scale != 1):
        print("incorrect argument ric_besselj (scale)")
        return

    np_besselj = np.frompyfunc(mp.besselj, 2, 1)
    for i in range(N):
        for j in range(M):
            if M == 1:
                rpw = x
            else:
                rpw = x[j]
            if (scale == 1):
                # y = np.sqrt(np.pi * rpw / 2) * mp.besselj(nu[i]+0.5, rpw) \
                #     * np.e**(-1*abs(np.imag(rpw)))
                y = np.sqrt(np.pi * rpw / 2) * np_besselj(nu[i]+0.5, rpw) \
                    * np.e**(-1*abs(np.imag(rpw)))
            elif (scale == 0):
                # y = np.sqrt(np.pi * rpw / 2) * mp.besselj(nu[i]+0.5, rpw)
                y = np.sqrt(np.pi * rpw / 2) * np_besselj(nu[i]+0.5, rpw)
            a[i, j] = complex(y.real, y.imag)

    return a


def ricBesselJDerivative(nu, x, flag = 1):
    '''
        translation of KZHU ric_besselj_derivative(nu,x,flag)
        Inputs:
            nu: order of Riccati-Bessel Function, integer sequence from 1:N 
            as an array
            x: arguments to Riccati-Bessel Function, as a vector with M 
            elements, where M is number of frequencies
            flag: 1 for first order derivative, 2 for second order derivative. 
    '''
    M = np.asarray(x).size

    tmp = np.matmul(np.ones((len(nu), 1)), np.reshape(np.array(x), (1, M)))

    if (flag == 1):
        J = 0.5*(ricBesselJ(nu-1, x) + (1/tmp)*ricBesselJ(nu, x) - ricBesselJ(nu+1, x))
    elif (flag == 2):
        J = 0.5 * (
            ricBesselJDerivative(nu-1, x)
            + (1/tmp)*ricBesselJDerivative(nu, x)
            - (1/(tmp**2)) * ricBesselJ(nu, x)
            - ricBesselJDerivative(nu+1, x)
        )
    else:
        print('error: check third argument passed to ric_besselj_derivative (flag)')

    # removing all the zeros from inside the matrix...
    # f x*nu was 0, then it should become 0 after calculation
    x = np.reshape(np.array(x), (1, M))
    nu = np.reshape(np.array(nu), (len(nu), 1))

    tmp1 = np.matmul(np.ones((len(nu), 1)), x)
    J[tmp1 == 0] = 0

    tmp2 = np.matmul(nu, np.ones((1, len(x))))
    if (flag == 1):
        J[tmp1+tmp2 == 0] = 1

    return J


def ricBesselY(nu, x, scale = 1):
    '''
        Implementation of Riccati-Neumann function
        Inputs: 
            nu: column vector of integers from 1 to N inclusive
            x: rpw vector of M complex-valued arguments, M = len(frequency)
            scale: equals 1 by default, scales the output of bessel function 
                    by e**(-abs(imag(x))). if zero, does not scale. 
                    (not recommended)
        Output: 
            Y_{nu}(x) = \sqrt{\frac{\pi x}{2}} Y_{nu+1/2}(x)
            returned as an N by M array for each N values of nu and each M 
            values of x
        Notes:    
            scale factors will cancel out in the end, and minimize numerical 
            issues due to arguments with very large complex arguments.
    '''
    M = np.asarray(x).size
    N = max(np.shape(nu))
    a = np.zeros((N, M), np.complex128)

    if (scale != 0 and scale != 1):
        print("incorrect argument ric_bessely (scale)")
        return

    np_bessely = np.frompyfunc(mp.bessely, 2, 1)
    for i in range(0, len(nu)):
        for j in range(0, M):
            if M == 1:
                rpw = x
            else:
                rpw = x[j]
            if (scale == 1):
                y = np.sqrt(np.pi * rpw / 2) * np_bessely(nu[i]+0.5, rpw) \
                    * np.e**(-1*abs(np.imag(rpw)))
                # y = np.sqrt(np.pi * rpw / 2) 
                # * mp.bessely(nu[i]+0.5, rpw) 
                # * np.e**(-1*abs(np.imag(rpw)))
            elif (scale == 0):
                y = np.sqrt(np.pi * (rpw) / 2) * np_bessely(nu[i]+0.5, rpw)
                # y = np.sqrt(np.pi * (rpw) / 2) * mp.bessely(nu[i]+0.5, rpw)
            a[i, j] = complex(y.real, y.imag)

    # handling the case where x is zero because bessely is poorly defined 
    tmp = np.matmul(np.ones((N, 1)), np.reshape(np.array(x), (1, M)))
    a[tmp == 0] = float('-inf')

    return a


def ricBesselYDerivative(nu, x, flag = 1):
    '''
        Y = ric_bessely_derivative(nu, x, flag) using the recursive
        relationship to calculate the first derivative of the
        Riccati-Neumann's function.

        The Riccati-Neumann function is defined as
            Y_{nu}(x) = \sqrt{\frac{\pi x}{2}} Y_{nu+1/2}(x)

        Inputs:
        nu: order of Riccati-Bessel Function, integer sequence from 1:N as an array
            x: arguments to Ric-Bessel Function, as a vector with M elements, where M is 
                number of frequencies 
                Note: if x == 0, then a ValueError is raised (because bessely(nu,0)= -inf)
                      This should not happen because size parameters are non-zero
            flag: 1 for first order derivative, 2 for second order derivative. 
    '''
    M = np.asarray(x).size
    N = max(np.shape(nu))

    tmp = np.matmul(np.ones((N, 1)), np.reshape(np.array(x), (1, M)))
    if (flag == 1):
        Y = 0.5 * (ricBesselY(nu-1, x) + (1.0/tmp) * ricBesselY(nu, x) - ricBesselY(nu+1, x))
    elif (flag == 2):
        Y = 0.5 * (
            ricBesselYDerivative(nu-1, x) + (1/tmp)*ricBesselYDerivative(nu, x) 
            - (1/(tmp**2)) * ricBesselY(nu, x)
            - ricBesselYDerivative(nu+1, x)
        )
    else:
        print('error: third argument passed to ric_bessely_derivative must be 1 or 2')

    x = np.reshape(np.array(x), (1, M))
    nu = np.reshape(np.array(nu), (N, 1))

    tmp2 = np.matmul(np.ones((N, 1)), x)
    Y[tmp2 == 0] = float('-inf')
    tmp1 = np.matmul(nu, np.ones((1, M)))
    if (flag == 1):
        Y[tmp1+tmp2 == 0] = 1
    elif (flag == 2):
        Y[tmp1+tmp2 == 0] = -1

    return Y


def ricBesselH(nu, x, K):
    '''
        H = ric_besselh(nu, K, x) implement the Hankel function,
        which is defined as
            H_{nu}(x) = \sqrt{\frac{\pi x}{2}} H_{nu+1/2}(x)
        Inputs:
            nu  order of the spherical Bessel's function. Must be a column vector.
            x   Must be a row vector.
            K   1 for Hankel's function of the first kind; 2 for Hankel's
                function of the second kind.
    '''
    if K == 1:
        H = ricBesselJ(nu, x) + 1j*ricBesselY(nu, x)
    elif K == 2:
        H = ricBesselJ(nu, x) - 1j*ricBesselY(nu, x)
    else:
        print('error: third argument passed to ric_besselh must be 1 or 2')

    return H


def ricBesselHDerivative(nu, x, K, flag = 1):
    '''
        H = ric_besselh_derivative(nu, K, x) use the recursive relationship
        to calculate the first derivative of the Hankel function.
            H_{nu}(x) = \sqrt{\frac{\pi x}{2}} H_{nu+1/2}(x)

        Inputs:
            nu      order of the riccati-Hankel's function. Must be a column vector

            K = 1   if it is Hankel's function of the first kind; K=2 if it is 
                    Hankel's function of the second kind.  
            x       Must be a row evector
            flag    1 for the first order derivative; 2 for the second order derivative
    '''
    if K == 1:
        H = ricBesselJDerivative(nu, x, flag) + 1j*ricBesselYDerivative(nu, x, flag)
    elif K == 2:
        H = ricBesselJDerivative(nu, x, flag) - 1j*ricBesselYDerivative(nu, x, flag)
    else:
        print('error: argument K passed to ric_besselh_derivative must be 1 or 2')

    return H
