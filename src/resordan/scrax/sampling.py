# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 11:48:12 2023

@author: stia
"""

import numpy as np
import matplotlib.pyplot as plt
from resordan.scrax.tumbling import randomUnitSphere, sph2cart, cart2sph

# Systematically generate uniformly sampled points on unit sphere
def fibonacciSphereSampling(num_points=1, # no. data points
                              repeatable=False): # repeatable sample (or add random rotation)
    
    phi = np.pi * (np.sqrt(5.) - 1.) # golden angle [rad]
    
    x = np.zeros(num_points)
    y = np.zeros(num_points)
    z = np.zeros(num_points)
    
    for idx in range(num_points):
        y[idx] = 1 - (idx / float(num_points - 1)) * 2 # y goes from 1 to -1
        radius = np.sqrt(1 - y[idx]**2) # radius at y
        theta = phi * idx # golden angle increment  
        x[idx] = np.cos(theta) * radius
        z[idx] = np.sin(theta) * radius        

    if not repeatable:
        theta_rnd, phi_rnd = randomUnitSphere(1) # random rotations
        points = np.stack((x,y,z),axis=0)
        phi, theta = cart2sph(points)
        phi = phi + phi_rnd # rotated azimuth angles
        theta = theta + theta_rnd # rotated elevation angles
        points = sph2cart(phi, theta)
        [x, y, z] = [points[idx,:].T for idx in range(3)]
        
    return x, y, z

# Systematically generate uniformly sampled points on unit sphere
def goldenSpiralSampling(num_points=1, # no. data points
                           repeatable=False): # repeatable sample (or add random rotation)
    
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = np.pi * (1 + np.sqrt(5)) * indices
    
    if not repeatable:
        phi_rnd = np.random.rand() * 2*np.pi # random azimuth angle rotation
        theta_rnd = np.random.rand() * np.pi # random elevation angle rotation
        phi = np.mod(phi + phi_rnd, 2*np.pi) # rotated azimuth angles
        theta = np.mod(theta + np.pi/2 + theta_rnd, np.pi) - np.pi/2 # rotated elevation angles
    
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    
    return x, y, z

# Random generate uniformly distributed points on unit sphere
def uniformRandomSampling(num_points=1):
    
    # Random generate spherical coordinates on unit sphere
    azimuth, elevation = randomUnitSphere(num_points)
    
    # Convert to Cartesian coordinates
    points = sph2cart(azimuth, elevation).T
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    
    return x, y, z

#
# Unit test
#

if __name__=="__main__":
    
    num_points = 100
    
    # Fibonacci sphere sampling
    xf, yf, zf = fibonacciSphereSampling(num_points)
    
    # Golden section spiral sampling
    xg, yg, zg = goldenSpiralSampling(num_points)
    
    # Uniform random sampling of sphere
    xu, yu, zu = uniformRandomSampling(num_points)
    
    # Create a sphere
    r = 1
    phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2*np.pi:100j]
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title('Fibonacci sphere')
    ax1.plot_surface(
        x, y, z, rstride=1, cstride=1, color='c', alpha=0.4, linewidth=0)
    ax1.scatter(xf, yf, zf, color='k', s=20)
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title('Golden spiral')
    ax2.plot_surface(
        x, y, z, rstride=1, cstride=1, color='c', alpha=0.4, linewidth=0)
    ax2.scatter(xg, yg, zg, color='k', s=20)
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title('Uniform random')
    ax3.plot_surface(
        x, y, z, rstride=1, cstride=1, color='c', alpha=0.4, linewidth=0)
    ax3.scatter(xu, yu, zu, color='k', s=10)
    plt.tight_layout()
    plt.show()
    

