# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 09:44:09 2023

@author: stia
"""

import numpy as np
from resordan.scrax.get_wire_rcs import getWireRCS
from resordan.scrax.get_circular_plate_rcs import getCircularPlateRCS
from resordan.scrax.get_square_plate_rcs import getSquarePlateRCS
from resordan.scrax.get_sphere_rcs import getSphereRCS
from resordan.scrax.get_simplified_open_cylinder_rcs import getSimplifiedOpenCylinderRCS
from resordan.scrax.get_open_cylinder_rcs import getOpenCylinderRCS
from resordan.scrax.get_closed_cylinder_rcs import getClosedCylinderRCS
from resordan.scrax.dielectric_material import DielectricMaterial as DM
from resordan.scrax.sampling import fibonacciSphereSampling, goldenSpiralSampling, uniformRandomSampling
#from resordan.scrax.tumbling import randomTumbling, tumbling, randomUnitSphere
from resordan.scrax.tumbling import cart2sph

def sample_angles(num_points, # number of samples
                  sampling_method): # sampling method
    
    if sampling_method == 'fibonacci':
        # Fibonacci sphere sampling
        x, y, z = fibonacciSphereSampling(num_points)
    elif sampling_method == 'golden':
        # Golden section spiral sampling
        x, y, z = goldenSpiralSampling(num_points)
    elif sampling_method == 'random':
        # Uniform random sampling
        x, y, z = uniformRandomSampling(num_points)
    else:
        Warning('Unknown sampling method.')
        return None
    
    positions = np.array([x, y, z])
    azimuth_rad, elevation_rad = cart2sph(positions)
    
    return azimuth_rad, elevation_rad

def sampleWireRCS(num_points, # number of samples
                  sampling_method, # sampling method
                  length, # wire length [m]
                  radius, # wire radius [m]
                  frequency, # microwave frequency [Hz]
                  polangle): # polarisation angle [rad]
    
    azimuth, elevation = sample_angles(num_points, sampling_method)
    RCS = np.zeros(num_points)
    
    for (az, el, idx) in zip(azimuth, elevation, range(num_points)):
        RCS[idx] = getWireRCS(length, radius, frequency, polangle, az, el)
    
    return RCS

def sampleCircularPlateRCS(num_points, # number of samples
                           sampling_method, # sampling method
                           radius, # plate radius [m]
                           frequency): # microwave frezuency [Hz]
    
    azimuth, elevation = sample_angles(num_points, sampling_method)
    RCS = np.zeros(num_points)
    
    for (az, el, idx) in zip(azimuth, elevation, range(num_points)):
        RCS[idx] = getCircularPlateRCS(radius, frequency, az, el)
    
    return RCS

def sampleSquarePlateRCS(num_points, # number of samples
                         sampling_method, # sampling method
                         side, # side length [m]
                         frequency): # microwave frezuency [Hz]
    
    azimuth, elevation = sample_angles(num_points, sampling_method)
    RCS = np.zeros(num_points)
    
    for (az, el, idx) in zip(azimuth, elevation, range(num_points)):
        RCS[idx] = getSquarePlateRCS(side, frequency, az, el)
    
    return RCS

def sampleSphereRCS(num_points, # number of samples
                    sampling_method, # sampling method
                    radius, # sphere radius [m]
                    background_material, # DielectricMaterial class object (propagation medium)
                    sphere_material, # DielectricMaterial class object (sphere)                    
                    frequency): # microwave frezuency [Hz]
    
    sensor_location = [0,0,-1e7]
    ratio = radius * frequency / background_material.c_0
    RCS, fc = getSphereRCS(radius, background_material, sphere_material, \
                           ratio, sensor_location)
    
    return RCS * np.ones(num_points)

def sampleSimplifiedOpenCylinderRCS(num_points, # number of samples
                                    sampling_method, # sampling method
                                    radius1, # radius or first semiaxis (if elliptical) [m]
                                    radius2, # second semiaxis or None (if spherical) [m]
                                    height, # cylinder height [m]
                                    frequency): # microwave frequency [Hz]
    
    azimuth, elevation = sample_angles(num_points, sampling_method)
    RCS = np.zeros(num_points)
    
    for (az, el, idx) in zip(azimuth, elevation, range(num_points)):
        RCS[idx] = getSimplifiedOpenCylinderRCS(radius1, radius2, height, frequency, az, el)
    
    return RCS

def sampleClosedCylinderRCS(num_points, # number of samples
                            sampling_method, # sampling method
                            radius, # cylinder radius [m]
                            height, # cylinder height [m]
                            frequency): # microwave frequency [Hz]
    
    azimuth, elevation = sample_angles(num_points, sampling_method)
    RCS = np.zeros(num_points)
    
    for (az, el, idx) in zip(azimuth, elevation, range(num_points)):
        RCS[idx] = getClosedCylinderRCS(radius, height, frequency, az, el)
    
    return RCS

def sampleOpenCylinderRCS(num_points, # number of samples
                          sampling_method, # sampling method
                          radius, # cylinder radius [m]
                          height, # cylinder height [m]
                          frequency): # microwave frequency [Hz]
    
    azimuth, elevation = sample_angles(num_points, sampling_method)
    RCS = np.zeros(num_points)
    
    for (az, el, idx) in zip(azimuth, elevation, range(num_points)):
        RCS[idx] = getOpenCylinderRCS(radius, height, frequency, az, el)
    
    return RCS

#
# Unit test
#

if __name__=="__main__":
    
    # Object properties
    radius = 0.1
    radius1 = radius
    radius2 = 0.01
    side = radius
    length = 1.0
    height = 1.0
    polangle = np.pi/4.
    
    # Speed of light (in vacuum)
    c = 2.99792458e8
    
    # EISCAT radar frequency [Hz]
    vhf = 2.24e8 # EISCAT VHF radar frequency
    uhf = 9.30e8 # EISCAT UHF radar frequency
    frequency = uhf
    ratio = radius * frequency / c
    
    # Define sphere material and propagation medium
    PEC = DM(1e8,0,1e-8,0)
    vacuum = DM(1,0)
    
    #
    # Sample RCS measurements
    #
    
    num_points = 10 # Number of samples
    sampling_method = 'fibonacci'
    
    rcs_wire = sampleWireRCS(num_points, sampling_method, length, radius2, frequency, polangle)
    rcs_cp = sampleCircularPlateRCS(num_points, sampling_method, radius, frequency)
    rcs_sp = sampleSquarePlateRCS(num_points, sampling_method, side, frequency)
    rcs_soc = sampleSimplifiedOpenCylinderRCS(num_points, sampling_method, radius1, None, height, frequency)
    rcs_oc = sampleOpenCylinderRCS(num_points, sampling_method, radius, height, frequency)
    rcs_cc = sampleClosedCylinderRCS(num_points, sampling_method, radius, height, frequency)
    rcs_sph = sampleSphereRCS(num_points, sampling_method, radius, vacuum, PEC, frequency)
    
