# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 09:42:53 2023

@author: stia
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

# Convert Cartesian coordinates on unit sphere to azimuth and elevation angles
def cart2sph(cart_pos): # Cartesian positions [3 x N]
    
    x = cart_pos[0,:] # x coordinate
    y = cart_pos[1,:] # y coordinate
    z = cart_pos[2,:] # z coordinate
    
    #r = np.sqrt(x**2 + y**2 + z**2)
    az = np.arctan2(y, x) # azimuth angle
    el = np.arctan2(z, np.sqrt(x**2 + y**2)) # elevation angle
    
    return az, el

# Convert azimuth and elevation angles on unit sphere to Cartesian coordinates
def sph2cart(azimuth_rad, # Azimuth coordinate [rad]
             elevation_rad): # Elevation coordinate [rad]
    
    x = np.sin(elevation_rad) * np.cos(azimuth_rad)
    y = np.sin(elevation_rad) * np.sin(azimuth_rad)
    z = np.cos(elevation_rad)
    points = np.stack((x, y, z), axis=0)
    
    return points

# Calculate rotation axis for given azimuth and elevation angles
def rotation_axis(azimuth_rad, # azimuth angle [rad]
                  elevation_rad): # elevation angle [rad]
    
    # 3D rotation axis as unit sphere vector
    x = np.sin(elevation_rad) * np.cos(azimuth_rad)
    y = np.sin(elevation_rad) * np.sin(azimuth_rad)
    z = np.cos(elevation_rad)
    k = np.array([x, y, z])
    
    return k

# Calculate rotation matrix for given rotation axis and angle
def rotation_matrix(rot_axis, # rotation axis (unit sphere vector)
                    theta): # rotation angle [rad]

    # Rotation matrix for given rotation axis
    cos = np.cos(theta)
    sin = np.sin(theta)
    kx = rot_axis[0]
    ky = rot_axis[1]
    kz = rot_axis[2]
    rot_matrix = np.array([
        [cos+kx**2*(1-cos), kx*ky*(1-cos)-kz*sin, kx*kz*(1-cos)+ky*sin],
        [ky*kz*(1-cos)+kz*sin, cos+ky**2*(1-cos), ky*kz*(1-cos)-kz*sin],
        [kz*kx*(1-cos)-ky*sin, kz*ky*(1-cos)+kx*sin, cos+kz**2*(1-cos)]
    ])
        
    return rot_matrix

# Rotate point on a unit sphere with specified rotation axis and random 
# initial rotation angle. Calculate rotating points' position at a given time
def pointPositions(time, # time axis
                    omega, # angular speed [rad/s]
                    rotation_axis): # specified as unit sphere vector
    
    # Rotation axis (unit sphere vector elements)
    kx = rotation_axis[0]
    ky = rotation_axis[1]
    kz = rotation_axis[2]
    
    # Rotation angles [rad]
    theta_0 = np.random.uniform() * 2 * np.pi # Random initial rotation angle
    theta_rad = time * omega + theta_0
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    N = len(time)

    positions = np.zeros((3,N))
    for cos, sin, n in zip(cos_theta, sin_theta, range(N)):
        # Compute rotation matrix
        rotation_matrix = np.array([
            [cos+kx**2*(1-cos), kx*ky*(1-cos)-kz*sin, kx*kz*(1-cos)+ky*sin],
            [ky*kx*(1-cos)+kz*sin, cos+ky**2*(1-cos), ky*kz*(1-cos)-kx*sin],
            [kz*kx*(1-cos)-ky*sin, kz*ky*(1-cos)+kx*sin, cos+kz**2*(1-cos)]
            ])
        positions[:,n] = rotation_matrix.dot(np.array([1,0,0]))
    
    return positions

# Random generate points on unit sphere (as (azimuth, elevation))
def randomUnitSphere(num_points=1):
    
    points = []
    for n in range(num_points):
        azimuth = np.random.rand() * 2 * np.pi # Random azimuth angle [rad]
        z = 1 - np.random.rand() * 2 # Random sample uniformly from [-1,1]
        elevation = np.arccos(z) # Random elevation angle [rad]
        points.append((azimuth, elevation))
        
    points = np.array(points).T
    azimuth = points[0,:]
    elevation = points[1,:]
    
    return azimuth, elevation

# Simulate tumbling around random rotation axis
def randomTumbling(angular_speed, # angular speed [rad/s]
                    sample_rate, # sampling rate [Hz]
                    num_points, # number of samples []
                    in_degrees=False): # return angles in degrees (or radians)

    '''
    Simulates a tumbling space debris object by random generating a rotation
    axis and returning a given number of azimuth and elevation angles of this 
    object as it tumbles around this axis at the given angular speed and 
    sampling rate. 
    Azimuth angles are measured in the xy-plane from the x-axis.
    Elevation angles are measured between the xy-plane and the z-axis.
    '''

    # Sampling times
    time = np.linspace(0, num_points/sample_rate, num=num_points, endpoint=False)
    
    # Specify random rotation axis
    azimuth_rad = np.random.uniform() * 2 * np.pi # Random azimuth angle [rad]
    z = 1 - np.random.rand() * 2 # Random sample uniformly from [-1,1]
    elevation_rad = np.arccos(z) # Random elevation angle [rad]
    
    # Compute positions during rotation
    k = rotation_axis(azimuth_rad, elevation_rad)
    positions = pointPositions(time, angular_speed, k)

    # Convert to azimuth and elevation angles
    azimuth_angles, elevation_angles = cart2sph(positions)
    
    if in_degrees:
        azimuth_angles = np.rad2deg(azimuth_angles)
        elevation_angles = np.rad2deg(elevation_angles)
    
    return azimuth_angles, elevation_angles

# Simulate tumbling around specified rotation axis
def fixedTumbling(angular_speed, # angular speed [rad/s]
                   sample_rate, # sampling rate [Hz]
                   num_points, # number of samples []
                   rotation_axis, # specified as unit sphere vector
                   in_degrees=False): # return angles in degrees (or radians)

    '''
    Simulates a tumbling space debris object rotating around a rotation axis 
    specified by the given azimuth and elevation angles. Returns a given 
    number of azimuth and elevation angles of this object as it tumbles around
    this axis at the given angular speed and sampling rate. 
    Azimuth angles are measured in the xy-plane from the x-axis.
    Elevation angles are measured between the xy-plane and the z-axis.
    '''

    # Sampling times
    time = np.linspace(0, num_points/sample_rate, num=num_points, endpoint=False)
    
    # Compute positions during rotation
    positions = pointPositions(time, angular_speed, rotation_axis)

    # Convert to azimuth and elevation angles
    azimuth_angles, elevation_angles = cart2sph(positions)
    
    if in_degrees:
        azimuth_angles = np.rad2deg(azimuth_angles)
        elevation_angles = np.rad2deg(elevation_angles)

    return azimuth_angles, elevation_angles

#
# Unit test
#

if __name__=="__main__":
    
    # Time parameters
    num_points = 1000  # Number of points to plot
    angular_speed = 0.5  # Angular speed [rad/s]
    sample_rate = 50 # Sampling rate
    time = np.linspace(0, 20, num_points)  # Time interval
    
    # Generate a random azimuth and elevation angle
    azimuth_rad = np.random.uniform() * 2 * np.pi # Random azimuth angle in radians
    elevation_rad = np.random.uniform() * np.pi # Random elevation angle in radians

    # Rotation axis specified as Cartesian elements of unit sphere vector
    k = rotation_axis(azimuth_rad, elevation_rad)
    
    # Compute positions during rotation
    positions = pointPositions(time, angular_speed, k)
    
    # Convert to azimuth and elevation angles
    az, el = cart2sph(positions)
    
    #az, el = randomTumbling(angular_speed, sample_rate, num_points)
    #az, el = fixedTumbling(angular_speed, sample_rate, num_points, k)
    
    # Plot the rotating point on the unit sphere
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot(positions[0, :], positions[1, :], positions[2, :])
    ax1.set_title('Point rotating on a unit sphere')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.show()
    
    # Plot the corresponding azimuth and elevation angles
    fig2, axs2 = plt.subplots(2)
    axs2[0].plot(time, az/np.pi)
    axs2[0].set_xlabel('Time t')
    axs2[0].set_ylabel('Azimuth angle [rad]')
    #axs2[0].set_ylim([-np.pi, np.pi])
    axs2[0].set_ylim([-1, 1])
    axs2[0].yaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    axs2[0].yaxis.set_major_locator(tck.MultipleLocator(base=1.0))
    axs2[1].plot(time, el/np.pi)
    axs2[1].set_xlabel('Time t')
    axs2[1].set_ylabel('Elevation angle [rad]')
    #axs2[1].set_ylim([-np.pi/2, np.pi/2])
    axs2[1].set_ylim([-1, 1])
    axs2[1].yaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    axs2[1].yaxis.set_major_locator(tck.MultipleLocator(base=1.0))
    plt.show()
    
