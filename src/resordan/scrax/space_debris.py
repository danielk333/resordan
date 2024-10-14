# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:57:20 2023

@author: stia
"""

import numpy as np
from DielectricMaterial import DielectricMaterial
from getSphereRCS import getSphereRCS
from getWireRCS import getWireRCS
from getCircularPlateRCS import getCircularPlateRCS
from getSquarePlateRCS import getSquarePlateRCS
from getOpenCylinderRCS import getOpenCylinderRCS
from getClosedCylinderRCS import getClosedCylinderRCS
from sampling import uniform_random_sampling, fibonacci_sphere_sampling, \
    golden_spiral_sampling
from tumbling import fixed_tumbling, random_tumbling

"""
Class: DebrisMeasurement

Defines a space debris measurement in terms of the wave frequency, the wave 
polarisation and the propagation medium. The propagation medium is given in 
terms of a DielectricMaterial class object.

Example: A measurement with the EISCAT radar is defined as:

    fc = 224.0e6 # Eiscat radar frequency [Hz]
    polangle = np.pi / 4. # Polarisation angle [rad]
    vacuum = DielectricMaterial(1., 0.)
    name = "Eiscat radar"
    DM = DebrisMeasurement(frequency=fc, 
                           pol_angle=polangle, 
                           medium=vacuum, 
                           name=name)
"""
class DebrisMeasurement:
    
    # Initialise with wave frequency, wave polarisation angle and 
    # propagation medium properties defined by DielectricMaterial class object.
    def __init__(self, 
                 frequency=None,
                 pol_angle=None,
                 medium=DielectricMaterial(1.,0.), # default: vacuum
                 name=""):
        '''
        Parameters
        ----------
        frequency : float64
            Microwave frequency
        pol_angle : float64
            Polarisation angle [rad]
        medium : DielectricMaterial class
            Propagation medium (default = vacuum)
        name : string, optional
            Name tag. The default is "".
        '''
        self.frequency = frequency
        self.pol_angle = pol_angle
        self.medium = medium
        self.name = name
        self.wavelength = medium.c_0 / self.frequency
        
        if (self.frequency == None):
            raise TypeError("Frequency must be defined for DebrisMeasurement class object.")

class ObjectOrientation:
    
    def __init__(self,
                 azimuth=None,
                 elevation=None,
                 aspect=None,
                 x=None,
                 y=None,
                 z=None):
        
        self.azimuth = azimuth
        self.elevation = elevation
        self.aspect = aspect
        self.x = x
        self.y = y
        self.z = z

    # Convert unit sphere vector defining observation angle 
    # from Cartesian to spherical coordinates
    def cart2sph(self):
        
        if (self.x==None) or (self.y==None) or (self.z==None):
            raise TypeError("Method cart2sph requires that Cartesian coordinates exist.")
        
        x = self.x # x coordinate
        y = self.y # y coordinate
        z = self.z # z coordinate
        
        self.azimuth = np.arctan2(y, x) # azimuth angle
        self.elevation = np.arctan2(z, np.sqrt(x**2 + y**2)) # elevation angle
    
    # Convert unit sphere vector defining observation angle
    # from spherical to Cartesian coordinates
    def sph2cart(self):
        
        if (self.azimuth==None) or (self.elevation==None):
            raise TypeError("Method sph2cart requires that spherical coordinates exist.")

        az = self.azimuth
        el = self.elevation
        
        self.x = np.sin(el) * np.cos(az)
        self.y = np.sin(el) * np.sin(az)
        self.z = np.cos(el)

"""
Class: DebrisObject

Defines a space debris object in terms of its dielectric material, geometric 
object type, and the dimensions associated with that object.

Examples:

    r = 0.5 # radius [m]
    h = 2.0 # height [m]
    name = "Perfectly conducting closed cylinder"
    PCCC = DebrisObject(radius=r, height=h, objtype='ClosedCylinder', name=name)
    
    PEC = DielectricMaterial(1.e8, 0., 1.e-8, 0.)
    r = 0.1 # radius [m]
    name = "Perfectly conducting sphere"
    PCS = DebrisObject(material=PEC, radius=r, objtype='Sphere', name=name)
    
    epsilon_r = 200. # dielectric permittivity
    sigma_e = 50. # electrical conductivity
    mu_r = 1. # relative magnetic permeability
    sigma_m = 0. # magnetic conductivity
    mission_panel = DielectricMaterial(mu_r=mu_r,
                                       sigma_m=sigma_m,
                                       epsilon_r=epsilon_r,
                                       sigma_e=sigma_e)
    name = "Square mission panel"
    SMP = DebrisObject(material=mission_panel, length=h, objtype='SquarePlate', name=name)
"""
class DebrisObject:
    
    # Initialise with geometric object type, associated dimensions and 
    # material properties defined in DielectricMaterial class object
    def __init__(self, 
                 material=DielectricMaterial(1e8,0,1e-8,0),
                 radius=None,
                 height=None,
                 length=None,
                 objtype=None,
                 orientation=None,
                 name=""):

        # Set parameters        
        self.material=material
        self.radius=radius
        self.height=height
        self.length=length
        self.objtype=objtype
        self.orientation=orientation
        self.name = name
        
        # Initialise with an empty ObjectOrientation object, if not provided
        if (self.orientation==None):
            self.orientation = ObjectOrientation()
        
        # Check for required parameters
        if self.objtype=='Sphere':
            if (self.material==None or self.radius==None):
                raise TypeError("Sphere object must have material and radius parameters.")
        elif self.objtype=='Wire':
            if (self.length==None or self.radius==None):
                raise TypeError("Wire object must have length and radius parameters.")
        elif self.objtype=='CircularPlate':
            if (self.radius==None):
                raise TypeError("Cirlular plate object must have radius parameter.")
        elif self.objtype=='SquarePlate':
            if (self.length==None):
                raise TypeError("Square plate object must have length parameter.")
        elif self.objtype=='OpenCylinder':
            if (self.radius==None or self.height==None):
                raise TypeError("Open cylinder object must have radius and height parameters.")
        elif self.objtype=='ClosedCylinder':
            if (self.radius==None or self.height==None):
                raise TypeError("Closed cylinder object must have radius and height parameters.")
    
    # Collect a sample of orientation angles (azimuth and elevation) using
    # the specified sample size and one of the available sampling methods:
    # 'random' (random uniform sampling on sphere);
    # 'fibonacci' (systematic Fibonacci sphere sampling); or
    # 'golden' (systematic golden spiral sampling on sphere).
    def sample(self,
               num_points, # number of samples
               method='random', # sampling method
               repeatable=False): # repeatable sample (or add random angle)
        
        if method=='random':
            x, y, z = uniform_random_sampling(num_points)
        elif method=='fibonacci':
            x, y, z = fibonacci_sphere_sampling(num_points, repeatable)
        elif method=='golden':
            x, y, z = golden_spiral_sampling(num_points, repeatable)
        else:
            raise TypeError("Unknown sampling method in DebrisObject.sample().")
        
        # Convert Cartesian coordinates to azimuth and elevation angles
        self.orientation.azimuth = np.arctan2(y, x)
        self.orientation.elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
        
        return

    # Collect a sample of orientation angles (azimuth and elevation)
    # by simulated tumbling of the object.
    def tumble(self,
               num_points,
               angular_speed,
               sample_rate,
               rotation_axis=[]):
        
        if rotation_axis==[]:
            # Tumbling around random rotation axis
            x, y, z = random_tumbling(angular_speed, 
                                      sample_rate,
                                      num_points,
                                      in_degrees=False)
        else:
            # Tumbling around specified rotation axis
            x, y, z = fixed_tumbling(angular_speed,
                                     sample_rate,
                                     num_points,
                                     rotation_axis,
                                     in_degrees=False)
        
        # Convert Cartesian coordinates to azimuth and elevation angles
        self.orientation.azimuth = np.arctan2(y, x)
        self.orientation.elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
        
        return

    # Perform RCS measurement with sensor properties specified in 
    # DebrisMeasurement class object and at the object orientation specified 
    # by the given azimuth and elevation angles
    def measure(self,
                measurement, # DebrisMeasurement class object
                num_points, # number of samples
                method): # sampling method

        # Sample orientation angles for simulated measurements
        self.sample(num_points=num_points, method=method)

        # Orientation angles
        azimuth = self.orientation.azimuth
        elevation = self.orientation.elevation
        
        #if ((azimuth==None or elevation==None) and (self.objtype!='Sphere')):
        #    raise TypeError("ObjectOrientation object must contain azimuth and elevation angles.")
        
        if self.objtype=='Sphere':
            # if (measurement.frequency==None or measurement.medium==None):
            #     raise TypeError("Measurement object must provide frequency \
            #                     and medium parameters.")
            sensor_location = [0,0,-1e7] # default sensor location
            rcs_sphere = getSphereRCS(self.radius, 
                                      measurement.medium, 
                                      self.material, 
                                      self.radius / measurement.wavelength, 
                                      sensor_location)
            Naz = np.asarray(azimuth).size
            Nel = np.asarray(elevation).size
            if Naz == Nel:
                rcs = rcs_sphere * np.ones(Naz)
            else:
                rcs = rcs_sphere * np.ones((Naz,Nel))
        elif self.objtype=='Wire':
            # if (measurement.frequency==None or measurement.pol_angle==None or
            #     azimuth==None or elevation==None):
            #     raise TypeError("Measurement and orientation objects must \
            #                     provide frequency, polarisation angle, \
            #                     azimuth angle and elevation angle parameters.")
            rcs = getWireRCS(self.length, 
                             self.radius, 
                             measurement.frequency, 
                             measurement.pol_angle,
                             azimuth,
                             elevation)
        elif self.objtype=='CircularPlate':
            # if (measurement.frequency==None or azimuth==None or elevation==None):
            #     raise TypeError("Measurement and orientation objects must \
            #                     provide frequency, azimuth angle and \
            #                     elevation angle parameters.")
            rcs = getCircularPlateRCS(self.radius, 
                                      measurement.frequency, 
                                      azimuth, 
                                      elevation)
        elif self.objtype=='SquarePlate':
            # if (measurement.frequency==None or azimuth==None or elevation==None):
            #     raise TypeError("Measurement and orientation objects must \
            #                     provide frequency, azimuth angle and \
            #                     elevation angle parameters.")
            rcs = getSquarePlateRCS(self.length, 
                                    measurement.frequency, 
                                    azimuth, 
                                    elevation)
        elif self.objtype=='OpenCylinder':
            # if (measurement.frequency==None or azimuth==None or elevation==None):
            #     raise TypeError("Measurement and orientation objects must \
            #                     provide frequency, azimuth angle and \
            #                     elevation angle parameters.")
            rcs = getOpenCylinderRCS(self.radius,
                                     self.height, 
                                     measurement.frequency,
                                     azimuth, 
                                     elevation)
        elif self.objtype=='ClosedCylinder':
            # if (measurement.frequency==None or azimuth==None or elevation==None):
            #     raise TypeError("Measurement and orientation objects must \
            #                     provide frequency, azimuth angle and \
            #                     elevation angle parameters.")
            rcs = getClosedCylinderRCS(self.radius,
                                       self.height,
                                       measurement.frequency, 
                                       azimuth, 
                                       elevation)
        
        return rcs

#
# Unit test
#

if __name__=="__main__":
    
    #
    # Sample RCS measurements
    #

    # Define perfectly conducting closed cylinder
    r = 0.1 # radius
    h = 1.0 # height
    name = "Perfectly conducting closed cylinder"
    PCCC = DebrisObject(radius=r, height=h, objtype='ClosedCylinder', name=name)

    # Define measurement setup
    vhf = 2.24e8 # EISCAT VHF radar frequency
    uhf = 9.30e8 # EISCAT UHF radar frequency
    fc = uhf
    c = 2.99792458e8 # speed of light (in vacuum)
    #ratio = r * fc / c
    polangle = np.pi/4.
    vacuum = DielectricMaterial(1., 0.) # propagation medium: vacuum
    eiscat = DebrisMeasurement(frequency=fc, 
                               pol_angle=polangle, 
                               medium=vacuum,
                               name='EISCAT radar')
    
    # Simulate RCS measurement sample
    ns = 100 # Number of samples
    sampling_method = 'fibonacci' # sampling method
    rcs_pccc = PCCC.measure(eiscat, num_points=ns, method=sampling_method)
    
    # Define perfectly conducting sphere
    PEC = DielectricMaterial(1.e8, 0., 1.e-8, 0.)
    r = 0.1 # radius [m]
    PCS = DebrisObject(material=PEC, 
                       radius=r, 
                       objtype='Sphere', 
                       name='Perfectly conducting sphere')

    # Simulate RCS measurement sample
    ns = 20 # Number of samples
    sampling_method = 'fibonacci' # sampling method
    rcs_pcs = PCS.measure(eiscat, num_points=ns, method=sampling_method)
    
    # Define spherical mission panel
    epsilon_r = 200. # dielectric permittivity
    sigma_e = 50. # electrical conductivity
    mu_r = 1. # relative magnetic permeability
    sigma_m = 0. # magnetic conductivity
    mission_panel = DielectricMaterial(mu_r=mu_r, # composite material
                                       sigma_m=sigma_m,
                                       epsilon_r=epsilon_r,
                                       sigma_e=sigma_e)
    SMP = DebrisObject(material=mission_panel, 
                       radius=r, 
                       objtype='Sphere', 
                       name='Spherical mission panel')
    
    # Simulate RCS measurement sample
    ns = 20 # Number of samples
    sampling_method = 'fibonacci' # sampling method
    rcs_smp = SMP.measure(eiscat, num_points=ns, method=sampling_method)
