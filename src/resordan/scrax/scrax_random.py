# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:48:36 2023

@author: stia
"""

import numpy as np

def rndsphere(N):
    """
    Random azimuth and elevation angles for uniform sampling of sphere
    """
    
    phi = np.random.Generator.vonmises(0., 0., N)
    theta = np.random.Generator.uniform(0., 1., N) * 2 * np.pi
    
    return (phi, theta)
