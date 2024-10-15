import pytest
import numpy as np

from resordan.scrax.get_circular_plate_rcs import getCircularPlateRCS


def test_circular_plate():

    import matplotlib
    matplotlib.use('TkAgg')

    from matplotlib import pyplot as plt


    vhf = 2.24e8 # EISCAT VHF radar frequency [Hz]
    uhf = 9.30e8 # EISCAT UHF radar frequency [Hz]
    fc = uhf
    r = 0.225 # radius [m]
    c = 2.99792458e8 # speed of light (in vacuum)
    h = 1.0 # height [m]
    #el = np.deg2rad(np.asfarray([-90,-60,-45,-30,0,30,45,60,90]))
    el = np.deg2rad(np.asfarray(list(range(-90,90)))) # elevation angles
    #el = np.deg2rad(np.asfarray([45,46,47,48,49,50])#,51,52,53,54,55]))
    eps = 1e-6
    
    rcs = getCircularPlateRCS(r,fc,0,el)
    rcs_dB = 10*np.log10(rcs+eps)
    
    #print(rcs)
    plt.plot(el.flatten(),rcs_dB.flatten())
    plt.show()

    assert True
    



