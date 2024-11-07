import pytest
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from resordan.scrax.get_circular_plate_rcs import getCircularPlateRCS

# Fixture to set up matplotlib for all tests
@pytest.fixture(autouse=True)
def setup_matplotlib():
    # Set the matplotlib backend to 'Agg' (or any other backend)
    # 'Agg' is typically used for non-interactive testing (without GUI)
    matplotlib.use('TkAgg')

    # You could also reset figures before each test, if needed
    yield
    # Cleanup code (if necessary), e.g., closing all figures
    plt.close('all')


@pytest.mark.interactive
def test_circular_plate(setup_matplotlib):

    # vhf = 2.24e8  # EISCAT VHF radar frequency [Hz]
    uhf = 9.30e8  # EISCAT UHF radar frequency [Hz]
    fc = uhf
    r = 0.225  # radius [m]
    # c = 2.99792458e8  # speed of light (in vacuum)
    # h = 1.0  # height [m]
    # el = np.deg2rad(np.asfarray([-90,-60,-45,-30,0,30,45,60,90]))
    el = np.deg2rad(np.asarray(list(range(-90, 90))))  # elevation angles
    # el = np.deg2rad(np.asfarray([45,46,47,48,49,50])#,51,52,53,54,55]))
    eps = 1e-6

    rcs = getCircularPlateRCS(r, fc, 0, el)
    rcs_dB = 10*np.log10(rcs+eps)

    # print(rcs)
    plt.plot(el.flatten(), rcs_dB.flatten())
    plt.show()

    assert True

@pytest.mark.interactive
def test_size_shape_estimator(rcs, model):
    
    # Size prediction
    max_dim = model.predict(rcs)
    
    return {"xSectMaxPred": max_dim}
