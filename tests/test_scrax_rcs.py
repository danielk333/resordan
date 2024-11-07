import pytest
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from resordan.scrax.get_circular_plate_rcs import getCircularPlateRCS
from resordan.scrax.scrax import process_size_shape_estimator
import pickle
import tempfile
from pathlib import Path


@pytest.mark.interactive
def test_circular_plate(setup_matplotlib):

    matplotlib.use('TkAgg')


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

    plt.close('all')

    assert True


def test_snr_prediction():

    rcs_data = np.array([
        1.26037036e+00, 8.49878611e+00, 4.74602680e+01, 1.63479929e+01,
        3.27654639e+00, 1.93519575e+00, 7.28008166e-01, 3.10218118e+00,
        6.60246862e+00, 1.77719047e+00, 1.70862592e+00, 8.05324053e-01,
        6.60713679e-01, 7.82819190e-01, 7.69305285e-01, 2.76347415e-01,
        9.87335502e+01, 1.75266724e+01, 5.76228048e+00, 3.12110456e+00,
        2.55969778e+00, 5.71462026e-02, 6.16913940e-02, 2.71260586e-02,
        np.nan, 3.17948739e-02, 8.58343048e-01, 1.41288590e-01
    ])

    # make mockup filt
    data_dict = {"rcs_data": rcs_data}    
    
    with tempfile.TemporaryDirectory() as temp_dir:

        model_file = Path(temp_dir) / "model.pickle"
        snr_file = Path(temp_dir) / "snr.pickle"

        # make mockup file for rcs data
        with open(snr_file, "wb") as f:
            pickle.dump(data_dict, f)

        # make mockup file for model
        with open(model_file, "wb") as f:
            pickle.dump(data_dict, f)

        # make task list
        tasks = [("satid", "passid", str(snr_file))]

        # load model
        with open(snr_file, "rb") as f:
            model = pickle.load(f)

        # process
        results = process_size_shape_estimator(tasks, model)
        assert len(results) > 0
        satid, passid, fname, res = results[0]
        assert res is not None


def test_size_shape_estimator(rcs, model):
    
    # Size prediction
    max_dim = model.predict(rcs)
    
    return {"xSectMaxPred": max_dim}
