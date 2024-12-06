import numpy as np
import pickle
from pathlib import Path
from resordan.scrax.features import test_statistics

SNR_PREDICTION_FILENAME = "correlated_snr_prediction.pickle"


def size_shape_estimator(rcs_data, model):
    
    # Compute feature vector
    fvec = test_statistics(rcs_data)
    fvec[np.isnan(fvec)] = 0 # Replace NaN values with zeros
    fvec[fvec<1e-6] = 0 # Replace small values with zeros
    
    # Reshape as (n_objects=1, n_features)
    fvec = fvec.reshape(1, fvec.size)
    
    # Size prediction
    max_dim = model.predict(fvec)
    
    return {"xSectMaxPred": max_dim}

def tasks_size_shape_estimator(src):
    """generate estimation tasks for a src product"""
    tasks = []
    for satdir in [d for d in Path(src).iterdir() if d.is_dir()]:
        for passdir in [d for d in satdir.iterdir() if d.is_dir()]:
            snr_prediction_file = passdir / SNR_PREDICTION_FILENAME
            if snr_prediction_file.is_file():
                tasks.append((satdir.name, passdir.name, str(snr_prediction_file)))
    return tasks

def process_size_shape_estimator(tasks, model):
    """
    process size shape estimation for a list of tasks
    (satid, passid, snr_prediction_file)
    """    
    results = []
    for satid, passid, size_prediction_file in tasks:
        with open(size_prediction_file, 'rb') as file:
            data = pickle.load(file)
            estimate = size_shape_estimator(data["rcs_data"], model)
            results.append((satid, passid, size_prediction_file, estimate))
    return results


def scrax(data_dir, model_file, logger=None):
    """
    main entrypoint scrax
    """
    if logger:
        logger.info("Hellow World, Scrax")

    # generate estimation tasks
    tasks = tasks_size_shape_estimator(data_dir)

    # load model
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    # process
    results = process_size_shape_estimator(tasks, model)

    # store results
    for satid, passid, snr_prediction_file, estimate in results:
        with open(snr_prediction_file, 'rb') as file:
            data = pickle.load(file)
            data.update(**estimate)
        with open(snr_prediction_file, 'wb') as file:
            pickle.dump(data, file)




if __name__ == '__main__':

    from pathlib import Path
    import importlib.resources

    PROJECT = Path("/cluster/projects/p106119-SpaceDebrisRadarCharacterization")
    SOURCE = PROJECT / "rcs/leo_bpark_2.1u_EI-20240822-UHF"

    with importlib.resources.path('resordan.scrax.models', "size_predict_n10.pickle") as MODEL:
        scrax(str(SOURCE), str(MODEL))


    
