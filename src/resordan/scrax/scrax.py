import numpy as np
import pickle
from pathlib import Path
from resordan.scrax.features import compute_feature_vector

SNR_PREDICTION_FILENAME = "correlated_snr_prediction.pickle"


def size_shape_estimator(rcs_data, model):
    
    # Compute feature vector
    fvec = compute_feature_vector(rcs_data)
    fvec[np.isnan(fvec)] = 0 # Replace NaN values with zeros
    fvec[fvec<1e-6] = 0 # Replace small values with zeros
    
    # Reshape as (n_objects=1, n_features)
    fvec = fvec.reshape(1, fvec.size)
    
    # Size prediction
    max_dim = model.predict(fvec)
    
    return {"xSectMaxPred": max_dim}


def new_size_shape_estimator(rcs_vector, model):
    return {"xSectMaxPred": 0.88}


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
    for satid, passid, data_file in tasks:
        with open(data_file, 'rb') as file:
            data = pickle.load(file)
            estimate = size_shape_estimator(data["rcs_data"], model)
            results.append((satid, passid, data_file, estimate))
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


def new_scrax(rcs_dir, model_file):

    def get_data(pass_dir):
        pickle_file = pass_dir / SNR_PREDICTION_FILENAME
        if not pickle_file.is_file():
            print("pickle file does not exist")
            return None, None
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)
            return data['catid'], data['rcs_data']

    # load model
    model = model_file
    # with open(model_file, 'rb') as f:
    #     model = pickle.load(f)

    # generate estimation tasks
    map = {}
    for pass_dir in [d for d in Path(rcs_dir).iterdir() if d.is_dir()]:        
        catid, data = get_data(pass_dir) 
        if catid is None:
            continue
        if catid not in map:
            map[catid] = {"catid": catid, "passes": [pass_dir], "vector": [data]}
        else:
            item = map[catid]
            item["passes"].append(pass_dir)
            item["vector"].append(data)

    # process
    for item in map.values():
        item["result"] = new_size_shape_estimator(item["vector"], model)

    # store results
    for item in map.values():
        print(f"[{item['catid']}] [{len(item['passes'])}] : {item['result']}")



if __name__ == '__main__':

    from pathlib import Path
    import importlib.resources

    PROJECT = Path("/cluster/projects/p106119-SpaceDebrisRadarCharacterization")
    SOURCE = PROJECT / "rcs/leo_bpark_2.1u_EI-20240822-UHF"

    NEW_SOURCE = PROJECT / "rcs/leo_mpark_2.1u_EI-20240704-UHF"


    MODEL = Path("/cluster/home/inar/Dev/Git/resordan/tests/data/size_predict_n10.pickle")


    #with importlib.resources.path('resordan.scrax.models', "size_predict_n10.pickle") as MODEL:
    #    scrax(str(SOURCE), str(MODEL))

    # scrax(str(SOURCE), str(MODEL))

    new_scrax(str(NEW_SOURCE), str(MODEL))






