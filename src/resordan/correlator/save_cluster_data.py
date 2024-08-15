import glob, sys
import logging
import argparse
import matplotlib.pyplot as plt
import numpy as np
import cluster_plotting as plotting
from resordan.clustering import algorithm as clustering
from resordan.data.gmf import GMFDataset
from datetime import datetime

logger = logging.getLogger("resordan")
logging.basicConfig(level=logging.INFO)

def main(input_args=None):
    Parser = argparse.ArgumentParser(description='Getting clustering events and saving')
    Parser.add_argument('-gmf', type=str, action='store', dest='GMF', help='Path to GMF directory')
    Parser.add_argument('-sf', type=str, action='store', dest='SF', help='Path to save data')
    Parser.add_argument('-p', type=str, action='store', dest='P', help='True if plot is displayed')
    
    if len(sys.argv) < 2:
        Parser.print_help()
        sys.exit(1)
    else:
        pass
    
    results = Parser.parse_args()
    gmf_dir = results.GMF
    save_path = results.SF
    plot = results.P

    gmf_files = list(sorted(glob.glob(f"{gmf_dir}/*.h5")))
    gmf_dataset = GMFDataset.from_files(gmf_files)

    detector_params = dict(
        # loss_weights=(1e-3, 1e-3),
        segment_split_time=1.5,
        snr_db_threshold=20,
        # loss_threshold=10,
    )
    events_dataset = clustering.snr_peaks_detection(gmf_dataset, **detector_params)
    GMFDataset.to_pickle(events_dataset,save_path)

    if events_dataset.events:
        if plot:
            detected_inds = np.concatenate([e.idx for e in events_dataset.events])
            fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex="all")
            plotting.plot_peaks(axes=ax, data=gmf_dataset, detected_inds=detected_inds)
            plt.show()

    else:
        print("No detections found.")

if __name__ == '__main__':
    main()
