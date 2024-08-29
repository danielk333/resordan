import glob, sys, os
import logging
import argparse
import matplotlib.pyplot as plt
import numpy as np
import cluster_plotting as plotting
from resordan.clustering import algorithm as clustering
from resordan.data.gmf import GMFDataset
from datetime import datetime
from pathlib import Path

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
    save_dir = results.SF
    plot = results.P
    pattern = '*.h5'

    for root, dirs, files in os.walk(gmf_dir):
        dirs.sort()
        for dirname in dirs:
            print(dirname)
            dirsave = (dirname[0:13]).replace('-','')
            filesave = 'ce_uhf_' + dirsave + '.pkl'
            dirsasve = dirsave[0:8]
            
            gmf_files = list(sorted(glob.glob(os.path.join(root, dirname, pattern))))
            gmf_dataset = GMFDataset.from_files(gmf_files)

            detector_params = dict(
                # loss_weights=(1e-3, 1e-3),
                segment_split_time=1.5,
                snr_db_threshold=20,
                # loss_threshold=10,
            )
            events_dataset = clustering.snr_peaks_detection(gmf_dataset, **detector_params)
            Path(os.path.join(save_dir,dirsasve)).mkdir(parents=True, exist_ok=True)
            GMFDataset.to_pickle(events_dataset,str(os.path.join(save_dir,dirsasve,filesave)))

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
