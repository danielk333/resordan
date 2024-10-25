import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from resordan.clustering import algorithm
from resordan.clustering import plotting
from resordan.data.gmf import GMFDataset
from resordan.data.events import EventsDataset


def main(input_args=None):
    parser = argparse.ArgumentParser(description='Cluster GMF Data')
    parser.add_argument('src', type=str, help='Path to GMF directory')
    parser.add_argument('-o', '--output', type=str, help='Path to output file')
    parser.add_argument('-v', "--verbose", action='store_true', help='Print results to screen')
    parser.add_argument('-p', '--plot', type=str, action='store', help='To display plots')

    args = parser.parse_args()

    # do the thing
    events_dataset, gmf_dataset = algorithm.event_detection(args.src)

    if args.verbose:
        print(events_dataset)

    # write detections
    if Path(args.out).exists():
        print (f"File {args.out} already exists")
    else:
        EventsDataset.to_pickle(events_dataset, args.out)

    if events_dataset.events:
        if args.plot:
            detected_inds = np.concatenate([e.idx for e in events_dataset.events])
            fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex="all")
            plotting.plot_peaks(axes=ax, data=gmf_dataset, detected_inds=detected_inds)
            plt.show()


if __name__ == '__main__':
    main()
