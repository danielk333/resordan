import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from resordan.clustering import algorithm
from resordan.clustering import plotting
from resordan.data.events import EventsDataset


def main(input_args=None):
    parser = argparse.ArgumentParser(description='Cluster GMF Data')
    parser.add_argument('src', type=str, help='Path to GMF directory')
    parser.add_argument('-o', '--output', type=str, help='Path to output file')
    parser.add_argument('-v', "--verbose", action='store_true', help='Print results to screen')
    parser.add_argument('-p', '--plot', action='store_true', help='To display plots')
    parser.add_argument('-spt', '--segment_split_time', type=float,
                        help = 'Min time (in s) to discern SNR peaks distiinct detections. Default = 5')
    parser.add_argument('-sdt', '--snr_db_threshold', type=float,
                        help = 'Threshold (in dB) for detecting peaks in SNR. Default = 15')
    parser.add_argument('-lw', '--loss_weights', type=float,
                        help = 'Loss function weights. Default = (1e-3, 1e-3)')
    parser.add_argument('-lt', '--loss_threshold', type=float,
                        help = 'Loss function threshold. Default = 10')
    parser.add_argument('-mns', '--min_n_samples', type=float,
                        help = 'Minimum number of samples in an event. Default = 5')

    args = parser.parse_args()
    # Updated parameters for detection
    detector_params = {}
    if args.loss_weights is not None: detector_params['loss_weights'] = args.loss_weights
    if args.segment_split_time is not None: detector_params['segment_split_time'] = args.segment_split_time
    if args.snr_db_threshold is not None: detector_params['snr_db_threshold'] = args.snr_db_threshold
    if args.loss_threshold is not None: detector_params['loss_threshold'] = args.loss_threshold
    if args.min_n_samples is not None: detector_params['min_n_samples'] = args.min_n_samples

    # Perform cluster detection. The selection parameters can be modified
    if not detector_params:
        events_dataset, gmf_dataset = algorithm.event_detection(args.src)
    else:
        print(detector_params)
        events_dataset, gmf_dataset = algorithm.event_detection(args.src,**detector_params)

    if args.verbose:
        print(events_dataset)

    # write detections
    if Path(args.output).exists():
        print (f"File {args.output} already exists")
    else:
        EventsDataset.to_pickle(events_dataset, args.output)

    if events_dataset.events:
        if args.plot:
            detected_inds = np.concatenate([e.idx for e in events_dataset.events])
            fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex="all")
            plotting.plot_peaks(axes=ax, data=gmf_dataset, detected_inds=detected_inds)
            plt.show()


if __name__ == '__main__':
    main()
