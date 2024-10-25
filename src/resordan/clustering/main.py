import argparse
from pathlib import Path

from resordan.clustering import algorithm
from resordan.clustering import plotting
from resordan.data.gmf import GMFDataset
from resordan.data.events import EventsDataset

DETECTOR_PARAMS = dict(
    loss_weights=(1e-3, 1e-3),
    segment_split_time=1.5,
    snr_db_threshold=20,
    loss_threshold=10,
)

def main(input_args=None):
    parser = argparse.ArgumentParser(description='Cluster GMF Data')
    parser.add_argument('src', type=str, help='Path to GMF directory')
    parser.add_argument('-o', '--output', type=str, help='Path to output file')
    parser.add_argument('-v', "--verbose", action='store_true', help='Print results to screen')
    parser.add_argument('-p', '--plot', type=str, action='store', help='To display plots')


    args = parser.parse_args()

    src = Path(args.src)
    for root, dirs, files in os.walk(args.src):
        if root == args.src:
            i = 0
            #src = Path(root)
            if len(dirs) and len(files) == 0:
                return print('WARNIG: no GMF files')
            if len(dirs) > 0:
                events_list = []
                for dir in sorted(dirs):
                    srcfor = Path(os.path.join(root,dir))
                    gmf_files = list(sorted([file for file in srcfor.rglob('*.h5') if file.is_file()]))
                    gmf_dataset = GMFDataset.from_files(gmf_files)
                    datas = algorithm.snr_peaks_detection(gmf_dataset, **DETECTOR_PARAMS)
                    for events in datas.events:
                        events.event_number = i
                        events_list.append(events)
                        i = i+1
                events_dataset = EventsDataset(meta=datas.meta, detector_config=datas.detector_config, events=events_list)
            else:
                gmf_files = list(sorted([file for file in src.rglob('*.h5') if file.is_file()]))
                gmf_dataset = GMFDataset.from_files(gmf_files)
                events_dataset = algorithm.snr_peaks_detection(gmf_dataset, **DETECTOR_PARAMS)


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
