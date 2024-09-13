import argparse
from pathlib import Path

from resordan.clustering import algorithm as clustering
from resordan.data.gmf import GMFDataset
from resordan.data.events import EventsDataset

DETECTOR_PARAMS = dict(
    # loss_weights=(1e-3, 1e-3),
    segment_split_time=1.5,
    snr_db_threshold=20,
    # loss_threshold=10,
)

def main(input_args=None):
    parser = argparse.ArgumentParser(description='Cluster GMF Data')
    parser.add_argument('src', type=str, help='Path to GMF directory')
    parser.add_argument('-o', '--output', type=str, help='Path to output file')
    parser.add_argument('-v', "--verbose", action='store_true', help='Print results to screen')


    args = parser.parse_args()

    src = Path(args.src)

    # find gmf files
    gmf_files = list(sorted([file for file in src.rglob('*.h5') if file.is_file()]))
    gmf_dataset = GMFDataset.from_files(gmf_files)
    # cluster
    events_dataset = clustering.snr_peaks_detection(gmf_dataset, **DETECTOR_PARAMS)

    if args.verbose:
        print(events_dataset)

    # write detections
    if Path(args.out).exists():
        print (f"File {args.out} already exists")
    else:
        EventsDataset.to_pickle(events_dataset, args.out)

if __name__ == '__main__':
    main()
