import argparse
from pathlib import Path

from resordan.clustering import algorithm as clustering
from resordan.data.gmf import GMFDataset
from resordan.data.events import EventDataset

DETECTOR_PARAMS = dict(
    # loss_weights=(1e-3, 1e-3),
    segment_split_time=1.5,
    snr_db_threshold=20,
    # loss_threshold=10,
)

def main(input_args=None):
    parser = argparse.ArgumentParser(description='Cluster GMF Data')
    parser.add_argument('src', type=str, help='Path to GMF directory')
    parser.add_argument('--dst', type=str, help='Path to output directory')

    args = parser.parse_args()

    src = Path(args.src)

    # find gmf files
    gmf_files = list(sorted([file for file in src.rglob('*.h5') if file.is_file()]))
    gmf_dataset = GMFDataset.from_files(gmf_files)
    # cluster
    events_dataset = clustering.snr_peaks_detection(gmf_dataset, **DETECTOR_PARAMS)

    print(events_dataset)

    # write detections
    if args.dst:
        outfile = Path(dst) / f"{src.name}.pkl"
        EventDataset.to_pickle(events_dataset, outfile)

if __name__ == '__main__':
    main()
