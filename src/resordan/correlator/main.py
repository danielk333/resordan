import argparse
from pathlib import Path

def main(input_args=None):
    parser = argparse.ArgumentParser(description='Correlate Detections')
    parser.add_argument('src', type=str, help='Path to detections')
    parser.add_argument('dst', type=str, help='Path to output directory')

    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    print("WORK IN PROGRESS")


if __name__ == '__main__':
    main()
