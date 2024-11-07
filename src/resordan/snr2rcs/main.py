#!/usr/bin/env python

import argparse
from pathlib import Path
import configparser
from resordan.snr2rcs.snr2rcs import snr2rcs

def main():
    parser = argparse.ArgumentParser(description='SNR2RCS')
    parser.add_argument('src', type=str, help='Path to GMF product (directory)')
    parser.add_argument('cfg', type=str, help='Path to snr2rcs config file')
    parser.add_argument('dst', type=str, help='Path to target directory')
    parser.add_argument('-v', "--verbose", action='store_true', help='Print results to screen')
    parser.add_argument('-c', "--clobber", action='store_true', help='Overwrite pre-existing files')
    parser.add_argument("--tmp", help='Temporary directory')
    parser.add_argument("--cleanup", action='store_true', help='Cleanup temporary directory')

    args = parser.parse_args()

    # gmf product
    src = Path(args.src)
    if not src.is_dir():
        print(f"Path to GMF product is not a directory: {src}")
        return

    # target
    dst = Path(args.dst)
    if not dst.exists():
        dst.mkdir(parents=True, exist_ok=True)

    # snr2rcs config file
    configfile = Path(args.cfg)
    if not configfile.is_file():
        print(f"Path to snr2rcs config file is missing: {configfile}")
        return

    # Read config
    cfg = configparser.ConfigParser()
    cfg.read(configfile)

    # Run
    snr2rcs(src, cfg, dst, verbose=args.verbose, clobber=args.clobber, tmp=args.tmp, cleanup=args.cleanup)


if __name__ == '__main__':
    main()
