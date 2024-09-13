#!/usr/bin/env python

import argparse
from pathlib import Path
import configparser
from resordan.snr2rcs.snr2rcs import snr2rcs

def main():
    parser = argparse.ArgumentParser(description='SNR2RCS')
    parser.add_argument('src', type=str, help='Path to GMF product (directory)')
    parser.add_argument('cfg', type=str, help='Path to snr2rcs config file')
    parser.add_argument('-o', '--output', type=str, help='Path to output file')
    parser.add_argument('-v', "--verbose", action='store_true', help='Print results to screen')
    parser.add_argument('-c', "--clobber", action='store_true', help='Overwrite pre-existing files')
    parser.add_argument("--tmp", help='Temporary directory')
    
    args = parser.parse_args()

    # gmf product
    src = Path(args.src)
    if not src.is_dir():
        print(f"Path to GMF produc is not a directory: {src}")
        return

    # snr2rcs config file
    configfile = Path(args.cfg)
    if not configfile.is_file():
        print(f"Path to snr2rcs config file is missing: {configfile}")

    # Read config
    cfg = configparser.ConfigParser()
    cfg.read(configfile)
    snr2rcs(src, cfg, verbose=args.verbose, clobber=args.clobber, tmp=args.tmp)


if __name__ == '__main__':
    main()
