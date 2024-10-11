import argparse
from pathlib import Path
import logging
import sys

from resordan.scrax.scrax import scrax

"""
Command Line Interface for Scrax
"""

def main():
    parser = argparse.ArgumentParser(description='Scrax')
    parser.add_argument('input', type=str, help='Path to scrax input')
    parser.add_argument('-o', '--output', type=str, help='Path to output file')
    parser.add_argument('-v', "--verbose", action='store_true', help='Print results to screen')

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # check input
    if not Path(args.input).exists():
        print(f"File {args.input} does not exist")
        return

    # check output
    #if Path(args.output).exists():
    #    print (f"File {args.output} already exists")
    #    return

    # scrax
    scrax(args.input, logger=logger)


if __name__ == '__main__':
    main()
