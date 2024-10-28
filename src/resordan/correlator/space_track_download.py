#!/usr/bin/env python
import sys
import re
import getpass
import argparse
from datetime import datetime, timedelta
import datetime as dt
import subprocess
import spacetrack 


"""
Get TLE's published for the time period 24h before the observation day
or TLE for given objects.
"""

def fetch_publish_tle(epoch_dt, st_user, st_passwd):
    # Get published TLE's for the period 24 hour before the start of the experiment
    st = spacetrack.SpaceTrackClient(identity=st_user, password=st_passwd)
    dt0 = epoch_dt - dt.timedelta(hours=24)
    dt1 = epoch_dt - dt.timedelta(hours=1)
    drange = spacetrack.operators.inclusive_range(dt0, dt1)
    return list(st.tle_publish(
        iter_lines=True, 
        orderby='TLE_LINE1', 
        format='tle',
        publish_epoch=drange
    ))

def fetch_object_tle(norad_cat_id, drange, st_user, st_passwd):
    # Get published TLE's for the given objects
    st = spacetrack.SpaceTrackClient(identity=st_user, password=st_passwd)
    return list(st.tle(
        norad_cat_id=norad_cat_id,
        iter_lines=True, 
        orderby='TLE_LINE1', 
        format='tle',
        epoch=drange
    ))




_iso_fmt = '%Y-%m-%dT%H:%M:%S'
_td_regx = re.compile(r'^((?P<days>[\.\d]+?)d)? *((?P<hours>[\.\d]+?)h)? ' +
                      r'*((?P<minutes>[\.\d]+?)m)? *((?P<seconds>[\.\d]+?)s)?$')

def parse_timedelta(time_str):
    """
    Parse a time string e.g. (2h13m) into a timedelta object.

    Modified from virhilo's answer at https://stackoverflow.com/a/4628148/851699

    :param time_str: A string identifying a duration.  (eg. 2h13m)
    :return datetime.timedelta: A datetime.timedelta object
    """
    parts = _td_regx.match(time_str)
    assert parts is not None, f"Could not parse any time information from '{time_str}'.  " + \
                                "Examples of valid strings: '8h', '2d8h5m20s', '2m4s'"
    time_params = {name: float(param) for name, param in parts.groupdict().items() if param}
    return timedelta(**time_params)


def main(input_args=None):
    parser = argparse.ArgumentParser(description='Download tle snapshot from space-track')
    parser.add_argument('start_date', type=str, nargs='?', default='7d',
                    help='Start date of snapshot [ISO] or timedelta ("24h", "12d", etc)')
    parser.add_argument('end_date', type=str, nargs='?', default='now',
                    help='End date of snapshot [ISO]')
    parser.add_argument('output', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    parser.add_argument('--credentials', '-c', nargs=1, type=str, help='File containing username and password for space-track.org')
    parser.add_argument('--name', '-n', default=None, help='Name of the object to match with the "like" operator')

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    # end date argument can be ISO format datetime or 'now'
    if args.end_date == "now":
        dt1 = datetime.now()
    else:
        dt1 = datetime.strptime(args.end_date, _iso_fmt)

    # start date argument can be absolute or timedelta
    try:
        dt0 = dt1 - parse_timedelta(args.start_date)
    except AssertionError:
        dt0 = datetime.strptime(args.start_date, _iso_fmt)

    # Read credentials
    if args.credentials is not None:
        sourcefile = (args.credentials)[0]
        proc = subprocess.Popen("sed -n '1p' "+sourcefile, stdout=subprocess.PIPE, shell=True)
        user = proc.stdout.read()
        proc = subprocess.Popen("sed -n '2p' "+sourcefile, stdout=subprocess.PIPE, shell=True)
        passwd = proc.stdout.read()
        user = user.strip().decode( "utf-8" )
        passwd = passwd.strip().decode( "utf-8" )
    else:
        user = input("Username for space-track.org:")
        passwd = getpass.getpass("Password for " + user + ":")


    # get published TLE for the time period 24h before the observation day
    # get TLE for the given objects 
    if args.name is not None:
        drange = spacetrack.operators.inclusive_range(dt0, dt1)
        name_op = spacetrack.operators.like(args.name)
        lines = fetch_object_tle(name_op, drange, user, passwd)
    else:
        lines = fetch_publish_tle(dt1, user, passwd)

    # Write results
    lineno = 0
    for line in lines:
        args.output.write(line + '\n')
        lineno += 1



if __name__ == '__main__':
    main()
