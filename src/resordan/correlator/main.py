import argparse
from pathlib import Path
import spacetrack
import datetime as dt

ISO_FMT = '%Y-%m-%dT%H:%M:%S'

def download_tle(start_dt, end_dt, st_credentials, out=None):
    """
    download TLE for time interval, store to folder dst if given
    """
    drange = spacetrack.operators.inclusive_range(start_dt, end_dt)
    user = st_credentials["user"]
    passwd = st_credentials["passwd"]
    st = spacetrack.SpaceTrackClient(identity=user, password=passwd)
    lines = st.tle_publish(
        iter_lines=True, 
        orderby='TLE_LINE1', 
        format='tle',
        publish_epoch=drange
    )
    if not out:
        return lines
    else:
        start = dt.datetime.strftime(start_dt, ISO_FMT)
        end = dt.datetime.strftime(end_dt, ISO_FMT)
        if Path(out).exists():
            print(f"TLE file already exists {out}")
        else:
            # write to file
            with open(out, 'w') as file:
                for line in lines:
                    file.write(line + "\n")


def main(input_args=None):
    parser = argparse.ArgumentParser(description='Correlate Detections')
    parser.add_argument('src', type=str, help='Path to detections')
    parser.add_argument('dst', type=str, help='Path to output directory')

    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    print("WORK IN PROGRESS")






if __name__ == '__main__':


    def test_discos():
        # DISCOS
        TOKEN = "ImE2NGE5Y2YwLWUxMTEtNDg2NS04ZTYxLTRhZmQ0OTBiN2VjNyI.pN0jXjkaLyz0GsRFmYliLcQTtl4"



