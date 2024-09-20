import pytest
from pathlib import Path
from resordan.correlator.beam_rcs_estimator import rcs_estimator

WORKDIR = Path("/cluster/work/users/inar/usecase")
TOKEN = "ImE2NGE5Y2YwLWUxMTEtNDg2NS04ZTYxLTRhZmQ0OTBiN2VjNyI.pN0jXjkaLyz0GsRFmYliLcQTtl4"

def prerequisites():
    return WORKDIR.is_dir()

@pytest.mark.skipif(not prerequisites() , reason="Local file is missing")
def test_rcs_estimator():
    tle_file = WORKDIR / "tle.txt"
    radarid = "eiscat_uhf"
    output = WORKDIR / "out"
    rcs_estimator(radarid, str(tle_file), str(WORKDIR), str(WORKDIR), str(output), TOKEN, verbose=True)
    assert 1 == 1