import pytest
from resordan.correlator.discos_cat import get_discos_objects

def test_discos():

    OBJECT_IDS = [
        "43763", "30768", "46320", "43831", "22356", "4868", "5683", "4154", 
        "25560", "43476", "43477", "7663", "30017", "31147", "15938", "31097", 
        "48717", "44890", "15055", "18790", "30649", "43740", "43733", "46480", 
        "43348", "39320", "21106", "42534"
    ]

    TOKEN = "ImE2NGE5Y2YwLWUxMTEtNDg2NS04ZTYxLTRhZmQ0OTBiN2VjNyI.pN0jXjkaLyz0GsRFmYliLcQTtl4"

    results = get_discos_objects(OBJECT_IDS, TOKEN)
    assert len(OBJECT_IDS) == len(results)



