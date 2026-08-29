##########################################################################################
# oops/inst/juno/junocam/__init__.py
##########################################################################################
import pytest
import programs.gold_master as gm

from oops.unittester_support import TEST_DATA_PREFIX

@pytest.fixture(autouse=True)
def _standard_obs():
    # Imported for its side effect: it defines the standard observations.
    from tests.hosts.juno.junocam import standard_obs     # noqa: F401


def test_JNCR_2016347_03C00192_V01():
    gm.execute_as_pytest('JNCR_2016347_03C00192_V01')


def test_JNCR_2020366_31C00065_V01():
    gm.execute_as_pytest('JNCR_2020366_31C00065_V01')


def test_JNCR_2019096_19M00012_V02():
    gm.execute_as_pytest('JNCR_2019096_19M00012_V02')


def test_JNCR_2019149_20G00008_V01():
    gm.execute_as_pytest('JNCR_2019149_20G00008_V01')
##########################################################################################
