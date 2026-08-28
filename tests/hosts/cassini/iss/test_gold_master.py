################################################################################
# tests/hosts/cassini/iss/test_gold_master.py
################################################################################

import pytest

import programs.gold_master as gm

from oops.body import Body


@pytest.fixture(autouse=True)
def _standard_obs():
    # Imported for its side effect: it defines the standard observations.
    from tests.hosts.cassini.iss import standard_obs     # noqa: F401

    yield

    Body._undefine_solar_system()
    Body.define_solar_system()


def test_W1573721822():
    gm.execute_as_pytest('W1573721822_1')

################################################################################
