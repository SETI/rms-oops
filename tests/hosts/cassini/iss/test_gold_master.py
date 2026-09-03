##########################################################################################
# tests/hosts/cassini/iss/test_gold_master.py
##########################################################################################

import pytest

import programs.gold_master as gm

from oops.body import Body


@pytest.fixture(autouse=True)
def _standard_obs():
    # Imported for its side effect: it defines the standard observations.
    from tests.hosts.cassini.iss import standard_obs     # noqa: F401
    from oops.hosts.cassini.iss import ISS

    # Start each test from a known SPICE state. The teardown below calls
    # Body.define_solar_system() with no time range, which furnishes the extended
    # ephemerides (jup310xl.bsp, ura111.xl.bsp); SPICE gives a later-furnished kernel
    # precedence, so without this the second and later tests in the module would compute
    # their geometry from kernels the first test never used. Undefining the solar system
    # unloads them, and ISS.reset() clears the host module's "initialized" flag so that
    # reading the observation furnishes the kernels for its own time range again.
    Body._undefine_solar_system()
    ISS.reset()

    yield

    Body._undefine_solar_system()
    Body.define_solar_system()


def test_W1573721822():
    gm.execute_as_pytest('W1573721822_1')

##########################################################################################
