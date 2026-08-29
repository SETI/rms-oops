##########################################################################################
# tests/spicedb/conftest.py: fixtures for the spicedb tests
##########################################################################################

import pytest

import spicedb


@pytest.fixture(autouse=True)
def _empty_kernel_registry():
    """Start and finish each test with no kernels furnished through spicedb.

    The tests assert on the exact contents of the furnished-kernel registries, so
    anything an earlier test left behind (oops.Body.define_solar_system() furnishes
    through spicedb) would otherwise be counted.
    """

    spicedb.unload_all()

    yield

    spicedb.unload_all()

##########################################################################################
