################################################################################
# tests/conftest.py: fixtures shared across the oops test suite
################################################################################

import cspyce
import pytest

from oops.frame import Frame
from oops.path  import Path
from oops.unittester_support import TEST_SPICE_PREFIX

# Leap seconds, planetary constants, and the planetary ephemeris: the kernels that
# every test of SPICE-derived geometry needs.
CORE_KERNELS = ['naif0009.tls', 'pck00010.tpc', 'de421.bsp']


@pytest.fixture
def core_kernels():
    """Furnish the core SPICE kernels with the Path and Frame registries cleared.

    The registries are cleared before and after the test, so a path or frame that
    one test module registers cannot be found by another.
    """

    for path in TEST_SPICE_PREFIX.retrieve(CORE_KERNELS):
        cspyce.furnsh(path)
    Path._reset_caches()
    Frame._reset_caches()

    yield

    Path._reset_caches()
    Frame._reset_caches()

################################################################################
