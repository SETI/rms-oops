##########################################################################################
# tests/conftest.py: fixtures shared across the oops test suite
##########################################################################################

import cspyce
import pytest
from filecache import FCPath

from oops.frame import Frame
from oops.path  import Path
from programs.gold_master.test_support import TEST_SPICE_PREFIX

# Leap seconds, planetary constants, and the planetary ephemeris: the kernels that
# every test of SPICE-derived geometry needs.
CORE_KERNELS = ('naif0009.tls', 'pck00010.tpc', 'de421.bsp')

# test_support resolves the prefix to None when neither $OOPS_RESOURCES nor
# $OOPS_TEST_DATA_PATH names a test data tree. Every test that reads a kernel needs one,
# so the precondition is checked once here: an unconfigured environment then fails at
# collection, naming what is missing, instead of raising AttributeError on None inside
# whichever test happened to run first. Tests import this narrowed name rather than the
# optional one that test_support publishes.
assert TEST_SPICE_PREFIX is not None, (
    'The oops test suite needs $OOPS_RESOURCES or $OOPS_TEST_DATA_PATH to be set')
SPICE_PREFIX: FCPath = TEST_SPICE_PREFIX


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add the command-line options of the oops test suite.

    Parameters:
        parser (Parser): The pytest command-line parser.
    """

    parser.addoption('--gold-master', action='store', default=None, metavar='DIR',
                     help='Root directory of the gold master files that the tests '
                          'under tests/hosts compare against, overriding '
                          '$OOPS_GOLD_MASTER_PATH and $OOPS_RESOURCES. The directory '
                          'must have the standard layout, with the files for one '
                          'observation in DIR/<mission>.<instrument>/<basename>, such '
                          'as DIR/cassini.iss/W1573721822_1.')


@pytest.fixture(scope='session', autouse=True)
def gold_master_path(request: pytest.FixtureRequest) -> None:
    """Point the gold master tests at the directory given by --gold-master.

    The option applies for the whole session and is a no-op when it is not given, so the
    tests then read the gold masters that the environment defines.
    """

    path = request.config.getoption('--gold-master')
    if path:
        # Imported here rather than at the top of the module: programs.gold_master pulls
        # in oops and scipy, and nothing else in the test suite needs it.
        import programs.gold_master as gm
        gm.set_gold_master_path(path)


@pytest.fixture
def core_kernels():
    """Furnish the core SPICE kernels with the Path and Frame registries cleared.

    The registries are cleared before and after the test, so a path or frame that
    one test module registers cannot be found by another.
    """

    for path in SPICE_PREFIX.retrieve(CORE_KERNELS):
        cspyce.furnsh(path)
    Path._reset_caches()
    Frame._reset_caches()

    yield

    Path._reset_caches()
    Frame._reset_caches()

##########################################################################################
