##########################################################################################
# programs/gold_master/test_support.pyi
##########################################################################################
"""Type stub for :mod:`programs.gold_master.test_support`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from filecache import FCPath, FileCache

__all__ = ['OOPS_RESOURCES', 'OOPS_TEST_DATA_PATH', 'OOPS_GOLD_MASTER_PATH',
           'OOPS_BACKPLANE_OUTPUT_PATH', 'TEST_DATA_FILECACHE', 'TEST_SPICE_FILECACHE',
           'TEST_DATA_PREFIX', 'TEST_SPICE_PREFIX', 'GOLD_MASTER_PREFIX',
           'BACKPLANE_OUTPUT_PREFIX']

OOPS_RESOURCES: str | None
OOPS_TEST_DATA_PATH: str | None
OOPS_GOLD_MASTER_PATH: str | None
OOPS_BACKPLANE_OUTPUT_PATH: str
TEST_DATA_FILECACHE: FileCache
TEST_SPICE_FILECACHE: FileCache
TEST_DATA_PREFIX: FCPath | None
TEST_SPICE_PREFIX: FCPath | None
GOLD_MASTER_PREFIX: FCPath | None
BACKPLANE_OUTPUT_PREFIX: FCPath

##########################################################################################
