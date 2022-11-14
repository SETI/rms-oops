################################################################################
# oops/backplane/unittester.py
################################################################################
#
# Usage:
#   python unittester.py [--output dir] [--no-output] [--exercises-only]
#                        [--no-exercises] [--reference] [--undersample #]
#                        [--test-level #] [--no-compare] [--verbose] [--log]
#                        [--diff old new] [--help] [--args arg1 arg2 ...]
#
#   --output dir         Directory in which to save backplane PNG images.
#                        Default is TESTDATA_PARENT_DIRECTORY/[data dir].
#                        If the directory does not exist, it is created.
#   --no-output          Disable saving of backplane PNG files.
#   --exercises-only     Execute only the backplane exercises.
#   --no-exercises       Execute all tests except the backplane exercises.
#   --reference          Generate reference backplanes in.
#                        [default output dir]/reference_[undersample]/.
#   --undersample #      Amount to undersample backplanes.  Default is 16.
#   --test-level #       Selects among pre-set parameter combinations:
#                         test_level 1: no printing, no saving, undersample 32.
#                         test_level 2: printing, no saving, undersample 16.
#                         test_level 3: printing, saving, no undersample.
#   --no-compare         Do not compare backplanes with references.
#   --verbose            Print output to the terminal.
#   --log                Enable the internal oops logging.
#   --diff old new       Compare new and old backplane logs.
#   --help               Print usage message.
#   --args arg1 arg2 ... Generic arguments to pass to the test modules.  Must
#                        occur last in the arumgnet list.


import unittest

from oops.backplane              import Test_Backplane_Surfaces
from oops.backplane              import Test_Backplane_Borders
from oops.backplane              import Test_Backplane_Empty_Events
from oops.backplane              import Test_Backplane_Exercises
from oops.backplane.ansa         import Test_Ansa
from oops.backplane.border       import Test_Border
from oops.backplane.distance     import Test_Distance
from oops.backplane.lighting     import Test_Lighting
from oops.backplane.limb         import Test_Limb
from oops.backplane.orbit        import Test_Orbit
from oops.backplane.pole         import Test_Pole
from oops.backplane.resolution   import Test_Resolution
from oops.backplane.ring         import Test_Ring
from oops.backplane.sky          import Test_Sky
from oops.backplane.spheroid     import Test_Spheroid
from oops.backplane.where        import Test_Where


########################################
from oops.backplane.unittester_support      import backplane_unittester_args

if __name__ == '__main__':

    backplane_unittester_args()
    unittest.main(verbosity=2)
################################################################################
