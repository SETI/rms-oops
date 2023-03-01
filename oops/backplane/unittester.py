################################################################################
# oops/backplane/unittester.py
################################################################################
#
# usage: unittester.py [-h] [--args [arg [arg ...]]] [--verbose]
#                     [--exercises-only] [--no-exercises] [--no-compare]
#                     [--output dir] [--no-output] [--log] [--undersample N]
#                     [--reference] [--test-level N]
#
#
# optional arguments:
#  -h, --help            show this help message and exit
#  --args [arg [arg ...]]
#                        Generic arguments to pass to the test modules. Must
#                        occur last in the argument list.
#  --verbose             Print output to the terminal.
#  --exercises-only      Execute only the backplane exercises.
#  --no-exercises        Execute all tests except the backplane exercises.
#  --no-compare          Do not compare backplanes with references.
#  --output dir          Directory in which to save backplane PNG images.
#                        Default is $OOPS_BACKPLANE_OUTPUT_PATH/[data dir]. If
#                        the directory does not exist, it is created.
#  --no-output           Disable saving of backplane PNG files.
#  --log                 Enable the internal oops logging.
#  --undersample N       Amount by which to undersample backplanes. Default is
#                        16.
#  --reference           Generate reference backplanes and exit.
#  --test-level N        Selects among pre-set parameter combinations:
#                        -test_level 1: no printing, no saving, undersample 32.
#                        -test_level 2: printing, no saving, undersample 16.
#                        -test_level 3: printing, saving, no undersampling.
#                        These behaviors are overridden by other arguments.

import unittest

from oops.backplane            import Test_Backplane_via_gold_master
from oops.backplane            import Test_Backplane_Surfaces
from oops.backplane            import Test_Backplane_Borders
from oops.backplane            import Test_Backplane_Empty_Events
from oops.backplane            import Test_Backplane_Exercises
from oops.backplane.ansa       import Test_Ansa
from oops.backplane.border     import Test_Border
from oops.backplane.distance   import Test_Distance
from oops.backplane.lighting   import Test_Lighting
from oops.backplane.limb       import Test_Limb
from oops.backplane.orbit      import Test_Orbit
from oops.backplane.pole       import Test_Pole
from oops.backplane.resolution import Test_Resolution
from oops.backplane.ring       import Test_Ring
from oops.backplane.sky        import Test_Sky
from oops.backplane.spheroid   import Test_Spheroid
from oops.backplane.where      import Test_Where

########################################
from oops.backplane.unittester_support import backplane_unittester_args

if __name__ == '__main__':
    backplane_unittester_args()
    unittest.main(verbosity=2)
################################################################################
