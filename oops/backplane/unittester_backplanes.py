################################################################################
# oops/unittester_backplanes.py: Unit-test most backplane methods
################################################################################
#
# Usage:
#   python unittester_backplanes.py [--png] [--out dir] [--log] [--silent]
#
#   --png           save a PNG image of each backplane array
#   --out dir       save a PNG image of each backplane array to this directory
#   --log           enable the internal oops logging
#   --silent        don't print any output to the terminal
#
# Note that backplanes_unittests.py now has basically the same functionality.
# However, by default, that program also executes the general backplane tests
# when executed from the command line. This program only executes the Backplane
# exercises.

import sys
import unittest
import oops
from oops.backplane_unittests import Test_Backplane_Exercises

if __name__ == '__main__':

    Test_Backplane_Exercises.EXERCISES = True

    if '--png' in sys.argv:
        Test_Backplane_Exercises.SAVING = True
        sys.argv.remove('--png')
    else:
        Test_Backplane_Exercises.SAVING = False

    if '--out' in sys.argv:
        EXERCISES_ONLY = True
        Test_Backplane_Exercises.EXERCISES = True
        Test_Backplane_Exercises.SAVING = True

        k = sys.argv.index('--out')
        Test_Backplane_Exercises.OUTPUT = sys.argv[k+1]
        del sys.argv[k:k+2]
    else:
        Test_Backplane_Exercises.OUTPUT = './'

    if '--log' in sys.argv:
        Test_Backplane_Exercises.LOGGING = True
        sys.argv.remove('--log')
    else:
        Test_Backplane_Exercises.LOGGING = False

    if '--silent' in sys.argv:
        Test_Backplane_Exercises.PRINTING = False
        sys.argv.remove('--silent')
    else:
        Test_Backplane_Exercises.PRINTING = True

    unittest.main(verbosity=2)

################################################################################
