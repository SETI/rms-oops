##########################################################################################
# programs/gold_master/all.py
##########################################################################################

"""Import of every standard gold master test suite.

Importing this module imports each test suite module in turn, and each of those calls
register_test_suite() as a side effect. After this module has been imported, every
standard suite is available from get_test_suite().
"""

# Define all gold_master tests suites
# flake8: noqa: F401 -- these imports exist for their side effects

import programs.gold_master.ansa
import programs.gold_master.border
import programs.gold_master.distance
import programs.gold_master.lighting
import programs.gold_master.limb
import programs.gold_master.orbit
import programs.gold_master.pole
import programs.gold_master.resolution
import programs.gold_master.ring
import programs.gold_master.sky
import programs.gold_master.spheroid
import programs.gold_master.where

##########################################################################################
