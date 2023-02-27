################################################################################
# oops/calibration/unittester.py
################################################################################

import unittest

from oops.calibration                import Test_Calibration
from oops.calibration.extendedsource import Test_ExtendedSource # DEPRECATED
from oops.calibration.flatcalib      import Test_FlatCalib
from oops.calibration.nullcalib      import Test_NullCalib
from oops.calibration.pointsource    import Test_PointSource    # DEPRECATED
from oops.calibration.radiance       import Test_Radiance
from oops.calibration.rawcounts      import Test_RawCounts

########################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
