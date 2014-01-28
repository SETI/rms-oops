################################################################################
# oops/calib_/flat.py: Subclass Flat of class Calibration
#
# 3/20/12 MRS - New.
################################################################################

from oops.calib_.extended import ExtendedSource
from oops.array_ import *

class Flat(ExtendedSource):
    """Subclass Flat of class Calibration is an alternative name for subclass
    ExtendedSource. It is identical. Its name is provided to make it clear that
    this is the proper calibration object to use for a flat (i.e., undistorted)
    field of view.
    """

    pass

################################################################################
