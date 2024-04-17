################################################################################
# oops/__init__.py
################################################################################

# Examples of import statements and how they work:
#
# >>> import oops
#   This imports the entire oops object tree (but not hosts), creating
#   classes such as oops.Scalar, ops.Event, oops.path.SpicePath, plus other
#   components such as oops.config. This is the recommended form of import.
#
# >>> import oops as abc
#   Imports the ENTIRE oops object tree but with the prefix "abc" replacing
#   "oops".
#
# >>> from oops import *
#   This imports all the oops components without the "oops" prefix. It can fill
#   up the default name space but there is nothing wrong with it.

import cspyce
import cspyce.aliases
cspyce.use_errors()
cspyce.use_aliases()

import oops.cadence.all     as cadence
import oops.calibration.all as calib
import oops.fov.all         as fov
import oops.gravity.all     as gravity
import oops.frame.all       as frame
import oops.observation.all as observation
import oops.path.all        as path
import oops.surface.all     as surface

# Add all abstract base classes to top level namespace
Cadence     = cadence.Cadence
Calibration = calibration.Calibration
FOV         = fov.FOV
Gravity     = gravity.Gravity
Frame       = frame.Frame
Observation = observation.Observation
Path        = path.Path
Surface     = surface.Surface

from oops.backplane import Backplane
from oops.body      import Body
from oops.event     import Event
from oops.fittable  import Fittable
from oops.meshgrid  import Meshgrid
from oops.transform import Transform

import oops.backplane.all           # define all Backplane methods

import oops.constants     as constants
import oops.spice_support as spice
import oops.config        as config
import oops.utils         as utils

from oops.constants import C, C_INVERSE, RPD, DPR, SPR, RPS, SPD, AU, \
                           PI, TWOPI, HALFPI

from polymath import Boolean, Matrix, Matrix3, Pair, Quaternion, Qube, Scalar, \
                     Units, Vector, Vector3

try:
    from ._version import __version__
except ImportError as err:
    __version__ = 'Version unspecified'


################################################################################
# Class cross-references and other class attributes to be defined after startup
################################################################################

Transform.FRAME_CLASS = frame.Frame
Transform.IDENTITY = Transform(Matrix3.IDENTITY,
                               Vector3.ZERO,
                               frame.Frame.J2000,
                               frame.Frame.J2000,
                               path.Path.SSB)

Event.PATH_CLASS = path.Path

################################################################################
