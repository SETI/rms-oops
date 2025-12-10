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

import oops.cadence
import oops.calibration
import oops.fov
import oops.gravity
import oops.frame
import oops.observation
import oops.path
import oops.surface

oops.obs = oops.observation         # handy abbreviation

# Add all abstract base classes to top level namespace
Cadence     = oops.cadence.Cadence
Calibration = oops.calibration.Calibration
FOV         = oops.fov.FOV
Gravity     = oops.gravity.Gravity
Frame       = oops.frame.Frame
Observation = oops.observation.Observation
Path        = oops.path.Path
Surface     = oops.surface.Surface

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
                     Vector, Vector3

try:
    from ._version import __version__
except ImportError as err:
    __version__ = 'Version unspecified'

################################################################################
# The hierarchy of imports is:
#   Body, Surface, Path, Gravity, Event, Frame, Transform
# Each class can reference classes later in the list, but any reference to a
# class earlier in the list requires this approach.
################################################################################

Transform.FRAME_CLASS = frame.Frame
Transform.IDENTITY = Transform(Matrix3.IDENTITY,
                               Vector3.ZERO,
                               frame.Frame.J2000,
                               frame.Frame.J2000,
                               path.Path.SSB)

Frame.EVENT_CLASS = Event
Frame.PATH_CLASS = Path
Frame.SPICEPATH_CLASS = oops.path.SpicePath

Event.PATH_CLASS = path.Path
Event.SSB = path.Path.SSB

Gravity.BODY_CLASS = Body
Surface.BODY_CLASS = Body
Path.BODY_CLASS = Body

################################################################################
