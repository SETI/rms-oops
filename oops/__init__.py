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
from oops.cache     import Cache
from oops.event     import Event
from oops.fittable  import Fittable
from oops.meshgrid  import Meshgrid
from oops.transform import Transform

import oops.backplane.all           # define all Backplane methods

import oops.constants     as constants
import oops.spice_support as spice
import oops.config        as config
import oops.utils         as utils
import oops.mutable       as mutable

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

Transform._Frame = frame.Frame
Transform.IDENTITY = Transform(Matrix3.IDENTITY,
                               Vector3.ZERO,
                               frame.Frame.J2000,
                               frame.Frame.J2000,
                               path.Path.SSB)

Cache._Frame = Frame
Cache._Path = Path

Frame._Event = Event
Frame._Path = Path

Event._Path = path.Path
Event.SSB = path.Path.SSB

Path._Body = Body

Surface._Body = Body

Fittable._MUTABLE = mutable

################################################################################
