##########################################################################################
# oops/__init__.py
##########################################################################################
"""An observation-geometry library for planetary science.

`oops` models instruments, spacecraft trajectories, and target bodies, and computes
per-pixel geometry ("backplanes") for real observations. It is built on SPICE, through
`cspyce`, and on the `polymath` array types.

The library is organized around a handful of abstract classes, each with a subpackage of
concrete implementations: :class:`~oops.Path` for the motion of a point in space,
:class:`~oops.Frame` for the orientation of a coordinate frame, :class:`~oops.Surface` for
a 2-D body surface, :class:`~oops.FOV` for the geometry of a field of view,
:class:`~oops.Cadence` for the timing of an observation, :class:`~oops.Calibration` for
the relationship between data numbers and physical units, :class:`~oops.Gravity` for the
gravity field of a planet, and :class:`~oops.Observation` for an instrument's data array.
:class:`~oops.Body` ties a target to its path, frame, surface and gravity;
:class:`~oops.Event` describes a photon arriving at or departing from a point in
spacetime; and :class:`~oops.Backplane` computes the geometry of every pixel of an
Observation.

Geometric values are `polymath` objects, not NumPy arrays, and all quantities are in km,
seconds TDB, and radians.
"""

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
import oops.mutable       as mutable

from oops.constants import C, C_INVERSE, RPD, DPR, SPR, RPS, SPD, AU, PI, TWOPI, HALFPI

from polymath import (Boolean, Matrix, Matrix3, Pair, Quaternion, Qube, Scalar, Vector,
                      Vector3)

__all__ = ['cadence', 'calibration', 'fov', 'gravity', 'frame', 'observation', 'path',
           'surface', 'obs', 'backplane', 'body', 'cache', 'event', 'fittable',
           'meshgrid', 'transform', 'Cadence', 'Calibration', 'FOV', 'Gravity', 'Frame',
           'Observation', 'Path', 'Surface', 'Backplane', 'Body', 'Cache', 'Event',
           'Fittable', 'Meshgrid', 'Transform', 'constants', 'spice', 'config', 'mutable',
           'C', 'C_INVERSE', 'RPD', 'DPR', 'SPR', 'RPS', 'SPD', 'AU', 'PI', 'TWOPI',
           'HALFPI', 'Boolean', 'Matrix', 'Matrix3', 'Pair', 'Quaternion', 'Qube',
           'Scalar', 'Vector', 'Vector3']

try:
    from ._version import __version__
except ImportError:
    __version__ = 'Version unspecified'

##########################################################################################
# The hierarchy of imports is:
#   Body, Surface, Path, Gravity, Event, Frame, Transform
# Each class can reference classes later in the list, but any reference to a class earlier
# in the list requires this approach. Cache and Fittable sit outside that hierarchy and
# are filled in here for the same reason.
##########################################################################################

Transform._Frame = frame.Frame
Transform.IDENTITY = Transform(Matrix3.IDENTITY,
                               Vector3.ZERO,
                               frame.Frame.J2000,
                               frame.Frame.J2000,
                               origin=path.Path.SSB)

Cache._Frame = Frame
Cache._Path = Path

Frame._Event = Event
Frame._Path = Path

Event._Path = path.Path
Event.SSB = path.Path.SSB

Path._Body = Body

Surface._Body = Body

Fittable._MUTABLE = mutable

##########################################################################################
