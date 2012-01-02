################################################################################
# oops/__init__.py
################################################################################

__all__ = ["Array", "Scalar", "Pair", "Vector3", "Matrix3", "Tuple",
           "Body", "Event", "Transform",
           "FOV", "FlatFOV", "PolynomialFOV", "SubarrayFOV", "SubsampledFOV",
           "Frame", "Cmatrix", "RingFrame", "SpiceFrame",
           "Observation",
           "Path", "SpicePath",
           "Surface", "RingPlane"]

DEBUG = False

C = 299792.458  # speed of light in km/s

################################################################################
# Programmer's Note
#
# Class interdependencies force Python to determine its load order for the
# individual modules. Because OOPS is spread out over so many files, care must
# be exercised to ensure there are no circular dependencies, where "a.py"
# requires "b.py" and vice-versa.
#
# The class interdependencies are as follows:
#
# - utils is self-contained.
# - Array and Scalar have interdependencies but are defined in the same file.
# - Pair, Tuple and Vector3 depend on Scalar and Array.
# - Matrix3 depends on Vector3.
#
# This completes the broadcastable classes. Everything else depends on these.
#
# - Transform depends only on broadcastables.
# - Frame depends on Transform.
# - Event depends on Path and Frame.
# - Path depends on Frame.
#
# Note that it is very easy to write innocent-looking code that violates the
# ordering of the four classes above.
#
# After this point, things get a bit more flexible:
#
# - Surface is independent.
# - FOV is independent.
# - Observation depends on FOV and just about everything else.
# - Instrument depends on Observation, but is not a class per se, just a
#   collection of useful routines.
#
# The class variable OOPS_CLASS is used here in there where the obvious but
# illegal alternative would be to use is_instance(). The latter would force a
# circular reference and oops could not load.
#
# Similarly, note that the Path and Frame registries are defined below in this
# file so they can be accessed globally and circular references
################################################################################

################################################################################
# These are for the registries. However, they need to be defined before Frame
# and Path are imported.
################################################################################

J2000FRAME = None
FRAME_REGISTRY = {}

SSBPATH = None
PATH_REGISTRY = {}

########################################

def is_id(item):
    abbr = item.__class__.__name__[0:3]
    return abbr in ("int", "str")

################################################################################
# Imports...
################################################################################

# Basics
import utils

from broadcastable.Array    import Array
from broadcastable.Empty    import Empty
from broadcastable.Scalar   import Scalar
from broadcastable.Pair     import Pair
from broadcastable.Vector3  import Vector3
from broadcastable.Matrix3  import Matrix3

# Single-instance classes
from Event                  import Event
from Transform              import Transform

# Multiple-instance classes
from fov.FOV                import FOV
from fov.FlatFOV            import FlatFOV
from fov.PolynomialFOV      import PolynomialFOV
from fov.SubarrayFOV        import SubarrayFOV
from fov.SubsampledFOV      import SubsampledFOV

from frame.Frame            import Frame
from frame.Cmatrix          import Cmatrix
from frame.RingFrame        import RingFrame
from frame.SpiceFrame       import SpiceFrame

from observation.Observation import Observation
from observation.Snapshot   import Snapshot

from path.Path              import Path, Waypoint
from path.MultiPath         import MultiPath
from path.SpicePath         import SpicePath

from surface.Surface        import Surface
from surface.RingPlane      import RingPlane

import instrument
import instrument.hst
from instrument.hst.acs     import *
from instrument.hst.nicmos  import *
from instrument.hst.wfc3    import *
from instrument.hst.wfpc2   import *

################################################################################
# Global Frames Registry
#
# See frame/Frame.py for more information. The registry gets initialized when
# this file is loaded. The registry is defined here rather than in Frame.py to
# avoid circularities that can arise in the order that modules are loaded.
################################################################################

def as_frame(frame):
    """Returns a Frame object given the registered name or the object
    itself."""

    if frame is None: return None

    try:
        test = frame.frame_id
        return frame
    except AttributeError:
        return FRAME_REGISTRY[frame]

def as_frame_id(frame):
    """Returns a Frame ID given the object or a registered ID."""

    if frame is None: return None

    try:
        return frame.frame_id
    except AttributeError: 
        return frame

def as_primary_frame(frame):
    """Returns the primary definition of a Frame object, based on a
    registered name or a Frame object."""

    try:
        return FRAME_REGISTRY[frame.frame_id]
    except AttributeError:
        return FRAME_REGISTRY[frame]

################################################################################
# Global Path Registry
#
# See path/Path.py for more information. The registry gets initialized when
# this file is loaded. The registry is defined here rather than in Path.py to
# avoid circularities that can arise in the order that modules are loaded.
################################################################################

def as_path(path):
    """Returns a Path object given the registered name or the object
    itself."""

    if path is None: return None

    try:
        test = path.path_id
        return path
    except AttributeError:
        return PATH_REGISTRY[path]

def as_path_id(path):
    """Returns a path ID given the object or a registered ID."""

    if path is None: return None

    try:
        return path.path_id
    except AttributeError:
        return path

def as_primary_path(path):
    """Returns the primary definition of a Path object, based on a
    registered name or a Path object."""

    try:
        return PATH_REGISTRY[path.path_id]
    except AttributeError:
        return PATH_REGISTRY[path]

################################################################################
