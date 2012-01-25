################################################################################
# oops/__init__.py
#
# 1/11/12 (MRS) - Updated define_solar_system() and added load_leap_seconds().
#   Removed define_saturn_cassni() because it is now part of instrument.cassini.
################################################################################

__all__ = ["Array", "Empty", "Scalar", "Pair", "Vector3", "Matrix3", "Tuple",
                "UnitScalar",
           "Body", "Event", "Transform",
           "Calibration", "AreaScaling", "Scaling",
           "FOV", "FlatFOV", "PolynomialFOV", "SubarrayFOV", "SubsampledFOV",
           "Frame", "Cmatrix", "MatrixFrame", "RingFrame", "SpiceFrame",
           "Observation", "Snapshot",
           "Path", "SpicePath",
           "Surface", "RingPlane"]

import numpy as np
import os
import spicedb
import julian

DEBUG = False

# Handy constants

C = 299792.458                  # speed of light in km/s
RPD = np.pi / 180.              # radians per degree
DPR = 180. / np.pi              # degrees per radian
SPR = 3600. * 180. / np.pi      # arcseconds per radian
RPS = np.pi / (3600. * 180.)    # radians per arcsecond
SPD = 86400.                    # seconds per day

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
import cspice
import utils

from broadcastable.Array    import Array
from broadcastable.Empty    import Empty
from broadcastable.Scalar   import Scalar
from broadcastable.Pair     import Pair
from broadcastable.Vector3  import Vector3
from broadcastable.Matrix3  import Matrix3
from broadcastable.UnitScalar import UnitScalar

# Single-instance classes
from Event                  import Event
from Transform              import Transform

# Multiple-instance classes
from calibration.Calibration import Calibration
from calibration.AreaScaling import AreaScaling
from calibration.Scaling     import Scaling

from fov.FOV                import FOV
from fov.FlatFOV            import FlatFOV
from fov.PolynomialFOV      import PolynomialFOV
from fov.SubarrayFOV        import SubarrayFOV
from fov.SubsampledFOV      import SubsampledFOV

from frame.Frame            import Frame
from frame.Cmatrix          import Cmatrix
from frame.MatrixFrame      import MatrixFrame
from frame.RingFrame        import RingFrame
from frame.SpiceFrame       import SpiceFrame

from observation.Observation import Observation
from observation.Snapshot   import Snapshot

from path.Path              import Path, Waypoint
from path.MultiPath         import MultiPath
from path.SpicePath         import SpicePath

from surface.Surface        import Surface
from surface.RingPlane      import RingPlane

# NOTE: Instrument modules have to be loaded last. See bottom of file.
# import instrument
# import instrument.hst
# from instrument.hst.acs     import *
# from instrument.hst.nicmos  import *
# from instrument.hst.wfc3    import *
# from instrument.hst.wfpc2   import *
# 
# import instrument.cassini
# import instrument.cassini.iss

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
# Useful class methods to define SPICE-related quantities
################################################################################

LSK_LOADED = False

def load_leap_seconds():
    """Loads the most recent leap seconds kernel if it was not already loaded.
    """

    global LSK_LOADED

    if LSK_LOADED: return

    # Query for the most recent LSK
    spicedb.open_db()
    lsk = spicedb.select_kernels("LSK")
    spicedb.close_db()

    # Furnish the LSK to the SPICE toolkit
    spicedb.furnish_kernels(lsk)

    # Initialize the Julian toolkit
    julian.load_from_kernel(os.path.join(spicedb.get_spice_path(),
                                         lsk[0].filespec))

    LSK_LOADED = True

def define_solar_system(start_time, stop_time, asof=None):
    """Constructs SpicePaths for all the planets and moons in the solar system
    (including Pluto). Each planet is defined relative to the SSB. Each moon
    is defined relative to its planet. Names are as defined within the SPICE
    toolkit.

    Input:
        start_time      start_time of the period to be convered, in ISO date or
                        date-time format.
        stop_time       stop_time of the period to be covered, in ISO date or
                        date-time format.
        asof            a UTC date such that only kernels released earlier than
                        that date will be included, in ISO format.
    """

    # Always load the most recent Leap Seconds kernel, but only once
    load_leap_seconds()

    # Convert the formats to times as recognized by spicedb.
    (day, sec) = julian.day_sec_from_iso(start_time)
    start_time = julian.ymdhms_format_from_day_sec(day, sec)

    (day, sec) = julian.day_sec_from_iso(stop_time)
    stop_time = julian.ymdhms_format_from_day_sec(day, sec)

    if asof is not None:
        (day, sec) = julian.day_sec_from_iso(stop_time)
        asof = julian.ymdhms_format_from_day_sec(day, sec)

    # Load the necessary SPICE kernels
    spicedb.open_db()
    spicedb.furnish_solar_system(start_time, stop_time, asof)
    spicedb.close_db()

    # Sun...
    ignore = oops.SpicePath("SUN","SSB")

    # Mercury...
    define_planet("MERCURY", [], [])

    # Venus...
    define_planet("VENUS", [], [])

    # Earth...
    define_planet("EARTH", ["MOON"], [])

    # Mars...
    define_planet("MARS", spicedb.MARS_ALL_MOONS, [])

    # Jupiter...
    define_planet("JUPITER", spicedb.JUPITER_REGULAR,
                             spicedb.JUPITER_IRREGULAR)

    # Saturn...
    define_planet("SATURN", spicedb.SATURN_REGULAR,
                            spicedb.SATURN_IRREGULAR)

    # Uranus...
    define_planet("URANUS", spicedb.URANUS_REGULAR,
                            spicedb.URANUS_IRREGULAR)

    # Neptune...
    define_planet("NEPTUNE", spicedb.NEPTUNE_REGULAR,
                             spicedb.NEPTUNE_IRREGULAR)

    # Pluto...
    define_planet("PLUTO", spicedb.PLUTO_ALL_MOONS, [])

def define_planet(planet, regular_ids, irregular_ids):

    # Define the planet's path and frame
    ignore = oops.SpicePath(planet, "SSB")
    ignore = oops.SpiceFrame(planet)

    # Define the SpicePaths of individual regular moons
    regulars = []
    for id in regular_ids:
        regulars += [oops.SpicePath(id, planet)]
        try:        # Some frames for small moons are not defined
            ignore = oops.SpiceFrame(id)
        except RuntimeError: pass
        except LookupError: pass

    # Define the Multipath of the regular moons, with and without the planet
    ignore = oops.MultiPath(regulars, planet,
                            id=(planet + "_REGULARS"))
    ignore = oops.MultiPath([planet] + regulars, "SSB",
                            id=(planet + "+REGULARS"))

    # Without irregulars, we're just about done
    if irregular_ids == []:
        ignore = oops.MultiPath([planet] + regulars, "SSB",
                                 id=(planet + "+MOONS"))
        return

    # Define the SpicePaths of individual irregular moons
    irregulars = []
    for id in irregular_ids:
        irregulars += [oops.SpicePath(id, planet)]
        try:        # Some frames for small moons are not defined
            ignore = oops.SpiceFrame(id)
        except RuntimeError: pass
        except LookupError: pass

    # Define the Multipath of all the irregular moons
    ignore = oops.MultiPath(regulars, planet, id=(planet + "_IRREGULARS"))

    # Define the Multipath of all the moons, with and without the planet
    ignore = oops.MultiPath(regulars + irregulars, planet,
                            id=(planet + "_MOONS"))
    ignore = oops.MultiPath([planet] + regulars + irregulars, "SSB",
                            id=(planet + "+MOONS"))

################################################################################

import instrument
import instrument.hst
from instrument.hst.acs     import *
from instrument.hst.nicmos  import *
from instrument.hst.wfc3    import *
from instrument.hst.wfpc2   import *

import instrument.cassini
import instrument.cassini.iss

################################################################################
