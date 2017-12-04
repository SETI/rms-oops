################################################################################
# oops/body.py: Body class
################################################################################

import numpy as np
import os

import spicedb
import julian
import gravity
import cspice

from polymath import *

from oops.path_.path      import Path
from oops.path_.multipath import MultiPath
from oops.path_.spicepath import SpicePath

from oops.frame_.frame       import Frame, AliasFrame
from oops.frame_.ringframe   import RingFrame
from oops.frame_.poleframe   import PoleFrame
from oops.frame_.spiceframe  import SpiceFrame
from oops.frame_.synchronous import Synchronous

from oops.surface_.nullsurface import NullSurface
from oops.surface_.ringplane   import RingPlane
from oops.surface_.orbitplane  import OrbitPlane
from oops.surface_.spicebody   import spice_body

import oops.constants     as constants
import oops.spice_support as spice_support

################################################################################
# A list of known changes in SPICE names and IDs
# This also standardizes the SPICE names of provisionally-named bodies.
################################################################################

JUPITER_ALIASES = [
    # Jupiter [new code, old code], [formal name, provisional name]
    [[     55060], [         'S2003_J_2' ]],
    [[     55061], [         'S2003_J_3' ]],
    [[     55062], [         'S2003_J_4' ]],
    [[557, 55063], [         'S2003_J_5' ]],
    [[     55064], [         'S2003_J_9' ]],
    [[     55065], [         'S2003_J_10']],
    [[     55066], [         'S2003_J_12']],
    [[558, 55067], [         'S2003_J_15']],
    [[     55068], [         'S2003_J_16']],
    [[550       ], ['HERSE', 'S2003_J_17']],
    [[555, 55069], [         'S2003_J_18']],
    [[     55070], [         'S2003_J_19']],
    [[     55071], [         'S2003_J_23']],
    [[551, 55072], [         'S2010_J_1' ]],
    [[552, 55073], [         'S2010_J_2' ]],
    [[     55074], [         'S2011_J_1' ]],
    [[556, 55075], [         'S2011_J_2' ]],
    [[554       ], [         'S2016_J_1' ]],
    [[553, 55076], ['DIA',   'S2000_J_11']],
]

SATURN_ALIASES = [
    # Saturn [code], [preferred name, old name]
    [[65035], ['S2004_S_7' , 'S7_2004' ]],
    [[65040], ['S2004_S_12', 'S12_2004']],
    [[65041], ['S2004_S_13', 'S13_2004']],
    [[65045], ['S2004_S_17', 'S17_2004']],
    [[65048], ['S2006_S_1' , 'S01_2006']],
    [[65055], ['S2007_S_2' , 'S02_2007']],
    [[65050], ['S2006_S_3' , 'S03_2006']],
    [[65056], ['S2007_S_3' , 'S03_2007']],
]

ALIASES = JUPITER_ALIASES + SATURN_ALIASES

# Define within CSPICE
if cspice.VERSION == 2:
    for (codes, names) in ALIASES:
        cspice.boddef_aliases(names, codes)
else:
    for (codes, names) in ALIASES:
        while len(codes) < len(names):
            codes = [codes[0]] + codes
        for (c,n) in zip(codes, names)[::-1]:   # reverse; later boddefs win
            cspice.boddef(n, c)

################################################################################

class Body(object):
    """Defines the properties and relationships of solar system bodies.

    Bodies include planets, dwarf planets, satellites and rings. Each body has
    these attributes:
        name            the name of this body.
        spice_id        the ID from the SPICE toolkit, if the body found in
                        SPICE; otherwise, None.
        path            a Waypoint for the body's path
        frame           a Wayframe for the body's frame.
        ring_frame      the Wayframe of a "despun" RingFrame relevant to a ring
                        that might orbit this body. None if not (yet) defined.
        ring_is_retrograde  True if the ring frame is retrograde relative to
                            IAU-defined north.

        parent          the physical body (not necessarily the barycenter) about
                        which this body orbits. If a string is given, the parent
                        is found by looking it up in the BODY_REGISTRY
                        dictionary.
        barycenter      the body defining the barycenter of motion and the
                        gravity field defining this body's motion. If a string
                        is given, the barycenter is found by looking it up in
                        the BODY_REGISTRY dictionary.

        surface         the Surface object defining the body's surface. None if
                        the body is a point and has no surface.
        radius          a single value in km, defining the radius of a sphere
                        that encloses the entire body. Zero for bodies that have
                        no surface.
        inner_radius    a single value in km, defining the radius of a sphere
                        that is entirely enclosed by the body. Zero for bodies
                        that have no surface.
        gravity         the gravity field of the body; None if the gravity field
                        is undefined or negligible.
        keywords        a list of keywords associated with the body. Typical
                        values are "PLANET", "BARYCENTER", "SATELLITE",
                        "SPACECRAFT", "RING", and for satellites, "REGULAR",
                        "IRREGULAR", "CLASSICAL". The name of each body appears
                        as a keyword in its own keyword list. In addition, every
                        planet appears as a keyword for its system barycenter
                        and for each of its satellites and rings.

        children        a list of child bodies associated with this body. Every
                        Body object should appear on the list of the children of
                        its parent and also the children of its barycenter.
    """

    BODY_REGISTRY = {}      # global dictionary of body objects

    def __init__(self, name, path, frame, parent, barycenter,
                 spice_name=None):
        """Constructor for a Body object."""

        self.name = name

        if spice_name is None:
            spice_name = name

        try:
            self.spice_id = cspice.bodn2c(spice_name)
        except (KeyError, ValueError):
            self.spice_id = None

        self.ring_frame = None

        self.path = Path.as_waypoint(path)
        self.frame = Frame.as_wayframe(frame)

        if type(parent) == type(""):
            self.parent = Body.lookup(parent)
        else:
            self.parent = parent

        if type(barycenter) == type(""):
            self.barycenter = Body.lookup(barycenter)
        else:
            self.barycenter = barycenter

        self.surface = None
        self.radius = 0.
        self.inner_radius = 0.
        self.gravity = None
        self.keywords = [self.name]

        self.children = []

        # Append this to the appropriate child lists
        if self.parent is not None:
            if self not in self.parent.children:
                self.parent.children.append(self)

        if self.barycenter is not None:
            if self not in self.barycenter.children:
                self.barycenter.children.append(self)

        # Save it in the Solar System dictionary
        Body.BODY_REGISTRY[self.name] = self

    ########################################

    def apply_surface(self, surface, radius, inner_radius=0.):
        """Add the surface attribute to a Body."""

        self.surface = surface
        self.radius = radius
        self.inner_radius = inner_radius
        # assert self.surface.origin == self.path
        # This assertion is not strictly necessary

    def apply_ring_frame(self, epoch=None, retrograde=False, pole=None):
        """Add the ring and ring_frame attributes to a Body."""

        # On a repeat call, make sure the frames match
        if type(self.ring_frame) == RingFrame and pole is None:
            assert self.ring_frame.epoch == epoch
            assert self.ring_frame.retrograde == retrograde
            return

        if type(self.ring_frame) == PoleFrame and pole is not None:
            assert self.ring_frame.retrograde == retrograde
            assert self.ring_frame.invariable_pole == pole
            return

        if pole is not None:
            self.ring_frame = PoleFrame(self.frame, pole=pole,
                                                    retrograde=retrograde)
        else:
            self.ring_frame = RingFrame(self.frame, epoch=epoch,
                                                    retrograde=retrograde)

        self.ring_is_retrograde = retrograde

    def apply_gravity(self, gravity):
        """Add the gravity attribute to a Body."""

        self.gravity = gravity

    def add_keywords(self, keywords):
        """Add one or more keywords to the list associated with this Body."""

        if type(keywords) == type(""): keywords = [keywords]

        for keyword in keywords:
            # Avoid duplicates...
            if keyword not in self.keywords:
                self.keywords.append(keyword)

    ############################################################################
    # Tools for selecting the children of a body
    ############################################################################

    def select_children(self, include_all=None, include_any=None,
                              exclude=None, radius=None, recursive=False):
        """Return a list of body objects based on keywords and size."""

        if recursive:
            bodies = []
            self._recursive_children(bodies)
        else:
            bodies = self.children

        if include_all is not None:
            bodies = Body.keywords_include_all(bodies, include_all)

        if include_any is not None:
            bodies = Body.keywords_include_any(bodies, include_any)

        if exclude is not None:
            bodies = Body.keywords_do_not_include(bodies, exclude)

        if radius is not None:
            if type(radius) == type(0) or type(radius) == type(0.):
                radius = (radius, np.inf)
            elif len(radius) == 1:
                radius = (radius[0], np.inf)
            bodies = Body.radius_in_range(bodies, radius[0], radius[1])

        return bodies

    def _recursive_children(self, list):

        for child in self.children:
            if child not in list:
                list.append(child)
            child._recursive_children(list)

    @staticmethod
    def name_in(bodies, names):
        """Retain bodies if their names ARE found in the list provided."""

        if type(names) == type(""): names = [names]

        list = []
        for body in bodies:
            if body.name in names and body not in list:
                list.append(body)
        return list

    @staticmethod
    def name_not_in(bodies, names):
        """Retain bodies only if their names are NOT in the list provided."""

        if type(names) == type(""): names = [names]

        list = []
        for body in bodies:
            if body.name not in names and body not in list:
                list.append(body)
        return list

    @staticmethod
    def radius_in_range(bodies, min, max=np.inf):
        """Retain bodies if their radii fall INSIDE the range (min,max)."""

        list = []
        for body in bodies:
            if body.radius >= min and body.radius <= max and body not in list:
                list.append(body)
        return list

    @staticmethod
    def radius_not_in_range(bodies, min, max=np.inf):
        """Retain bodies if their radii fall OUTSIDE the range (min,max)."""

        list = []
        for body in bodies:
            if body.radius < min or body.radius > max and body not in list:
                list.append(body)
        return list

    @staticmethod
    def surface_class_in(bodies, class_names):
        """Retain bodies if the their surface class IS found in the list.

        Note that the name of the surface class is "NoneType" for cases where
        the surface has not been specified."""

        if type(class_names) == type(""): class_names = [class_names]

        list = []
        for body in bodies:
            name = type(body.surface).__name__
            if name in class_names and body not in list:
                list.append(body)
        return list

    @staticmethod
    def surface_class_not_in(bodies, class_names):
        """Retain bodies if their surface class is NOT found in the list.

        Note that the name of the surface class is "NoneType" for cases where
        the surface has not been specified."""

        if type(class_names) == type(""): class_names = [class_names]

        list = []
        for body in bodies:
            name = type(body.surface).__name__
            if name not in class_names and body not in list:
                list.append(body)
        return list

    @staticmethod
    def has_gravity(bodies):
        """Retain bodies on the list if they HAVE a defined gravity."""

        list = []
        for body in bodies:
            if body.gm is not None and body not in list:
                list.append(body)
        return list

    @staticmethod
    def has_no_gravity(bodies):
        """Retain bodies on the list if they have NO gravity."""

        list = []
        for body in bodies:
            if body.gm is None and body not in list:
                list.append(body)
        return list

    @staticmethod
    def has_children(bodies):
        """Retain bodies on the list if they HAVE children."""

        list = []
        for body in bodies:
            if body.children != [] and body not in list:
                list.append(body)
        return list

    @staticmethod
    def has_no_children(bodies):
        """Retain bodies on the list if they have NO children."""

        list = []
        for body in bodies:
            if body.children == [] and body not in list:
                list.append(body)
        return list

    @staticmethod
    def keywords_include_any(bodies, keywords):
        """Retain bodies that have at least one of the specified keywords."""

        if type(keywords) == type(""): keywords = [keywords]

        list = []
        for body in bodies:
            for keyword in keywords:
                if keyword in body.keywords and body not in list:
                    list.append(body)
                    break
        return list

    @staticmethod
    def keywords_include_all(bodies, keywords):
        """Retain bodies if they have all of the specified keywords."""

        if type(keywords) == type(""): keywords = [keywords]

        list = []
        for body in bodies:
            is_match = True
            for keyword in keywords:
                if keyword not in body.keywords:
                    is_match = False
                    break

            if is_match and body not in list:
                list.append(body)

        return list

    @staticmethod
    def keywords_do_not_include(bodies, keywords):
        """Retain bodies if they DO NOT have any of the specified keywords."""

        if type(keywords) == type(""): keywords = [keywords]

        list = []
        for body in bodies:
            is_found = False
            for keyword in keywords:
                if keyword in body.keywords:
                    is_found = True
                    break

            if not is_found and body not in list:
                list.append(body)

        return list

    ########################################

    @staticmethod
    def define_multipath(bodies, origin="SSB", frame="J2000", id=None):
        """Construct a multipath for the centers of the given list of bodies.

        The default ID of the path returned is the name of the first body with
        a "+" appended."""

        paths = []
        for body in bodies:
            paths.append(body.path)

        return MultiPath(paths, origin, frame, id)

    ############################################################################
    # Body registry
    ############################################################################

    @staticmethod
    def lookup(key):
        """Return a body from the registry given its name."""

        return Body.BODY_REGISTRY[key.upper()]

    @staticmethod
    def exists(key):
        """Return True if the body's name exists in the registry."""

        return key.upper() in Body.BODY_REGISTRY

    @staticmethod
    def as_body(body):
        """Return a body object given the registered name or the object itself.
        """

        if type(body) == Body: return body
        return Body.lookup(body)

    @staticmethod
    def as_body_name(body):
        """Return a body name given the registered name or the object itself."""

        if type(body) == Body: return body.name
        return body

    @staticmethod
    def reset_registry():
        """Initialize the registry.
    
        It is not generally necessary to call this function, but it can be used
        to reset the registry for purposes of debugging.
        """

        Body.BODY_REGISTRY.clear()

        spice_support.initialize()

        Path.reset_registry()
        Frame.reset_registry()

################################################################################
# General function to load Solar System components
################################################################################

def define_solar_system(start_time=None, stop_time=None, asof=None,
                        irregulars=True, planets=(1,2,3,4,5,6,7,8,9)):
    """Construct bodies, paths and frames for planets and their moons.

    Each planet is defined relative to the SSB. Each moon is defined relative to
    its planet. Names are as defined within the SPICE toolkit. Body associations
    are defined within the spicedb library.

    Input:
        start_time      start_time of the period to be convered, in ISO date
                        or date-time format.
        stop_time       stop_time of the period to be covered, in ISO date
                        or date-time format.
        asof            a UTC date such that only kernels released earlier
                        than that date will be included, in ISO format.
        irregulars      True to include the outer irregular satellites.
        planets         1-9 to load kernels for a particular planet and its
                        moons. 0 or None to load nine planets (including Pluto).
                        Use a tuple to list more than one planet number.

    Return              an ordered list of SPICE kernel names
    """

    if planets is None or planets == 0:
        planets = (1,2,3,4,5,6,7,8,9)
    if type(planets) == int:
        planets = (planets,)

    # Load the necessary SPICE kernels
    spicedb.open_db()
    names = spicedb.furnish_solar_system(start_time, stop_time, asof,
                                         planets=planets)
    spicedb.close_db()

    # Define B1950 in addition to J2000
    ignore = SpiceFrame("B1950", "J2000")

    # SSB and Sun
    define_bodies(["SSB"], None, None, ["SUN", "BARYCENTER"])
    define_bodies(["SUN"], None, None, ["SUN"])

    # Mercury, Venus, Earth orbit the Sun
    define_bodies([199, 299, 399], "SUN", "SUN", ["PLANET"])

    # Add Earth's Moon
    define_bodies([301], "EARTH", "EARTH",
                  ["SATELLITE", "CLASSICAL", "REGULAR"])

    # Define planetary systems
    if 4 in planets:
        _define_mars(start_time, stop_time, asof, irregulars)
    if 5 in planets:
        _define_jupiter(start_time, stop_time, asof, irregulars)
    if 6 in planets:
        _define_saturn(start_time, stop_time, asof, irregulars)
    if 7 in planets:
        _define_uranus(start_time, stop_time, asof, irregulars)
    if 8 in planets:
        _define_neptune(start_time, stop_time, asof, irregulars)
    if 9 in planets:
        _define_pluto(start_time, stop_time, asof, irregulars)

    return names

################################################################################
# Mars System
################################################################################

MARS_ALL_MOONS = range(401,403)

def _define_mars(start_time, stop_time, asof=None, irregulars=False):
    """Define components of the Mars system."""

    # Mars and the Mars barycenter orbit the Sun
    define_bodies([499], "SUN", "SUN", ["PLANET"])
    define_bodies([4], "SUN", "SUN", ["BARYCENTER"])

    # Moons and rings of Mars
    define_bodies(MARS_ALL_MOONS, "MARS", "MARS",
                  ["SATELLITE", "CLASSICAL", "REGULAR"])
    define_ring("MARS", "MARS_RING_PLANE", None, [])

################################################################################
# Jupiter System
################################################################################

JUPITER_CLASSICAL = range(501,505)
JUPITER_REGULAR   = [505] + range(514,517)
JUPITER_IRREGULAR = range(506,514) + range(517,551) + [554] + \
                    [55060, 55061, 55062, 55064, 55065, 55066, 55068, 55070,
                     55071, 55074]
# See definition of JUPITER_ALIASES at the top of the file for the list of
# additional, ambiguous irregular moons

JUPITER_MAIN_RING_LIMIT = 128940.

def _define_jupiter(start_time, stop_time, asof=None, irregulars=False):
    """Define components of the Jupiter system."""

    # Jupiter and the Jupiter barycenter orbit the Sun
    define_bodies([599], "SUN", "SUN", ["PLANET"])
    define_bodies([5], "SUN", "SUN", ["BARYCENTER"])

    # Moons and rings of Jupiter
    define_bodies(JUPITER_CLASSICAL, "JUPITER", "JUPITER",
                  ["SATELLITE", "CLASSICAL", "REGULAR"])
    define_bodies(JUPITER_REGULAR, "JUPITER", "JUPITER",
                  ["SATELLITE", "REGULAR"])

    if irregulars:
        define_bodies(JUPITER_IRREGULAR, "JUPITER", "JUPITER BARYCENTER",
                      ["SATELLITE", "IRREGULAR"])

        # For backwards compatibility, test alternative IDs for certain moons
        if cspice.VERSION < 2:      # Search for the aliased ID that is in SPK
          for (ids, names) in JUPITER_ALIASES:
            try:
                _ = cspice.spkez(ids[0], 0., 'J2000', 'NONE', 1) # did it work?
                cspice.boddef(names[0], ids[0])
                define_bodies([ids[0]], "JUPITER", "JUPITER BARYCENTER",
                                        ["SATELLITE", "IRREGULAR"])
            except RuntimeError:
                cspice.boddef(names[-1], ids[1])
                define_bodies([ids[1]], "JUPITER", "JUPITER BARYCENTER",
                                        ["SATELLITE", "IRREGULAR"])
        else:                       # cspice handles aliases; one ID is enough
            for (ids, names) in JUPITER_ALIASES:
                define_bodies([ids[0]], "JUPITER", "JUPITER BARYCENTER",
                                        ["SATELLITE", "IRREGULAR"])

    define_ring("JUPITER", "JUPITER_RING_PLANE", None, [])
    define_ring("JUPITER", "JUPITER_RING_SYSTEM", JUPITER_MAIN_RING_LIMIT, [])

################################################################################
# Saturn System
################################################################################

SATURN_CLASSICAL_INNER = range(601,607)     # Mimas through Titan orbit Saturn
SATURN_CLASSICAL_OUTER = range(607,609)     # Hyperion, Iapetus orbit barycenter
SATURN_CLASSICAL_IRREG = [609]              # Phoebe
SATURN_REGULAR   = range(610,619) + range(632,636) + [649,653]
SATURN_IRREGULAR = (range(619,632) + range(636,649) + range(650,653) +
                    [65035, 65040, 65041, 65045, 65048, 65050, 65055, 65056])

SATURN_MAIN_RINGS = ( 74658., 136780.)
SATURN_D_RING =     ( 66900.,  74658.)
SATURN_C_RING =     ( 74658.,  91975.)
SATURN_B_RING =     ( 91975., 117507.)
SATURN_A_RING =     (122340., 136780.)
SATURN_F_RING_CORE =  140220.
SATURN_F_RING_LIMIT = 140612.
SATURN_RINGS  = (SATURN_MAIN_RINGS[0], SATURN_F_RING_LIMIT)

def _define_saturn(start_time, stop_time, asof=None, irregulars=False):
    """Define components of the Saturn system."""

    # Saturn and the Saturn barycenter orbit the SSB
    define_bodies([699], "SUN", "SSB", ["PLANET"])
    define_bodies([6], "SUN", "SSB", ["BARYCENTER"])

    # Moons and rings of Saturn
    define_bodies(SATURN_CLASSICAL_INNER, "SATURN", "SATURN",
                  ["SATELLITE", "CLASSICAL", "REGULAR"])
    define_bodies(SATURN_CLASSICAL_OUTER, "SATURN", "SATURN BARYCENTER",
                  ["SATELLITE", "CLASSICAL", "REGULAR"])
    define_bodies(SATURN_CLASSICAL_IRREG, "SATURN", "SATURN BARYCENTER",
                  ["SATELLITE", "CLASSICAL", "IRREGULAR"])
    define_bodies(SATURN_REGULAR, "SATURN", "SATURN",
                  ["SATELLITE", "REGULAR"])

    if irregulars:
        define_bodies(SATURN_IRREGULAR, "SATURN", "SATURN BARYCENTER",
                      ["SATELLITE", "IRREGULAR"])

    define_ring("SATURN", "SATURN_RING_PLANE", None, [])
    define_ring("SATURN", "SATURN_RING_SYSTEM", SATURN_F_RING_LIMIT, [])
    define_ring("SATURN", "SATURN_RINGS", SATURN_RINGS, [])
    define_ring("SATURN", "SATURN_MAIN_RINGS", SATURN_MAIN_RINGS, [])
    define_ring("SATURN", "SATURN_A_RING", SATURN_A_RING, [])
    define_ring("SATURN", "SATURN_B_RING", SATURN_B_RING, [])
    define_ring("SATURN", "SATURN_C_RING", SATURN_C_RING, [])

################################################################################
# Uranus System
################################################################################

URANUS_CLASSICAL  = range(701,706)
URANUS_INNER      = range(706,716) + [725,726,727]
URANUS_IRREGULAR  = range(716,726)

URANUS_EPSILON_LIMIT = 51604.
URANUS_MU_LIMIT = [97700. - 17000./2, 97700. + 17700./2]
URANUS_NU_LIMIT = [67300. -  3800./2, 67300. +  3800./2]

# Special definitions of Uranian eccentric/inclined rings
URANUS_OLD_GRAVITY = gravity.Gravity(5793939., [3.34343e-3, -2.885e-5], 26200.)

# Local function used to adapt the tabulated elements from French et al. 1991.
def _uranus_ring_elements(a, e, peri, i, node, da):
    n = URANUS_OLD_GRAVITY.n(a)
    prec = URANUS_OLD_GRAVITY.combo(a, (1,-1, 0))
    regr = URANUS_OLD_GRAVITY.combo(a, (1, 0,-1))
    peri *= constants.RPD
    node *= constants.RPD
    i *= constants.RPD
    return (a, 0., n, e, peri, prec, i, node, regr, (a+da/2))
    # The extra item returned is the outer radius in the orbit's frame

URANUS_SIX_ELEMENTS  = _uranus_ring_elements(
                        41837.15, 1.013e-3, 242.80, 0.0616,  12.12, 2.8)
URANUS_FIVE_ELEMENTS = _uranus_ring_elements(
                        42234.82, 1.899e-3, 170.31, 0.0536, 286.57, 2.8)
URANUS_FOUR_ELEMENTS = _uranus_ring_elements(
                        42570.91, 1.059e-3, 127.28, 0.0323,  89.26, 2.7)
URANUS_ALPHA_ELEMENTS = _uranus_ring_elements(
                        44718.45, 0.761e-3, 333.24, 0.0152,  63.08, 7.15+3.52)
URANUS_BETA_ELEMENTS = _uranus_ring_elements(
                        45661.03, 0.442e-3, 224.88, 0.0051, 310.05, 8.15+3.07)
URANUS_ETA_ELEMENTS = _uranus_ring_elements(
                        47175.91, 0.004e-3, 228.10, 0.0000,   0.00, 1.7)
URANUS_GAMMA_ELEMENTS = _uranus_ring_elements(
                        47626.87, 0.109e-3, 132.10, 0.0000,   0.00, 3.8)
URANUS_DELTA_ELEMENTS = _uranus_ring_elements(
                        48300.12, 0.004e-3, 216.70, 0.0011, 260.70, 7.0)
URANUS_LAMBDA_ELEMENTS = _uranus_ring_elements(
                        50023.94, 0.000e-3,   0.00, 0.0000,   0.00, 2.5)
URANUS_EPSILON_ELEMENTS = _uranus_ring_elements(
                        51149.32, 7.936e-3, 214.97, 0.0000,   0.00, 58.1+37.6)

def _define_uranus(start_time, stop_time, asof=None, irregulars=False):
    """Define components of the Uranus system."""

    # Uranus and the Uranus barycenter orbit the SSB
    define_bodies([799], "SUN", "SSB", ["PLANET"])
    define_bodies([7], "SUN", "SSB", ["BARYCENTER"])

    # Moons and rings of Uranus
    define_bodies(URANUS_CLASSICAL, "URANUS", "URANUS",
                  ["SATELLITE", "CLASSICAL", "REGULAR"])
    define_bodies(URANUS_INNER, "URANUS", "URANUS",
                  ["SATELLITE", "REGULAR"])
    define_bodies(URANUS_IRREGULAR, "URANUS", "URANUS",
                  ["SATELLITE", "IRREGULAR"])

    define_ring("URANUS", "URANUS_RING_PLANE", None,  [], retrograde=True)
    define_ring("URANUS", "URANUS_RING_SYSTEM", URANUS_EPSILON_LIMIT, [],
                                                          retrograde=True)
    define_ring("URANUS", "MU_RING", URANUS_MU_LIMIT, [], retrograde=True)
    define_ring("URANUS", "NU_RING", URANUS_NU_LIMIT, [], retrograde=True)

    URANUS_EPOCH = cspice.utc2et("1977-03-10T20:00:00")

    uranus_wrt_b1950 = AliasFrame("IAU_URANUS").wrt("B1950")
    ignore = RingFrame(uranus_wrt_b1950, URANUS_EPOCH, retrograde=True,
                       id="URANUS_RINGS_B1950")

    define_orbit("URANUS", "SIX_RING", URANUS_SIX_ELEMENTS,
                           URANUS_EPOCH, "URANUS_RINGS_B1950", [])
    define_orbit("URANUS", "FIVE_RING", URANUS_FIVE_ELEMENTS,
                           URANUS_EPOCH, "URANUS_RINGS_B1950", [])
    define_orbit("URANUS", "FOUR_RING", URANUS_FOUR_ELEMENTS,
                           URANUS_EPOCH, "URANUS_RINGS_B1950", [])
    define_orbit("URANUS", "ALPHA_RING", URANUS_ALPHA_ELEMENTS,
                           URANUS_EPOCH, "URANUS_RINGS_B1950", ["MAIN"])
    define_orbit("URANUS", "BETA_RING", URANUS_BETA_ELEMENTS,
                           URANUS_EPOCH, "URANUS_RINGS_B1950", ["MAIN"])
    define_orbit("URANUS", "ETA_RING", URANUS_ETA_ELEMENTS,
                           URANUS_EPOCH, "URANUS_RINGS_B1950", ["MAIN"])
    define_orbit("URANUS", "GAMMA_RING", URANUS_GAMMA_ELEMENTS,
                           URANUS_EPOCH, "URANUS_RINGS_B1950", ["MAIN"])
    define_orbit("URANUS", "DELTA_RING", URANUS_DELTA_ELEMENTS,
                           URANUS_EPOCH, "URANUS_RINGS_B1950", ["MAIN"])
    define_orbit("URANUS", "LAMBDA_RING", URANUS_LAMBDA_ELEMENTS,
                           URANUS_EPOCH, "URANUS_RINGS_B1950", [])
    define_orbit("URANUS", "EPSILON_RING", URANUS_EPSILON_ELEMENTS,
                           URANUS_EPOCH, "URANUS_RINGS_B1950", ["MAIN"])

################################################################################
# Neptune System
################################################################################

NEPTUNE_CLASSICAL_INNER = [801]             # Triton
NEPTUNE_CLASSICAL_OUTER = [802]             # Nereid orbits barycenter
NEPTUNE_REGULAR   = range(803,809)
NEPTUNE_IRREGULAR = range(809,814)

NEPTUNE_ADAMS_LIMIT = 62940.

def _define_neptune(start_time, stop_time, asof=None, irregulars=False):
    """Define components of the Neptune system."""

    # Neptune and the Neptune barycenter orbit the SSB
    define_bodies([899], "SUN", "SSB", ["PLANET"])
    define_bodies([8], "SUN", "SSB", ["BARYCENTER"])

    # Moons and rings of Neptune
    define_bodies(NEPTUNE_CLASSICAL_INNER, "NEPTUNE", "NEPTUNE",
                  ["SATELLITE", "CLASSICAL", "REGULAR"])
    define_bodies(NEPTUNE_CLASSICAL_OUTER, "NEPTUNE", "NEPTUNE BARYCENTER",
                  ["SATELLITE", "CLASSICAL", "IRREGULAR"])
    define_bodies(NEPTUNE_REGULAR, "NEPTUNE", "NEPTUNE",
                  ["SATELLITE", "REGULAR"])

    if irregulars:
        define_bodies(NEPTUNE_IRREGULAR, "NEPTUNE", "NEPTUNE BARYCENTER",
                      ["SATELLITE", "IRREGULAR"])

    ra  = cspice.bodvrd('NEPTUNE', 'POLE_RA')[0]  * np.pi/180
    dec = cspice.bodvrd('NEPTUNE', 'POLE_DEC')[0] * np.pi/180
    pole = Vector3.from_ra_dec_length(ra,dec)

    define_ring("NEPTUNE", "NEPTUNE_RING_PLANE",  None, [], pole=pole)
    define_ring("NEPTUNE", "NEPTUNE_RING_SYSTEM", NEPTUNE_ADAMS_LIMIT, [],
                                                  pole=pole)

################################################################################
# Pluto System
################################################################################

CHARON        = [901]
PLUTO_REGULAR = range(902,906)

def _define_pluto(start_time, stop_time, asof=None, irregulars=False):
    """Define components of the Pluto system."""

    # Pluto and the Pluto barycenter orbit the SSB
    define_bodies([999], "SUN", "SSB", ["PLANET"])
    define_bodies([9], "SUN", "SSB", ["BARYCENTER"])

    # Moons and rings of Pluto
    define_bodies(CHARON, "PLUTO", "PLUTO",
                  ["SATELLITE", "CLASSICAL", "REGULAR"])
    define_bodies(PLUTO_REGULAR, "PLUTO", "PLUTO BARYCENTER",
                  ["SATELLITE", "REGULAR"])

    define_ring("PLUTO", "PLUTO_RING_PLANE", None, [],
                barycenter_name="PLUTO BARYCENTER")
    define_ring("PLUTO", "PLUTO_INNER_RING_PLANE", None, [],
                barycenter_name="PLUTO")

    barycenter = Body.BODY_REGISTRY["PLUTO BARYCENTER"]
    barycenter.ring_frame = Body.BODY_REGISTRY["PLUTO"].ring_frame

################################################################################
# Define bodies and rings...
################################################################################

def define_bodies(spice_ids, parent, barycenter, keywords):
    """Define the path, frame, surface for bodies by name or SPICE ID.

    All must share a common parent and barycenter."""

    for spice_id in spice_ids:

        # Define the body's path
        path = SpicePath(spice_id, "SSB")

        # The name of the path is the name of the body
        name = path.path_id

        # If the body already exists, skip it
        if name in Body.BODY_REGISTRY: continue

        # Sometimes a frame is undefined for a new moon; in this case assume it
        # is synchronous
        try:
            frame = SpiceFrame(spice_id)
        except LookupError:
            if ('BARYCENTER' in keywords) or ('IRREGULAR' in keywords):
                frame = Frame.J2000
            else:
                frame = Synchronous(path, parent, id='SYNCHRONOUS_' + name)

        # Define the planet's body
        # Note that this will overwrite any registered body of the same name
        body = Body(name, name, frame.frame_id, parent, barycenter)
        body.add_keywords(keywords)

        # Add the gravity object if it exists
        try:
            body.apply_gravity(gravity.LOOKUP[name])
        except KeyError: pass

        # Add the surface object if shape information is available
        try:
            shape = spice_body(spice_id, frame.frame_id, (1.,1.,1.))
            body.apply_surface(shape, shape.req, shape.rpol)
        except RuntimeError:
            shape = NullSurface(path, frame)
            body.apply_surface(shape, 0., 0.)
        except LookupError:
            shape = NullSurface(path, frame)
            body.apply_surface(shape, 0., 0.)

        # Add a planet name to any satellite or barycenter
        if "SATELLITE" in body.keywords and parent is not None:
            body.add_keywords(parent)

        if "BARYCENTER" in body.keywords and parent is not None:
            body.add_keywords(parent)

def define_ring(parent_name, ring_name, radii, keywords, retrograde=False,
                barycenter_name=None, pole=None):
    """Define the path, frame, surface and body for ring, given radial limits.

    A single radius value is used to define the outer limit of rings. Note that
    a ring has limits but a defined ring plane does not.

    Input:
        parent_name     the name of the central planet for the ring surface.
        ring_name       the name of the surface.
        radii           if this is a tuple with two values, these are the radial
                        limits of the ring; if it is a scalar, then the ring
                        plane has no defined radial limits, but the radius
                        attribute of the body will be set to this value; if
                        None, then the radius attribute of the body will be set
                        to zero.
        keywords        the list of keywords under which this surface is to be 
                        registered. Every ring is also registered under its own
                        name and under the keyword "RING".
        retrograde      True if the ring is retrograde relative to the central
                        planet's IAU-defined pole.
        barycenter_name the name of the ring's barycenter if this is not the
                        same as the name of the central planet.
        pole            if not None, the pole of the invariable plane to be used
                        in the PoleFrame (instead of a RingFrame).
    """

    # If the ring body already exists, skip it
    if ring_name in Body.BODY_REGISTRY: return

    # Identify the parent
    parent = Body.lookup(parent_name)
    parent.apply_ring_frame(retrograde=retrograde, pole=pole)

    if barycenter_name is None:
        barycenter = parent
    else:
        barycenter = Body.lookup(barycenter_name)

    # Interpret the radii
    try:
        rmax = radii[1]
    except IndexError:
        rmax = radii[0]
        radii = None
    except TypeError:
        if radii is None:
            rmax = 0.
        else:
            rmax = radii
            radii = None

    # Create the ring body
    # Note that this will overwrite any registered ring of the same name
    body = Body(ring_name, barycenter.path, parent.ring_frame,
                parent, parent)
    body.apply_gravity(barycenter.gravity)
    body.apply_ring_frame(retrograde=retrograde, pole=pole)

    shape = RingPlane(barycenter.path, parent.ring_frame, radii,
                      gravity=barycenter.gravity)

    body.apply_surface(shape, rmax, 0.)

    body.add_keywords([parent, "RING", ring_name])
    body.add_keywords(keywords)

def define_orbit(parent_name, ring_name, elements, epoch, reference, keywords):
    """Define the path, frame, surface and body for ring given orbital elements.

    The ring can be inclined or eccentric.
    """

    parent = Body.lookup(parent_name)

    orbit = OrbitPlane(elements, epoch, parent.path, reference, id=ring_name)

    body = Body(ring_name, orbit.internal_origin, orbit.internal_frame,
                parent, parent)
    body.apply_surface(orbit, elements[9], 0.)

    body.add_keywords([parent, "RING", "ORBIT", ring_name])
    body.add_keywords(keywords)

def define_small_body(spice_id, name=None, spk=None, keywords=[],
                                parent='SUN', barycenter='SSB'):
    """Define the path, frame, surface for a body by SPICE ID.

    This body treats the Sun as its parent body and barycenter."""

    # Load the SPK if necessary
    if spk:
        cspice.furnsh(spk)

    # Define the body's path
    path = SpicePath(spice_id, "SSB", id=name)

    # The name of the path is the name of the body
    name = name or path.path_id

    # If the body already exists, skip it
    if name in Body.BODY_REGISTRY: return

    # Sometimes a frame is undefined for a new moon; in this case assume it
    # is synchronous
    try:
        frame = SpiceFrame(spice_id)
    except LookupError:
        if ('BARYCENTER' in keywords) or ('IRREGULAR' in keywords):
            frame = Frame.J2000
        else:
            frame = Synchronous(path, parent, id='SYNCHRONOUS_' + name)

    # Define the planet's body
    # Note that this will overwrite any registered body of the same name
    body = Body(name, path.path_id, frame.frame_id,
                      parent=Body.lookup(parent),
                      barycenter=Body.lookup(barycenter))
    body.add_keywords(keywords)

    # Add the gravity object if it exists
    try:
        body.apply_gravity(gravity.LOOKUP[name])
    except KeyError: pass

    # Add the surface object if shape information is available
    try:
        shape = spice_body(spice_id, frame.frame_id, (1.,1.,1.))
        body.apply_surface(shape, shape.req, shape.rpol)
    except RuntimeError:
        shape = NullSurface(path, frame)
        body.apply_surface(shape, 0., 0.)
    except LookupError:
        shape = NullSurface(path, frame)
        body.apply_surface(shape, 0., 0.)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Body(unittest.TestCase):

    def runTest(self):

        # Imports are here to avoid conflicts
        Path.reset_registry()
        Frame.reset_registry()
        Body.reset_registry()

        define_solar_system("2000-01-01", "2020-01-01")

        self.assertEqual(Body.lookup("DAPHNIS").barycenter.name,
                         "SATURN")
        self.assertEqual(Body.lookup("PHOEBE").barycenter.name,
                         "SATURN BARYCENTER")

        mars = Body.lookup("MARS")
        moons = mars.select_children(include_all=["SATELLITE"])
        self.assertEqual(len(moons), 2)     # Phobos, Deimos

        saturn = Body.lookup("SATURN")
        moons = saturn.select_children(include_all=["CLASSICAL", "IRREGULAR"])
        self.assertEqual(len(moons), 1)     # Phoebe

        moons = saturn.select_children(exclude=["IRREGULAR","RING"], radius=160)
        self.assertEqual(len(moons), 8)     # Mimas-Iapetus

        rings = saturn.select_children(include_any=("RING"))
        self.assertEqual(len(rings), 7)     # A, B, C, Main, all, plane, system

        moons = saturn.select_children(include_all="SATELLITE",
                                       exclude=("IRREGULAR"), radius=1000)
        self.assertEqual(len(moons), 1)     # Titan only

        sun = Body.lookup("SUN")
        planets = sun.select_children(include_any=["PLANET"])
        self.assertEqual(len(planets), 9)

        sun = Body.lookup("SUN")
        planets = sun.select_children(include_any=["PLANET", "EARTH"])
        self.assertEqual(len(planets), 9)

        sun = Body.lookup("SUN")
        planets = sun.select_children(include_any=["PLANET", "EARTH"],
                                      recursive=True)
        self.assertEqual(len(planets), 10)  # 9 planets plus Earth's moon

        sun = Body.lookup("SUN")
        planets = sun.select_children(include_any=["PLANET", "JUPITER"],
                                      exclude=["IRREGULAR", "BARYCENTER", "IO"],
                                      recursive=True)
        self.assertEqual(len(planets), 16)  # 9 planets + 7 Jovian moons

        Path.reset_registry()
        Frame.reset_registry()
        Body.reset_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
