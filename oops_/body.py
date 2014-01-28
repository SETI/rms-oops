################################################################################
# oops_/body.py: Body class
#
# 2/18/12 Created (MRS).
# 8/8/12 MRS - Added inner_radius attribute.
# 1/4/13 MRS - Added ring_is_retrograde attribute.
################################################################################

import numpy as np
import os
import spicedb
import julian
import gravity
import cspice

import oops_.path.all as path_
import oops_.frame.all as frame_
import oops_.surface.all as surface
import oops_.spice_support as spice
import oops_.registry as registry
import oops_.constants as constants

class Body(object):
    """Body is a class that defines the properties of, and relationships
    between, solar system objects such as planets, satellites and rings.

    Each body has these attributes:
        name            the name of this body.
        path_id         the ID of the Path this body follows.
        path            a Waypoint for the body's path
        frame_id        the ID of the coordinate frame describing this body.
        frame           a Wayframe for the body's frame.
        ring_frame_id   the ID of a "despun" frame relevant to a ring that might
                        orbit this body. None if not (yet) defined.
        ring_frame      a RingFrame for the body.
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

    def __init__(self, name, path_id, frame_id, parent, barycenter):
        """Constructor for a Body object."""

        self.name = name
        self.path_id = path_id
        self.frame_id = frame_id
        self.ring_frame_id = None

        self.path = path_.Waypoint(self.path_id)
        self.frame = frame_.Wayframe(self.frame_id)

        if type(parent) == type(""):
            self.parent = registry.body_lookup(parent)
        else:
            self.parent = parent

        if type(barycenter) == type(""):
            self.barycenter = registry.body_lookup(barycenter)
        else:
            self.barycenter = barycenter

        self.surface = None
        self.radius = 0.
        self.inner_radius = 0.
        self.gravity = None
        self.keywords = [self.name]

        self.children = []

        # Append this to the appopriate child lists
        if self.parent is not None:
            if self not in self.parent.children:
                self.parent.children.append(self)

        if self.barycenter is not None:
            if self not in self.barycenter.children:
                self.barycenter.children.append(self)

        # Save it in the Solar System dictionary
        registry.BODY_REGISTRY[self.name] = self

########################################

    def apply_surface(self, surface, radius, inner_radius=0.):
        """Adds the surface attribute to a Body."""

        self.surface = surface
        self.radius = radius
        self.inner_radius = inner_radius
        # assert self.surface.origin_id == self.path_id
        # This assertion is not strictly necessary

    def apply_ring_frame(self, epoch=None, retrograde=False):
        """Adds the and ring_frame and ring_frame_id attributes to a Body."""

        # On a repeat call, make sure they match
        if self.ring_frame_id is not None:
            ringframe = registry.as_frame(self.ring_frame_id)
            assert ringframe.epoch == epoch
            assert ringframe.retrograde == retrograde
            return

        self.ring_frame = frame_.RingFrame(self.frame_id, epoch=epoch,
                                           retrograde=retrograde)
        self.ring_frame_id = self.ring_frame.frame_id
        self.ring_is_retrograde = retrograde

    def apply_gravity(self, gravity):
        """Adds the gravity attribute to a Body."""

        self.gravity = gravity

    def add_keywords(self, keywords):
        """Adds one or more keywords to the list associated with this Body."""

        if type(keywords) == type(""): keywords = [keywords]

        for keyword in keywords:
            # Avoid duplicates...
            if keyword not in self.keywords:
                self.keywords.append(keyword)

################################################################################
# Tools for selecting the children of an body
################################################################################

    def select_children(self, include_all=None, include_any=None,
                              exclude=None, radius=None, recursive=False):
        """Returns a list of body object satisfying the given constraints on
        keywords and size."""

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
        """Retains bodies on this list only if their names ARE found in the list
        provided."""

        if type(names) == type(""): names = [names]

        list = []
        for body in bodies:
            if body.name in names and body not in list:
                list.append(body)
        return list

    @staticmethod
    def name_not_in(bodies, names):
        """Retains bodies on this list only if their names are NOT found in the
        list provided."""

        if type(names) == type(""): names = [names]

        list = []
        for body in bodies:
            if body.name not in names and body not in list:
                list.append(body)
        return list

    @staticmethod
    def radius_in_range(bodies, min, max=np.inf):
        """Retains bodies on this list only if their radii fall INSIDE the
        specified range (min,max)."""

        list = []
        for body in bodies:
            if body.radius >= min and body.radius <= max and body not in list:
                list.append(body)
        return list

    @staticmethod
    def radius_not_in_range(bodies, min, max=np.inf):
        """Retains bodies on this list only if their radii fall OUTSIDE the
        specified range (min,max)."""

        list = []
        for body in bodies:
            if body.radius < min or body.radius > max and body not in list:
                list.append(body)
        return list

    @staticmethod
    def surface_class_in(bodies, class_names):
        """Retains bodies on this list only if the name of their surface class
        IS found in the specified list. Note that the name of the surface class
        is "NoneType" for cases where the surface has not been specified."""

        if type(class_names) == type(""): class_names = [class_names]

        list = []
        for body in bodies:
            name = type(body.surface).__name__
            if name in class_names and body not in list:
                list.append(body)
        return list

    @staticmethod
    def surface_class_not_in(bodies, class_names):
        """Retains bodies on this list only if the name of their surface class
        is NOT found in the specified list. Note that the name of the surface
        class is "NoneType" for cases where the surface has not been specified."""

        if type(class_names) == type(""): class_names = [class_names]

        list = []
        for body in bodies:
            name = type(body.surface).__name__
            if name not in class_names and body not in list:
                list.append(body)
        return list

    @staticmethod
    def has_gravity(bodies):
        """Retains bodies on the list only if they HAVE a defined gravity."""

        list = []
        for body in bodies:
            if body.gm is not None and body not in list:
                list.append(body)
        return list

    @staticmethod
    def has_no_gravity(bodies):
        """Retains bodies on the list only if they have NO gravity."""

        list = []
        for body in bodies:
            if body.gm is None and body not in list:
                list.append(body)
        return list

    @staticmethod
    def has_children(bodies):
        """Retains bodies on the list only if they HAVE children."""

        list = []
        for body in bodies:
            if body.children != [] and body not in list:
                list.append(body)
        return list

    @staticmethod
    def has_no_children(bodies):
        """Retains bodies on the list only if they have NO children."""

        list = []
        for body in bodies:
            if body.children == [] and body not in list:
                list.append(body)
        return list

    @staticmethod
    def keywords_include_any(bodies, keywords):
        """Retains bodies on this list only if they have at least one of the
        specified keywords."""

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
        """Retains bodies on this list only if they have all of the specified
        keywords."""

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
        """Retains bodies on this list only if they DO NOT have any of the
        specified keywords."""

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
        """Constructs a multipath defining the centers of the given list of
        bodies. The default ID is the name of the first body with a "+"
        appended."""

        paths = []
        for body in bodies:
            paths.append(body.path_id)

        return path_.MultiPath(paths, origin, frame, id)

    @staticmethod
    def lookup(name):
        return registry.body_lookup(name)

################################################################################
# Definitions of satellite systems using SPICE body IDs
# These are disjoint sets, organized based on the keywords to be applied.
# They are also separated into groups based on whether they orbit the planet
# or the barycenter.
################################################################################

MARS_ALL_MOONS = range(401,403)

JUPITER_CLASSICAL = range(501,505)
JUPITER_REGULAR   = [505] + range(514,517)
JUPITER_IRREGULAR = range(506,514) + range(517,550) + [55062, 55063]

SATURN_CLASSICAL_INNER = range(601,607)     # Mimas through Titan orbit Saturn
SATURN_CLASSICAL_OUTER = range(607,609)     # Hyperion, Iapetus orbit barycenter
SATURN_CLASSICAL_IRREG = [609]              # Phoebe
SATURN_REGULAR   = range(610,619) + range(632,636) + [649,653]
SATURN_IRREGULAR = (range(619,632) + range(636,649) + range(650,653) +
                    [65035, 65040, 65041, 65045, 65048, 65050, 65055, 65056])

URANUS_CLASSICAL  = range(701,706)
URANUS_INNER      = range(706,716) + [725,726,727]
URANUS_IRREGULAR  = range(716,726)

NEPTUNE_CLASSICAL_INNER = [801]             # Triton
NEPTUNE_CLASSICAL_OUTER = [802]             # Nereid orbits barycenter
NEPTUNE_REGULAR   = range(803,809)
NEPTUNE_IRREGULAR = range(809,814)

CHARON        = [901]
PLUTO_REGULAR = range(902,906)

################################################################################
# Definitions of ring systems
################################################################################

JUPITER_MAIN_RING_LIMIT = 128940.

SATURN_MAIN_RINGS = (74658., 136780.)
SATURN_C_RING = ( 74658.,  91975.)
SATURN_B_RING = ( 91975., 117507.)
SATURN_A_RING = (122340., 136780.)
SATURN_F_RING_LIMIT = 140612.
SATURN_RINGS  = (SATURN_MAIN_RINGS[0], SATURN_F_RING_LIMIT)

URANUS_EPSILON_LIMIT = 51604.
URANUS_MU_LIMIT = [97700. - 17000./2, 97700. + 17700./2]
URANUS_NU_LIMIT = [67300. -  3800./2, 67300. +  3800./2]

NEPTUNE_ADAMS_LIMIT = 62940.

# Special definitions of Uranian eccentric/inclined rings
URANUS_OLD_GRAVITY = gravity.Gravity(5793939., [3.34343e-3, -2.885e-5], 26200.)

# Local function used to adapt the tabulated elements from French et al. 1991.
def uranus_elements(a, e, peri, i, node, da):
    n = URANUS_OLD_GRAVITY.n(a)
    prec = URANUS_OLD_GRAVITY.combo(a, (1,-1, 0))
    regr = URANUS_OLD_GRAVITY.combo(a, (1, 0,-1))
    peri *= constants.RPD
    node *= constants.RPD
    i *= constants.RPD
    return (a, 0., n, e, peri, prec, i, node, regr, (a+da/2))
    # The extra item returned is the outer radius in the orbit's frame

URANUS_SIX_ELEMENTS  = uranus_elements(
                        41837.15, 1.013e-3, 242.80, 0.0616,  12.12, 2.8)
URANUS_FIVE_ELEMENTS = uranus_elements(
                        42234.82, 1.899e-3, 170.31, 0.0536, 286.57, 2.8)
URANUS_FOUR_ELEMENTS = uranus_elements(
                        42570.91, 1.059e-3, 127.28, 0.0323,  89.26, 2.7)
URANUS_ALPHA_ELEMENTS = uranus_elements(
                        44718.45, 0.761e-3, 333.24, 0.0152,  63.08, 7.15+3.52)
URANUS_BETA_ELEMENTS = uranus_elements(
                        45661.03, 0.442e-3, 224.88, 0.0051, 310.05, 8.15+3.07)
URANUS_ETA_ELEMENTS = uranus_elements(
                        47175.91, 0.004e-3, 228.10, 0.0000,   0.00, 1.7)
URANUS_GAMMA_ELEMENTS = uranus_elements(
                        47626.87, 0.109e-3, 132.10, 0.0000,   0.00, 3.8)
URANUS_DELTA_ELEMENTS = uranus_elements(
                        48300.12, 0.004e-3, 216.70, 0.0011, 260.70, 7.0)
URANUS_LAMBDA_ELEMENTS = uranus_elements(
                        50023.94, 0.000e-3,   0.00, 0.0000,   0.00, 2.5)
URANUS_EPSILON_ELEMENTS = uranus_elements(
                        51149.32, 7.936e-3, 214.97, 0.0000,   0.00, 58.1+37.6)

################################################################################
# Convenient procedure to load the entire Solar System
################################################################################

def define_solar_system(start_time, stop_time, asof=None):
    """Constructs bodies, paths and frames for all the planets and moons in the
    solar system (including Pluto). Each planet is defined relative to the SSB.
    Each moon is defined relative to its planet. Names are as defined within the
    SPICE toolkit. Body associations are defined within the spicedb library.

    Input:
        start_time      start_time of the period to be convered, in ISO date
                        or date-time format.
        stop_time       stop_time of the period to be covered, in ISO date
                        or date-time format.
        asof            a UTC date such that only kernels released earlier
                        than that date will be included, in ISO format.
    """

    # If the solar system was already loaded, just return
    if "SUN" in registry.BODY_REGISTRY.keys(): return

    # Always load the most recent Leap Seconds kernel, but only once
    spice.load_leap_seconds()

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

    # We might need B1950 in addition to J2000
    ignore = frame_.SpiceFrame("B1950", "J2000")

    # SSB and Sun and SSB
    define_bodies(["SSB"], None, None, ["SUN", "BARYCENTER"])
    define_bodies(["SUN"], None, None, ["SUN"])

    # Mercury-Jupiter and Jupiter barycenter orbit the Sun
    define_bodies([199, 299, 399, 499, 599], "SUN", "SUN", ["PLANET"])
    define_bodies([5], "SUN", "SUN", ["BARYCENTER"])

    # Saturn-Pluto planets and barycenters orbit the SSB
    define_bodies([699, 799, 899, 999], "SUN", "SSB", ["PLANET"])
    define_bodies([6, 7, 8, 9], "SUN", "SSB", ["BARYCENTER"])

    # Earth's Moon
    define_bodies([301], "EARTH", "EARTH",
                  ["SATELLITE", "CLASSICAL", "REGULAR"])

    # Moons and rings of Mars
    define_bodies(MARS_ALL_MOONS, "MARS", "MARS",
                  ["SATELLITE", "CLASSICAL", "REGULAR"])
    define_ring("MARS", "MARS_RING_PLANE", None, [])

    # Moons and rings of Jupiter
    define_bodies(JUPITER_CLASSICAL, "JUPITER", "JUPITER",
                  ["SATELLITE", "CLASSICAL", "REGULAR"])
    define_bodies(JUPITER_REGULAR, "JUPITER", "JUPITER",
                  ["SATELLITE", "REGULAR"])
    define_bodies(JUPITER_IRREGULAR, "JUPITER", "JUPITER BARYCENTER",
                  ["SATELLITE", "IRREGULAR"])

    define_ring("JUPITER", "JUPITER_RING_PLANE", None, [])
    define_ring("JUPITER", "JUPITER_RING_SYSTEM", JUPITER_MAIN_RING_LIMIT, [])

    # Moons and rings of Saturn
    define_bodies(SATURN_CLASSICAL_INNER, "SATURN", "SATURN",
                  ["SATELLITE", "CLASSICAL", "REGULAR"])
    define_bodies(SATURN_CLASSICAL_OUTER, "SATURN", "SATURN BARYCENTER",
                  ["SATELLITE", "CLASSICAL", "REGULAR"])
    define_bodies(SATURN_CLASSICAL_IRREG, "SATURN", "SATURN BARYCENTER",
                  ["SATELLITE", "CLASSICAL", "IRREGULAR"])
    define_bodies(SATURN_REGULAR, "SATURN", "SATURN",
                  ["SATELLITE", "REGULAR"])
    define_bodies(SATURN_IRREGULAR, "SATURN", "SATURN BARYCENTER",
                  ["SATELLITE", "IRREGULAR"])

    define_ring("SATURN", "SATURN_RING_PLANE", None, [])
    define_ring("SATURN", "SATURN_RING_SYSTEM", SATURN_F_RING_LIMIT, [])
    define_ring("SATURN", "SATURN_RINGS", SATURN_RINGS, [])
    define_ring("SATURN", "SATURN_MAIN_RINGS", SATURN_MAIN_RINGS, [])
    define_ring("SATURN", "SATURN_A_RING", SATURN_A_RING, [])
    define_ring("SATURN", "SATURN_B_RING", SATURN_B_RING, [])
    define_ring("SATURN", "SATURN_C_RING", SATURN_C_RING, [])

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
    uranus_wrt_b1950 = registry.connect_frames("IAU_URANUS", "B1950")
    ignore = frame_.RingFrame(uranus_wrt_b1950, URANUS_EPOCH,
                                         "URANUS_RINGS_B1950", retrograde=True)

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

    # Moons and rings of Neptune
    define_bodies(NEPTUNE_CLASSICAL_INNER, "NEPTUNE", "NEPTUNE",
                  ["SATELLITE", "CLASSICAL", "REGULAR"])
    define_bodies(NEPTUNE_CLASSICAL_OUTER, "NEPTUNE", "NEPTUNE BARYCENTER",
                  ["SATELLITE", "CLASSICAL", "IRREGULAR"])
    define_bodies(NEPTUNE_REGULAR, "NEPTUNE", "NEPTUNE",
                  ["SATELLITE", "REGULAR"])
    define_bodies(NEPTUNE_IRREGULAR, "NEPTUNE", "NEPTUNE BARYCENTER",
                  ["SATELLITE", "IRREGULAR"])

    define_ring("NEPTUNE", "NEPTUNE_RING_PLANE",  None, [])
    define_ring("NEPTUNE", "NEPTUNE_RING_SYSTEM", NEPTUNE_ADAMS_LIMIT, [])

    # Moons and rings of Pluto
    define_bodies(CHARON, "PLUTO", "PLUTO",
                  ["SATELLITE", "CLASSICAL", "REGULAR"])
    define_bodies(PLUTO_REGULAR, "PLUTO", "PLUTO BARYCENTER",
                  ["SATELLITE", "REGULAR"])

    define_ring("PLUTO", "PLUTO_RING_PLANE", None, [],
                barycenter_name="PLUTO BARYCENTER")
    define_ring("PLUTO", "PLUTO_INNER_RING_PLANE", None, [],
                barycenter_name="PLUTO")

    barycenter = registry.BODY_REGISTRY["PLUTO BARYCENTER"]
    barycenter.ring_frame_id = registry.BODY_REGISTRY["PLUTO"].ring_frame_id

def define_bodies(spice_ids, parent, barycenter, keywords):
    """Defines the path, frame, surface and body for a given list of bodies
    identified by name or SPICE ID. All must share a common parent and
    barycenter."""

    for spice_id in spice_ids:

        # Define the body's path and frame
        path = path_.SpicePath(spice_id, "SSB")

        # Sometimes a frame is undefined for a new body; in this case any frame
        # will do.
        try:
            frame = frame_.SpiceFrame(spice_id)
        except LookupError:
            frame = frame_.Wayframe("J2000", path.path_id)

        # The name of the path is the name of the body
        name = path.path_id

        # Define the planet's body, assuming an orbit around the Sun
        body = Body(name, name, frame.frame_id, parent, barycenter)
        body.add_keywords(keywords)

        # Add the gravity object if it exists
        try:
            body.apply_gravity(gravity.LOOKUP[name])
        except KeyError: pass

        # Add the surface object if shape information is available
        try:
            shape = surface.spice_body(spice_id)
            body.apply_surface(shape, shape.req, shape.rpol)
        except RuntimeError:
            shape = surface.NullSurface(path, frame)
            body.apply_surface(shape, 0., 0.)
        except LookupError:
            shape = surface.NullSurface(path, frame)
            body.apply_surface(shape, 0., 0.)

        # Add a planet name to any satellite or barycenter
        if "SATELLITE" in body.keywords and parent is not None:
            body.add_keywords(parent)

        if "BARYCENTER" in body.keywords and parent is not None:
            body.add_keywords(parent)

def define_ring(parent_name, ring_name, radii, keywords, retrograde=False,
                barycenter_name=None):
    """Defines the path, frame, surface and body for a given ring, given its
    inner and outer radii. A single radius value is used to define the outer
    limit of rings, but the ring plane itself has no boundaries.

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
    """

    parent = registry.body_lookup(parent_name)
    parent.apply_ring_frame(retrograde=retrograde)

    if barycenter_name is None:
        barycenter = parent
    else:
        barycenter = registry.body_lookup(barycenter_name)

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

    body = Body(ring_name, barycenter.path_id, parent.ring_frame_id,
                parent, parent)
    shape = surface.RingPlane(barycenter.path_id, parent.ring_frame_id,
                              radii, gravity=parent.gravity)

    body.apply_surface(shape, rmax, 0.)

    body.add_keywords([parent, "RING", ring_name])
    body.add_keywords(keywords)

def define_orbit(parent_name, ring_name, elements, epoch, reference, keywords):
    """Defines the path, frame, surface and body for a given eccentric and/or
    inclined ring as defined by a set of orbital elements.
    """

    parent = registry.body_lookup(parent_name)

    orbit = surface.OrbitPlane(elements, epoch, parent.path_id, reference,
                               id=ring_name)

    body = Body(ring_name, orbit.internal_origin_id, orbit.internal_frame_id,
                parent, parent)
    body.apply_surface(orbit, elements[9], 0.)

    body.add_keywords([parent, "RING", "ORBIT", ring_name])
    body.add_keywords(keywords)

################################################################################
# Initialize the registry
################################################################################

registry.BODY_CLASS = Body

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Body(unittest.TestCase):

    def runTest(self):

        # Imports are here to avoid conflicts
        import oops_.registry as registry
        registry.initialize_frame_registry()
        registry.initialize_path_registry()
        registry.initialize_body_registry()

        define_solar_system("2000-01-01", "2010-01-01")

        self.assertEqual(registry.body_lookup("DAPHNIS").barycenter.name,
                         "SATURN")
        self.assertEqual(registry.body_lookup("PHOEBE").barycenter.name,
                         "SATURN BARYCENTER")

        mars = registry.body_lookup("MARS")
        moons = mars.select_children(include_all=["SATELLITE"])
        self.assertEqual(len(moons), 2)     # Phobos, Deimos

        saturn = registry.body_lookup("SATURN")
        moons = saturn.select_children(include_all=["CLASSICAL", "IRREGULAR"])
        self.assertEqual(len(moons), 1)     # Phoebe

        moons = saturn.select_children(exclude=["IRREGULAR","RING"], radius=170)
        self.assertEqual(len(moons), 8)     # Mimas-Iapetus

        rings = saturn.select_children(include_any=("RING"))
        self.assertEqual(len(rings), 6)     # A, B, C, Main, Saturn all, plane

        moons = saturn.select_children(include_all="SATELLITE",
                                       exclude=("IRREGULAR"), radius=1000)
        self.assertEqual(len(moons), 1)     # Titan only

        sun = registry.body_lookup("SUN")
        planets = sun.select_children(include_any=["PLANET"])
        self.assertEqual(len(planets), 9)

        sun = registry.body_lookup("SUN")
        planets = sun.select_children(include_any=["PLANET", "EARTH"])
        self.assertEqual(len(planets), 9)

        sun = registry.body_lookup("SUN")
        planets = sun.select_children(include_any=["PLANET", "EARTH"],
                                      recursive=True)
        self.assertEqual(len(planets), 10)  # 9 planets plus Earth's moon

        sun = registry.body_lookup("SUN")
        planets = sun.select_children(include_any=["PLANET", "JUPITER"],
                                      exclude=["IRREGULAR", "BARYCENTER", "IO"],
                                      recursive=True)
        self.assertEqual(len(planets), 16)  # 9 planets + 7 Jovian moons

        registry.initialize_frame_registry()
        registry.initialize_path_registry()
        registry.initialize_body_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
