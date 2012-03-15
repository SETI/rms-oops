################################################################################
# oops_/body.py: Body class
#
# 2/18/12 Created (MRS)
################################################################################

import numpy as np
import os
import spicedb
import julian
import gravity

import oops_.path.all as path_
import oops_.frame.all as frame_
import oops_.surface.all as surface
import oops_.spice_support as spice
import oops_.registry as registry

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
        self.radius  = 0.
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

    def apply_surface(self, surface, radius):
        """Adds the and surface attribute to a Body."""

        self.surface = surface
        self.radius  = radius
        assert self.surface.origin_id == self.path_id

    def apply_ring_frame(self, epoch=None):
        """Adds the and ring_frame and ring_frame_id attributes to a Body."""

        # Make sure the epochs match
        if self.ring_frame_id is not None:
            ringframe = registry.as_frame(self.ring_frame_id)
            assert ringframe.epoch == epoch
            return

        self.ring_frame = frame_.RingFrame(self.frame_id, epoch)
        self.ring_frame_id = self.ring_frame.frame_id

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
            if body.name in names:
                list.append(body)
        return list

    @staticmethod
    def name_not_in(bodies, names):
        """Retains bodies on this list only if their names are NOT found in the
        list provided."""

        if type(names) == type(""): names = [names]

        list = []
        for body in bodies:
            if body.name not in names:
                list.append(body)
        return list

    @staticmethod
    def radius_in_range(bodies, min, max=np.inf):
        """Retains bodies on this list only if their radii fall INSIDE the
        specified range (min,max)."""

        list = []
        for body in bodies:
            if body.radius >= min and body.radius <= max:
                list.append(body)
        return list

    @staticmethod
    def radius_not_in_range(bodies, min, max=np.inf):
        """Retains bodies on this list only if their radii fall OUTSIDE the
        specified range (min,max)."""

        list = []
        for body in bodies:
            if body.radius < min or body.radius > max:
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
            if name in class_names:
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
            if name not in class_names:
                list.append(body)
        return list

    @staticmethod
    def has_gravity(bodies):
        """Retains bodies on the list only if they HAVE a defined gravity."""

        list = []
        for body in bodies:
            if body.gm is not None:
                list.append(body)
        return list

    @staticmethod
    def has_no_gravity(bodies):
        """Retains bodies on the list only if they have NO gravity."""

        list = []
        for body in bodies:
            if body.gm is None:
                list.append(body)
        return list

    @staticmethod
    def has_children(bodies):
        """Retains bodies on the list only if they HAVE children."""

        list = []
        for body in bodies:
            if body.children != []:
                list.append(body)
        return list

    @staticmethod
    def has_no_children(bodies):
        """Retains bodies on the list only if they have NO children."""

        list = []
        for body in bodies:
            if body.children == []:
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
                if keyword in body.keywords:
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

            if is_match:
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

            if not is_found:
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
################################################################################

MARS_ALL_MOONS = range(401,403)

JUPITER_CLASSICAL = range(501,505)
JUPITER_REGULAR   = [505] + range(514,517)
JUPITER_IRREGULAR = range(506,514) + range(517,550) + [55062, 55063]

SATURN_CLASSICAL_INNER = range(601,607)     # Mimas through Titan
SATURN_CLASSICAL_OUTER = range(607,609)     # Hyperion, Iapetus
SATURN_CLASSICAL_IRREG = [609]              # Phoebe
SATURN_REGULAR   = range(610,619) + range(632,636) + [649,653]
SATURN_IRREGULAR = (range(636,649) + range(650,653) +
                    [65035, 65040, 65041, 65045, 65048, 65050, 65055, 65056])

URANUS_CLASSICAL  = range(701,706)
URANUS_REGULAR    = range(701,716) + [726,727]
URANUS_IRREGULAR  = range(716,726)

NEPTUNE_CLASSICAL = range(801,803)
NEPTUNE_REGULAR   = range(801,809)
NEPTUNE_IRREGULAR = range(809,814)

PLUTO_CLASSICAL   = [901]
PLUTO_REGULAR     = range(901,905)

################################################################################
# Definitions of ring systems
################################################################################

JUPITER_MAIN_RING_LIMIT = 128940.

SATURN_MAIN_RINGS = (74658., 136780.)
SATURN_A_RING = ( 74658.,  91975.)
SATURN_B_RING = ( 91975., 117507.)
SATURN_C_RING = (122340., 136780.)
SATURN_F_RING_LIMIT = 140612.

URANUS_EPSILON_LIMIT = 51604.

NEPTUNE_ADAMS_LIMIT = 62940.

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
    define_ring("JUPITER", "JUPITER_RING_PLANE", JUPITER_MAIN_RING_LIMIT, [])

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
    define_ring("SATURN", "SATURN_RING_PLANE", SATURN_F_RING_LIMIT, [])
    define_ring("SATURN", "SATURN_MAIN_RINGS", SATURN_MAIN_RINGS, [])
    define_ring("SATURN", "SATURN_A_RING", SATURN_A_RING, [])
    define_ring("SATURN", "SATURN_B_RING", SATURN_B_RING, [])
    define_ring("SATURN", "SATURN_C_RING", SATURN_C_RING, [])

    # Moons and rings of Uranus
    define_bodies(URANUS_CLASSICAL, "URANUS", "URANUS",
                  ["SATELLITE", "CLASSICAL", "REGULAR"])
    define_bodies(URANUS_REGULAR,   "URANUS", "URANUS",
                  ["SATELLITE", "REGULAR"])
    define_bodies(URANUS_IRREGULAR, "URANUS", "URANUS",
                  ["SATELLITE", "IRREGULAR"])
    define_ring("URANUS", "URANUS_RING_PLANE", URANUS_EPSILON_LIMIT, [])

    # Moons and rings of Neptune
    define_bodies(NEPTUNE_CLASSICAL, "NEPTUNE", "NEPTUNE",
                  ["SATELLITE", "CLASSICAL", "REGULAR"])
    define_bodies(NEPTUNE_REGULAR,   "NEPTUNE", "NEPTUNE",
                  ["SATELLITE", "REGULAR"])
    define_bodies(NEPTUNE_IRREGULAR, "NEPTUNE", "NEPTUNE",
                  ["SATELLITE", "IRREGULAR"])
    define_ring("NEPTUNE", "NEPTUNE_RING_PLANE", NEPTUNE_ADAMS_LIMIT, [])

    # Moons and rings of Pluto
    define_bodies(PLUTO_CLASSICAL, "PLUTO", "PLUTO",
                  ["SATELLITE", "CLASSICAL", "REGULAR"])
    define_bodies(PLUTO_REGULAR,   "PLUTO", "PLUTO",
                  ["SATELLITE", "REGULAR"])
    define_ring("PLUTO", "PLUTO_RING_PLANE", None, [])

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
            body.apply_surface(shape, shape.req)
        except RuntimeError: pass
        except LookupError: pass

        # Add a planet name to any satellite or barycenter
        if "SATELLITE" in body.keywords and parent is not None:
            body.add_keywords(parent)

        if "BARYCENTER" in body.keywords and parent is not None:
            body.add_keywords(parent)

def define_ring(parent_name, ring_name, radii, keywords):
    """Defines the path, frame, surface and body for a given ring, given its
    inner and outer radii. A single radius value is used to define the outer
    limit of rings, but the ring plane itself has no boundaries.
    """

    parent = registry.body_lookup(parent_name)
    parent.apply_ring_frame()

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

    body = Body(ring_name, parent.path_id, parent.ring_frame_id, parent, parent)
    shape = surface.RingPlane(parent.path_id, parent.ring_frame_id,
                              radii, gravity=parent.gravity)

    body.apply_surface(shape, rmax)

    body.add_keywords([parent, "RING", ring_name])

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
        self.assertEqual(len(rings), 5)     # A, B, C, Main rings, plane

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

        registry.BODY_REGISTRY = {}

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
