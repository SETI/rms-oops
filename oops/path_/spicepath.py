################################################################################
# oops/path_/spicepath.py: Subclass SpicePath of class Path
################################################################################

import numpy as np
import cspice

from oops.path_.path import Path, Waypoint, RotatedPath
from oops.array_     import *
from oops.config     import QUICK
from oops.event      import Event
import oops.registry as registry
import oops.spice_support as spice

class SpicePath(Path):
    """Subclass SpicePath of class Path returns a path based on an SP kernel in
    the SPICE toolkit. It represents the geometric position of a single target
    body with respect to a single origin."""

    # Set False to confirm that SpicePaths return the same results without
    # shortcuts and with shortcuts
    USE_SPICEPATH_SHORTCUTS = True

    # Set False to call cspice library sequentially rather than with array
    # inputs
    VECTORIZE_CSPICE = True

    def __init__(self, spice_id, spice_origin="SSB", spice_frame="J2000",
                       id=None, shortcut=None):
        """Constructor for a SpicePath object.

        Input:
            spice_id        the name or integer ID of the target body as used
                            in the SPICE toolkit.
            spice_origin    the name or integer ID of the origin body as
                            used in the SPICE toolkit; "SSB" for the Solar
                            System Barycenter by default. It may also be the
                            registered name of another SpicePath.
            spice_frame     the name or integer ID of the reference frame or of
                            the a body with which the frame is primarily
                            associated, as used in the SPICE toolkit.
            id              the name or ID under which the path will be
                            registered. By default, this will be the value of
                            spice_id if that is given as a string; otherwise
                            it will be the name as used by the SPICE toolkit.
            shortcut        If a shortcut is specified, then this is registered
                            as a shortcut definition; the other registered path
                            definitions are unchanged.
        """

        (self.spice_target_id,
         self.spice_target_name) = spice.body_id_and_name(spice_id)

        (self.spice_origin_id,
         self.spice_origin_name) = spice.body_id_and_name(spice_origin)

        self.spice_frame_name = spice.frame_id_and_name(spice_frame)[1]

        # Fill in the path_id
        if id is None:
            if type(spice_id) == type(""):
                self.path_id = spice_id
            else:
                self.path_id = self.spice_target_name
        else:
            self.path_id = id

        # Fill in the origin_id, which should already be in the dictionary
        self.origin_id = spice.PATH_TRANSLATION[spice_origin]

        # Fill in the frame_id, which should already be in the SpicePath
        # dictionary
        self.frame_id = spice.FRAME_TRANSLATION[spice_frame]

        self.shape = []
        self.shortcut = shortcut

        if shortcut is None:
            # Save it in the global dictionary of Spice translations under
            # alternative names
            spice.PATH_TRANSLATION[self.spice_target_name] = self.path_id
            spice.PATH_TRANSLATION[self.spice_target_id]   = self.path_id
            spice.PATH_TRANSLATION[spice_id]               = self.path_id
            spice.PATH_TRANSLATION[id]                     = self.path_id

        # Register the SpicePath
        self.register(shortcut)

########################################

    def event_at_time(self, time, quick=None):
        """Returns an Event object corresponding to a specified Scalar time on
        this path.

        Input:
            time        a time Scalar at which to evaluate the path.

        Return:         an Event object containing (at least) the time, position
                        and velocity of the path.
        """

        # A single input time can be handled quickly
        time = Scalar.as_scalar(time)
        if time.shape == []:
            (state,
             lighttime) = cspice.spkez(self.spice_target_id,
                                       float(time.vals),
                                       self.spice_frame_name,
                                       "NONE", # no aberration or light time fix
                                       self.spice_origin_id)

            return Event(time, state[0:3], state[3:6], self.origin_id,
                                                       self.frame_id)

        # Use a QuickPath if warranted, possibly making a recursive call
        if quick is None: quick = QUICK.flag
        if quick:
            return self.quick_path(time, quick).event_at_time(time, False)

        # Fill in the states and light travel times using CSPICE
        if SpicePath.VECTORIZE_CSPICE:
            (state,
             lighttime) = cspice.spkez_vector(self.spice_target_id,
                                              time.vals.ravel(),
                                              self.spice_frame_name,
                                              "NONE", # no aberration/light time
                                              self.spice_origin_id)
            pos = state[:,0:3].reshape(time.shape + [3])
            vel = state[:,3:6].reshape(time.shape + [3])

            # Convert to an Event and return
            return Event(time, pos, vel, self.origin_id, self.frame_id)

        else:
            # Create the buffers
            state     = np.empty(time.shape + [6])
            lighttime = np.empty(time.shape)

            # Iterate through the array
            for i,t in np.ndenumerate(time.vals):
                (state[i],
                 lighttime[i]) = cspice.spkez(self.spice_target_id,
                                              t,
                                              self.spice_frame_name,
                                              "NONE",
                                              self.spice_origin_id)

            # Convert to an Event and return
            return Event(time, state[...,0:3], state[...,3:6],
                               self.origin_id, self.frame_id)

########################################

    def connect_to(self, origin, frame=None):
        """Returns a Path object in which event point from an arbitrary origin
        path to this path. SpicePath overrides the default method to create
        quicker "shortcuts" between SpicePaths.

        Input:
            origin          an origin Path object or its registered name.
            frame           a frame object or its registered ID. Default is
                            to use the frame of the origin's path.
        """

        # Use the slow method if necessary, for debugging
        if not SpicePath.USE_SPICEPATH_SHORTCUTS:
            return Path.connect_to(self, origin, frame)

        # Derive the origin info and make sure it is SPICE-related
        origin = registry.as_path(origin)
        origin_id = origin.path_id
        if type(origin) is SpicePath:
            spice_origin_id = origin.spice_target_id
        elif origin.path_id == "SSB":
            spice_origin_id = "SSB"
        else:
            # If the origin frame is not a SpiceFrame, use the default procedure
            return Path.connect_to(self, origin, frame)

        # Derive the frame info and see if it is SPICE-related
        frame = registry.as_frame(frame)
        if frame is None: frame = origin.frame

        frame_id = frame.frame_id
        if type(frame).__name__ == "SpiceFrame":
            spice_frame_name = frame.spice_frame_name
        else:
            spice_frame_name = "J2000"

        # Construct the shortcut frame and register it, using J2000 if necessary
        spiceframe_as_registered = (spice.FRAME_TRANSLATION[spice_frame_name])
        shortcut = ("SHORTCUT_" + str(self.path_id) + "_" +
                                  str(origin_id)    + "_" +
                                  str(spiceframe_as_registered))
        result = SpicePath(self.spice_target_id, spice_origin_id,
                           spice_frame_name, self.path_id, shortcut)

        # If the path uses a non-spice frame, add a rotated version
        if spiceframe_as_registered != frame_id:
            shortcut = ("SHORTCUT_" + str(self.path_id) + "_" +
                                      str(origin_id)    + "_" +
                                      str(frame_id))
            result = RotatedPath(result, frame_id)
            result.register(shortcut)

        return result

################################################################################
# UNIT TESTS
################################################################################

# This is the opportunity to show that all Path and Frame operations produce
# results that are consistent with the well-tested SPICE toolkit.

import unittest

class Test_SpicePath(unittest.TestCase):

    def runTest(self):

      # Imports are here to avoid conflicts
      from oops.edelta import Edelta
      from oops.frame_.frame import Frame
      from oops.frame_.spiceframe import SpiceFrame
      import oops.constants as constants

      Path.USE_QUICKPATHS = False
      Frame.USE_QUICKFRAMES = False

      # Repeat the tests without and then with shortcuts
      for SpicePath.USE_SPICEPATH_SHORTCUTS in (False, True):

        registry.initialize_path_registry()
        registry.initialize_frame_registry()

        sun   = SpicePath("SUN", "SSB")
        earth = SpicePath("EARTH", "SSB")
        moon  = SpicePath("MOON", "EARTH")

        self.assertEqual(moon, registry.path_lookup("MOON"))
        self.assertEqual(sun, registry.path_lookup("SUN"))

        # Validate state vectors using event_at_time()
        times = np.arange(-3.e8, 3.01e8, 0.5e7)
        moon_event = moon.event_at_time(times)
        for i in range(len(times)):
            (state, lt) = cspice.spkez(301,times[i],"J2000","NONE",399)
            self.assertEqual(moon_event.pos[i], state[0:3])
            self.assertEqual(moon_event.vel[i], state[3:6])

        # Check light travel time corrections to/from SSB
        saturn = SpicePath(6, "SSB", id="SATURN")
        times = np.arange(-3.e8, 3.01e8, 0.5e8)
        ssb_event = registry.as_path("SSB").event_at_time(times)

        saturn_event = saturn.photon_to_event(ssb_event, iters=99)
        for i in range(len(times)):
            (state, lt) = cspice.spkez(6,times[i],"J2000","CN",0)
            self.assertTrue(abs(lt + saturn_event.dep_lt[i]) < 1.e-11)
            self.assertTrue(abs(saturn_event.time[i] + lt - ssb_event.time[i])
                                                                       < 1.e-11)
            self.assertTrue(abs(ssb_event.arr[i] + state[0:3]) < 1.e-8)
            self.assertTrue(abs(saturn_event.dep[i] + state[0:3]) < 1.e-7)
            self.assertEqual(abs(saturn_event.pos[i]), (0.,0.,0.))
            self.assertEqual(abs(saturn_event.vel[i]), (0.,0.,0.))

        saturn_event = saturn.photon_from_event(ssb_event, iters=99)
        for i in range(len(times)):
            (state, lt) = cspice.spkez(6,times[i],"J2000","XCN",0)
            self.assertTrue(abs(lt - saturn_event.arr_lt[i]) < 1.e-11)
            self.assertTrue(abs(ssb_event.time[i] + lt - saturn_event.time[i])
                                                                       < 1.e-11)
            self.assertTrue(abs(ssb_event.dep[i] - state[0:3]) < 1.e-8)
            self.assertTrue(abs(Vector3(state[0:3]) - ssb_event.dep[i]) < 1.e-8)

        # Check instantaneous geometry using linked paths

        # Moon wrt Earth
        times = np.arange(-3.e8, 3.01e8, 0.5e8)
        moon_event = moon.event_at_time(times).wrt_path("EARTH")
        for i in range(len(times)):
            (state, lt) = cspice.spkez(301,times[i],"J2000","NONE",399)
            self.assertTrue(np.all(np.abs(state[0:3] -
                                          moon_event.pos.vals[i]) < 1.e-8))
            self.assertTrue(np.all(np.abs(state[3:6] -
                                          moon_event.vel.vals[i]) < 1.e-8))

        # Moon to SSB
        moon_event = moon.event_at_time(times).wrt_path("SSB")
        for i in range(len(times)):
            (state, lt) = cspice.spkez(301,times[i],"J2000","NONE",0)
            self.assertTrue(np.all(np.abs(state[0:3] -
                                          moon_event.pos.vals[i]) < 1.e-6))
            self.assertTrue(np.all(np.abs(state[3:6] -
                                          moon_event.vel.vals[i]) < 1.e-6))

        # Moon to Saturn
        moon_event = moon.event_at_time(times).wrt_path("SATURN")
        for i in range(len(times)):
            (state, lt) = cspice.spkez(301,times[i],"J2000","NONE",6)
            self.assertTrue(np.all(np.abs(state[0:3] -
                                          moon_event.pos.vals[i]) < 1.e-6))
            self.assertTrue(np.all(np.abs(state[3:6] -
                                          moon_event.vel.vals[i]) < 1.e-6))

        ####################################
        # Tests of combined paths but no frame rotation
        registry.initialize_path_registry()
        times = np.arange(-3.e8, 3.01e8, 0.5e7)

        sun    = SpicePath("SUN", "SSB")
        earth  = SpicePath("EARTH", "SSB")
        moon   = SpicePath("MOON", "EARTH")
        mars   = SpicePath("MARS", "MOON")
        saturn = SpicePath(6, "MOON", id="SATURN")
        self.assertEqual(registry.path_lookup("SATURN"), saturn)

        xsun   = registry.path_lookup("SUN")
        xssb   = registry.path_lookup("SSB")
        xearth = registry.path_lookup("EARTH")
        xmoon  = registry.path_lookup("MOON")
        xmars  = registry.path_lookup("MARS")
        xsat   = registry.path_lookup(saturn.path_id)

        self.assertEqual(Path.common_ancestry(xsun, xssb),
                         ([xsun, xssb], [xssb]))
        self.assertEqual(Path.common_ancestry(xssb, xsun),
                         ([xssb], [xsun, xssb]))
        self.assertEqual(Path.common_ancestry(xmars, xearth),
                         ([xmars, xmoon, xearth], [xearth]))
        self.assertEqual(Path.common_ancestry(xmars, xsun),
                         ([xmars, xmoon, xearth, xssb], [xsun, xssb]))

        path = registry.connect_paths("MARS","SUN")
        # print Path.str_ancestry(Path.common_ancestry(xmars,xsun))
        # ([MARS", "MOON", "EARTH", "SSB"], ["SUN", "SSB"])

        event = registry.as_path("MARS").event_at_time(times).wrt_path("SUN")
        self.assertEqual(event.frame_id, "J2000")
        self.assertEqual(event.origin_id, "SUN")

        for i in range(len(times)):
            (state, lt) = cspice.spkez(499, times[i], "J2000", "NONE", 10)
            dpos = event.pos[i] - state[0:3]
            dvel = event.vel[i] - state[3:6]
            self.assertTrue(np.all(np.abs(dpos.vals) < 1.e-7))
            self.assertTrue(np.all(np.abs(dvel.vals) < 1.e-14))

        # Tests using different frames
        registry.initialize_path_registry()
        registry.initialize_frame_registry()

        ignore = SpiceFrame("IAU_EARTH", "J2000")
        earth  = SpicePath("EARTH", "SSB", "IAU_EARTH")

        path = registry.connect_paths("EARTH","SSB", "J2000")
        event = path.event_at_time(times)
        for i in range(len(times)):
            (state, lt) = cspice.spkez(399, times[i], "J2000", "NONE", 0)
            dpos = event.pos[i] - state[0:3]
            dvel = event.vel[i] - state[3:6]
            self.assertTrue(np.all(np.abs(dpos.vals) < 1.e-7))
            self.assertTrue(np.all(np.abs(dvel.vals) < 1.e-9))

        ####################################
        registry.initialize_path_registry()
        registry.initialize_frame_registry()

        ignore = SpiceFrame("IAU_EARTH", "J2000")
        ignore = SpiceFrame("IAU_MARS", "IAU_EARTH")
        ignore = SpiceFrame("IAU_MOON", "IAU_EARTH")
        ignore = SpiceFrame("B1950", "J2000")

        earth  = SpicePath("EARTH", "SSB", "IAU_EARTH")
        moon   = SpicePath("MOON", "EARTH", "IAU_MOON")
        sun    = SpicePath("SUN", "SSB", "J2000")
        mars   = SpicePath("MARS", "SUN", "J2000")

        path = registry.connect_paths("EARTH", "SSB", "J2000")
        event = path.event_at_time(times)
        for i in range(len(times)):
            (state, lt) = cspice.spkez(399, times[i], "J2000", "NONE", 0)
            dpos = event.pos[i] - state[0:3]
            dvel = event.vel[i] - state[3:6]
            self.assertTrue(np.all(np.abs(dpos.vals) < 1.e-7))
            self.assertTrue(np.all(np.abs(dvel.vals) < 1.e-9))

        path = registry.connect_paths("SSB", "EARTH", "IAU_MARS")
        event = path.event_at_time(times)
        for i in range(len(times)):
            (state, lt) = cspice.spkez(0, times[i], "IAU_MARS", "NONE", 399)
            dpos = event.pos[i] - state[0:3]
            dvel = event.vel[i] - state[3:6]
            self.assertTrue(np.all(np.abs(dpos.vals) < 1.e-7))
            self.assertTrue(np.all(np.abs(dvel.vals) < 1.e-9))

        path = registry.connect_paths("EARTH", "SUN", "IAU_EARTH")
        event = path.event_at_time(times)
        for i in range(len(times)):
            (state, lt) = cspice.spkez(399, times[i], "IAU_EARTH", "NONE", 10)
            dpos = event.pos[i] - state[0:3]
            dvel = event.vel[i] - state[3:6]
            self.assertTrue(np.all(np.abs(dpos.vals) < 1.e-6))
            self.assertTrue(np.all(np.abs(dvel.vals) < 1.e-9))

        path = registry.connect_paths("MOON", "MARS", "IAU_EARTH")
        event = path.event_at_time(times)
        for i in range(len(times)):
            (state, lt) = cspice.spkez(301, times[i], "IAU_EARTH", "NONE", 499)
            dpos = event.pos[i] - state[0:3]
            dvel = event.vel[i] - state[3:6]
            self.assertTrue(np.all(np.abs(dpos.vals) < 1.e-6))
            self.assertTrue(np.all(np.abs(dvel.vals) < 1.e-9))

        ####################################
        registry.initialize_path_registry()
        registry.initialize_frame_registry()

        ignore = SpiceFrame("IAU_MARS", "J2000")
        ignore = SpiceFrame("IAU_EARTH", "J2000")
        ignore = SpiceFrame("B1950", "IAU_EARTH")
        ignore = SpiceFrame("IAU_MOON", "B1950")

        sun    = SpicePath("SUN", "SSB", "J2000")
        earth  = SpicePath("EARTH", "SSB", "IAU_EARTH")
        moon   = SpicePath("MOON", "EARTH", "IAU_MOON")
        mars   = SpicePath("MARS", "MOON", "B1950")

        times = np.arange(-3.e8, 3.01e8, 0.5e7)
        path = registry.connect_paths("MARS","MOON", "IAU_MOON")
        event = path.event_at_time(times)
        for i in range(len(times)):
            (state, lt) = cspice.spkez(499, times[i], "IAU_MOON", "NONE", 301)
            dpos = event.pos[i] - state[0:3]
            dvel = event.vel[i] - state[3:6]
            self.assertTrue(np.all(np.abs(dpos.vals) < 1.e-6))
            self.assertTrue(np.all(np.abs(dvel.vals) < 1.e-9))

        ####################################
        registry.initialize_path_registry()
        registry.initialize_frame_registry()

        sun    = SpicePath("SUN", "SSB")
        earth  = SpicePath("EARTH", "SSB")
        moon   = SpicePath("MOON", "EARTH")
        saturn = SpicePath(6, "SSB", id="SATURN")
        ssb    = registry.as_path("SSB")

        times = np.arange(-3.e8, 3.01e8, 0.5e8)

        # Check light travel time corrections, Saturn to Earth wrt SSB
        earth_event = earth.event_at_time(times)
        saturn_event = saturn.photon_to_event(earth_event)

        saturn_rel = Edelta.sub_events(saturn_event, earth_event)
        self.assertTrue(abs(saturn_rel.dep - earth_event.arr) < 1.e-6)

        saturn_rel_ssb = saturn_rel.wrt_ssb()
        saturn_abs_ssb = saturn_event.wrt_ssb()
        self.assertTrue(abs(saturn_rel_ssb.time - saturn_abs_ssb.time) < 1.e-6)
        self.assertTrue(abs(saturn_rel_ssb.pos  - saturn_abs_ssb.pos)  < 1.e-6)
        self.assertTrue(abs(saturn_rel_ssb.vel  - saturn_abs_ssb.vel)  < 1.e-6)
        self.assertTrue(abs(saturn_rel_ssb.dep  - saturn_abs_ssb.dep)  < 1.e-6)

        for i in range(len(times)):
            (state, lt) = cspice.spkez(6,times[i],"J2000","CN",399)
            self.assertTrue(abs(lt + saturn_rel.time[i]) < 1.e-7)
            self.assertTrue(abs(saturn_event.time[i] + lt
                                - earth_event.time[i]) < 1.e-11)
            self.assertTrue(abs(earth_event.arr[i] + state[0:3]) < 1.e-8)
            self.assertTrue(abs(saturn_rel.pos[i]  - state[0:3]) < 1.e-6)
            self.assertTrue(abs(saturn_rel.vel[i]  - state[3:6]) < 1.e-3)

        # Check light travel time corrections, Saturn from Earth wrt SSB
        earth_event = earth.event_at_time(times)
        saturn_event = saturn.photon_from_event(earth_event)
        saturn_rel = Edelta.sub_events(saturn_event, earth_event)

        for i in range(len(times)):
            (state, lt) = cspice.spkez(6,times[i],"J2000","XCN",399)
            self.assertTrue(np.abs(lt - saturn_rel.time.vals[i]) < 1.e-7)
            self.assertTrue(np.abs(earth_event.time.vals[i] + lt
                                   - saturn_event.time.vals[i]) < 1.e-11)
            self.assertTrue(np.all(np.abs(state[0:3] -
                                          earth_event.dep[i].vals) < 1.e-8))
            self.assertTrue(np.all(np.abs(state[0:3]
                                          - saturn_rel.pos[i].vals) < 1.e-6))
            self.assertTrue(np.all(np.abs(state[3:6]
                                          - saturn_rel.vel[i].vals) < 1.e-3))

        # Check light travel time corrections, Saturn wrt Earth, Earth-centered
        earth_event = Waypoint("EARTH").event_at_time(times)
        self.assertEqual(earth_event.pos, (0.,0.,0.))
        self.assertEqual(earth_event.vel, (0.,0.,0.))

        saturn = registry.connect_paths("SATURN", "EARTH", "J2000")
        saturn_event = saturn.photon_to_event(earth_event)
        saturn_rel = Edelta.sub_events(saturn_event, earth_event)

        self.assertEqual(saturn_event.origin_id, "SATURN")
        self.assertEqual(saturn_event.pos, (0.,0.,0.))
        self.assertEqual(saturn_event.vel, (0.,0.,0.))

        self.assertEqual(saturn_rel.event.origin_id, "EARTH")
        self.assertEqual(saturn_rel.origin_id, "EARTH")

        for i in range(len(times)):
            (state, lt) = cspice.spkez(6,times[i],"J2000","CN",399)
            self.assertTrue(np.abs(lt + saturn_rel.time.vals[i] < 1.e-7))
            self.assertTrue(np.abs(saturn_event.time.vals[i] + lt
                                        - earth_event.time.vals[i]) < 1.e-11)
            self.assertTrue(np.all(np.abs(state[0:3]
                                        + earth_event.arr[i].vals) < 1.e-8))
            self.assertTrue(np.all(np.abs(state[0:3]
                                        - saturn_rel.pos[i].vals) < 1.e-6))
            self.assertTrue(np.abs(saturn_rel.pos[i].norm()/constants.C +
                                        + saturn_rel.time[i]) < 1.e-7)
            self.assertTrue(np.all(np.abs(state[3:6]
                                        - saturn_rel.vel[i].vals) < 1.e-3))

        ####################################
        # Fixed and then rotating frames, forward calculation

        times = np.arange(0., 86401., 8640.)

        for frame in ["J2000", "IAU_EARTH"]:
          registry.initialize_path_registry()
          registry.initialize_frame_registry()

          ignore = SpiceFrame("IAU_EARTH", "J2000")
          earth  = SpicePath("EARTH", "SSB", frame)
          pluto  = SpicePath(9, "SSB", frame, id="PLUTO")

          pluto = Waypoint("PLUTO", frame)
          earth_event = Waypoint("EARTH", frame).event_at_time(times)
          pluto_event = pluto.photon_to_event(earth_event)

          self.assertTrue(abs(earth_event.arr_lt - pluto_event.dep_lt) < 1.e-12)
          self.assertTrue(abs(earth_event.wrt_ssb().arr -
                              pluto_event.wrt_ssb().dep) < 1.e-5)

          # Erase the wrt_ssb() cache and check again
          pluto_event.filled_ssb = None
          self.assertTrue(abs(earth_event.wrt_ssb().arr -
                              pluto_event.wrt_ssb().dep) < 1e-5)

          pluto_rel = Edelta.sub_events(pluto_event, earth_event)
          self.assertTrue(abs(pluto_rel.pos.norm() -
                            pluto_event.dep.norm()) < 1.e-5)
          self.assertTrue(abs(pluto_rel.pos + pluto_rel.dep) < 1.e-5)

          pluto_rel_ssb = pluto_rel.wrt_ssb()
          pluto_event_ssb = pluto_event.wrt_ssb()
          self.assertTrue(abs(pluto_event_ssb.time - pluto_rel_ssb.time) < 1e-2)
          self.assertTrue(abs(pluto_event_ssb.pos  - pluto_rel_ssb.pos)  < 1e-2)
          self.assertTrue(abs(pluto_event_ssb.vel  - pluto_rel_ssb.vel)  < 1e-2)
          self.assertTrue(abs(pluto_event_ssb.dep  - pluto_rel_ssb.dep)  < 1e-2)

          for i in range(len(times)):
            (state, lt) = cspice.spkez(9, times[i], frame, "CN", 399)
            self.assertTrue(abs(pluto_rel.time[i] + lt) < 1.e-6)
            self.assertTrue(abs(pluto_event.time[i] + lt
                                          - earth_event.time[i]) < 1.e-10)
            self.assertTrue(abs(earth_event.arr[i] + state[0:3]) < 1.e-5)
            self.assertTrue(abs(pluto_rel.pos[i]   - state[0:3]) < 1.e-5)
            self.assertTrue(abs(pluto_rel.vel[i]   - state[3:6]) < 1.e-3)

        ####################################
        # Fixed and then rotating frames, reverse calculation

        times = np.arange(0., 86401., 8640.)

        for frame in ["J2000", "IAU_EARTH"]:
          registry.initialize_path_registry()
          registry.initialize_frame_registry()

          ignore = SpiceFrame("IAU_EARTH", "J2000")
          earth  = SpicePath("EARTH", "SSB", frame)
          pluto  = SpicePath(9, "SSB", frame, id="PLUTO")

          pluto = Waypoint("PLUTO", frame)
          earth_event = Waypoint("EARTH", frame).event_at_time(times)
          pluto_event = pluto.photon_from_event(earth_event)

          self.assertTrue(abs(earth_event.dep_lt - pluto_event.arr_lt) < 1.e-12)
          self.assertTrue(abs(earth_event.wrt_ssb().dep -
                              pluto_event.wrt_ssb().arr) < 1.e-5)

          # Erase the wrt_ssb() cache and check again
          pluto_event.filled_ssb = None
          self.assertTrue(abs(earth_event.wrt_ssb().dep -
                              pluto_event.wrt_ssb().arr) < 1e-5)

          pluto_rel = Edelta.sub_events(pluto_event, earth_event)
          self.assertTrue(abs(pluto_rel.pos.norm() -
                              pluto_event.arr.norm()) < 1.e-5)
          self.assertTrue(abs(pluto_rel.pos - pluto_rel.arr) < 1.e-5)

          for i in range(len(times)):
            (state, lt) = cspice.spkez(9, times[i], frame, "XCN", 399)
            self.assertTrue(abs(pluto_rel.time[i] - lt) < 1.e-6)
            self.assertTrue(abs(earth_event.time[i] + lt
                                          - pluto_event.time[i]) < 1.e-10)
            self.assertTrue(abs(earth_event.dep[i] - state[0:3]) < 1.e-5)
            self.assertTrue(abs(pluto_rel.pos[i]   - state[0:3]) < 1.e-5)
            self.assertTrue(abs(pluto_rel.vel[i]   - state[3:6]) < 1.e-3)

        ####################################
        # More linked frames...
        registry.initialize_path_registry()
        registry.initialize_frame_registry()

        times = np.arange(0., 864001., 8640.)

        ignore = SpiceFrame("IAU_MARS", "J2000")
        ignore = SpiceFrame("B1950", "J2000")
        ignore = SpiceFrame("IAU_EARTH", "B1950")

        ignore = SpicePath("MARS", "SSB")
        ignore = SpicePath("EARTH", "MARS", "IAU_MARS")

        mars = Waypoint("MARS", "J2000")
        earth_event = Waypoint("EARTH","B1950").event_at_time(times)
        mars_event = mars.photon_to_event(earth_event)
        mars_rel = Edelta.sub_events(mars_event, earth_event)

        for i in range(len(times)):
            (state, lt) = cspice.spkez(499,times[i],"B1950","CN",399)
            self.assertTrue(np.abs(lt + mars_rel.time.vals[i]) < 1.e-7)
            self.assertTrue(np.abs(mars_event.time.vals[i] + lt
                                        - earth_event.time.vals[i]) < 1.e-9)
            self.assertTrue(abs(mars_rel.pos[i] - state[0:3]) < 1.e-5)
            self.assertTrue(abs(mars_rel.vel[i] - state[3:6]) < 1.e-3)

        ####################################
        # The IAU_EARTH frame works fine on Earth

        registry.initialize_path_registry()
        registry.initialize_frame_registry()

        times = np.arange(0., 864001., 86400.)

        ignore = SpiceFrame("IAU_EARTH", "J2000")

        ignore = SpicePath("EARTH", "SSB", "J2000")
        ignore = SpicePath(9, "SSB", "J2000", id="PLUTO")

        pluto = Waypoint("PLUTO","J2000")
        earth_event = Waypoint("EARTH","IAU_EARTH").event_at_time(times)
        pluto_event = pluto.photon_to_event(earth_event)
        pluto_rel = Edelta.sub_events(pluto_event, earth_event)

        for i in range(len(times)):
            (state, lt) = cspice.spkez(9,times[i],"IAU_EARTH","CN",399)
            self.assertTrue(np.abs(lt + pluto_rel.time.vals[i]) < 1.e-7)
            self.assertTrue(np.abs(pluto_event.time.vals[i] + lt
                                        - earth_event.time.vals[i]) < 1.e-9)
            self.assertTrue(abs(pluto_rel.pos[i] - state[0:3]) < 1.e-5)
            self.assertTrue(abs(pluto_rel.vel[i] - state[3:6]) < 1.e-3)

        ####################################
        # IAU_MARS on Mars

        registry.initialize_path_registry()
        registry.initialize_frame_registry()

        times = np.arange(0., 864001., 86400.)

        ignore = SpiceFrame("IAU_MARS", "J2000")

        ignore = SpicePath("EARTH", "SSB", "J2000")
        ignore = SpicePath(4, "SSB", "J2000", id="MARS")
        ignore = SpicePath(9, "SSB", "J2000", id="PLUTO")

        pluto = Waypoint("PLUTO","J2000")
        earth_event = Waypoint("MARS","IAU_MARS").event_at_time(times)
        pluto_event = pluto.photon_to_event(earth_event)
        pluto_rel = Edelta.sub_events(pluto_event, earth_event)

        for i in range(len(times)):
            (state, lt) = cspice.spkez(9,times[i],"IAU_MARS","CN",4)
            self.assertTrue(np.abs(lt + pluto_rel.time.vals[i]) < 1.e-7)
            self.assertTrue(np.abs(pluto_event.time.vals[i] + lt
                                        - earth_event.time.vals[i]) < 1.e-9)
            self.assertTrue(abs(pluto_rel.pos[i] - state[0:3]) < 1.e-5)
            self.assertTrue(abs(pluto_rel.vel[i] - state[3:6]) < 1.e-3)

        ####################################
        # Check stellar aberration calculation in J2000

        registry.initialize_path_registry()
        registry.initialize_frame_registry()

        times = np.arange(0., 365*86400., 86400.)

        earth  = SpicePath("EARTH", "SSB", "J2000")
        pluto  = SpicePath(9, "SSB", "J2000", id="PLUTO")

        pluto = Waypoint("PLUTO","J2000")
        earth_event = Waypoint("EARTH","J2000").event_at_time(times)
        pluto_event = pluto.photon_to_event(earth_event)
        pluto_rel = Edelta.sub_events(pluto_event, earth_event)

        (ra,dec) = earth_event.ra_and_dec(aberration=False)
        for i in range(len(times)):
            (state, lt) = cspice.spkez(9,times[i],"J2000","CN",399)
            (ra_test, dec_test) = cspice.recrad(state[0:3])[1:3]
            self.assertTrue(abs(ra[i]  - ra_test)  < 1.e-7)
            self.assertTrue(abs(dec[i] - dec_test) < 1.e-7)

        (ra,dec) = earth_event.ra_and_dec(aberration=True)
        for i in range(len(times)):
            (state, lt) = cspice.spkez(9,times[i],"J2000","CN+S",399)
            (ra_test, dec_test) = cspice.recrad(state[0:3])[1:3]
            self.assertTrue(abs(ra[i]  - ra_test)  < 1.e-7)
            self.assertTrue(abs(dec[i] - dec_test) < 1.e-7)

        # Time-reversed
        pluto = Waypoint("PLUTO","J2000")
        earth_event = Waypoint("EARTH","J2000").event_at_time(times)
        pluto_event = pluto.photon_from_event(earth_event)
        pluto_rel = Edelta.sub_events(pluto_event, earth_event)

        (ra,dec) = earth_event.ra_and_dec(aberration=False, subfield="dep")
        for i in range(len(times)):
            (state, lt) = cspice.spkez(9,times[i],"J2000","XCN",399)
            (ra_test, dec_test) = cspice.recrad(state[0:3])[1:3]
            self.assertTrue(abs(ra[i]  - ra_test)  < 1.e-7)
            self.assertTrue(abs(dec[i] - dec_test) < 1.e-7)

        (ra,dec) = earth_event.ra_and_dec(aberration=True, subfield="dep")
        for i in range(len(times)):
            (state, lt) = cspice.spkez(9,times[i],"J2000","XCN+S",399)
            (ra_test, dec_test) = cspice.recrad(state[0:3])[1:3]
            self.assertTrue(abs(ra[i]  - ra_test)  < 1.e-7)
            self.assertTrue(abs(dec[i] - dec_test) < 1.e-7)

        ####################################
        # Check stellar aberration calculation in a rotating frame

        registry.initialize_path_registry()
        registry.initialize_frame_registry()

        times = np.arange(0., 365*86400., 86400.)

        ignore = SpiceFrame("IAU_EARTH", "J2000")
        earth  = SpicePath("EARTH", "SSB", "IAU_EARTH")
        pluto  = SpicePath(9, "SSB", "IAU_EARTH", id="PLUTO")

        pluto = Waypoint("PLUTO","IAU_EARTH")
        earth_event = Waypoint("EARTH","IAU_EARTH").event_at_time(times)
        pluto_event = pluto.photon_to_event(earth_event)
        pluto_rel = Edelta.sub_events(pluto_event, earth_event)

        (ra,dec) = earth_event.ra_and_dec(aberration=False)
        for i in range(len(times)):
            (state, lt) = cspice.spkez(9,times[i],"J2000","CN",399)
            (ra_test, dec_test) = cspice.recrad(state[0:3])[1:3]
            self.assertTrue(abs(ra[i]  - ra_test)  < 1.e-7)
            self.assertTrue(abs(dec[i] - dec_test) < 1.e-7)

        (ra,dec) = earth_event.ra_and_dec(aberration=True)
        for i in range(len(times)):
            (state, lt) = cspice.spkez(9,times[i],"J2000","CN+S",399)
            (ra_test, dec_test) = cspice.recrad(state[0:3])[1:3]
            self.assertTrue(abs(ra[i]  - ra_test)  < 1.e-7)
            self.assertTrue(abs(dec[i] - dec_test) < 1.e-7)

        registry.initialize_path_registry()
        registry.initialize_frame_registry()

      spice.initialize()
      Path.USE_QUICKPATHS = True
      Frame.USE_QUICKFRAMES = True

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
