################################################################################
# oops/path/spicepath.py: Subclass SpicePath of class Path
################################################################################

import cspyce
import numpy as np

from polymath import Scalar, Vector3

from .       import AliasPath, Path, RotatedPath
from ..event import Event
from ..frame import Frame
from ..path  import ReversedPath
import oops.spice_support as spice

class SpicePath(Path):
    """A Path subclass that returns information based on an SPICE SP kernel.

    It represents the geometric position of a single target body with respect to
    a single origin.
    """

    # Set False to confirm that SpicePaths return the same results without
    # shortcuts and with shortcuts
    USE_SPICEPATH_SHORTCUTS = True

    #===========================================================================
    def __init__(self, spice_id, spice_origin="SSB", spice_frame="J2000",
                       path_id=None, shortcut=None, unpickled=False):
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
            path_id         the name or ID under which the path will be
                            registered. By default, this will be the value of
                            spice_id if that is given as a string; otherwise
                            it will be the name as used by the SPICE toolkit.
            shortcut        If a shortcut is specified, then this is registered
                            as a shortcut definition; the other registered path
                            definitions are unchanged.
            unpickled       True if this object was read from a pickle file. If
                            so, then it will be treated as a duplicate of a
                            pre-existing SpicePath for the same SPICE ID.
        """

        # Preserve the inputs
        self.spice_id = spice_id
        self.spice_origin = spice_origin
        self.spice_frame = spice_frame
        self.shortcut = shortcut

       # Interpret the SPICE IDs
        (self.spice_target_id,
         self.spice_target_name) = spice.body_id_and_name(spice_id)

        (self.spice_origin_id,
         self.spice_origin_name) = spice.body_id_and_name(spice_origin)

        self.spice_frame_name = spice.frame_id_and_name(spice_frame)[1]

        # Fill in the Path ID and save it in the global dictionary
        if path_id is None:
            if isinstance(spice_id, str):
                self.path_id = spice_id
            else:
                self.path_id = self.spice_target_name
        else:
            self.path_id = path_id

        # Only save info in the PATH_TRANSLATION dictionary if it is not already
        # there. We do not want to overwrite original definitions with those
        # just read from pickle files.
        if not shortcut:
            if self.spice_target_id not in spice.PATH_TRANSLATION:
                spice.PATH_TRANSLATION[self.spice_target_id] = self.path_id
            if self.spice_target_name not in spice.PATH_TRANSLATION:
                spice.PATH_TRANSLATION[self.spice_target_name] = self.path_id

        # Fill in the origin waypoint, which should already be in the dictionary
        origin_id = spice.PATH_TRANSLATION[self.spice_origin_id]
        self.origin = Path.as_waypoint(origin_id)

        # Fill in the frame wayframe, which should already be in the dictionary
        frame_id = spice.FRAME_TRANSLATION[self.spice_frame_name]
        self.frame = Frame.as_wayframe(frame_id)

        # No shape, no keys
        self.shape = ()
        self.keys = set()
        self.shortcut = shortcut

        # Register the SpicePath; fill in the waypoint
        self.register(shortcut, unpickled=unpickled)

    #===========================================================================
    def __getstate__(self):
        return (self.spice_target_id, self.spice_origin_id,
                self.spice_frame_name)

    #===========================================================================
    def __setstate__(self, state):

        (spice_target_id, spice_origin_id, spice_frame_name) = state

        # If this is a duplicate of a pre-existing SpicePath, make sure it gets
        # assigned the pre-existing path ID and Waypoint
        path_id = spice.PATH_TRANSLATION.get(spice_target_id, None)
        self.__init__(spice_target_id, spice_origin_id, spice_frame_name,
                      path_id=path_id, unpickled=True)

    #===========================================================================
    def event_at_time(self, time, quick={}):
        """An Event corresponding to a specified Scalar time on this path.

        Input:
            time        a time Scalar at which to evaluate the path.

        Return:         an Event object containing (at least) the time, position
                        and velocity of the path.
        """

        time = Scalar.as_scalar(time).as_float()

        # A fully-masked time can be handled quickly
        if time.mask is True:
            return Event(time, Vector3.ZERO, self.origin, self.frame)

        # A single unmasked time can be handled quickly
        if time.shape == ():
            (state,
             lighttime) = cspyce.spkez(self.spice_target_id,
                                       time.vals,
                                       self.spice_frame_name,
                                       'NONE',
                                       self.spice_origin_id)

            return Event(time, (state[0:3],state[3:6]), self.origin, self.frame)

        # Use a QuickPath if warranted, possibly making a recursive call
        if isinstance(quick, dict):
            return self.quick_path(time, quick).event_at_time(time, False)

        # Fill in the states and light travel times using cspyce
        if np.any(time.mask):
            state = cspyce.spkez_vector(self.spice_target_id,
                                        time.vals[time.antimask],
                                        self.spice_frame_name,
                                        'NONE',
                                        self.spice_origin_id)[0]

            pos = np.zeros(time.shape + (3,))
            vel = np.zeros(time.shape + (3,))
            pos[time.antimask] = state[...,0:3]
            vel[time.antimask] = state[...,3:6]

        else:
            state = cspyce.spkez_vector(self.spice_target_id,
                                        time.vals.ravel(),
                                        self.spice_frame_name,
                                        'NONE',
                                        self.spice_origin_id)[0]
            pos = state[:,0:3].reshape(time.shape + (3,))
            vel = state[:,3:6].reshape(time.shape + (3,))

        # Convert to an Event and return
        return Event(time, (pos,vel), self.origin, self.frame)

    #===========================================================================
    def wrt(self, origin, frame=None):
        """Construct a path pointing from an origin to this target in any frame.

        SpicePath overrides the default method to create quicker "shortcuts"
        between SpicePaths.

        Input:
            origin      an origin Path object or its registered name.
            frame       a frame object or its registered ID. Default is to use
                        the frame of the origin's path.
        """

        # Use the slow method if necessary, for debugging
        if not SpicePath.USE_SPICEPATH_SHORTCUTS:
            return Path.wrt(self, origin, frame)

        # Interpret the origin path
        origin = Path.as_primary_path(origin)
        if origin in (Path.SSB, None):
            spice_origin_id = 0
        elif isinstance(origin, SpicePath):
            spice_origin_id = origin.spice_target_id
        else:
            # If the origin is not a SpicePath, seek from the other direction
            return ReversedPath(origin.wrt(self, frame))

        origin_id = spice.PATH_TRANSLATION[spice_origin_id]

        # Interpret the frame
        frame = Frame.as_primary_frame(frame)
        if frame in (Frame.J2000, None):
            spice_frame_name = 'J2000'
            uses_spiceframe = True
        elif type(frame).__name__ == 'SpiceFrame':  # avoids a circular load
            spice_frame_name = frame.spice_frame_name
            uses_spiceframe = True
        else:
            uses_spiceframe = False     # not a SpiceFrame
            spice_frame_name = 'J2000'

        if uses_spiceframe:
            frame_id = spice.FRAME_TRANSLATION[spice_frame_name]
        else:
            frame_id = 'J2000'

        shortcut = ('SPICE_SHORTCUT[' + str(self.path_id) + ',' +
                                        str(origin_id)    + ',' +
                                        str(frame_id)     + ']')

        result = SpicePath(self.spice_target_id, spice_origin_id,
                           spice_frame_name, self.path_id, shortcut)

        # If the path uses a non-spice frame, add a rotated version
        if not uses_spiceframe:
            shortcut = ('SHORTCUT_' + str(self.path_id) + '_' +
                                      str(origin_id)    + '_' +
                                      str(frame.frame_id))
            result = RotatedPath(result, frame)
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
      from ..frame import Frame
      from ..frame.spiceframe import SpiceFrame
      import oops.constants as constants
      import os
      from ..unittester_support import TESTDATA_PARENT_DIRECTORY

      Path.USE_QUICKPATHS = False
      Frame.USE_QUICKFRAMES = False

      cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE/pck00010.tpc"))
      cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE/de421.bsp"))

      # Repeat the tests without and then with shortcuts
      for SpicePath.USE_SPICEPATH_SHORTCUTS in (False, True):

        Path.reset_registry()
        Frame.reset_registry()

        _     = SpicePath("SUN", "SSB")
        earth = SpicePath("EARTH", "SSB")
        moon  = SpicePath("MOON", "EARTH")

        # Validate state vectors using event_at_time()
        times = np.arange(-3.e8, 3.01e8, 0.5e7)
        moon_event = moon.event_at_time(times)
        for i in range(len(times)):
            (state, lt) = cspyce.spkez(301,times[i],"J2000","NONE",399)
            self.assertEqual(moon_event.pos[i], state[0:3])
            self.assertEqual(moon_event.vel[i], state[3:6])

        # Check light travel time corrections to/from SSB
        saturn = SpicePath(6, "SSB", path_id="SATURN")
        times = np.arange(-3.e8, 3.01e8, 0.5e8)
        ssb_event = Path.as_primary_path("SSB").event_at_time(times)

        (saturn_event, ssb_event) = saturn.photon_to_event(ssb_event,
                                                 converge={'max_iterations':99})
        for i in range(len(times)):
            (state, lt) = cspyce.spkez(6,times[i],"J2000","CN",0)
            self.assertTrue(abs(lt - saturn_event.dep_lt[i]) < 1.e-11)
            self.assertTrue(abs(saturn_event.time[i] + lt - ssb_event.time[i])
                                                                       < 1.e-11)
            self.assertTrue((ssb_event.arr[i] + state[0:3]).norm() < 1.e-8)
            self.assertTrue((saturn_event.dep[i] + state[0:3]).norm() < 1.e-7)
            self.assertEqual(saturn_event.pos[i], (0.,0.,0.))
            self.assertEqual(saturn_event.vel[i], (0.,0.,0.))

        (saturn_event, ssb_event) = saturn.photon_from_event(ssb_event,
                                                converge={'max_iterations':99})
        for i in range(len(times)):
            (state, lt) = cspyce.spkez(6,times[i],"J2000","XCN",0)
            self.assertTrue(abs(lt + saturn_event.arr_lt[i]) < 1.e-11)
            self.assertTrue(abs(ssb_event.time[i] + lt - saturn_event.time[i])
                                                                       < 1.e-11)
            self.assertTrue((ssb_event.dep[i] - state[0:3]).norm() < 1.e-8)
            self.assertTrue((Vector3(state[0:3]) - ssb_event.dep[i]).norm() < 1.e-8)

        # Check instantaneous geometry using linked paths

        # Moon wrt Earth
        times = np.arange(-3.e8, 3.01e8, 0.5e8)
        moon_event = moon.event_at_time(times).wrt_path("EARTH")
        for i in range(len(times)):
            (state, lt) = cspyce.spkez(301,times[i],"J2000","NONE",399)
            self.assertTrue(np.all(np.abs(state[0:3] -
                                          moon_event.pos.vals[i]) < 1.e-8))
            self.assertTrue(np.all(np.abs(state[3:6] -
                                          moon_event.vel.vals[i]) < 1.e-8))

        # Moon to SSB
        moon_event = moon.event_at_time(times).wrt_path("SSB")
        for i in range(len(times)):
            (state, lt) = cspyce.spkez(301,times[i],"J2000","NONE",0)
            self.assertTrue(np.all(np.abs(state[0:3] -
                                          moon_event.pos.vals[i]) < 1.e-6))
            self.assertTrue(np.all(np.abs(state[3:6] -
                                          moon_event.vel.vals[i]) < 1.e-6))

        # Moon to Saturn
        moon_event = moon.event_at_time(times).wrt_path("SATURN")
        for i in range(len(times)):
            (state, lt) = cspyce.spkez(301,times[i],"J2000","NONE",6)
            self.assertTrue(np.all(np.abs(state[0:3] -
                                          moon_event.pos.vals[i]) < 1.e-6))
            self.assertTrue(np.all(np.abs(state[3:6] -
                                          moon_event.vel.vals[i]) < 1.e-6))

        ####################################
        # Tests of combined paths but no frame rotation

        Path.reset_registry()
        Frame.reset_registry()

        times = np.arange(-3.e8, 3.01e8, 0.5e7)

        _      = SpicePath("SUN", "SSB")
        earth  = SpicePath("EARTH", "SSB")
        moon   = SpicePath("MOON", "EARTH")
        mars   = SpicePath("MARS", "MOON")
        saturn = SpicePath(6, "MOON", path_id="SATURN")

        self.assertEqual(Path.as_primary_path("SATURN"), saturn)

        path = Path.as_path("MARS").wrt("SUN")

        event = Path.as_path("MARS").event_at_time(times).wrt_path("SUN")
        self.assertEqual(event.frame_id, "J2000")
        self.assertEqual(event.origin_id, "SUN")

        for i in range(len(times)):
            (state, lt) = cspyce.spkez(499, times[i], "J2000", "NONE", 10)
            dpos = event.pos[i] - state[0:3]
            dvel = event.vel[i] - state[3:6]
            self.assertTrue(np.all(np.abs(dpos.vals) < 1.e-7))
            self.assertTrue(np.all(np.abs(dvel.vals) < 1.e-14))

        # Tests using different frames
        Path.reset_registry()
        Frame.reset_registry()

        _ = SpicePath("MARS", "SSB")
        _ = SpiceFrame("IAU_MARS", "J2000")
        earth = SpicePath("EARTH", "SSB", "IAU_MARS")

        path = Path.as_path("EARTH").wrt("SSB", "J2000")
        event = path.event_at_time(times)
        for i in range(len(times)):
            (state, lt) = cspyce.spkez(399, times[i], "J2000", "NONE", 0)
            dpos = event.pos[i] - state[0:3]
            dvel = event.vel[i] - state[3:6]
            self.assertTrue(np.all(np.abs(dpos.vals) < 1.e-7))
            self.assertTrue(np.all(np.abs(dvel.vals) < 1.e-9))

        ####################################

        Path.reset_registry()
        Frame.reset_registry()

        _ = SpiceFrame("IAU_EARTH", "J2000")
        _ = SpiceFrame("IAU_MARS", "IAU_EARTH")
        _ = SpiceFrame("IAU_MOON", "IAU_EARTH")
        _ = SpiceFrame("B1950", "J2000")

        earth = SpicePath("EARTH", "SSB", "IAU_EARTH")
        moon  = SpicePath("MOON", "EARTH", "IAU_MOON")
        _     = SpicePath("SUN", "SSB", "J2000")
        mars  = SpicePath("MARS", "SUN", "J2000")

        path = Path.as_path("EARTH").wrt("SSB", "J2000")
        event = path.event_at_time(times)
        for i in range(len(times)):
            (state, lt) = cspyce.spkez(399, times[i], "J2000", "NONE", 0)
            dpos = event.pos[i] - state[0:3]
            dvel = event.vel[i] - state[3:6]
            self.assertTrue(np.all(np.abs(dpos.vals) < 1.e-7))
            self.assertTrue(np.all(np.abs(dvel.vals) < 1.e-9))

        path = Path.as_path("SSB").wrt("EARTH", "IAU_MARS")
        event = path.event_at_time(times)
        for i in range(len(times)):
            (state, lt) = cspyce.spkez(0, times[i], "IAU_MARS", "NONE", 399)
            dpos = event.pos[i] - state[0:3]
            dvel = event.vel[i] - state[3:6]
            self.assertTrue(np.all(np.abs(dpos.vals) < 1.e-7))
            self.assertTrue(np.all(np.abs(dvel.vals) < 1.e-9))

        path = Path.as_path("EARTH").wrt("SUN", "IAU_EARTH")
        event = path.event_at_time(times)
        for i in range(len(times)):
            (state, lt) = cspyce.spkez(399, times[i], "IAU_EARTH", "NONE", 10)
            dpos = event.pos[i] - state[0:3]
            dvel = event.vel[i] - state[3:6]
            self.assertTrue(np.all(np.abs(dpos.vals) < 1.e-6))
            self.assertTrue(np.all(np.abs(dvel.vals) < 1.e-9))

        path = Path.as_path("MOON").wrt("MARS", "IAU_EARTH")
        event = path.event_at_time(times)
        for i in range(len(times)):
            (state, lt) = cspyce.spkez(301, times[i], "IAU_EARTH", "NONE", 499)
            dpos = event.pos[i] - state[0:3]
            dvel = event.vel[i] - state[3:6]
            self.assertTrue(np.all(np.abs(dpos.vals) < 1.e-6))
            self.assertTrue(np.all(np.abs(dvel.vals) < 1.e-9))

        ####################################

        Path.reset_registry()
        Frame.reset_registry()

        _ = SpiceFrame("IAU_MARS", "J2000")
        _ = SpiceFrame("IAU_EARTH", "J2000")
        _ = SpiceFrame("B1950", "IAU_EARTH")
        _ = SpiceFrame("IAU_MOON", "B1950")

        _      = SpicePath("SUN", "SSB", "J2000")
        earth  = SpicePath("EARTH", "SSB", "IAU_EARTH")
        moon   = SpicePath("MOON", "EARTH", "IAU_MOON")
        mars   = SpicePath("MARS", "MOON", "B1950")

        times = np.arange(-3.e8, 3.01e8, 0.5e7)
        path = Path.as_path("MARS").wrt("MOON", "IAU_MOON")
        event = path.event_at_time(times)
        for i in range(len(times)):
            (state, lt) = cspyce.spkez(499, times[i], "IAU_MOON", "NONE", 301)
            dpos = event.pos[i] - state[0:3]
            dvel = event.vel[i] - state[3:6]
            self.assertTrue(np.all(np.abs(dpos.vals) < 1.e-6))
            self.assertTrue(np.all(np.abs(dvel.vals) < 1.e-9))

        ####################################

        Path.reset_registry()
        Frame.reset_registry()

        _      = SpicePath("SUN", "SSB")
        earth  = SpicePath("EARTH", "SSB")
        moon   = SpicePath("MOON", "EARTH")
        saturn = SpicePath(6, "SSB", path_id="SATURN")
        _      = Path.as_path("SSB")

        times = np.arange(-3.e8, 3.01e8, 0.5e8)

        # Check light travel time corrections, Saturn to Earth wrt SSB
        earth_event = earth.event_at_time(times)
        (saturn_event, earth_event) = saturn.photon_to_event(earth_event)

        saturn_rel = saturn_event.sub(earth_event)
        self.assertTrue(abs(saturn_rel.dep - earth_event.arr).max() < 1.e-6)

        saturn_rel_ssb = saturn_rel.wrt_ssb()
        saturn_abs_ssb = saturn_event.wrt_ssb()
        self.assertTrue(abs(saturn_rel_ssb.time - saturn_abs_ssb.time).max() < 2.e-6)
        self.assertTrue(abs(saturn_rel_ssb.pos  - saturn_abs_ssb.pos).max()  < 2.e-6)
        self.assertTrue(abs(saturn_rel_ssb.vel  - saturn_abs_ssb.vel).max()  < 2.e-6)
        self.assertTrue(abs(saturn_rel_ssb.dep  - saturn_abs_ssb.dep).max()  < 2.e-6)

        for i in range(len(times)):
            (state, lt) = cspyce.spkez(6,times[i],"J2000","CN",399)
            self.assertTrue(abs(lt + saturn_rel.time[i]) < 1.e-7)
            self.assertTrue(abs(saturn_event.time[i] + lt
                                - earth_event.time[i]) < 1.e-11)
            self.assertTrue(abs(earth_event.arr[i] + state[0:3]) < 1.e-8)
            self.assertTrue(abs(saturn_rel.pos[i]  - state[0:3]) < 1.e-6)
            self.assertTrue(abs(saturn_rel.vel[i]  - state[3:6]) < 1.e-3)

        # Check light travel time corrections, Saturn from Earth wrt SSB
        earth_event = earth.event_at_time(times)
        (saturn_event,earth_event) = saturn.photon_from_event(earth_event)
        saturn_rel = saturn_event.sub(earth_event)

        for i in range(len(times)):
            (state, lt) = cspyce.spkez(6,times[i],"J2000","XCN",399)
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
        earth_event = Path.as_path("EARTH").event_at_time(times)
        self.assertEqual(earth_event.pos, (0.,0.,0.))
        self.assertEqual(earth_event.vel, (0.,0.,0.))

        saturn = Path.as_path("SATURN").wrt("EARTH", "J2000")
        (saturn_event,earth_event) = saturn.photon_to_event(earth_event)
        saturn_rel = saturn_event.sub(earth_event)

        self.assertEqual(saturn_event.origin_id, "SATURN")
        self.assertEqual(saturn_event.pos, (0.,0.,0.))
        self.assertEqual(saturn_event.vel, (0.,0.,0.))

        self.assertEqual(saturn_rel.event.origin_id, "SATURN")
        self.assertEqual(saturn_rel.origin_id, "EARTH")

        for i in range(len(times)):
            (state, lt) = cspyce.spkez(6,times[i],"J2000","CN",399)
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

        # Apparent case
        for i in range(len(times)):
            (state, lt) = cspyce.spkez(6,times[i],"J2000","CN+S",399)
            self.assertTrue(np.abs(lt + saturn_rel.time.vals[i] < 1.e-7))
            self.assertTrue(np.abs(saturn_event.time.vals[i] + lt
                                        - earth_event.time.vals[i]) < 1.e-11)

            length = np.sqrt(np.sum(state[0:3]**2))
            self.assertTrue(np.all(np.abs(state[0:3] / length +
                                   earth_event.arr_ap[i].unit().vals) < 1.e-8))

        ####################################
        # Fixed and then rotating frames, forward calculation

        times = np.arange(0., 86401., 8640.)

        for frame in ["J2000", "IAU_EARTH"]:
            Path.reset_registry()
            Frame.reset_registry()

            _ = SpiceFrame("IAU_EARTH", "J2000")
            earth = SpicePath("EARTH", "SSB", frame)
            pluto = SpicePath(9, "SSB", frame, path_id="PLUTO")

            pluto = AliasPath("PLUTO", frame)
            earth_event = AliasPath("EARTH", frame).event_at_time(times)
            (pluto_event,earth_event) = pluto.photon_to_event(earth_event)

            self.assertTrue(abs(earth_event.arr_lt + pluto_event.dep_lt).max() < 1.e-12)
            self.assertTrue(abs(earth_event.wrt_ssb().arr -
                                pluto_event.wrt_ssb().dep).max() < 1.e-5)

            # Erase the wrt_ssb() cache and check again
            if iter in range(2):
              if iter > 0:
                pluto_event._Event__ssb_ = None
                pluto_event._Event__ssb_xform_ = None

              self.assertTrue(abs(earth_event.wrt_ssb().arr -
                                  pluto_event.wrt_ssb().dep).max() < 1.e-5)

              pluto_rel = pluto_event.sub(earth_event)
              self.assertTrue(abs(pluto_rel.pos.norm() -
                                  pluto_event.dep.norm()).max() < 1.e-5)
              self.assertTrue(abs(pluto_rel.pos + pluto_rel.dep).max() < 1.e-5)

              pluto_rel_ssb = pluto_rel.wrt_ssb()
              pluto_event_ssb = pluto_event.wrt_ssb()
              self.assertTrue(abs(pluto_event_ssb.time - pluto_rel_ssb.time).max() < 1e-2)
              self.assertTrue(abs(pluto_event_ssb.pos - pluto_rel_ssb.pos).max() < 1e-2)
              self.assertTrue(abs(pluto_event_ssb.vel - pluto_rel_ssb.vel).max() < 1e-2)
              self.assertTrue(abs(pluto_event_ssb.dep - pluto_rel_ssb.dep).max() < 1e-2)

              for i in range(len(times)):
                (state, lt) = cspyce.spkez(9, times[i], frame, "CN", 399)
                self.assertTrue(abs(pluto_rel.time[i] + lt) < 1.e-6)
                self.assertTrue(abs(pluto_event.time[i] + lt
                                              - earth_event.time[i]) < 1.e-10)
                self.assertTrue(abs(earth_event.arr[i] + state[0:3]) < 1.e-5)
                self.assertTrue(abs(pluto_rel.pos[i]   - state[0:3]) < 1.e-5)
                self.assertTrue(abs(pluto_rel.vel[i]   - state[3:6]) < 1.e-3)

              # Apparent case
              for i in range(len(times)):
                (state, lt) = cspyce.spkez(9, times[i], frame, "CN+S", 399)
                self.assertTrue(np.abs(lt + pluto_rel.time.vals[i]) < 1.e-6)
                self.assertTrue(np.abs(pluto_event.time.vals[i] + lt -
                                       earth_event.time.vals[i]) < 1.e-11)

                length = np.sqrt(np.sum(state[0:3]**2))
                self.assertTrue(np.all(np.abs(state[0:3] / length +
                                    earth_event.arr_ap[i].unit().vals) < 1.e-8))

        ####################################
        # Fixed and then rotating frames, reverse calculation

        times = np.arange(0., 86401., 8640.)

        for frame in ["J2000", "IAU_EARTH"]:
            Path.reset_registry()
            Frame.reset_registry()

            _ = SpiceFrame("IAU_EARTH", "J2000")
            earth = SpicePath("EARTH", "SSB", frame)
            pluto = SpicePath(9, "SSB", frame, path_id="PLUTO")

            pluto = AliasPath("PLUTO", frame)
            earth_event = AliasPath("EARTH", frame).event_at_time(times)
            (pluto_event,earth_event) = pluto.photon_from_event(earth_event)

            self.assertTrue(abs(earth_event.dep_lt + pluto_event.arr_lt).max() < 1.e-12)
            self.assertTrue(abs(earth_event.wrt_ssb().dep -
                                pluto_event.wrt_ssb().arr).max() < 1.e-5)

            # Erase the wrt_ssb() cache and check again
            if iter in range(2):
              if iter > 0:
                pluto_event._Event__ssb_ = None
                pluto_event._Event__ssb_xform_ = None

              self.assertTrue(abs(earth_event.wrt_ssb().dep -
                                  pluto_event.wrt_ssb().arr).max() < 1e-5)

              pluto_rel = pluto_event.sub(earth_event)
              self.assertTrue(abs(pluto_rel.pos.norm() -
                                  pluto_event.arr.norm()).max() < 1.e-5)
              self.assertTrue(abs(pluto_rel.pos - pluto_rel.arr).max() < 1.e-5)

              for i in range(len(times)):
                (state, lt) = cspyce.spkez(9, times[i], frame, "XCN", 399)
                self.assertTrue(abs(pluto_rel.time[i] - lt) < 1.e-6)
                self.assertTrue(abs(earth_event.time[i] + lt
                                              - pluto_event.time[i]) < 1.e-10)
                self.assertTrue(abs(earth_event.dep[i] - state[0:3]) < 1.e-5)
                self.assertTrue(abs(pluto_rel.pos[i]   - state[0:3]) < 1.e-5)
                self.assertTrue(abs(pluto_rel.vel[i]   - state[3:6]) < 1.e-3)

              # Apparent case
              for i in range(len(times)):
                (state, lt) = cspyce.spkez(9, times[i], frame, "XN+S", 399)
                self.assertTrue(np.abs(pluto_rel.time[i] - lt) < 1.e-6)
                self.assertTrue(np.abs(pluto_event.time.vals[i] + lt -
                                       earth_event.time.vals[i]) < 1.e-11)

                length = np.sqrt(np.sum(state[0:3]**2))
                self.assertTrue(np.all(np.abs(state[0:3] / length +
                                    earth_event.dep_ap[i].unit().vals) < 1.e-8))

        ####################################
        # More linked frames...

        Path.reset_registry()
        Frame.reset_registry()

        times = np.arange(0., 864001., 8640.)

        _ = SpiceFrame("IAU_MARS", "J2000")
        _ = SpiceFrame("B1950", "J2000")
        _ = SpiceFrame("IAU_EARTH", "B1950")

        _ = SpicePath("MARS", "SSB")
        _ = SpicePath("EARTH", "MARS", "IAU_MARS")

        mars = AliasPath("MARS", "J2000")
        earth_event = AliasPath("EARTH","B1950").event_at_time(times)
        (mars_event,earth_event) = mars.photon_to_event(earth_event)
        mars_rel = mars_event.sub(earth_event)

        for i in range(len(times)):
            (state, lt) = cspyce.spkez(499,times[i],"B1950","CN",399)
            self.assertTrue(np.abs(lt + mars_rel.time.vals[i]) < 1.e-7)
            self.assertTrue(np.abs(mars_event.time.vals[i] + lt
                                        - earth_event.time.vals[i]) < 1.e-9)
            self.assertTrue((mars_rel.pos[i] - state[0:3]).norm() < 1.e-5)
            self.assertTrue((mars_rel.vel[i] - state[3:6]).norm() < 1.e-3)

        ####################################
        # The IAU_EARTH frame works fine on Earth

        Path.reset_registry()
        Frame.reset_registry()

        times = np.arange(0., 864001., 86400.)

        _ = SpiceFrame("IAU_EARTH", "J2000")
        _ = SpicePath("EARTH", "SSB", "J2000")
        _ = SpicePath(9, "SSB", "J2000", path_id="PLUTO")

        pluto = AliasPath("PLUTO", "J2000")
        earth_event = AliasPath("EARTH","IAU_EARTH").event_at_time(times)
        (pluto_event,earth_event) = pluto.photon_to_event(earth_event)
        pluto_rel = pluto_event.sub(earth_event)

        for i in range(len(times)):
            (state, lt) = cspyce.spkez(9,times[i],"IAU_EARTH","CN",399)
            self.assertTrue(np.abs(lt + pluto_rel.time.vals[i]) < 1.e-7)
            self.assertTrue(np.abs(pluto_event.time.vals[i] + lt
                                        - earth_event.time.vals[i]) < 1.e-9)
            self.assertTrue((pluto_rel.pos[i] - state[0:3]).norm() < 1.e-5)
            self.assertTrue((pluto_rel.vel[i] - state[3:6]).norm() < 1.e-3)

        ####################################
        # IAU_MARS on Mars

        Path.reset_registry()
        Frame.reset_registry()

        times = np.arange(0., 864001., 86400.)

        _ = SpiceFrame("IAU_MARS", "J2000")
        _ = SpicePath("EARTH", "SSB", "J2000")
        _ = SpicePath(4, "SSB", "J2000", path_id="MARS")
        _ = SpicePath(9, "SSB", "J2000", path_id="PLUTO")

        pluto = AliasPath("PLUTO","J2000")
        earth_event = AliasPath("MARS","IAU_MARS").event_at_time(times)
        (pluto_event,earth_event) = pluto.photon_to_event(earth_event)
        pluto_rel = pluto_event.sub(earth_event)

        for i in range(len(times)):
            (state, lt) = cspyce.spkez(9,times[i],"IAU_MARS","CN",4)
            self.assertTrue(np.abs(lt + pluto_rel.time.vals[i]) < 1.e-7)
            self.assertTrue(np.abs(pluto_event.time.vals[i] + lt
                                        - earth_event.time.vals[i]) < 1.e-9)
            self.assertTrue((pluto_rel.pos[i] - state[0:3]).norm() < 1.e-5)
            self.assertTrue((pluto_rel.vel[i] - state[3:6]).norm() < 1.e-3)

        ####################################
        # Check stellar aberration calculation in J2000

        Path.reset_registry()
        Frame.reset_registry()

        times = np.arange(0., 365*86400., 86400.)

        earth  = SpicePath("EARTH", "SSB", "J2000")
        pluto  = SpicePath(9, "SSB", "J2000", path_id="PLUTO")

        pluto = AliasPath("PLUTO","J2000")
        earth_event = AliasPath("EARTH","J2000").event_at_time(times)
        (pluto_event,earth_event) = pluto.photon_to_event(earth_event)
        pluto_rel = pluto_event.sub(earth_event)

        (ra,dec) = earth_event.ra_and_dec(apparent=False)
        for i in range(len(times)):
            (state, lt) = cspyce.spkez(9,times[i],"J2000","CN",399)
            (ra_test, dec_test) = cspyce.recrad(state[0:3])[1:3]
            self.assertTrue(abs(ra[i]  - ra_test)  < 1.e-7)
            self.assertTrue(abs(dec[i] - dec_test) < 1.e-7)

        (ra,dec) = earth_event.ra_and_dec(apparent=True)
        for i in range(len(times)):
            (state, lt) = cspyce.spkez(9,times[i],"J2000","CN+S",399)
            (ra_test, dec_test) = cspyce.recrad(state[0:3])[1:3]
            self.assertTrue(abs(ra[i]  - ra_test)  < 1.e-7)
            self.assertTrue(abs(dec[i] - dec_test) < 1.e-7)

        # Time-reversed
        pluto = AliasPath("PLUTO","J2000")
        earth_event = AliasPath("EARTH","J2000").event_at_time(times)
        (pluto_event,earth_event) = pluto.photon_from_event(earth_event)
        pluto_rel = pluto_event.sub(earth_event)

        (ra,dec) = earth_event.ra_and_dec(apparent=False, subfield="dep")
        for i in range(len(times)):
            (state, lt) = cspyce.spkez(9, times[i], "J2000", "XCN", 399)
            (ra_test, dec_test) = cspyce.recrad(state[0:3])[1:3]
            self.assertTrue(abs(ra[i]  - ra_test)  < 1.e-7)
            self.assertTrue(abs(dec[i] - dec_test) < 1.e-7)

        (ra,dec) = earth_event.ra_and_dec(apparent=True, subfield="dep")
        for i in range(len(times)):
            (state, lt) = cspyce.spkez(9,times[i], "J2000", "XCN+S", 399)
            (ra_test, dec_test) = cspyce.recrad(state[0:3])[1:3]
            self.assertTrue(abs(ra[i]  - ra_test)  < 1.e-7)
            self.assertTrue(abs(dec[i] - dec_test) < 1.e-7)

        ####################################
        # Check stellar aberration calculation in a rotating frame

        Path.reset_registry()
        Frame.reset_registry()

        times = np.arange(0., 365*86400., 86400.)

        _ = SpiceFrame("IAU_EARTH", "J2000")
        earth = SpicePath("EARTH", "SSB", "IAU_EARTH")
        pluto = SpicePath(9, "SSB", "IAU_EARTH", path_id="PLUTO")

        pluto = AliasPath("PLUTO", "IAU_EARTH")
        earth_event = AliasPath("EARTH", "IAU_EARTH").event_at_time(times)
        (pluto_event,earth_event) = pluto.photon_to_event(earth_event)
        pluto_rel = pluto_event.sub(earth_event)

        # Note: These "RA,dec" values are in the IAU_EARTH frame, not J2000!
        (ra,dec) = earth_event.ra_and_dec(apparent=False, frame=None)
        for i in range(len(times)):
            (state, lt) = cspyce.spkez(9, times[i], "IAU_EARTH", "CN", 399)
            (ra_test, dec_test) = cspyce.recrad(state[0:3])[1:3]
            self.assertTrue(abs(ra[i]  - ra_test)  < 1.e-8)
            self.assertTrue(abs(dec[i] - dec_test) < 1.e-8)

        (ra,dec) = earth_event.ra_and_dec(apparent=True, frame=None)
        for i in range(len(times)):
            (state, lt) = cspyce.spkez(9, times[i], "IAU_EARTH", "CN+S", 399)
            (ra_test, dec_test) = cspyce.recrad(state[0:3])[1:3]
            self.assertTrue(abs(ra[i]  - ra_test)  < 1.e-8)
            self.assertTrue(abs(dec[i] - dec_test) < 1.e-8)

        Path.reset_registry()
        Frame.reset_registry()

      spice.initialize()
      Path.USE_QUICKPATHS = True
      Frame.USE_QUICKFRAMES = True

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
