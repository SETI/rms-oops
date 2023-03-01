################################################################################
# oops/frame/ringframe.py: Subclass RingFrame of class Frame
################################################################################

import numpy as np
from polymath       import Matrix3, Qube, Scalar, Vector3
from oops.frame     import Frame
from oops.transform import Transform

class RingFrame(Frame):
    """A Frame subclass describing a non-rotating frame centered on the Z-axis
    of another frame, but oriented with the X-axis fixed along the ascending
    node of the equator within the reference frame.
    """

    FRAME_IDS = {}  # frame_id to use if a frame already exists upon un-pickling

    #===========================================================================
    def __init__(self, frame, epoch=None, retrograde=False, aries=False,
                       frame_id='+', cache_size=1000, unpickled=False):
        """Constructor for a RingFrame Frame.

        Input:
            frame       a frame describing the central planet of the ring plane
                        relative to J2000.

            epoch       the time TDB at which the frame is to be evaluated. If
                        this is specified, then the frame will be precisely
                        inertial, based on the orientation of the pole at the
                        specified epoch. If it is unspecified, then the frame
                        could wobble or rotate slowly due to precession of the
                        planet's pole.

            retrograde  True to flip the sign of the Z-axis. Necessary for
                        retrograde systems like Uranus.

            aries       True to use the First Point of Aries as the longitude
                        reference; False to use the ascending node of the ring
                        plane. Note that the former might be preferred in a
                        situation where the ring plane is uncertain, wobbles, or
                        is nearly parallel to the celestial equator. In these
                        situations, using Aries as a reference will reduce the
                        uncertainties related to the pole orientation.

            frame_id    the ID under which the frame will be registered. None to
                        leave the frame unregistered. If the value is "+", then
                        the registered name is the planet frame's name with the
                        suffix "_DESPUN" if epoch is None, or "_INERTIAL" if an
                        epoch is specified.

            cache_size  number of transforms to cache. This can be useful
                        because it avoids unnecessary SPICE calls when the frame
                        is being used repeatedly at a finite set of times.

            unpickled   True if this frame has been read from a pickle file.
        """

        self.planet_frame = Frame.as_frame(frame).wrt(Frame.J2000)
        self.reference = Frame.J2000
        self.epoch = None if epoch is None else Scalar.as_scalar(epoch)
        self.retrograde = bool(retrograde)
        self.shape = Qube.broadcasted_shape(self.planet_frame, self.epoch)
        self.keys = set()

        self.aries = bool(aries)

        # The frame might not be exactly inertial due to polar precession, but
        # it is good enough
        self.origin = None

        # Define cache
        self.cache = {}
        self.trim_size = max(cache_size//10, 1)
        self.given_cache_size = cache_size
        self.cache_size = cache_size + self.trim_size
        self.cache_counter = 0
        self.cached_value_returned = False          # Just used for debugging

        # Fill in the frame ID
        if frame_id is None:
            self.frame_id = Frame.temporary_frame_id()
        elif frame_id == '+':
            if self.epoch is None:
                self.frame_id = self.planet_frame.frame_id + "_DESPUN"
            else:
                self.frame_id = self.planet_frame.frame_id + "_INERTIAL"
        else:
            self.frame_id = frame_id

        # Register if necessary
        self.register(unpickled=unpickled)

        # For a fixed epoch, derive the inertial tranform now
        self.transform = None
        if self.epoch is not None:
            self.transform = self.transform_at_time(self.epoch)

        # Save in internal dict for name lookup upon serialization
        if (not unpickled and self.shape == ()
            and self.frame_id in Frame.WAYFRAME_REGISTRY):
                key = (self.planet_frame.frame_id,
                       None if self.epoch is None else self.epoch.vals,
                       self.retrograde, self.aries)
                RingFrame.FRAME_IDS[key] = self.frame_id

    # Unpickled frames will always have temporary IDs to avoid conflicts
    def __getstate__(self):
        return (Frame.as_primary_frame(self.planet_frame), self.epoch,
                self.retrograde, self.aries, self.given_cache_size, self.shape)

    def __setstate__(self, state):
        # If this frame matches a pre-existing frame, re-use its ID
        (frame, epoch, retrograde, aries, cache_size, shape) = state
        if shape == ():
            key = (frame.frame_id,
                   None if epoch is None else epoch.vals,
                   retrograde, aries)
            frame_id = RingFrame.FRAME_IDS.get(key, None)
        else:
            frame_id = None

        self.__init__(frame, epoch, retrograde, aries, frame_id=frame_id,
                      cache_size=cache_size, unpickled=True)

    #===========================================================================
    def transform_at_time(self, time, quick={}):
        """The Transform into the this Frame at a Scalar of times."""

        # For a fixed epoch, return the fixed transform
        if self.transform is not None:
            return self.transform

        time = Scalar.as_scalar(time)

        # Check cache first if time is a Scalar
        if time.shape == ():
            key = time.values

            if key in self.cache:
                self.cached_value_returned = True
                (count, key, xform) = self.cache[key]
                self.cache_counter += 1
                count[0] = self.cache_counter
                return xform

        self.cached_value_returned = False

        # Otherwise, calculate it for the current time
        xform = self.planet_frame.transform_at_time(time, quick=quick)

        # The bottom row of the matrix is the Z-axis of the ring frame in J2000
        z_axis = xform.matrix.row_vector(2)

        # For a retrograde ring, reverse Z
        if self.retrograde:
            z_axis = -z_axis

        x_axis = Vector3.ZAXIS.cross(z_axis)
        matrix = Matrix3.twovec(z_axis, 2, x_axis, 0)

        # This is the RingFrame matrix. It rotates from J2000 to the frame where
        # the pole at epoch is along the Z-axis and the ascending node relative
        # to the J2000 equator is along the X-axis.

        if self.aries:
            (x,y,z) = x_axis.to_scalars()
            node_lon = y.arctan2(x)
            matrix = Matrix3.z_rotation(node_lon) * matrix

        # Create transform
        xform = Transform(matrix, Vector3.ZERO,
                          self.wayframe, self.reference, None)

        # Cache the transform if necessary
        if time.shape == () and self.given_cache_size > 0:

            # Trim the cache, removing the values used least recently
            if len(self.cache) >= self.cache_size:
                all_keys = self.cache.values()
                all_keys.sort()
                for (_, old_key, _) in all_keys[:self.trim_size]:
                    del self.cache[old_key]

            # Insert into the cache
            key = time.values
            self.cache_counter += 1
            count = np.array([self.cache_counter])
            self.cache[key] = (count, key, xform)

        return xform

    #===========================================================================
    def node_at_time(self, time, quick={}):
        """Angle from the frame's X-axis to the ring plane ascending node on
        the J2000 equator.
        """

        xform = self.transform_at_time(time, quick=quick)
        z_axis_wrt_j2000 = xform.unrotate(Vector3.ZAXIS)
        (x,y,_) = z_axis_wrt_j2000.to_scalars()

        if (x,y) == (0.,0.):
            return Scalar(0.)

        return (y.arctan2(x) + np.pi/2.) % Scalar.TWOPI

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_RingFrame(unittest.TestCase):

    def runTest(self):

        # Imports are here to reduce conflicts
        import os
        import cspyce
        from oops.frame.spiceframe   import SpiceFrame
        from oops.path.spicepath     import SpicePath
        from oops.event              import Event
        from oops.path               import Path
        from oops.unittester_support import TESTDATA_PARENT_DIRECTORY

        np.random.seed(2492)

        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE/naif0009.tls"))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE/pck00010.tpc"))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE/de421.bsp"))

        Path.reset_registry()
        Frame.reset_registry()

        _ = SpicePath("MARS", "SSB")
        planet = SpiceFrame("IAU_MARS", "J2000")
        rings  = RingFrame(planet)
        self.assertEqual(Frame.as_wayframe("IAU_MARS"), planet.wayframe)
        self.assertEqual(Frame.as_wayframe("IAU_MARS_DESPUN"), rings.wayframe)

        time = Scalar(np.random.rand(3,4,2) * 1.e8)
        posvel = np.random.rand(3,4,2,6)
        event = Event(time, (posvel[...,0:3], posvel[...,3:6]), "SSB", "J2000")
        rotated = event.wrt_frame("IAU_MARS")
        fixed   = event.wrt_frame("IAU_MARS_DESPUN")

        # Confirm Z axis is tied to planet's pole
        diff = Scalar(rotated.pos.mvals[...,2]) - Scalar(fixed.pos.mvals[...,2])
        self.assertTrue(np.all(np.abs(diff.values < 1.e-14)))

        # Confirm X-axis is always in the J2000 equator
        xaxis = Event(time, Vector3.XAXIS, "SSB", rings.frame_id)
        test = xaxis.wrt_frame("J2000")
        self.assertTrue(np.all(np.abs(test.pos.mvals[...,2] < 1.e-14)))

        # Confirm it's at the ascending node
        xaxis = Event(time, (1,1.e-13,0), "SSB", rings.frame_id)
        test = xaxis.wrt_frame("J2000")
        self.assertTrue(np.all(test.pos.mvals[...,1] > 0.))

        # Check that pole wanders when epoch is fixed
        rings2 = RingFrame(planet, 0.)
        self.assertEqual(Frame.as_wayframe("IAU_MARS_INERTIAL"), rings2.wayframe)
        inertial = event.wrt_frame("IAU_MARS_INERTIAL")

        diff = Scalar(rotated.pos.mvals[...,2]) - Scalar(inertial.pos.mvals[...,2])
        self.assertTrue(np.all(np.abs(diff.values) < 1.e-4))
        self.assertTrue(np.mean(np.abs(diff.values) > 1.e-8))

        Path.reset_registry()
        Frame.reset_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
