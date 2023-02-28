################################################################################
# oops/frame/laplaceframe.py: Subclass LaplaceFrame of class Frame
################################################################################

import numpy as np
from polymath             import Matrix3, Qube, Scalar, Vector3
from oops.frame           import Frame
from oops.frame.poleframe import PoleFrame
from oops.transform       import Transform

class LaplaceFrame(Frame):
    """A Frame subclass defined by a Kepler Path and a tilt angle.

    The new Z-axis is constructed by rotating the planet's pole by a specified,
    fixed angle toward the pole of the orbit. The rotation occurs around the
    ascending node of the orbit on the orbit's defined reference plane.

    As an example, use the Kepler Path of Triton, which is defined relative to
    Neptune's PoleFrame, to construct tilted Laplace Planes for each of
    Neptune's inner satellites. Note, however, that the tilt angles should be
    negative because Triton is retrograde, and therefore its orbital ascending
    node is the descending node for the orbits of the inner moons.
    """

    FRAME_IDS = {}  # frame_id to use if a frame already exists upon un-pickling

    #===========================================================================
    def __init__(self, orbit, tilt=0., frame_id='+', cache_size=1000,
                       unpickled=False):
        """Constructor for a LaplaceFrame.

        Input:
            orbit       a Kepler Path object.

            tilt        The tilt of the Laplace Plane's pole from the planet's
                        pole toward or beyond the invariable pole.

            frame_id    the ID under which the frame will be registered. None to
                        leave the frame unregistered. If the value is "+", then
                        the registered name is the name of the planet's
                        ring_frame with the suffix "_LAPLACE". Note that this
                        default ID will not be unique if frames are defined for
                        multiple Laplace Planes around the same planet.

            cache_size  number of transforms to cache. This can be useful
                        because it avoids unnecessary SPICE calls when the frame
                        is being used repeatedly at a finite set of times.

            unpickled   True if this frame has been read from a pickle file.
        """

        self.orbit = orbit
        self.planet = self.orbit.planet

        self.orbit_frame = self.orbit.frame.wrt(Frame.J2000)
        self.planet_frame = self.planet.frame.wrt(Frame.J2000)

        self.tilt = Scalar.as_scalar(tilt)
        self.cos_tilt = self.tilt.cos()
        self.sin_tilt = self.tilt.sin()

        self.reference = Frame.J2000
        self.origin = self.orbit.origin
        self.shape = Qube.broadcasted_shape(self.orbit.shape, self.tilt)
        self.keys = set()

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
            self.frame_id = self.orbit.planet.ring_frame.frame_id + '_LAPLACE'
        elif frame_id.startswith('+'):
            self.frame_id = (self.orbit.planet.ring_frame.frame_id + '_'
                             + frame_id[1:])
        else:
            self.frame_id = frame_id

        # Register if necessary
        self.register(unpickled=unpickled)

        # Save in internal dict for name lookup upon serialization
        if (not unpickled and self.shape == ()
            and self.frame_id in Frame.WAYFRAME_REGISTRY):
                key = (self.orbit.path_id, self.tilt.vals)
                LaplaceFrame.FRAME_IDS[key] = self.frame_id

    # Unpickled frames will always have temporary IDs to avoid conflicts
    def __getstate__(self):
        return (Path.as_primary_path(self.orbit),
                self.tilt, self.given_cache_size, self.shape)

    def __setstate__(self, state):
        # If this frame matches a pre-existing frame, re-use its ID
        (orbit, tilt, cache_size, shape) = state
        if shape == ():
            key = (orbit.path_id, tilt.vals)
            frame_id = PoleFrame.FRAME_IDS.get(key, None)
        else:
            frame_id = None

        self.__init__(orbit, tilt, frame_id=frame_id,
                      cache_size=cache_size, unpickled=True)

    #===========================================================================
    def transform_at_time(self, time, quick={}):
        """The Transform into the this Frame at a Scalar of times."""

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

        # All vectors below are in J2000 coordinates

        orbit_ref_xform = self.orbit.frame_wrt_j2000.transform_at_time(time,
                                                                quick=quick)
        orbit_ref_x_axis = orbit_ref_xform.unrotate(Vector3.XAXIS).wod
        orbit_ref_y_axis = orbit_ref_xform.unrotate(Vector3.YAXIS).wod
        orbit_ref_z_axis = orbit_ref_xform.unrotate(Vector3.ZAXIS).wod

        # Locate the node of the orbit on the orbit reference equator
        node_lon = self.orbit.node_at_time(time).wod
        cos_node = np.cos(node_lon)
        sin_node = np.sin(node_lon)
        orbit_node = cos_node * orbit_ref_x_axis + sin_node * orbit_ref_y_axis

        # This vector is 90 degrees behind of the node on the orbit reference
        # equator
        orbit_target = ( sin_node * orbit_ref_x_axis +
                        -cos_node * orbit_ref_y_axis)

        # This is the pole of the orbit
        orbit_pole = (self.orbit.cos_i * orbit_ref_z_axis +
                      self.orbit.sin_i * orbit_target)

        # Get the planet's pole in J2000
        planet_xform = self.planet_frame.transform_at_time(time, quick=quick)
        planet_pole = planet_xform.unrotate(Vector3.ZAXIS).wod

        # This is the vector we tilt toward
        # The projection of the orbit's pole perpendicular to the planet's pole
        tilt_target = orbit_pole.perp(planet_pole).unit()

        # Now, rotation is easy
        laplace_pole = (self.cos_tilt * planet_pole +
                        self.sin_tilt * tilt_target)

        # We still have to be very careful to match up the orbital longitude.

        # Angles are measured...
        # 1. From the reference direction in the orbit's reference frame.
        # 2. Along the equator plane of the orbit's reference frame to the
        #    ascending node of the Laplace plane
        # 3. Then along the Laplace plane

        # This vector is at the intersection of the reference plane and the
        # Laplace plane
        # common_node = orbit_ref_z_axis.cross(laplace_pole)
        # HOWEVER, the two vectors are very close (0.11 degrees apart in the
        # case of Proteus) so this is a very imprecise calculation.
        #
        # Instead, we use the orbital node. This is perpendicular to the Z-axis
        # of the orbit reference frame and it is _nearly_ perpendicular to the
        # Laplace pole; for the Neptune system, the angle is 90.07 degrees. The
        # error arising in the longitude by ignoring tilt of the plane by 0.07
        # degrees will be a factor of ~ cos(0.07 deg) ~ one part in 10^6.

        common_node = orbit_node

        # Create the rotation matrix
        matrix = Matrix3.twovec(laplace_pole, 2, common_node, 0)
        # This matrix rotates coordinates from J2000 to a frame in which the
        # Z-axis is along the Laplace pole and the X-axis is at the common node.

        # Get the longitude of the common node in the orbit reference frame
        common_node_wrt_orbit_ref = orbit_ref_xform.rotate(common_node).wod
        (x, y, _) = common_node_wrt_orbit_ref.to_scalars()
        common_node_lon = y.arctan2(x)

        # Rotate vectors around the Z-axis in the new frame to so that the
        # X-axis falls at this longitude
        matrix = Matrix3.z_rotation(common_node_lon) * matrix

        # Create the transform
        xform = Transform(matrix, Vector3.ZERO, self.wayframe, Frame.J2000,
                                                self.origin)

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

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_LaplaceFrame(unittest.TestCase):

    def runTest(self):

        pass

########################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
