##########################################################################################
# oops/frame/laplaceframe.py: Subclass LaplaceFrame of class Frame
##########################################################################################

import numpy as np

from polymath       import Matrix3, Qube, Scalar, Vector3
from oops.cache     import Cache
from oops.frame     import Frame
from oops.transform import Transform
import oops.mutable as mutable


class LaplaceFrame(Frame):
    """A Frame subclass defined by a Kepler Path and a tilt angle.

    The new Z-axis is constructed by rotating the planet's pole by a specified, fixed
    angle toward the pole of the orbit. The rotation occurs around the ascending node of
    the orbit on the orbit's defined reference plane.

    As an example, use the Kepler Path of Triton, which is defined relative to Neptune's
    PoleFrame, to construct tilted Laplace Planes for each of Neptune's inner satellites.
    Note, however, that the tilt angles should be negative because Triton is retrograde,
    and therefore its orbital ascending node is the descending node for the orbits of the
    inner moons.
    """

    _WAYFRAMES = {}

    def __init__(self, orbit, tilt=0., *, frame_id=None, cache_size=100):
        """Constructor for a LaplaceFrame.

        Parameters:
            orbit (KeplerPath): The orbit of the body for which a Laplace Plane is needed.
            tilt (Scalar, array-like, or float): The tilt of the Laplace Plane's pole from
                the planet's pole toward or beyond the invariable pole.
            frame_id (str, optional): The ID under which to register this Frame; None to
                leave this Frame unregistered. As a special case, use "+" to automatically
                generate a Frame ID by appending "_LAPLACE" to the Path ID of `orbit` (if
                it has an ID).
            cache_size (int, optional): Number of transforms to cache. This can be useful
                because it avoids unnecessary SPICE calls when the Frame is being used
                repeatedly at a limited set of times.

        Raises:
            ValueError: If `orbit` and `tilt` cannot be broadcasted to the same shape.
        """

        self._orbit = Frame._Path.as_path(orbit)
        self._planet = self._orbit._planet
        self._tilt = Scalar.as_scalar(tilt).wod.as_readonly()
        self._cos_tilt = self._tilt.cos()
        self._sin_tilt = self._tilt.sin()

        self._cache_size = cache_size

        self._reference = Frame.J2000
        self._origin = self._orbit._origin
        self._shape = Qube.broadcasted_shape(self._orbit._shape, self._tilt)

        if frame_id == '+' and self._orbit._path_id:
            frame_id = self._orbit._path_id + '_LAPLACE'

        self._register(frame_id)
        mutable.refresh(self)

    def _refresh(self):
        self._orbit_wrt_j2000 = self._orbit._frame.wrt_j2000
        self._planet_wrt_j2000 = self._planet._frame.wrt_j2000
        self._cache = Cache(self._cache_size)

    def _wayframe_key(self):
        return (self._orbit, self._tilt)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        mutable.refresh(self)
        return (self._orbit, self._tilt, self._stripped_id, self._cache_size)

    def __setstate__(self, state):
        (orbit, tilt, frame_id, cache_size) = state
        self.__init__(orbit, tilt, frame_id=frame_id, cache_size=cache_size)
        mutable.freeze(self)

    ######################################################################################
    # Frame API
    ######################################################################################

    def transform_at_time(self, time, *, quick=None):
        """Transform that rotates coordinates from the reference frame to this frame.

        If the frame is rotating, then the coordinates being transformed must be given
        relative to the center of rotation.

        Parameters:
            time (Scalar, array-like, or float): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            (Transform): The Tranform applicable at the specified time or times. It
                rotates vectors from the reference frame to this frame.

        Raises:
            ValueError: If the shapes of `time` and this object cannot be broadcasted.
        """

        time = Scalar.as_scalar(time)

        # Check cache first if time is shapeless
        if time.shape == ():
            xform = self._cache[time.vals]
            if xform:
                return xform

        # All vectors below are in J2000 coordinates

        orbit_ref_xform = self._orbit_wrt_j2000.transform_at_time(time, quick=quick).wod
        orbit_ref_x_axis = orbit_ref_xform.unrotate(Vector3.XAXIS)
        orbit_ref_y_axis = orbit_ref_xform.unrotate(Vector3.YAXIS)
        orbit_ref_z_axis = orbit_ref_xform.unrotate(Vector3.ZAXIS)

        # Locate the node of the orbit on the orbit reference equator
        node_lon = self._orbit.node_at_time(time).wod
        cos_node = np.cos(node_lon)
        sin_node = np.sin(node_lon)
        orbit_node = cos_node * orbit_ref_x_axis + sin_node * orbit_ref_y_axis

        # This vector is 90 degrees behind of the node on the orbit reference equator
        orbit_target = sin_node * orbit_ref_x_axis - cos_node * orbit_ref_y_axis

        # This is the pole of the orbit
        orbit_pole = (self._orbit._cos_i * orbit_ref_z_axis +
                      self._orbit._sin_i * orbit_target)

        # Get the planet's pole in J2000
        planet_xform = self._planet_wrt_j2000.transform_at_time(time, quick=quick).wod
        planet_pole = planet_xform.unrotate(Vector3.ZAXIS)

        # This is the vector we tilt toward
        # The projection of the orbit's pole perpendicular to the planet's pole
        tilt_target = orbit_pole.perp(planet_pole).unit()

        # Now, rotation is easy
        laplace_pole = self._cos_tilt * planet_pole + self._sin_tilt * tilt_target

        # We still have to be very careful to match up the orbital longitude.

        # Angles are measured...
        # 1. From the reference direction in the orbit's reference frame.
        # 2. Along the equator plane of the orbit's reference frame to the ascending node
        #    of the Laplace plane.
        # 3. Then along the Laplace plane.

        # This vector is at the intersection of the reference plane and the Laplace plane
        #     common_node = orbit_ref_z_axis.cross(laplace_pole)
        # HOWEVER, the two vectors are very close (0.11 degrees apart in the case of
        # Proteus) so this is a very imprecise calculation.
        #
        # Instead, we use the orbital node. This is perpendicular to the Z-axis of the
        # orbit reference frame and it is _nearly_ perpendicular to the Laplace pole; for
        # the Neptune system, the angle is 90.07 degrees. The error arising in the
        # longitude by ignoring tilt of the plane by 0.07 degrees will be a factor of
        # ~ cos(0.07 deg) ~ one part in 10^6.

        common_node = orbit_node

        # Create the rotation matrix
        matrix = Matrix3.twovec(laplace_pole, 2, common_node, 0)
        # This matrix rotates coordinates from J2000 to a frame in which the Z-axis is
        # along the Laplace pole and the X-axis is at the common node.

        # Get the longitude of the common node in the orbit reference frame
        common_node_wrt_orbit_ref = orbit_ref_xform.rotate(common_node)
        (x, y, _) = common_node_wrt_orbit_ref.to_scalars()
        common_node_lon = y.arctan2(x)

        # Rotate vectors around the Z-axis in the new frame to so that the X-axis falls at
        # this longitude
        matrix = Matrix3.z_rotation(common_node_lon) * matrix

        # Create the transform
        xform = Transform(matrix, Vector3.ZERO, self._wayframe, Frame.J2000, self._origin)

        # Cache the transform if necessary
        if time.shape == ():
            self._cache[time.vals] = xform

        return xform

##########################################################################################

Frame._FRAME_SUBCLASSES.append(LaplaceFrame)

##########################################################################################
