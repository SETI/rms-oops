##########################################################################################
# oops/frame/poleframe.py: Subclass PoleFrame of class Frame
##########################################################################################

from polymath       import Matrix3, Qube, Scalar, Vector3
from oops.cache     import Cache
from oops.frame     import Frame
from oops.transform import Transform
import oops.mutable as mutable


class PoleFrame(Frame):
    """A Frame subclass describing a non-rotating frame centered on the Z-axis of a body's
    pole vector.

    This differs from RingFrame in that the pole may precess around a separate, invariable
    pole for the system. Because of this behavior, the reference longitude is defined as
    the ascending node of the invariable plane rather than as the ascending node of the
    ring plane. This frame is recommended for Neptune in particular.
    """

    _WAYFRAMES = {}

    def __init__(self, frame, pole, *, retrograde=False, aries=False, frame_id=None,
                 cache_size=100):
        """Constructor for a PoleFrame.

        Input:
            frame (Frame or str): Frame or frame ID for a (possibly) rotating frame
                describing the central planet relative to J2000. This is typically the
                planet's rotating SpiceFrame.
            pole (Vector3 or array-like): The pole of the invariable plane, about which
                planet's pole precesses, in J2000 coordinates. This enables the reference
                longitude to be defined properly.
            retrograde (bool, optional): True to flip the sign of the Z-axis. This is
                necessary for retrograde systems like Uranus.
            aries (bool, optional): True to use the First Point of Aries as the longitude
                reference; False to use the ascending node of the invariable plane. Note
                that the former might be preferred in a situation where the invariable
                pole is uncertain, because small changes in the invariable pole will have
                only a limited effect on the absolute reference longitude.
            frame_id (str, optional): The ID under which to register this Frame; None to
                leave this Frame unregistered. As a special case, use "+" to automatically
                generate an ID by appending "_POLE" to the ID of the `frame` (if it has an
                ID).
            cache_size (int, optional): The number of transforms to cache. This can be
                useful because it avoids unnecessary SPICE calls when the Frame is being
                used repeatedly at a finite set of times.

        Raises:
            ValueError: If `frame` and `pole` cannot be broadcasted to the same shape.
        """

        # Rotates from J2000 to the invariable frame
        pole = Vector3.as_vector3(pole).wod.as_readonly()
        (ra, dec, _) = pole.to_ra_dec_length(recursive=False)
        self._invariable_matrix = Matrix3.pole_rotation(ra, dec)
        # ^Rotates J2000 coordinates into a frame where the Z-axis is the invariable pole
        # and the X-axis is the ascending node of the invariable plane on J2000
        self._invariable_pole = pole
        self._invariable_node = Vector3.ZAXIS.ucross(pole)

        self._aries = bool(aries)
        if self._aries:
            # The ascending node of the invariable plane falls 90 degrees ahead pole's RA
            self._invariable_node_lon = ra + Scalar.HALFPI
        else:
            self._invariable_node_lon = 0.

        self._planet_frame = Frame.as_frame(frame).wrt_j2000
        self._retrograde = bool(retrograde)
        self._cache_size = cache_size

        self._reference = Frame.J2000       # always non-rotating
        self._origin = self._planet_frame._origin
        self._shape = Qube.broadcasted_shape(self._invariable_pole, self._planet_frame)

        if frame_id == '+' and self._planet_frame._frame_id:
            frame_id = self._planet_frame._frame_id + '_POLE'

        self._register(frame_id)
        mutable.refresh(self)

    def _refresh(self):
        self._planet_wrt_j2000 = self._planet_frame.wrt(Frame.J2000)
        self._cache = Cache(self._cache_size)

    def _wayframe_key(self):
        return (self._planet_frame, self._invariable_pole, self._retrograde, self._aries)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        mutable.refresh(self)
        return (self._planet_frame, self._invariable_pole, self._retrograde, self._aries,
                self.stripped_id, self._cache_size)

    def __setstate__(self, state):
        (frame, pole, retrograde, aries, frame_id, cache_size) = state
        self.__init__(frame, pole, retrograde=retrograde, aries=aries, frame_id=frame_id,
                      cache_size=cache_size)
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

        # Check cache first if time is a Scalar
        if time.shape == ():
            xform = self._cache[time.vals]
            if xform:
                return xform

        # Calculate the planet frame for the current time in J2000
        xform = self._planet_wrt_j2000.transform_at_time(time, quick=quick)

        # The bottom row of the matrix is the Z-axis of the ring frame in J2000
        z_axis = xform.matrix.row_vector(2)

        # For a retrograde ring, reverse Z
        if self._retrograde:
            z_axis = -z_axis

        planet_matrix = Matrix3.twovec(z_axis, 2, Vector3.ZAXIS.cross(z_axis), 0)

        # This is the RingFrame matrix. It rotates from J2000 to the frame where the pole
        # at epoch is along the Z-axis and the ascending node relative to the J2000
        # equator is along the X-axis.

        # Locate the J2000 ascending node of the RingFrame on the invariable plane.
        planet_pole_j2000 = planet_matrix.inverse() * Vector3.ZAXIS
        joint_node_j2000 = self._invariable_pole.cross(planet_pole_j2000)

        joint_node_wrt_planet = planet_matrix * joint_node_j2000
        joint_node_wrt_frame = self._invariable_matrix * joint_node_j2000

        node_lon_wrt_planet = joint_node_wrt_planet.to_ra_dec_length()[0]
        node_lon_wrt_frame = joint_node_wrt_frame.to_ra_dec_length()[0]

        # Align the X-axis with the node of the invariable plane
        matrix = Matrix3.z_rotation(node_lon_wrt_planet - node_lon_wrt_frame +
                                    self._invariable_node_lon) * planet_matrix

        # Create the transform
        xform = Transform(Matrix3(matrix, xform.matrix.mask), Vector3.ZERO,
                          self._wayframe, self._reference, self._origin)

        # Cache the transform if necessary
        if time.shape == ():
            self._cache[time.vals] = xform

        return xform

    def node_at_time(self, time, *, quick=None):
        """The vector defining the ascending node of this frame's XY plane relative to
        the XY frame of its reference.

        Parameters:
            time (Scalar, array-like, or float): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            (Vector3): The unit vector pointing in the direction of the ascending node.

        Notes:
            TwoVector is a fixed frame, so its node vector relative to the `reference`
            frame is independent of time.

        Raises:
            ValueError: If the shapes of `time` and this object cannot be broadcasted.
        """

        # Calculate the pole for the current time
        xform = self._planet_wrt_j2000.transform_at_time(time, quick=quick)

        # The bottom row of the matrix is the pole in J2000 coordinates
        z_axis = xform.matrix.row_vector(2)
        if self._retrograde:
            z_axis = -z_axis

        # Locate this pole relative to the invariable plane
        z_axis_wrt_invariable = self._invariable_matrix * z_axis

        # The ascending node is 90 degrees ahead of the pole
        (x, y, _) = z_axis_wrt_invariable.to_scalars()

        node = y.arctan2(x) + Scalar.HALFPI + self._invariable_node_lon
        return node % Scalar.TWOPI

##########################################################################################

Frame._FRAME_SUBCLASSES.append(PoleFrame)

##########################################################################################
