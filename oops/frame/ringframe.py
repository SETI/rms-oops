##########################################################################################
# oops/frame/ringframe.py: Subclass RingFrame of class Frame
##########################################################################################

from polymath       import Matrix3, Qube, Scalar, Vector3
from oops.cache     import Cache
from oops.frame     import Frame
from oops.transform import Transform
import oops.mutable as mutable


class RingFrame(Frame):
    """A Frame subclass describing a non-rotating frame centered on the Z-axis of another
    frame, but oriented with the X-axis fixed along the ascending node of the equator
    within the reference frame.
    """

    _WAYFRAMES = {}

    def __init__(self, frame, epoch=None, *, aries=False, retrograde=False, frame_id=None,
                 cache_size=100):
        """Constructor for a RingFrame Frame.

        Parameters:
            frame (Frame or str): The frame or frame ID describing the central planet of
                the ring plane relative to J2000.
            epoch (Scalar or float, optional): The time TDB at which this Frame is to be
                evaluated. If this is specified, then the Frame will be precisely
                inertial, based on the orientation of the pole at the specified epoch. If
                it is unspecified, then the Frame could wobble and/or rotate slowly due to
                precession of the planet's pole.
            aries (bool, optional): True to use the First Point of Aries as the longitude
                reference; False to use the ascending node of the ring plane. Note that
                the former might be preferred in a situation where the ring plane is
                uncertain, wobbles, or is nearly parallel to the celestial equator. In
                these situations, using Aries as a reference will reduce the uncertainties
                related to the pole orientation.
            retrograde (bool, optional): True to flip the sign of the Z-axis. Necessary
                for retrograde systems like Uranus.
            frame_id (str, optional): The ID under which to register this Frame; None to
                leave this Frame unregistered. As a special case, use "+" to automatically
                generate a Frame ID by appending "_DESPUN" (if `epoch` is None) or
                "_INERTIAL" (if `epoch` is specified) to the ID of `frame` (if it has an
                ID).
            cache_size (int, optinal): The number of transforms to cache. This can be
                useful because it avoids unnecessary SPICE calls when the Frame is being
                used repeatedly at a finite set of times.

        Raises:
            ValueError: If `frame` and `epoch` cannot be broadcasted to the same shape.
        """

        self._planet_frame = Frame.as_frame(frame)
        self._epoch = epoch and Scalar.as_scalar(epoch).wod.as_readonly()
        self._retrograde = bool(retrograde)
        self._aries = bool(aries)
        self._cache_size = cache_size

        self._reference = Frame.J2000
        self._is_inertial = self._epoch is not None
        self._origin = self._planet_frame._origin if self._epoch is None else None
        self._shape = Qube.broadcasted_shape(self._planet_frame, self._epoch)

        if frame_id == '+' and self._planet_frame._frame_id:
            if self._is_inertial:
                frame_id = self._planet_frame._frame_id + '_INERTIAL'
            else:
                frame_id = self._planet_frame._frame_id + '_DESPUN'

        self._register(frame_id)
        mutable.refresh(self)

    def _refresh(self):
        self._planet_wrt_j2000 = self._planet_frame.wrt(Frame.J2000)
        self._cache = Cache(self._cache_size)
        self._transform = None
        self._node = None

        # For a fixed epoch, derive the inertial tranform now
        if self._is_inertial:
            transform = self.transform_at_time(self._epoch)
            self._transform = transform

            z_axis_wrt_j2000 = transform.unrotate(Vector3.ZAXIS)
            (x, y, _) = z_axis_wrt_j2000.to_scalars()
            if (x, y) == (0., 0.):
                self._node = Scalar(0.)
            else:
                self._node = (y.arctan2(x) + Scalar.HALFPI) % Scalar.TWOPI

    def _wayframe_key(self):
        return (self._planet_frame, self._epoch, self._retrograde, self._aries)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        mutable.refresh(self)
        return (self._planet_frame, self._epoch, self._retrograde,  self._aries,
                self.stripped_id, self._cache_size)

    def __setstate__(self, state):
        (frame, epoch, retrograde, aries, frame_id, cache_size) = state
        self.__init__(frame, epoch, retrograde=retrograde, aries=aries,
                      frame_id=frame_id, cache_size=cache_size)
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

        Notes:
            If the `epoch` defined for this Frame is None, then the returned Transform is
            independent of time. In this case, the returned Transform always has the shape
            of this object, regardless of the shape of `time`.
        """

        # For a fixed epoch, return the fixed transform
        if self._transform is not None:
            return self._transform

        time = Scalar.as_scalar(time)

        # Check cache first if time is a Scalar
        if time.shape == ():
            transform = self._cache[time.vals]
            if transform:
                return transform

        # Otherwise, calculate it for the current time
        transform = self._planet_wrt_j2000.transform_at_time(time, quick=quick)

        # The bottom row of the matrix is the Z-axis of the ring frame in J2000
        z_axis = transform.matrix.row_vector(2)

        # For a retrograde ring, reverse Z
        if self._retrograde:
            z_axis = -z_axis

        x_axis = Vector3.ZAXIS.cross(z_axis)
        matrix = Matrix3.twovec(z_axis, 2, x_axis, 0)

        # This is the RingFrame matrix. It rotates from J2000 to the frame where the pole
        # at epoch is along the Z-axis and the ascending node relative to the J2000
        # equator is along the X-axis.

        if self._aries:
            (x,y,z) = x_axis.to_scalars()
            node_lon = y.arctan2(x)
            matrix = Matrix3.z_rotation(node_lon) * matrix

        # Create transform
        transform = Transform(matrix, Vector3.ZERO, self._wayframe, self._reference, None)

        # Cache the transform if necessary
        if time.shape == ():
            self._cache[time.vals] = transform

        return transform

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

        Raises:
            ValueError: If the shapes of `time` and this object cannot be broadcasted.

        Notes:
            If the `epoch` defined for this Frame is None, then the returned node is
            independent of time. In this case, it has the shape of this object, regardless
            of the shape of `time`.
        """

        if self._is_inertial:
            return self._node

        transform = self.transform_at_time(time, quick=quick)
        z_axis_wrt_j2000 = transform.unrotate(Vector3.ZAXIS)
        (x, y, _) = z_axis_wrt_j2000.to_scalars()

        if (x, y) == (0., 0.):
            return Scalar(0.)

        return (y.arctan2(x) + Scalar.HALFPI) % Scalar.TWOPI

##########################################################################################

Frame._FRAME_SUBCLASSES.append(RingFrame)

##########################################################################################
