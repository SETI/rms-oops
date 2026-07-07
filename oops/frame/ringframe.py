##########################################################################################
# oops/frame/ringframe.py: Subclass RingFrame of class Frame
##########################################################################################

import numpy as np

from polymath       import Matrix3, Qube, Scalar, Vector3
from oops.cache     import Cache
from oops.fittable  import Fittable_
from oops.frame     import Frame
from oops.transform import Transform


class RingFrame(Frame):
    """A Frame subclass describing a non-rotating frame centered on the Z-axis of another
    frame, but oriented with the X-axis fixed along the ascending node of the equator
    within the reference frame.
    """

    FRAME_IDS = {}

    def __init__(self, frame, epoch=None, *, retrograde=False, aries=False, frame_id='+',
                 cache_size=100):
        """Constructor for a RingFrame Frame.

        Parameters:
            frame (Frame or str): The frame or frame ID describing the central planet of
                the ring plane relative to J2000.
            epoch (Scalar or float): The time TDB at which the frame is to be evaluated.
                If this is specified, then the frame will be precisely inertial, based on
                the orientation of the pole at the specified epoch. If it is unspecified,
                then the frame could wobble or rotate slowly due to precession of the
                planet's pole.
            retrograde (bool, optional): True to flip the sign of the Z-axis. Necessary
                for retrograde systems like Uranus.
            aries (bool, optional): True to use the First Point of Aries as the longitude
                reference; False to use the ascending node of the ring plane. Note that
                the former might be preferred in a situation where the ring plane is
                uncertain, wobbles, or is nearly parallel to the celestial equator. In
                these situations, using Aries as a reference will reduce the uncertainties
                related to the pole orientation.
            frame_id (str, optional): The ID to use; None to leave the frame unregistered.
                If the value is "+", then the registered name is the planet frame's ID
                with "_DESPUN" appended if `epoch` is None or "_INERTIAL" if an epoch is
                specified.
            cache_size (int, optinal): The number of transforms to cache. This can be
                useful because it avoids unnecessary SPICE calls when the frame is being
                used repeatedly at a finite set of times.
        """

        self.planet_frame = Frame.as_frame(frame).wrt(Frame.J2000)
        self.epoch = None if epoch is None else Scalar.as_scalar(epoch)
        self.retrograde = bool(retrograde)
        self.aries = bool(aries)
        self._cache_size = cache_size

        # Required attributes
        self.reference = Frame.J2000
        # The frame might not be exactly inertial due to polar precession, but it is close
        # enough for all practical purposes.
        self.origin = None
        self.shape = Qube.broadcasted_shape(self.planet_frame, self.epoch)

        # Fill in the frame ID
        if frame_id == '+':
            if self.epoch is None:
                self.frame_id = self.planet_frame.frame_id + '_DESPUN'
            else:
                self.frame_id = self.planet_frame.frame_id + '_INERTIAL'
        else:
            self.frame_id = self._recover_id(frame_id)

        # Register if necessary
        self.register()
        self._refresh()
        self._cache_id()

    def _refresh(self):
        self._cache = Cache(self._cache_size)

        # For a fixed epoch, derive the inertial tranform now
        self.transform = None
        if self.epoch is not None:
            self.transform = self.transform_at_time(self.epoch)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def _frame_key(self):
        return (self.planet_frame, self.epoch, self.retrograde, self.aries)

    def __getstate__(self):
        Fittable_.refresh(self)
        self._cache_id()
        return (Frame.as_primary_frame(self.planet_frame), self.epoch, self.retrograde,
                self.aries, self._state_id(), self._cache_size)

    def __setstate__(self, state):
        (frame, epoch, retrograde, aries, frame_id, cache_size) = state
        self.__init__(frame, epoch, retrograde=retrograde, aries=aries,
                      frame_id=frame_id, cache_size=cache_size)
        Fittable_.freeze(self)

    ######################################################################################
    # Frame API
    ######################################################################################

    def transform_at_time(self, time, quick={}):
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

        Notes:
            TwoVector is a fixed frame, so the transform relative to the `reference` frame
            is independent of time.
        """

        # For a fixed epoch, return the fixed transform
        if self.transform is not None:
            return self.transform

        time = Scalar.as_scalar(time)

        # Check cache first if time is a Scalar
        if time.shape == ():
            xform = self._cache[time.vals]
            if xform:
                return xform

        # Otherwise, calculate it for the current time
        xform = self.planet_frame.transform_at_time(time, quick=quick)

        # The bottom row of the matrix is the Z-axis of the ring frame in J2000
        z_axis = xform.matrix.row_vector(2)

        # For a retrograde ring, reverse Z
        if self.retrograde:
            z_axis = -z_axis

        x_axis = Vector3.ZAXIS.cross(z_axis)
        matrix = Matrix3.twovec(z_axis, 2, x_axis, 0)

        # This is the RingFrame matrix. It rotates from J2000 to the frame where the pole
        # at epoch is along the Z-axis and the ascending node relative to the J2000
        # equator is along the X-axis.

        if self.aries:
            (x,y,z) = x_axis.to_scalars()
            node_lon = y.arctan2(x)
            matrix = Matrix3.z_rotation(node_lon) * matrix

        # Create transform
        xform = Transform(matrix, Vector3.ZERO, self.wayframe, self.reference, None)

        # Cache the transform if necessary
        if time.shape == ():
            self._cache[time.vals] = xform

        return xform

    def node_at_time(self, time, quick={}):
        """The vector defining the ascending node of this frame's XY plane relative to
        the XY frame of its reference.

        Parameters:
            time (Scalar, array-like, or float): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            (Vector3): The unit vector pointing in the direction of the ascending node.
        """

        xform = self.transform_at_time(time, quick=quick)
        z_axis_wrt_j2000 = xform.unrotate(Vector3.ZAXIS)
        (x, y, _) = z_axis_wrt_j2000.to_scalars()

        if (x, y) == (0., 0.):
            return Scalar(0.)

        return (y.arctan2(x) + np.pi/2.) % Scalar.TWOPI

##########################################################################################
