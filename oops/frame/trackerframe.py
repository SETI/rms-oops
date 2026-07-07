##########################################################################################
# oops/frame/trackerframe.py: Subclass TrackerFrame of class Frame
##########################################################################################

from polymath       import Qube, Scalar, Vector3, Matrix3
from oops.cache     import Cache
from oops.fittable  import Fittable_
from oops.frame     import Frame
from oops.transform import Transform


class TrackerFrame(Frame):
    """A Frame subclass that ensures, via a small rotation, that a designated target path
    will remain in a fixed direction.

    The primary use of this frame is for observing moving targets with HST. Normally, HST
    images of the same target, obtained during the same visit and orbit, will have a
    common pointing offset and can be navigated as a group. This is not generally true
    when using the pointing information in the FITS headers, because that pointing refers
    to the start time of each frame rather than the midtime.
    """

    FRAME_IDS = {}

    def __init__(self, frame, target, observer, epoch, frame_id=None, cache_size=100):
        """Constructor for a Tracker Frame.

        Input:
            frame (Frame or str): The frame or frame ID that defines the initial pointing.
            target (Path or str): The target's path or path ID.
            observer (Path or str): The observer's path or path ID.
            epoch (Scalar, array-like, or float): The epoch for which the given frame is
                defined.
            frame_id (str, optional): The ID to use; None to leave the frame unregistered.
            cache_size (int, optional): Number of transforms to cache. This can be useful
                because it avoids unnecessary SPICE calls when the frame is being used
                repeatedly at a finite set of times.
        """

        self.fixed_frame = Frame.as_frame(frame).wrt(Frame.J2000)
        self.target_path = Frame.PATH_CLASS.as_path(target)
        self.observer_path = Frame.PATH_CLASS.as_path(observer)
        self.epoch = Scalar.as_scalar(epoch)
        self._cache_size = cache_size

        # Required attributes
        self.reference = Frame.as_wayframe(self.fixed_frame.reference)
        self.origin = self.fixed_frame.origin
        self.shape = Qube.broadcasted_shape(self.fixed_frame, self.target_path,
                                            self.observer_path, self.epoch)
        self.frame_id = self._recover_id(frame_id)

        # Update wayframe and frame_id; register if not temporary
        self.register()
        self._refresh()
        self._cache_id()

    def _refresh(self):

        # Determine the apparent direction to the target path at epoch
        obs_event = Frame.EVENT_CLASS(self.epoch, Vector3.ZERO, self.observer_path,
                                      Frame.J2000)
        (path_event, obs_event) = self.target_path.photon_to_event(obs_event)
        self._trackpoint = obs_event.neg_arr_ap.unit()

        # Determine the transform at epoch
        fixed_xform = self.fixed_frame.transform_at_time(self.epoch)
        self.reference_xform = Transform(fixed_xform.matrix, Vector3.ZERO, self.wayframe,
                                          self.reference, self.origin)

        # Convert the matrix to three axis vectors
        self.reference_rows = Vector3(self.reference_xform.matrix.vals)

        self._cache = Cache(self._cache_size)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def _frame_key(self):
        return (self.fixed_frame, self.target_path, self.observer_path, self.epoch)

    def __getstate__(self):
        Fittable_.refresh(self)
        self._cache_id()
        return (Frame.as_primary_frame(self.fixed_frame),
                Frame.PATH_CLASS.as_primary_path(self.target_path),
                Frame.PATH_CLASS.as_primary_path(self.observer_path),
                self.epoch, self._state_id(), self._cache_size)

    def __setstate__(self, state):
        (frame, target, observer, epoch, frame_id, cache_size) = state
        self.__init__(frame, target, observer, epoch, frame_id=frame_id,
                      cache_size=cache_size)
        Fittable_.freeze(self)

    ######################################################################################
    # Frame API
    ######################################################################################

    def transform_at_time(self, time, quick=False):
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
        """

        time = Scalar.as_scalar(time)

        # Check cache first if time is shapeless
        if time.shape == ():
            xform = self._cache[time.vals]
            if xform:
                return xform

        # Determine the needed rotation
        obs_event = Frame.EVENT_CLASS(time, Vector3.ZERO, self.observer_path, Frame.J2000)
        (path_event, obs_event) = self.target_path.photon_to_event(obs_event)
        newpoint = obs_event.neg_arr_ap.unit()

        rotation = self._trackpoint.cross(newpoint)
        rotation = rotation.reshape(rotation.shape + (1,))

        # Rotate the three axis vectors accordingly
        new_rows = self.reference_rows.spin(rotation)
        xform = Transform(Matrix3(new_rows.vals),
                          Vector3.ZERO,     # neglect the slow frame rotation
                          self.wayframe, self.reference, self.origin)

        # Cache the transform if necessary
        if time.shape == ():
            self._cache[time.vals] = xform

        return xform

##########################################################################################
