##########################################################################################
# oops/frame/trackerframe.py
##########################################################################################

from polymath       import Qube, Scalar, Vector3, Matrix3
from oops.cache     import Cache
from oops.frame     import Frame
from oops.transform import Transform


class TrackerFrame(Frame):
    """A Frame that holds a designated target path in a fixed direction.

    The frame applies a small rotation to another frame to achieve this.

    The primary use of this frame is for observing moving targets with HST. Normally, HST
    images of the same target, obtained during the same visit and orbit, will have a
    common pointing offset and can be navigated as a group. This is not generally true
    when using the pointing information in the FITS headers, because that pointing refers
    to the start time of each frame rather than the midtime.
    """

    _WAYFRAMES = {}
    _USE_QUICKFRAMES = True     # always a slowly-changing frame

    def __init__(self, frame, target, observer, epoch, *, frame_id=None,
                 cache_size=100):
        """Constructor for a TrackerFrame.

        Parameters:
            frame (Frame or str): The Frame or the ID of the Frame to be modified so that
                `target` holds a fixed direction.
            target (Path or str): The Path or the ID of the Path of the moving target.
            observer (Path or str): The Path or the ID of the Path of the observer.
            epoch (Scalar, array-like, or float): The time in seconds TDB at which the
                direction to `target` is to be held fixed.
            frame_id (str, optional): The ID under which to register this Frame; None to
                leave this Frame unregistered. As a special case, use "+" to automatically
                generate a Frame ID by appending "_TRACKER" to the ID of `frame` (if it
                has an ID).
            cache_size (int, optional): The number of transforms to cache. This can be
                useful because it avoids unnecessary SPICE calls when the Frame is being
                used repeatedly at a finite set of times.

        Raises:
            KeyError: If `frame`, `target`, or `observer` is an ID string that has not
                been registered.
            ValueError: If `frame`, `target`, `observer`, and `epoch` cannot be
                broadcasted to the same shape.
        """

        self._fixed_frame = Frame.as_frame(frame)
        self._target_path = Frame._Path.as_waypoint(target)
        self._observer_path = Frame._Path.as_waypoint(observer)
        self._epoch = Scalar.as_scalar(epoch).wod.as_readonly()
        self._cache_size = cache_size

        self._reference = Frame.J2000
        self._origin = self._fixed_frame._origin
        self._shape = Qube.broadcasted_shape(self._fixed_frame, self._target_path,
                                             self._observer_path, self._epoch)

        if frame_id == '+' and self._fixed_frame._frame_id:
            frame_id = self._fixed_frame._frame_id + '_TRACKER'

        self._register(frame_id)
        self.refresh()

    def _refresh(self):

        # Determine the apparent direction to the target path at epoch
        self._target_wrt_ssb = self._target_path.wrt_ssb
        obs_event = Frame._Event(self._epoch, Vector3.ZERO, self._observer_path,
                                 Frame.J2000)
        (_, obs_event) = self._target_wrt_ssb.photon_to_event(obs_event)
        self._trackpoint = obs_event.neg_arr_ap.unit()

        # Determine the transform at epoch
        fixed_xform = self._fixed_frame.wrt_j2000.transform_at_time(self._epoch)
        self._reference_xform = Transform(fixed_xform.matrix, Vector3.ZERO,
                                          self._wayframe, self._reference,
                                          origin=self._origin)

        # Convert the matrix to three axis vectors
        self._reference_rows = Vector3(self._reference_xform.matrix.vals)

        self._cache = Cache(self._cache_size)

    def _wayframe_key(self):
        return (self._fixed_frame, self._target_path, self._observer_path, self._epoch)

    def _show(self, level, indent=0):
        name = type(self).__name__
        skip = indent + len(name) + 1
        blanks = skip * ' '

        return (f'{name}(frame = {self._fixed_frame.show(level-1, skip)},\n'
                f'{blanks}target = {self._target_path.show(level-1, skip)},\n'
                f'{blanks}observer = {self._observer_path.show(level-1, skip)},\n'
                f'{blanks}epoch = {self._epoch})')

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        self.refresh()
        return (self._fixed_frame, self._target_path, self._observer_path, self._epoch,
                self.stripped_id, self._cache_size)

    def __setstate__(self, state):
        (frame, target, observer, epoch, frame_id, cache_size) = state
        self.__init__(frame, target, observer, epoch, frame_id=frame_id,
                      cache_size=cache_size)
        self.freeze()

    ######################################################################################
    # Frame API
    ######################################################################################

    def transform_at_time(self, time, *, quick=None):
        """Transform that rotates coordinates from the reference to this frame.

        If the frame is rotating, then the coordinates being transformed must be given
        relative to the center of rotation.

        Parameters:
            time (Scalar): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            Transform: Rotates vectors from the reference frame to this frame at the
            specified time.

        Raises:
            ValueError: If the shapes of `time` and this object cannot be broadcasted.
        """

        time = Scalar.as_scalar(time)

        # Check cache first if time is shapeless
        if time.shape == ():
            xform = self._cache[time.vals]
            if xform:
                return xform

        # Determine the needed rotation
        obs_event = Frame._Event(time, Vector3.ZERO, self._observer_path, Frame.J2000)
        (_, obs_event) = self._target_wrt_ssb.photon_to_event(obs_event, quick=quick)
        newpoint = obs_event.neg_arr_ap.unit()

        rotation = self._trackpoint.cross(newpoint)
        rotation = rotation.reshape(rotation.shape + (1,))

        # Rotate the three axis vectors accordingly
        new_rows = self._reference_rows.spin(rotation)
        xform = Transform(Matrix3(new_rows.vals),
                          Vector3.ZERO,     # neglect the slow frame rotation
                          self._wayframe, self._reference, origin=self._origin)

        # Cache the transform if necessary
        if time.shape == ():
            self._cache[time.vals] = xform

        return xform

##########################################################################################

Frame._FRAME_SUBCLASSES.append(TrackerFrame)

##########################################################################################
