##########################################################################################
# oops/frame/frameshift.py: Subclass FrameShift of class Frame
##########################################################################################

from polymath      import Scalar
from oops.fittable import Fittable
from oops.frame    import Frame


class FrameShift(Frame, Fittable):
    """A path defined by a time-shift of another frame.

    PLACEHOLDER CODE. "CONCEPTUALLY" CORRECT BUT NOT YET TESTED.
    """

    _FRAME_IDS = {}

    def __init__(self, dt, /, frame, *, frame_id=None):
        """Constructor for a FrameShift.

        Parameters:
            dt (float): The initial time shift in seconds.
            frame (Frame or str): The Framee or ID to which the time shift applies.
            frame_id (str, optional): The new frame ID; None to leave this frame
                unregistered.
        """

        self.dt = dt
        self.frame = frame

        # Required attributes
        self.reference = self.frame.reference
        self.origin = self.reference.origin
        self.shape = self.frame.shape
        self.frame_id = self._recover_id(frame_id)

        self.register()
        self._cache_id()

    ######################################################################################
    # Fittable interface
    ######################################################################################

    def _set_params(self, params):
        self.dt = params[0]

    @property
    def _params(self):
        return (self.dt,)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def _frame_key(self):
        return (self.dt, self.frame)

    def __getstate__(self):
        self.refresh(self)
        self._cache_id()
        return (self.dt, Frame.as_primary_frame(self.frame), self._state_id())

    def __setstate__(self, state):
        (dt, frame, frame_id) = state
        self.__init__(dt, frame, frame_id=frame_id)
        self.freeze()

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
        """

        time = Scalar.as_scalar(time)
        return self.frame.transform_at_time(time + self.dt, quick=quick)

##########################################################################################
