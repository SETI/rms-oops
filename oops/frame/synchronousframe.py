##########################################################################################
# oops/frame/synchronousframe.py: Subclass SynchronousFrame of class Frame
##########################################################################################

from polymath       import Matrix3
from oops.fittable  import Fittable_
from oops.frame     import Frame
from oops.transform import Transform

class SynchronousFrame(Frame):
    """A Frame subclass describing a a body that always keeps the x-axis pointed toward a
    central planet and the y-axis in the negative direction of motion.
    """

    FRAME_IDS = {}

    def __init__(self, body_path, planet_path, frame_id=None):
        """Constructor for a SynchronousFrame.

        Parameters:
            body_path (Path or str): The path or path ID of the body.
            planet_path (Path or str): The path or path ID of the central planet.
            frame_id (str, optional): The ID to use; None to leave the frame unregistered.
        """

        self.body_path = Frame.PATH_CLASS.as_path(body_path)
        self.planet_path = Frame.PATH_CLASS.as_path(planet_path)
        self.path = Frame.PATH_CLASS.wrt(self.planet_path, self.body_path)

        if self.planet_path.shape:
            raise ValueError('SynchronousFrame requires a shapeless body path')

        # Required attributes
        self.reference = Frame.as_wayframe(self.planet_path.frame)
        self.origin = self.planet_path.origin
        self.shape = self.body_path.shape
        self.frame_id = self._recover_id(frame_id)

        # Update wayframe and frame_id; register if not temporary
        self.register()
        self._cache_id()

    ######################################################################################
    # Serialization support
    ######################################################################################

    def _frame_key(self):
        return (self.body_path, self.planet_path)

    def __getstate__(self):
        Fittable_.refresh(self)
        self._cache_id()
        return (Frame.PATH_CLASS.as_primary_path(self.body_path),
                Frame.PATH_CLASS.as_primary_path(self.planet_path), self._state_id())

    def __setstate__(self, state):
        (body_path, planet_path, frame_id) = state
        self.__init__(body_path, planet_path, frame_id=frame_id)
        Fittable_.freeze(self)

    ######################################################################################
    # Frame API
    ######################################################################################

    def transform_at_time(self, time, *, quick=False):
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

        event = self.path.event_at_time(time, quick=quick)
        matrix = Matrix3.twovec(event.pos, 0, event.vel, 1)
        omega = event.pos.cross(event.vel) / event.pos.dot(event.pos)

        return Transform(matrix, omega, self.frame_id, self.reference, self.body_path)

##########################################################################################
