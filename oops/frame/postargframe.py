##########################################################################################
# oops/frame/postargframe.py: Subclass PosTargFrame of class Frame
##########################################################################################

import numpy as np

from polymath       import Matrix3, Vector3
from oops.fittable  import Fittable_
from oops.frame     import Frame
from oops.transform import Transform


class PosTargFrame(Frame):
    """A Frame subclass describing a fixed rotation about the X and Y axes, so the Z-axis
    of another frame falls at a slightly different position in this frame.
    """

    _FRAME_IDS = {}

    def __init__(self, xpos, ypos, reference, *, frame_id=None):
        """Constructor for a PosTarg Frame.

        Parameters:
            xpos (float): The X-position of the reference frame's Z-axis in this frame, in
                radians.
            ypos (float): The Y-position of the reference frame's Z-axis in this frame, in
                radians.
            reference (Frame or str): The frame or frame ID relative to which this frame
                is defined.
            frame_id (str, optional): The ID to use; None to leave the frame unregistered.
        """

        self.xpos = float(xpos)
        self.ypos = float(ypos)

        cos_x = np.cos(self.xpos)
        sin_x = np.sin(self.xpos)
        cos_y = np.cos(self.ypos)
        sin_y = np.sin(self.ypos)
        xmat = Matrix3([[1.,  0.,    0.   ],
                        [0.,  cos_y, sin_y],
                        [0., -sin_y, cos_y]])
        ymat = Matrix3([[ cos_x, 0., sin_x],
                        [ 0.,    1., 0.   ],
                        [-sin_x, 0., cos_x]])
        self._matrix = ymat * xmat

        self.reference = Frame.as_wayframe(reference)
        self.origin = self.reference.origin
        self.shape = self.reference.shape
        self.frame_id = self._recover_id(frame_id)

        self.register()
        self._cache_id()

        self.transform = Transform(self._matrix, Vector3.ZERO, self, self.reference,
                                   self.origin)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def _frame_key(self):
        return (self.xpos, self.ypos, self.reference)

    def __getstate__(self):
        self._cache_id()
        return (self.xpos, self.ypos, Frame.as_primary_frame(self.reference),
                self._state_id())

    def __setstate__(self, state):
        (xpos, ypos, reference, frame_id) = state
        self.__init__(xpos, ypos, reference, frame_id=frame_id)
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

        Notes:
            Navigation is a fixed frame, so the transform relative to the `reference`
            frame is independent of time.
        """

        return self.transform

##########################################################################################
