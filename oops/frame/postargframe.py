##########################################################################################
# oops/frame/postargframe.py: Subclass PosTargFrame of class Frame
##########################################################################################

import numpy as np

from polymath       import Matrix3, Vector3
from oops.frame     import Frame
from oops.transform import Transform
import oops.mutable as mutable


class PosTargFrame(Frame):
    """A Frame subclass describing a fixed rotation about the X and Y axes, so the Z-axis
    of another frame falls at a slightly different position in this frame.
    """

    _WAYFRAMES = {}

    def __init__(self, xpos, ypos, reference, *, frame_id=None):
        """Constructor for a PosTarg Frame.

        Parameters:
            xpos (float): The X-position of the reference Frame's Z-axis in the given
                referece frame, in radians.
            ypos (float): The Y-position of the reference Frame's Z-axis in the given
                referece frame, in radians.
            reference (Frame or str): The Frame or the ID of the Frame relative to which
                this Frame is defined.
            frame_id (str, optional): The ID under which to register this Frame; None to
                leave this Frame unregistered. As a special case, use "+" to automatically
                generate a Frame ID by appending "_POSTARG" to the ID of `reference` (if
                it has an ID).
        """

        self._xpos = float(xpos)
        self._ypos = float(ypos)

        cos_x = np.cos(self._xpos)
        sin_x = np.sin(self._xpos)
        cos_y = np.cos(self._ypos)
        sin_y = np.sin(self._ypos)
        xmat = Matrix3([[1.,  0.,    0.   ],
                        [0.,  cos_y, sin_y],
                        [0., -sin_y, cos_y]])
        ymat = Matrix3([[ cos_x, 0., sin_x],
                        [ 0.,    1., 0.   ],
                        [-sin_x, 0., cos_x]])
        self._matrix = ymat * xmat

        self._reference = Frame.as_frame(reference)
        self._origin = self._reference._origin
        self._shape = self._reference._shape

        if frame_id == '+' and self._reference._frame_id:
            frame_id = self._reference._frame_id + '_POSTARG'

        self._register(frame_id)
        mutable.refresh(self)

    def _refresh(self):
        self._transform = Transform(self._matrix, Vector3.ZERO, self, self._reference,
                                    self._origin)

    def _wayframe_key(self):
        return (self._xpos, self._ypos, self._reference)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        mutable.refresh(self)
        return (self._xpos, self._ypos, self._reference, self.stripped_id)

    def __setstate__(self, state):
        (xpos, ypos, reference, frame_id) = state
        self.__init__(xpos, ypos, reference, frame_id=frame_id)
        mutable.freeze(self)

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
                Use False to disable the use of QuickPaths and QuickFrames. Ignored by
                class PosTargFrame.

        Returns:
            (Transform): The Tranform applicable at the specified time or times. It
                rotates vectors from the reference frame to this frame.

        Notes:
            Navigation is a fixed frame, so the transform relative to the `reference`
            frame is independent of time. The returned Transform always has the shape of
            this object, regardless of the shape of `time`.
        """

        return self._transform

##########################################################################################

Frame._FRAME_SUBCLASSES.append(PosTargFrame)

##########################################################################################
