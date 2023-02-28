################################################################################
# oops/frame/postargframe.py: Subclass PosTargFrame of class Frame
################################################################################

import numpy as np
from polymath       import Matrix3, Vector3
from oops.frame     import Frame
from oops.transform import Transform

class PosTargFrame(Frame):
    """A Frame subclass describing a fixed rotation about the X and Y axes, so
    the Z-axis of another frame falls at a slightly different position in this
    frame.
    """

    FRAME_IDS = {}  # frame_id to use if a frame already exists upon un-pickling

    #===========================================================================
    def __init__(self, xpos, ypos, reference, frame_id=None, unpickled=False):
        """Constructor for a PosTarg Frame.

        Input:
            xpos        the X-position of the reference frame's Z-axis in this
                        frame, in radians.
            ypos        the Y-position of the reference frame's Z-axis in this
                        frame, in radians.
            reference   the frame relative to which this frame is defined.
            frame_id    the ID to use; None to leave the frame unregistered.
            unpickled   True if this frame has been read from a pickle file.
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

        mat = ymat * xmat

        self.frame_id  = frame_id
        self.reference = Frame.as_wayframe(reference)
        self.origin    = self.reference.origin
        self.shape     = self.reference.shape
        self.keys      = set()

        # Update wayframe and frame_id; register if not temporary
        self.register(unpickled=unpickled)

        # It needs a wayframe before we can define the transform
        self.transform = Transform(mat, Vector3.ZERO,
                                   self, self.reference, self.origin)

        # Save in internal dict for name lookup upon serialization
        if (not unpickled and self.shape == ()
            and self.frame_id in Frame.WAYFRAME_REGISTRY):
                key = (self.xpos, self.ypos, self.reference.frame_id)
                PosTargFrame.FRAME_IDS[key] = self.frame_id

    # Unpickled frames will always have temporary IDs to avoid conflicts
    def __getstate__(self):
        return (self.xpos, self.ypos,
                Frame.as_primary_frame(self.reference), self.shape)

    def __setstate__(self, state):
        # If this frame matches a pre-existing frame, re-use its ID
        (xpos, ypos, reference, shape) = state
        if shape == ():
            key = (xpos, ypos, reference.frame_id)
            frame_id = PosTargFrame.FRAME_IDS.get(key, None)
        else:
            frame_id = None

        self.__init__(xpos, ypos, reference, frame_id=frame_id, unpickled=True)

    #===========================================================================
    def transform_at_time(self, time, quick={}):
        """The Transform into the this Frame at a Scalar of times."""

        return self.transform

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_PosTargFrame(unittest.TestCase):

    def runTest(self):

        Frame.reset_registry()

        postarg = PosTargFrame(0.0001, 0.0002, "J2000")
        transform = postarg.transform_at_time(0.)
        rotated = transform.rotate(Vector3.ZAXIS)

        self.assertTrue(abs(rotated.vals[0] - 0.0001) < 1.e-8)
        self.assertTrue(abs(rotated.vals[1] - 0.0002) < 1.e-8)

        Frame.reset_registry()

#########################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
