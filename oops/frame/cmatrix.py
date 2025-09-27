##########################################################################################
# oops/frame/cmatrix.py: Subclass Cmatrix of class Frame
##########################################################################################

import numpy as np

from polymath       import Matrix3, Qube, Scalar, Vector3
from oops.fittable  import Fittable_
from oops.frame     import Frame
from oops.transform import Transform
from oops.constants import RPD


class Cmatrix(Frame):
    """Frame subclass in which the frame is defined by a fixed rotation matrix.

    Most commonly, it rotates J2000 coordinates into the frame of a camera, in
    which the Z-axis points along the optic axis, the X-axis points rightward,
    and the Y-axis points downward.
    """

    _FRAME_IDS = {}

    def __init__(self, cmatrix, reference=None, frame_id=None):
        """Constructor for a Cmatrix frame.

        Parameters:
            cmatrix (Matrix3): the C matrix.
            reference (Frame or str): Frame or Frame ID elative to which this frame is
                defined; None for J2000.
            frame_id (str, optional): The ID under which the frame will be registered;
                None to leave the frame unregistered
        """

        self.cmatrix = Matrix3.as_matrix3(cmatrix)

        # Required attributes
        self.reference = Frame.as_wayframe(reference) or Frame.J2000
        self.origin = self.reference.origin
        self.shape = Qube.broadcasted_shape(self.cmatrix, self.reference)
        self.frame_id = self._recover_id(frame_id)

       # Update wayframe and frame_id; register if not temporary
        self.register()
        self._refresh()
        self._cache_id()

    def _refresh(self):
        self.transform = Transform(self.cmatrix, Vector3.ZERO, self.wayframe,
                                   self.reference)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def _frame_key(self):
        return (self.cmatrix, self.reference)

    def __getstate__(self):
        Fittable_.refresh(self)
        self._cache_id()
        return (self.cmatrix, Frame.as_primary_frame(self.reference), self._state_id())

    def __setstate__(self, state):
        (cmatrix, reference, frame_id) = state
        self.__init__(cmatrix, reference, frame_id=frame_id)
        Fittable_.freeze(self)

    ######################################################################################
    # Alternative constructor
    ######################################################################################

    @staticmethod
    def from_ra_dec(ra, dec, clock, reference=None, frame_id=None):
        """Construct a Cmatrix from RA, dec and celestial north clock angles.

        Parameters:
            ra (Scalar, array-like, or float): The right ascension of the optic axis in
                degrees.
            dec (Scalar, array-like, or float): The declination of the optic axis in
                degrees.
            clock (Scalar, array-like, or float): The angle of celestial north in degrees,
                measured clockwise from the "up" direction in the observation.
            reference (Frame, optional): The reference frame or ID; None for J2000.
            frame_id (str, optional): The ID to use when registering this frame; None to
                leave it unregistered

        Note that this Frame can have an arbitrary shape. This shape is defined by
        broadcasting the shapes of the ra, dec, twist and reference.
        """

        ra = Scalar.as_scalar(ra)
        dec = Scalar.as_scalar(dec)
        clock = Scalar.as_scalar(clock)
        mask = Qube.or_(ra.mask, dec.mask, clock.mask)

        # The transform is fixed so save it now
        ra = RPD * ra.values
        dec = RPD * dec.values
        twist = RPD * (180. - clock.values)

        cosr = np.cos(ra)
        sinr = np.sin(ra)
        cosd = np.cos(dec)
        sind = np.sin(dec)
        cost = np.cos(twist)
        sint = np.sin(twist)
        (cosr, cosd, cost, sinr, sind, sint) = np.broadcast_arrays(cosr, cosd, cost,
                                                                   sinr, sind, sint)

        # Extracted from the PDS Data Dictionary definition, which is appended
        # below
        cmatrix_values = np.empty(cosr.shape + (3,3))
        cmatrix_values[..., 0, 0] = -sinr * cost - cosr * sind * sint
        cmatrix_values[..., 0, 1] =  cosr * cost - sinr * sind * sint
        cmatrix_values[..., 0, 2] =  cosd * sint
        cmatrix_values[..., 1, 0] =  sinr * sint - cosr * sind * cost
        cmatrix_values[..., 1, 1] = -cosr * sint - sinr * sind * cost
        cmatrix_values[..., 1, 2] =  cosd * cost
        cmatrix_values[..., 2, 0] =  cosr * cosd
        cmatrix_values[..., 2, 1] =  sinr * cosd
        cmatrix_values[..., 2, 2] =  sind
        cmatrix = Matrix3(cmatrix_values, mask)

        return Cmatrix(cmatrix, reference, frame_id=frame_id)

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
            Cmatrix is a fixed frame, so the transform relative to the `reference` frame
            is independent of time.
        """

        return self.transform

##########################################################################################
