##########################################################################################
# oops/frame/cmatrix.py: Subclass Cmatrix of class Frame
##########################################################################################

import numpy as np

from polymath       import Matrix3, Qube, Scalar, Vector3
from oops.frame     import Frame
from oops.transform import Transform
from oops.constants import RPD
import oops.mutable as mutable


class Cmatrix(Frame):
    """Frame subclass in which the frame is defined by a fixed rotation matrix.

    Most commonly, it rotates J2000 coordinates into the frame of a camera, in which the
    Z-axis points along the optic axis, the X-axis points rightward, and the Y-axis points
    downward.
    """

    _WAYFRAMES = {}

    def __init__(self, cmatrix, reference=None, *, frame_id=None):
        """Constructor for a Cmatrix frame.

        Parameters:
            cmatrix (Matrix3): the C matrix.
            reference (Frame or str): Frame or the ID of the Frame relative to which this
                frame is defined; default is J2000.
            frame_id (str, optional): The ID under which to register this Frame; None to
                leave this Frame unregistered.

        Raises:
            ValueError: If `cmatrix` and `reference` cannot be broadcasted to the same
                shape.
        """

        self._cmatrix = Matrix3.as_matrix3(cmatrix).wod.as_readonly()

        self._reference = Frame.as_wayframe(reference)
        self._origin = self._reference._origin
        self._shape = Qube.broadcasted_shape(self._cmatrix, self._reference)

        self._register(frame_id)
        mutable.refresh(self)

    @property
    def transform(self):
        return self._transform

    def _refresh(self):
        self._transform = Transform(self._cmatrix, Vector3.ZERO, self._wayframe,
                                    self._reference)

    def _wayframe_key(self):
        return (self._cmatrix, self._reference)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        mutable.refresh(self)
        return (self._cmatrix, self._reference, self.stripped_id)

    def __setstate__(self, state):
        (cmatrix, reference, frame_id) = state
        self.__init__(cmatrix, reference, frame_id=frame_id)
        mutable.freeze(self)

    ######################################################################################
    # Alternative constructor
    ######################################################################################

    @staticmethod
    def from_ra_dec(ra, dec, clock, reference=None, *, frame_id=None):
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
                leave it unregistered.

        Raises:
            ValueError: If `ra`, `dec`, `twist`, and `reference` cannot all be broadcasted
                to the same shape.
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
                Use False to disable the use of QuickPaths and QuickFrames. Ignored by
                class Cmatrix.

        Returns:
            (Transform): The Tranform applicable at the specified time or times. It
                rotates vectors from the reference frame to this frame.

        Notes:
            Cmatrix is a fixed frame, so the transform relative to the `reference` is
            independent of time. The returned Transform always matches the shape of this
            Frame, regardless of the shape of `time`.
        """

        return self._transform

##########################################################################################

Frame._FRAME_SUBCLASSES.append(Cmatrix)

##########################################################################################
