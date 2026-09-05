##########################################################################################
# oops/frame/cmatrix.py
##########################################################################################

import numpy as np

from polymath       import Matrix3, Qube, Scalar, Vector3
from oops.frame     import Frame
from oops.transform import Transform
from oops.constants import RPD


class Cmatrix(Frame):
    """A Frame subclass in which the Frame is defined by a fixed rotation matrix.

    Most commonly, it rotates J2000 coordinates into the frame of a camera, in which the
    *z*-axis points along the optic axis, the *x*-axis points rightward, and the *y*-axis
    points downward.
    """

    _WAYFRAMES = {}

    def __init__(self, cmatrix, reference=None, *, frame_id=None):
        """Constructor for a Cmatrix.

        Parameters:
            cmatrix (Matrix3): The 3x3 rotation matrix that rotates coordinates from
                `reference` into this Frame.
            reference (Frame or str, optional): The Frame or the ID of the Frame relative
                to which this Frame is defined; None for J2000.
            frame_id (str, optional): The ID under which to register this Frame; None to
                leave this Frame unregistered.

        Raises:
            KeyError: If `reference` is an ID string that has not been registered.
            ValueError: If `cmatrix` and `reference` cannot be broadcasted to the same
                shape.
        """

        self._cmatrix = Matrix3.as_matrix3(cmatrix).wod.as_readonly()

        self._reference = Frame.as_wayframe(reference)
        self._origin = self._reference._origin
        self._shape = Qube.broadcasted_shape(self._cmatrix, self._reference)

        self._register(frame_id)
        self.refresh()

    @property
    def transform(self):
        """The fixed Transform from the reference Frame to this Frame."""
        return self._transform

    def _refresh(self):
        self._transform = Transform(self._cmatrix, Vector3.ZERO, self._wayframe,
                                    self._reference)

    def _wayframe_key(self):
        return (self._cmatrix, self._reference)

    def _show(self, level, indent=0):
        skip = indent + 8
        blanks = skip * ' '

        if self._reference == Frame.J2000:
            ref_str = '"J2000"'
        else:
            ref_str = self._reference.show(level-1, skip)

        return (f'Cmatrix({self._cmatrix.mvals},\n'
                f'{blanks}{ref_str})')

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        self.refresh()
        return (self._cmatrix, self._reference, self.stripped_id)

    def __setstate__(self, state):
        (cmatrix, reference, frame_id) = state
        self.__init__(cmatrix, reference, frame_id=frame_id)
        self.freeze()

    ######################################################################################
    # Alternative constructor
    ######################################################################################

    @staticmethod
    def from_ra_dec(ra, dec, clock, reference=None, *, frame_id=None):
        """Construct a Cmatrix from RA, dec, and celestial north clock angles.

        Parameters:
            ra (Scalar): The right ascension of the optic axis in degrees.
            dec (Scalar): The declination of the optic axis in degrees.
            clock (Scalar): The angle of celestial north in degrees, measured clockwise
                from the "up" direction in the observation.
            reference (Frame or str, optional): The Frame or the ID of the Frame
                relative to which this Frame is defined; None for J2000.
            frame_id (str, optional): The ID under which to register this Frame; None to
                leave this Frame unregistered.

        Returns:
            Cmatrix: A Frame defined by the rotation matrix that these angles describe.

        Raises:
            KeyError: If `reference` is an ID string that has not been registered.
            ValueError: If `ra`, `dec`, `clock`, and `reference` cannot all be broadcasted
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

        # Extracted from the PDS Data Dictionary definition
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

    def transform_at_time(self, time, *, quick=False):
        """Transform that rotates coordinates from the reference to this frame.

        If the frame is rotating, then the coordinates being transformed must be given
        relative to the center of rotation.

        Parameters:
            time (Scalar): The time in seconds TDB.
            quick (dict or bool, optional): Ignored by class Cmatrix.

        Returns:
            Transform: Rotates vectors from the reference frame to this frame at the
            specified time.

        Notes:
            A Cmatrix is a fixed Frame, so the Transform relative to the `reference`
            Frame is independent of time. The returned Transform always has the shape of
            this Frame, regardless of the shape of `time`.
        """

        return self._transform

##########################################################################################

Frame._FRAME_SUBCLASSES.append(Cmatrix)

##########################################################################################
