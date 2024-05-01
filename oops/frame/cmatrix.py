################################################################################
# oops/frame/cmatrix.py: Subclass Cmatrix of class Frame
################################################################################

import numpy as np

from polymath       import Matrix3, Qube, Scalar, Vector3
from oops.frame     import Frame
from oops.transform import Transform
from oops.constants import RPD

class Cmatrix(Frame):
    """Frame subclass in which the frame is defined by a fixed rotation matrix.

    Most commonly, it rotates J2000 coordinates into the frame of a camera, in
    which the Z-axis points along the optic axis, the X-axis points rightward,
    and the Y-axis points downward.
    """

    # Note: Navigation frames are not generally re-used, so their IDs are
    # expendable. Frame IDs are not preserved during pickling.

    #===========================================================================
    def __init__(self, cmatrix, reference=None, frame_id=None):
        """Constructor for a Cmatrix frame.

        Input:
            cmatrix     a Matrix3 object.
            reference   the ID or frame relative to which this frame is defined;
                        None for J2000.
            frame_id    the ID under which the frame will be registered; None
                        to leave the frame unregistered
        """

        self.cmatrix = Matrix3.as_matrix3(cmatrix)

        # Required attributes
        self.frame_id  = frame_id
        self.reference = Frame.as_wayframe(reference) or Frame.J2000
        self.origin    = self.reference.origin
        self.shape     = Qube.broadcasted_shape(self.cmatrix, self.reference)
        self.keys      = set()

        # Update wayframe and frame_id; register if not temporary
        self.register()

        # It needs a wayframe before we can construct the transform
        self.transform = Transform(cmatrix, Vector3.ZERO,
                                   self.wayframe, self.reference)

    # Unpickled frames will always have temporary IDs to avoid conflicts
    def __getstate__(self):
        return (self.cmatrix, Frame.as_primary_frame(self.reference))

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    @staticmethod
    def from_ra_dec(ra, dec, clock, reference=None, frame_id=None):
        """Construct a Cmatrix from RA, dec and celestial north clock angles.

        Input:
            ra          a Scalar defining the right ascension of the optic axis
                        in degrees.
            dec         a Scalar defining the declination of the optic axis in
                        degrees.
            clock       a Scalar defining the angle of celestial north in
                        degrees, measured clockwise from the "up" direction in
                        the observation.
            reference   the reference frame or ID; None for J2000.
            frame_id    the ID to use when registering this frame; None to leave
                        it unregistered

        Note that this Frame can have an arbitrary shape. This shape is defined
        by broadcasting the shapes of the ra, dec, twist and reference.
        """

        ra    = Scalar.as_scalar(ra)
        dec   = Scalar.as_scalar(dec)
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

        (cosr, cosd, cost,
         sinr, sind, sint) = np.broadcast_arrays(cosr, cosd, cost,
                                                 sinr, sind, sint)

        # Extracted from the PDS Data Dictionary definition, which is appended
        # below
        cmatrix_values = np.empty(cosr.shape + (3,3))
        cmatrix_values[...,0,0] = -sinr * cost - cosr * sind * sint
        cmatrix_values[...,0,1] =  cosr * cost - sinr * sind * sint
        cmatrix_values[...,0,2] =  cosd * sint
        cmatrix_values[...,1,0] =  sinr * sint - cosr * sind * cost
        cmatrix_values[...,1,1] = -cosr * sint - sinr * sind * cost
        cmatrix_values[...,1,2] =  cosd * cost
        cmatrix_values[...,2,0] =  cosr * cosd
        cmatrix_values[...,2,1] =  sinr * cosd
        cmatrix_values[...,2,2] =  sind

        return Cmatrix(Matrix3(cmatrix_values,mask), reference, frame_id)

    #===========================================================================
    def transform_at_time(self, time, quick=False):
        """Transform into this Frame at a Scalar of times."""

        return self.transform

################################################################################
