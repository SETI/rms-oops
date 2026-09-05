##########################################################################################
# oops/fov/offsetfov.py
##########################################################################################

from polymath      import Pair
from oops.fittable import Fittable
from oops.fov      import FOV


class OffsetFOV(FOV, Fittable):
    """A field of view whose line of sight is shifted relative to another.

    A subclass of :class:`~oops.FOV` typically used for image navigation and pointing
    corrections.
    """

    def __init__(self, fov, uv_offset=None, xy_offset=None):
        """Constructor for an OffsetFOV.

        Parameters:
            fov (FOV): The FOV object relative to which this FOV is shifted.
            uv_offset (Pair, optional): The offset in *(u,v)* coordinates; the line of
                sight of this FOV falls at ``fov.uv_los - uv_offset``. At most one of
                `uv_offset` and `xy_offset` can be specified; if neither is given, the
                offset is zero.
            xy_offset (Pair, optional): The same offset expressed in *(x,y)* coordinates;
                the *(x,y)* values returned by this FOV are those of `fov` minus
                `xy_offset`.

        Raises:
            ValueError: If both `uv_offset` and `xy_offset` are specified.
        """

        self.fov = fov

        # Deal with alternative inputs:
        if (uv_offset is not None) and (xy_offset is not None):
            raise ValueError('only one of uv_offset and xy_offset can be '
                             + 'specified')

        # Coerce to Pair so that the Fittable interface, which reads
        # uv_offset.vals, works regardless of how the offset was given
        self.uv_offset = None if uv_offset is None else Pair.as_pair(uv_offset)
        self.xy_offset = None if xy_offset is None else Pair.as_pair(xy_offset)

        if self.uv_offset is not None:
            self.xy_offset = self.fov.xy_from_uv(self.uv_offset +
                                                 self.fov.uv_los)
        elif self.xy_offset is not None:
            self.uv_offset = (self.fov.uv_from_xy(self.xy_offset) -
                              self.fov.uv_los)
        else:                                   # default is a (0, 0) offset
            self.uv_offset = Pair.ZEROS
            self.xy_offset = Pair.ZEROS

        # Required attributes of an FOV
        self.uv_shape = self.fov.uv_shape
        self.uv_scale = self.fov.uv_scale
        self.uv_area  = self.fov.uv_area
        self.uv_los   = self.fov.uv_los - self.uv_offset

    ######################################################################################
    # Fittable interface
    ######################################################################################

    nparams = 2

    def _set_params(self, params):
        """Redefine the *(u,v)* offsets of this OffsetFOV."""

        self.uv_offset = Pair.as_pair(params)
        self.xy_offset = self.fov.xy_from_uv(self.uv_offset + self.fov.uv_los)
        self.uv_los = self.fov.uv_los - self.uv_offset

    @property
    def params(self):
        """The fitted parameters, the *(u,v)* offset as a tuple of two floats."""

        return tuple(self.uv_offset.vals)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        self.refresh()

        # Only one of the two offsets can be given to the constructor,
        # which derives the other from it; uv_offset is the one used by
        # the Fittable interface
        return (self.fov, self.uv_offset)

    def __setstate__(self, state):
        self.__init__(*state)
        self.freeze()

    ######################################################################################
    # FOV API
    ######################################################################################

    def xy_from_uvt(self, uv_pair, time=None, *, derivs=False, remask=False, **kwargs):
        """The camera coordinates *(x,y)* at FOV coordinates *(u,v)* and a given time.

        Parameters:
            uv_pair (Pair): *(u,v)* coordinates in this FOV.
            time (Scalar, optional): Absolute time in seconds TDB.
            derivs (bool, optional): If True, any derivatives in *(u,v)* get propagated
                into the returned *(x,y)* coordinates.
            remask (bool, optional): True to mask *(u,v)* coordinates outside the field of
                view; False to leave them unmasked.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Pair: The transformed *(x,y)* coordinates in the camera's frame, with the same
            shape as `uv_pair`.
        """

        uv_pair = Pair.as_pair(uv_pair, recursive=derivs)
        old_xy = self.fov.xy_from_uvt(uv_pair, time=time, derivs=derivs,
                                      remask=remask, **kwargs)
        return old_xy - self.xy_offset

    def uv_from_xyt(self, xy_pair, time=None, *, derivs=False, remask=False, **kwargs):
        """The FOV coordinates *(u,v)* at camera coordinates *(x,y)* and a given time.

        Parameters:
            xy_pair (Pair): *(x,y)* coordinates in this FOV, assuming *z = 1*.
            time (Scalar, optional): Absolute time in seconds TDB.
            derivs (bool, optional): If True, any derivatives in *(x,y)* get propagated
                into the returned *(u,v)* coordinates.
            remask (bool, optional): True to mask *(u,v)* coordinates outside the field of
                view; False to leave them unmasked.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Pair: The computed *(u,v)* FOV coordinates, with the same shape as `xy_pair`.
        """

        xy_pair = Pair.as_pair(xy_pair, recursive=derivs)
        return self.fov.uv_from_xyt(xy_pair + self.xy_offset, time=time,
                                    derivs=derivs, remask=remask, **kwargs)

##########################################################################################
