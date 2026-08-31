##########################################################################################
# oops/fov/tdifov.py
##########################################################################################

from polymath import Scalar, Pair
from oops.fov import FOV


class TDIFOV(FOV):
    """FOV subclass to apply TDI timing to another FOV."""

    IS_TIME_INDEPENDENT = False

    def __init__(self, fov, tstop, tdi_texp, tdi_axis):
        """Constructor for a TDIFOV.

        Parameters:
            fov (FOV): The time-independent FOV to which this TDI timing is applied.
            tstop (float): End time of the observation in seconds TDB.
            tdi_texp (float): Time interval between TDI shifts in seconds.
            tdi_axis (str): "u", "+u", "-u", "v", "+v", or "-v", the FOV axis along which
                the "Time Delay and Integration" applies, including the sign of the
                direction.

        Raises:
            ValueError: If `tdi_axis` is not one of the recognized values.
        """

        self.fov = fov
        self.tstop = float(tstop)
        self.tdi_texp = float(tdi_texp)
        self.tdi_axis = tdi_axis
        self.tdi_sign = -1 if '-' in tdi_axis else 1

        # Validation
        if tdi_axis not in ('u', 'v', '-u', '-v', '+u', '+v'):
            raise ValueError('invalid tdi_axis value: ' + repr(tdi_axis))

        # Interpret the axis
        if self.tdi_axis[-1] == 'u':
            self._duv_dshift = Pair((self.tdi_sign, 0))
            self._uv_line_index = 0
        else:
            self._duv_dshift = Pair((0, self.tdi_sign))
            self._uv_line_index = 1

        self._duv_dt = self._duv_dshift / self.tdi_texp

        # Required attributes
        self.uv_los   = self.fov.uv_los
        self.uv_scale = self.fov.uv_scale
        self.uv_shape = self.fov.uv_shape
        self.uv_area  = self.fov.uv_area

    def __getstate__(self):
        self.refresh()
        return (self.fov, self.tstop, self.tdi_texp, self.tdi_axis)

    def __setstate__(self, state):
        self.__init__(*state)
        self.freeze()

    def xy_from_uvt(self, uv_pair, time=None, *, derivs=False, remask=False, **kwargs):
        """The `(x,y)` camera frame coordinates given the FOV coordinates `(u,v)` at the
        specified time.

        Parameters:
            uv_pair (Pair): `(u,v)` coordinates in this FOV.
            time (Scalar): Absolute time in seconds TDB. Required, because a TDIFOV is
                time-dependent.
            derivs (bool, optional): If True, any derivatives in `(u,v)` get propagated
                into the returned `(x,y)` coordinates.
            remask (bool, optional): True to mask `(u,v)` coordinates outside the field of
                view; False to leave them unmasked.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Pair: The transformed `(x,y)` coordinates in the camera's frame, with the same
                shape as uv_pair.
        """

        # Update (u,v) based on the line and the number of TDI stages
        uv = Pair.as_pair(uv_pair, recursive=derivs).copy(recursive=False)
        line = uv.to_scalar(self._uv_line_index, recursive=False)
            # uv and line share memory, so updating line also updates uv.

        # Determine the number of TDI shifts
        time = Scalar.as_scalar(time, recursive=False)
        shifts = -1 - ((time - self.tstop) // self.tdi_texp).as_int()
        shifts[time == self.tstop] = 0

        # Apply the line shift to our copy of uv
        line -= self.tdi_sign * shifts

        # If a time derivative is present, we need to compensate for the TDI
        # readout
        if derivs:
            uv.insert_derivs(uv_pair.derivs.copy())  # copy dict but not derivs
            if 't' in uv.derivs:
                uv.derivs['t'] = uv.derivs['t'] - self._duv_dt

        return self.fov.xy_from_uvt(uv, derivs=derivs, remask=remask,
                                                       **kwargs)

    def uv_from_xyt(self, xy_pair, time=None, *, derivs=False, remask=False, **kwargs):
        """The `(u,v)` FOV coordinates given the `(x,y)` camera frame coordinates at the
        specified time.

        Parameters:
            xy_pair (Pair): `(x,y)` coordinates in this FOV, assuming `z = 1`.
            time (Scalar): Absolute time in seconds TDB. Required, because a TDIFOV is
                time-dependent.
            derivs (bool, optional): If True, any derivatives in `(x,y)` get propagated
                into the returned `(u,v)` coordinates.
            remask (bool, optional): True to mask `(u,v)` coordinates outside the field of
                view; False to leave them unmasked.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Pair: The computed `(u,v)` FOV coordinates, with the same shape as xy_pair.
        """

        # Apply the conversion for tstop
        uv = self.fov.uv_from_xyt(xy_pair, derivs=derivs, remask=remask, **kwargs)

        # Extract the line index from uv, sharing memory
        line = uv.to_scalar(self._uv_line_index, recursive=False)

        # Determine the number of TDI shifts
        time = Scalar.as_scalar(time, recursive=False)
        shifts = -1 - ((time - self.tstop) // self.tdi_texp).as_int()
        shifts[time == self.tstop] = 0

        # Apply the line shift to uv
        line += self.tdi_sign * shifts

        # If a time derivative is present, we need to compensate for the TDI
        # readout
        if 't' in uv.derivs:
            uv.derivs['t'] += self._duv_dt

        return uv

##########################################################################################
