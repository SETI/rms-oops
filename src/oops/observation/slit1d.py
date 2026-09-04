##########################################################################################
# oops/observation/slit1d.py
##########################################################################################

import numpy as np

from polymath                 import Scalar, Pair
from oops.observation         import Observation
from oops.cadence             import Cadence
from oops.cadence.snapcadence import SnapCadence
from oops.frame               import Frame
from oops.path                import Path


class Slit1D(Observation):
    """A 1-D slit measurement with no time-dependence.

    A subclass of :class:`~oops.Observation` that may still have additional axes, such as
    bands.
    """

    def __init__(self, axes, tstart, texp, fov, path, frame, **subfields):
        """Constructor for a Slit1D observation.

        Parameters:
            axes (list or tuple): Strings, with one value for each axis in the associated
                data array. A value of 'u' should appear at the location of the array's
                *u*-axis if any; 'v' should appear at the location of the array's *v*-axis
                if any. Only one of 'u' or 'v' can appear in a Slit1D.
            tstart (float): The start time of the observation in seconds TDB.
                Alternatively, a Cadence object with shape (1,) defining `tstart` and
                `texp`.
            texp (float): Exposure duration of the observation in seconds. Ignored if
                `tstart` is specified as a Cadence.
            fov (FOV): (field-of-view) object, which describes the field of view including
                any spatial distortion. It maps between spatial coordinates *(u,v)* and
                instrument coordinates *(x,y)*. For a Slit1D object, one of the axes of
                the FOV must have length 1.
            path (Path): The path waypoint co-located with the instrument.
            frame (Frame): The wayframe of a coordinate frame fixed to the optics of the
                instrument. This frame should have its *z*-axis pointing outward near the
                center of the line of sight, with the *x*-axis pointing rightward and the
                *y*-axis pointing downward.
            subfields (dict): All of the optional attributes. Additional subfields may be
                included as needed.

        Raises:
            ValueError: If `axes` does not contain exactly one of 'u' and 'v', if the
                cross-slit axis of the FOV does not have length 1, or if `tstart` is a
                Cadence whose shape is not (1,).
        """

        # Basic properties
        self.path = Path.as_waypoint(path)
        self.frame = Frame.as_wayframe(frame)

        # FOV
        self.fov = fov
        self.uv_shape = tuple(self.fov.uv_shape.vals)

        # Axes / Shape / Size
        self._axes = list(axes)
        if ('u' in self._axes) == ('v' in self._axes):
            raise ValueError('axes are incompatible with Slit1D: '
                             + repr(tuple(axes)))
        self.shape = len(axes) * [0]

        if 'u' in self._axes:
            self.u_axis = self._axes.index('u')
            self.v_axis = -1
            self.shape[self.u_axis] = self.uv_shape[0]
            self._along_slit_index = self.u_axis
            self._along_slit_uv_axis = 0
            self._cross_slit_uv_axis = 1
            self._along_slit_len = self.shape[self.u_axis]
        else:
            self.u_axis = -1
            self.v_axis = self._axes.index('v')
            self.shape[self.v_axis] = self.uv_shape[1]
            self._along_slit_index = self.v_axis
            self._along_slit_uv_axis = 1
            self._cross_slit_uv_axis = 0
            self._along_slit_len = self.shape[self.v_axis]

        self.swap_uv = False

        if self.uv_shape[self._cross_slit_uv_axis] != 1:
            raise ValueError('Slit1D cross-slit FOV axis must have length 1')

        self.t_axis = -1

        # Cadence
        if isinstance(tstart, Cadence):
            self.cadence = tstart
            if self.cadence.shape != (1,):
                raise ValueError("Shape of a Slit1D's cadence must be (1,)")
            self._texp = self.cadence.time[1] - self.cadence.time[0]
        else:
            self.cadence = SnapCadence(tstart, texp)
            self._texp = texp

        # Optional subfields
        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

    def __getstate__(self):
        self.refresh()
        return (self._axes, self.cadence, self._texp, self.fov, self.path, self.frame,
                self.subfields)

    def __setstate__(self, state):
        self.__init__(*state[:-1], **state[-1])
        self.freeze()

    def uvt(self, indices, *, remask=False, derivs=True):
        """Coordinates *(u,v)* and time *t* for indices into the data array.

        This method supports non-integer index values.

        Parameters:
            indices (Scalar or Vector): Array indices.
            remask (bool, optional): True to mask values outside the field of view.
            derivs (bool, optional): True to include derivatives in the returned values.

        Returns:
            tuple[Pair, Scalar]: The *(u,v)* location and the time in seconds TDB
            associated with the array `indices`.
        """

        # Interpret a 1-D index or a multi-D index
        slit_coord = Observation._scalar_from_indices(indices, self._along_slit_index,
                                                     derivs=derivs)

        if remask:
            is_outside = (slit_coord.vals < 0) | (slit_coord.vals > self._along_slit_len)
            slit_coord = slit_coord.remask_or(is_outside)

        # Create the (u,v) Pair
        uv_vals = np.empty(slit_coord.shape + (2,))
        uv_vals[..., self._along_slit_uv_axis] = slit_coord.vals
        uv_vals[..., self._cross_slit_uv_axis] = 0.5
        uv = Pair(uv_vals, mask=slit_coord.mask)

        # Create time Scalar; shapeless is OK unless there's a mask
        time = Scalar(self.cadence.midtime)

        # Apply mask to time if necessary
        if remask and np.any(slit_coord.mask):
            time = Scalar.filled(uv.shape, self.midtime, mask=slit_coord.mask)

        return (uv, time)

    def uvt_range(self, indices, *, remask=False):
        """Ranges of *(u,v)* spatial coordinates and time for integer array indices.

        Parameters:
            indices (Scalar or Vector): Array indices.
            remask (bool, optional): True to mask values outside the field of view.

        Returns:
            tuple[Pair, Pair, Scalar, Scalar]: The lower and upper *(u,v)* corners of the
            detector, followed by the earliest and latest time TDB, associated with the
            given array `indices`.
        """

        # Interpret a 1-D index or a multi-D index
        slit_coord = Observation._scalar_from_indices(indices, self._along_slit_index)

        slit_int = slit_coord.int(top=self._along_slit_len, remask=remask)

        # Create the (u,v) Pair
        uv_min_vals = np.zeros(slit_coord.shape + (2,), dtype='int')
        uv_min_vals[..., self._along_slit_uv_axis] = slit_int.vals
        uv_min = Pair(uv_min_vals, slit_int.mask)

        # Time
        time_min = Scalar.filled(uv_min.shape, self.time[0], mask=uv_min.mask)
        time_max = Scalar.filled(uv_min.shape, self.time[1], mask=uv_min.mask)

        return (uv_min, uv_min + Pair.INT11, time_min, time_max)

    def time_range_at_uv(self, uv_pair, *, remask=False):
        """The start and stop times of the specified spatial pixel *(u,v)*.

        A Slit1D observation has no time-dependence, so the times are those of the
        observation overall. The index along the cross-slit axis is generally ignored,
        although values outside the range 0 to 1 are masked if `remask` is True.

        Parameters:
            uv_pair (Pair): Spatial *(u,v)* data array coordinates, truncated to integers
                if necessary.
            remask (bool, optional): True to mask values outside the field of view.

        Returns:
            tuple[Scalar, Scalar]: The start time and stop time in seconds TDB for each
            `uv_pair`.
        """

        return self._time_range_at_uv_0d(uv_pair, remask=remask)

    def uv_range_at_time(self, time, *, remask=False):
        """The *(u,v)* range of spatial pixels observed at a specified time.

        A Slit1D observation has no time-dependence, so the entire slit is observed at
        every time within the limits of the observation.

        Parameters:
            time (Scalar): Time values in seconds TDB.
            remask (bool, optional): True to mask values outside the time limits.

        Returns:
            tuple[Pair, Pair]: The lower (inclusive) and upper (exclusive) corners of the
            observed *(u,v)* rectangle at `time`.
        """

        return Observation._uv_range_at_time_0d(self, time, uv_shape=self.fov.uv_shape,
                                                remask=remask)

    def uv_range_at_tstep(self, tstep, *, remask=False):
        """The range of spatial *(u,v)* pixels active at a particular time step.

        A Slit1D observation has no time-dependence, so the whole slit is active at every
        time step.

        Parameters:
            tstep (Scalar): Time step index.
            remask (bool, optional): True to mask time steps outside the cadence.

        Returns:
            tuple[Pair, Pair]: The lower (inclusive) and upper (exclusive) corners of the
            observed *(u,v)* rectangle at `tstep`.
        """

        return Observation._uv_range_at_tstep_0d(self, tstep, uv_shape=self.fov.uv_shape,
                                                 remask=remask)

    def time_shift(self, dtime):
        """A copy of the observation object with a time-shift.

        Parameters:
            dtime (float): The time offset to apply to the observation, in units of
                seconds. A positive value shifts the observation later.

        Returns:
            Observation: A (shallow) copy of the object with a new time.
        """

        cadence = self.cadence.time_shift(dtime)
        return Slit1D(axes=self._axes, tstart=cadence, texp=self._texp, fov=self.fov,
                      path=self.path, frame=self.frame, **self.subfields)

##########################################################################################
