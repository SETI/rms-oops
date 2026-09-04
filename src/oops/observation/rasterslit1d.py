##########################################################################################
# oops/observation/rasterslit1d.py
##########################################################################################

import numpy as np

from polymath               import Pair
from oops.observation       import Observation
from oops.cadence           import Cadence
from oops.cadence.metronome import Metronome
from oops.frame             import Frame
from oops.path              import Path


class RasterSlit1D(Observation):
    """A 1-D observation made by sweeping a single pixel along a slit.

    A subclass of :class:`~oops.Observation` whose one dimension is constructed by that
    sweep.

    The FOV describes the 1-D slit.
    """

    def __init__(self, axes, cadence, fov, path, frame, **subfields):
        """Constructor for a RasterSlit1D observation.

        Parameters:
            axes (list or tuple): Strings, with one value for each axis in the associated
                data array. A value of 'ut' should appear at the location of the array's
                u-axis if any; 'vt' should appear at the location of the array's v-axis if
                any. Only one of 'ut' or 'vt' can appear.
            cadence (Cadence): A 1-D Cadence object defining the start time and duration
                of each consecutive measurement. Alternatively, a tuple or dictionary
                providing the input arguments `(tstart, texp, [interstep_delay])` to the
                constructor Metronome.for_array1d(), except for the number of steps, which
                is defined by the FOV.
            fov (FOV): (field-of-view) object, which describes the field of view including
                any spatial distortion. It maps between spatial coordinates (u,v) and
                instrument coordinates (x,y). For a RasterSlit1D object, one of the axes
                of the FOV must have length 1.
            path (Path): The path waypoint co-located with the instrument.
            frame (Frame): The wayframe of a coordinate frame fixed to the optics of the
                instrument. This frame should have its Z-axis pointing outward near the
                center of the line of sight, with the X-axis pointing rightward and the
                Y-axis pointing downward.
            subfields (dict): All of the optional attributes. Additional subfields may be
                included as needed.

        Raises:
            ValueError: If `axes` does not contain exactly one of 'ut' and 'vt', if it
                also contains 't', if the cross-slit axis of the FOV does not have length
                1, or if the shapes of the cadence and the FOV are incompatible.
            TypeError: If `cadence` is not a Cadence, tuple, list, or dictionary.
        """

        # Basic properties
        self.path = Path.as_waypoint(path)
        self.frame = Frame.as_wayframe(frame)

        # FOV
        self.fov = fov
        fov_uv_shape = tuple(self.fov.uv_shape.vals)

        # Axes / Shape / Size
        self._axes = list(axes)
        count1 = ('ut' in self._axes) + ('vt' in self._axes)
        count2 = ('t' in self._axes)
        if (count1, count2) != (1,0):
            raise ValueError('invalid axes for RasterSlit1D: ' + repr(self._axes))

        self.shape = len(axes) * [0]

        if 'ut' in self._axes:
            self.u_axis = self._axes.index('ut')
            self.v_axis = -1
            self.t_axis = self.u_axis
            self.shape[self.u_axis] = fov_uv_shape[0]
            self.uv_shape = (fov_uv_shape[0], 1)
            self._along_slit_uv_index = 0
            self._cross_slit_uv_index = 1
        else:
            self.u_axis = -1
            self.v_axis = self._axes.index('vt')
            self.t_axis = self.v_axis
            self.shape[self.v_axis] = fov_uv_shape[1]
            self.uv_shape = (1, fov_uv_shape[1])
            self._along_slit_uv_index = 1
            self._cross_slit_uv_index = 0

        self.swap_uv = False

        self._along_slit_len = fov_uv_shape[self._along_slit_uv_index]
        if fov_uv_shape[self._cross_slit_uv_index] != 1:
            raise ValueError('RasterSlit1D cross-slit axis must have length 1')

        # Cadence
        samples = self._along_slit_len

        if isinstance(cadence, (tuple, list)):
            self.cadence = Metronome.for_array1d(samples, *cadence)
        elif isinstance(cadence, dict):
            self.cadence = Metronome.for_array1d(samples, **cadence)
        elif isinstance(cadence, Cadence):
            self.cadence = cadence
            if self.cadence.shape != (samples,):
                raise ValueError('RasterSlit1D input Cadence and FOV shapes '
                                 'are incompatible: %s, %s'
                                 % (cadence.shape, tuple(fov.uv_shape.vals)))
        else:
            raise TypeError('Invalid cadence class: ' + type(cadence).__name__)

        # Optional subfields
        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

    def __getstate__(self):
        self.refresh()
        return (self._axes, self.cadence, self.fov, self.path, self.frame, self.subfields)

    def __setstate__(self, state):
        self.__init__(*state[:-1], **state[-1])
        self.freeze()

    def uvt(self, indices, *, remask=False, derivs=True):
        """Coordinates `(u,v)` and time `t` for indices into the data array.

        This method supports non-integer index values.

        Parameters:
            indices (Scalar or Vector): Array indices.
            remask (bool, optional): True to mask values outside the field of view.
            derivs (bool, optional): True to include derivatives in the returned values.

        Returns:
            tuple[Pair, Scalar]: `(uv, time)`, where `uv` defines the values of `(u,v)`
            within the FOV that are associated with the array indices and `time` defines
            the time in seconds TDB associated with the array indices.
        """

        # Interpret a 1-D index or a multi-D index
        slit_coord = Observation.scalar_from_indices(indices, self.t_axis, derivs=derivs)

        # Create time Scalar
        time = self.cadence.time_at_tstep(slit_coord, remask=remask)
            # there's only one relevant axis and remask has it covered now

        # Create (u,v) Pair
        uv_vals = np.empty(slit_coord.shape + (2,))
        uv_vals[..., self._along_slit_uv_index] = slit_coord.vals
        uv_vals[..., self._cross_slit_uv_index] = 0.5
        uv = Pair(uv_vals, mask=time.mask)

        return (uv, time)

    def uvt_range(self, indices, *, remask=False):
        """Ranges of `(u,v)` spatial coordinates and time for integer array indices.

        Parameters:
            indices (Scalar or Vector): Array indices.
            remask (bool, optional): True to mask values outside the field of view.

        Returns:
            tuple[Pair, Pair, Scalar, Scalar]: `(uv_min, uv_max, time_min, time_max)`,
            where:

            * `uv_min`: The minimum values of (u,v) associated with the pixel.
            * `uv_max`: The maximum values of (u,v).
            * `time_min`: The minimum time associated with the pixel, in seconds TDB.
            * `time_max`: The maximum time value.
        """

        # Works for a 1-D index or a multi-D index
        slit_coord = Observation.scalar_from_indices(indices, self.t_axis, derivs=False)

        # Get the time range
        (time0, time1) = self.cadence.time_range_at_tstep(slit_coord, remask=remask)
            # there's only one relevant axis and remask has it covered now

        # Create uv_min from the slit index
        slit_int = slit_coord.int(top=self._along_slit_len, remask=False)

        uv_min_vals = np.zeros(slit_coord.shape + (2,), dtype='int')
        uv_min_vals[..., self._along_slit_uv_index] = slit_int.vals
        uv_min = Pair(uv_min_vals, mask=time0.mask)

        return (uv_min, uv_min + Pair.INT11, time0, time1)

    def time_range_at_uv(self, uv_pair, *, remask=False):
        """The start and stop times of the specified spatial pixel `(u,v)`.

        The index along the cross-slit axis is ignored, because that axis has length 1.

        Parameters:
            uv_pair (Pair): Spatial (u,v) data array coordinates, truncated to integers if
                necessary.
            remask (bool, optional): True to mask values outside the field of view.

        Returns:
            tuple[Scalar, Scalar]: Scalars of the start time and stop time of each `(u,v)`
            pair, as seconds TDB.
        """

        # We can't use super.time_range_at_uv_1d because the self.uv_shape is
        # not the FOV shape, as that routine expects.
        uv_pair = Pair.as_pair(uv_pair, recursive=False)
        tstep = uv_pair.to_scalar(self._along_slit_uv_index)
        return self.cadence.time_range_at_tstep(tstep, remask=remask)

    def uv_range_at_time(self, time, *, remask=False):
        """The `(u,v)` range of spatial pixels observed at a specified time.

        Because the slit is swept out one sample at a time, this range describes a single
        pixel.

        Parameters:
            time (Scalar): Time values in seconds TDB.
            remask (bool, optional): True to mask values outside the time limits.

        Returns:
            tuple[Pair, Pair]: `(uv_min, uv_max)`, where `uv_min` is the lower corner of
            the `(u,v)` rectangle observed and `uv_max` is the upper corner.
        """

        return Observation.uv_range_at_time_1d(self, time, self.uv_shape,
                                               axis=self._along_slit_uv_index,
                                               remask=remask)

    def uv_range_at_tstep(self, tstep, *, remask=False):
        """The range of spatial `(u,v)` pixels active at a particular time step.

        A RasterSlit1D samples one pixel of the slit at a time, so a single pixel is
        active at each time step.

        Parameters:
            tstep (Scalar): Time step index.
            remask (bool, optional): True to mask time steps outside the cadence.

        Returns:
            tuple[Pair, Pair]: `(uv_min, uv_max)`, where `uv_min` is the lower corner of
            the `(u,v)` rectangle active at this time step and `uv_max` is the upper
            corner, exclusive.
        """

        return Observation.uv_range_at_tstep_1d(self, tstep, self.uv_shape,
                                                axis=self._along_slit_uv_index,
                                                remask=remask)

    def time_shift(self, dtime):
        """A copy of the observation object with a time-shift.

        Parameters:
            dtime (float): The time offset to apply to the observation, in units of
                seconds. A positive value shifts the observation later.

        Returns:
            Observation: A (shallow) copy of the object with a new time.
        """

        obs = RasterSlit1D(axes=self._axes, cadence=self.cadence.time_shift(dtime),
                           fov=self.fov, path=self.path, frame=self.frame)

        for key in self.subfields.keys():
            obs.insert_subfield(key, self.subfields[key])

        return obs

##########################################################################################
