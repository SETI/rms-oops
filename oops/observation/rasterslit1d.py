################################################################################
# oops/observation/rasterslit1d.py: Subclass RasterSlit1D of class Observation
################################################################################

import numpy as np

from polymath               import Pair
from oops.observation       import Observation
from oops.cadence           import Cadence
from oops.cadence.metronome import Metronome
from oops.frame             import Frame
from oops.path              import Path

class RasterSlit1D(Observation):
    """A subclass of Observation consisting of a 1-D observation in which the
    one dimension is constructed by sweeping a single pixel along a slit.

    The FOV describes the 1-D slit.
    """

    #===========================================================================
    def __init__(self, axes, cadence, fov, path, frame, **subfields):
        """Constructor for a RasterSlit observation.

        Input:

            axes        a list or tuple of strings, with one value for each axis
                        in the associated data array. A value of 'ut' should
                        appear at the location of the array's u-axis if any;
                        'vt' should appear at the location of the array's v-axis
                        if any. Only one of 'ut' or 'vt' can appear.

            cadence     a 1-D Cadence object defining the start time and
                        duration of each consecutive measurement. Alternatively,
                        a tuple or dictionary providing the input arguments to
                        the constructor Metronome.for_array1d() (except for the
                        number of steps, which is defined by the FOV):
                            (tstart, texp, [interstep_delay])

            fov         a FOV (field-of-view) object, which describes the field
                        of view including any spatial distortion. It maps
                        between spatial coordinates (u,v) and instrument
                        coordinates (x,y). For a RasterSlit object, one of the
                        axes of the FOV must have length 1.

            path        the path waypoint co-located with the instrument.

            frame       the wayframe of a coordinate frame fixed to the optics
                        of the instrument. This frame should have its Z-axis
                        pointing outward near the center of the line of sight,
                        with the X-axis pointing rightward and the y-axis
                        pointing downward.

            subfields   a dictionary containing all of the optional attributes.
                        Additional subfields may be included as needed.
        """

        # Basic properties
        self.path = Path.as_waypoint(path)
        self.frame = Frame.as_wayframe(frame)

        # FOV
        self.fov = fov
        fov_uv_shape = tuple(self.fov.uv_shape.vals)

        # Axes / Shape / Size
        self.axes = list(axes)
        count1 = ('ut' in self.axes) + ('vt' in self.axes)
        count2 = ('t' in self.axes)
        if (count1, count2) != (1,0):
            raise ValueError('invalid axes for RasterSlit1D: '
                             + repr(self.axes))

        self.shape = len(axes) * [0]

        if 'ut' in self.axes:
            self.u_axis = self.axes.index('ut')
            self.v_axis = -1
            self.t_axis = self.u_axis
            self.shape[self.u_axis] = fov_uv_shape[0]
            self.uv_shape = (fov_uv_shape[0], 1)
            self._along_slit_uv_index = 0
            self._cross_slit_uv_index = 1
        else:
            self.u_axis = -1
            self.v_axis = self.axes.index('vt')
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

        # Timing
        self.time = self.cadence.time
        self.midtime = self.cadence.midtime

        # Optional subfields
        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

    def __getstate__(self):
        return (self.axes, self.cadence, self.fov, self.path, self.frame,
                self.subfields)

    def __setstate__(self, state):
        self.__init__(*state[:-1], **state[-1])

    #===========================================================================
    def uvt(self, indices, remask=False, derivs=True):
        """Coordinates (u,v) and time t for indices into the data array.

        This method supports non-integer index values.

        Input:
            indices     a Scalar or Vector of array indices.
            remask      True to mask values outside the field of view.
            derivs      True to include derivatives in the returned values.

        Return:         (uv, time)
            uv          a Pair defining the values of (u,v) within the FOV that
                        are associated with the array indices.
            time        a Scalar defining the time in seconds TDB associated
                        with the array indices.
        """

        # Interpret a 1-D index or a multi-D index
        slit_coord = Observation.scalar_from_indices(indices, self.t_axis,
                                                              derivs=derivs)

        # Create time Scalar
        time = self.cadence.time_at_tstep(slit_coord, remask=remask)
            # there's only one relevant axis and remask has it covered now

        # Create (u,v) Pair
        uv_vals = np.empty(slit_coord.shape + (2,))
        uv_vals[..., self._along_slit_uv_index] = slit_coord.vals
        uv_vals[..., self._cross_slit_uv_index] = 0.5
        uv = Pair(uv_vals, mask=time.mask)

        return (uv, time)

    #===========================================================================
    def uvt_range(self, indices, remask=False):
        """Ranges of (u,v) spatial coordinates and time for integer array
        indices.

        Input:
            indices     a Scalar or Vector of array indices.
            remask      True to mask values outside the field of view.

        Return:         (uv_min, uv_max, time_min, time_max)
            uv_min      a Pair defining the minimum values of FOV (u,v)
                        associated the pixel.
            uv_max      a Pair defining the maximum values of FOV (u,v)
                        associated the pixel.
            time_min    a Scalar defining the minimum time associated with the
                        array indices. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time value.
        """

        # Works for a 1-D index or a multi-D index
        slit_coord = Observation.scalar_from_indices(indices, self.t_axis,
                                                     derivs=False)

        # Get the time range
        (time0,
         time1) = self.cadence.time_range_at_tstep(slit_coord, remask=remask)
            # there's only one relevant axis and remask has it covered now

        # Create uv_min from the slit index
        slit_int = slit_coord.int(top=self._along_slit_len, remask=False)

        uv_min_vals = np.zeros(slit_coord.shape + (2,), dtype='int')
        uv_min_vals[..., self._along_slit_uv_index] = slit_int.vals
        uv_min = Pair(uv_min_vals, mask=time0.mask)

        return (uv_min, uv_min + Pair.INT11, time0, time1)

    #===========================================================================
    def time_range_at_uv(self, uv_pair, remask=False):
        """The start and stop times of the specified spatial pixel (u,v).

        Input:
            uv_pair     a Pair of spatial (u,v) data array coordinates,
                        truncated to integers if necessary.
            remask      True to mask values outside the field of view.

        Return:         a tuple containing Scalars of the start time and stop
                        time of each (u,v) pair, as seconds TDB.
        """

        # We can't use super.time_range_at_uv_1d because the self.uv_shape is
        # not the FOV shape, as that routine expects.
        uv_pair = Pair.as_pair(uv_pair, recursive=False)
        tstep = uv_pair.to_scalar(self._along_slit_uv_index)
        return self.cadence.time_range_at_tstep(tstep, remask=remask)

    #===========================================================================
    def uv_range_at_time(self, time, remask=False):
        """The (u,v) range of spatial pixels observed at the specified time.

        Input:
            time        a Scalar of time values in seconds TDB.
            remask      True to mask values outside the time limits.

        Return:         (uv_min, uv_max)
            uv_min      the lower (u,v) corner Pair of the area observed at the
                        specified time.
            uv_max      the upper (u,v) corner Pair of the area observed at the
                        specified time.
        """

        return Observation.uv_range_at_time_1d(self, time,
                                               uv_shape=Pair.INT11,
                                               axis=self._along_slit_uv_index,
                                               remask=remask)

    #===========================================================================
    def time_shift(self, dtime):
        """A copy of the observation object with a time-shift.

        Input:
            dtime       the time offset to apply to the observation, in units of
                        seconds. A positive value shifts the observation later.

        Return:         a (shallow) copy of the object with a new time.
        """

        obs = RasterSlit1D(axes=self.axes,
                           cadence=self.cadence.time_shift(dtime),
                           fov=self.fov, path=self.path, frame=self.frame)

        for key in self.subfields.keys():
            obs.insert_subfield(key, self.subfields[key])

        return obs

################################################################################
