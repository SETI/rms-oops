################################################################################
# oops/obs_/slit1d.py: Subclass Slit1D of class Observation
################################################################################

import numpy as np
from polymath import *

from oops.cadence_.metronome import Metronome
from oops.obs_.observation   import Observation

class Slit1D(Observation):
    """A Slit1D is subclass of Observation consisting of a 1-D slit measurement
    with no time-dependence.
    """

    def __init__(self, axes, det_size, tstart, texp,
                       fov, path_id, frame_id, **subfields):
        """Constructor for a Slit observation.

        Input:
            axes        a list or tuple of strings, with one value for each axis
                        in the associated data array. A value of "u" should
                        appear at the location of the array's u-axis if any;
                        "v" should appear at the location of the array's v-axis
                        if any. Only one of "u" or "v" can appear.
            det_size    the size of the detectors in FOV units parallel to the
                        slit. It will be < 1 if there are gaps between the
                        detectors.
            tstart      the start time of the observation in seconds TDB.
            texp        exposure time of the observation in seconds.

            fov         a FOV (field-of-view) object, which describes the field
                        of view including any spatial distortion. It maps
                        between spatial coordinates (u,v) and instrument
                        coordinates (x,y). For a Slit object, one of the axes of
                        the FOV must have length 1.
            path_id     the registered ID of a path co-located with the
                        instrument.
            frame_id    the registered ID of a coordinate frame fixed to the
                        optics of the instrument. This frame should have its
                        Z-axis pointing outward near the center of the line of
                        sight, with the X-axis pointing rightward and the y-axis
                        pointing downward.
            subfields   a dictionary containing all of the optional attributes.
                        Additional subfields may be included as needed.
        """

        self.cadence = Metronome(tstart, texp, texp, 1)
        self.fov = fov
        self.path_id = path_id
        self.frame_id = frame_id

        self.axes = list(axes)
        assert (("u" in self.axes and "v" not in self.axes) or
                ("v" in self.axes and "u" not in self.axes))

        self.shape = len(axes) * [0]

        if "u" in self.axes:
            self.u_axis = self.axes.index("u")
            self.v_axis = -1
            self.along_slit_index = self.u_axis
            self.along_slit_uv_index = 0
            self.cross_slit_uv_index = 1
            self.shape[self.u_axis] = self.fov.uv_shape.vals[0]
        else:
            self.u_axis = -1
            self.v_axis = self.axes.index("v")
            self.along_slit_index = self.v_axis
            self.along_slit_uv_index = 1
            self.cross_slit_uv_index = 0
            self.shape[self.v_axis] = self.fov.uv_shape.vals[1]

        self.uv_shape = self.fov.uv_shape.vals
        assert self.fov.uv_shape.vals[self.cross_slit_uv_index] == 1

        self.det_size = det_size
        self.slit_is_discontinuous = (self.det_size < 1)

        self.t_axis = -1
        self.tstart = tstart
        self.texp = texp
        self.time = self.cadence.time
        self.midtime = self.cadence.midtime

        self.scalar_time = (Scalar(self.time[0]), Scalar(self.time[1]))
        self.scalar_midtime = Scalar(self.midtime)

        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

    def uvt(self, indices, fovmask=False):
        """Returns the FOV coordinates (u,v) and the time in seconds TDB
        associated with the given indices into the data array. This method
        supports non-integer index values.

        Input:
            indices     a Tuple of array indices.
            fovmask     True to mask values outside the field of view.

        Return:         (uv, time)
            uv          a Pair defining the values of (u,v) associated with the
                        array indices.
            time        a Scalar defining the time in seconds TDB associated
                        with the array indices.
        """

        indices = Tuple.as_tuple(indices)

        slit_coord = indices.as_scalar(self.along_slit_index)
        if self.slit_is_discontinuous:
            slit_int = slit_coord.int()
            slit_coord = slit_int + (slit_coord - slit_int) * self.det_size

        uv_vals = np.empty(indices.shape + [2])
        uv_vals[..., self.along_slit_uv_index] = slit_coord.vals
        uv_vals[..., self.cross_slit_uv_index] = 0.5
        uv = Pair(uv_vals, indices.mask)

        time = self.scalar_midtime

        if fovmask:
            is_inside = self.uv_is_inside(uv, inclusive=True)
            if not np.all(is_inside):
                mask = indices.mask | np.logical_not(is_inside)
                uv.mask = mask

                time_vals = np.empty(indices.shape)
                time_vals[...] = self.midtime
                time = Scalar(time_vals, mask)

        return (uv, time)

    def uvt_range(self, indices, fovmask=False):
        """Returns the ranges of FOV coordinates (u,v) and the time range in
        seconds TDB associated with the given integer indices into the data
        array.

        Input:
            indices     a Tuple of integer array indices.
            fovmask     True to mask values outside the field of view.

        Return:         (uv_min, uv_max, time_min, time_max)
            uv_min      a Pair defining the minimum values of (u,v) associated
                        the pixel.
            uv_max      a Pair defining the maximum values of (u,v).
            time_min    a Scalar defining the minimum time associated with the
                        pixel. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time value.
        """

        indices = Tuple.as_int(indices)

        slit_coord = indices.as_scalar(self.along_slit_index)

        uv_vals = np.empty(indices.shape + [2], dtype="int")
        uv_vals[..., self.along_slit_uv_index] = slit_coord.vals
        uv_vals[..., self.cross_slit_uv_index] = 0
        uv_min = Pair(uv_vals, indices.mask)
        uv_max = uv_min + Pair.ONES

        time_min = self.scalar_time[0]
        time_max = self.scalar_time[1]

        if fovmask:
            is_inside = self.uv_is_inside(uv_min, inclusive=False)
            if not np.all(is_inside):
                mask = indices.mask | np.logical_not(is_inside)
                uv_min.mask = mask
                uv_max.mask = mask

                time_min_vals = np.empty(is_inside.shape)
                time_max_vals = np.empty(is_inside.shape)

                time_min_vals[...] = self.time[0]
                time_max_vals[...] = self.time[1]

                mask = np.logical_not(is_inside)
                time_min = Scalar(time_min_vals, mask)
                time_max = Scalar(time_max_vals, mask)

        return (uv_min, uv_max, time_min, time_max)

    def times_at_uv(self, uv_pair, fovmask=False, extras=None):
        """Returns the start and stop times of the specified spatial pixel
        (u,v).

        Input:
            uv_pair     a Pair of spatial (u,v) coordinates in and observation's
                        field of view. The coordinates need not be integers, but
                        any fractional part is truncated.
            fovmask     True to mask values outside the field of view.
            extras      an optional tuple or dictionary containing any extra
                        parameters required for the conversion from (u,v) to
                        time.

        Return:         a tuple containing Scalars of the start time and stop
                        time of each (u,v) pair, as seconds TDB.
        """

        if fovmask:
            is_inside = self.uv_is_inside(uv_pair, inclusive=True)
            if not np.all(is_inside):
                time_min_vals = np.empty(is_inside.shape)
                time_max_vals = np.empty(is_inside.shape)

                time_min_vals[...] = self.time[0]
                time_max_vals[...] = self.time[1]

                mask = np.logical_not(is_inside)
                time_min = Scalar(time_min_vals, mask)
                time_max = Scalar(time_max_vals, mask)

                return (time_min, time_max)

        return self.scalar_time

    def sweep_duv_dt(self, uv_pair, extras=None):
        """Returns the mean local sweep speed of the instrument in the (u,v)
        directions.

        Input:
            uv_pair     a Pair of spatial indices (u,v).
            extras      an optional tuple or dictionary containing any extra
                        parameters required to define the timing of array
                        elements.

        Return:         a Pair containing the local sweep speed in units of
                        pixels per second in the (u,v) directions.
        """

        return Pair.ZERO

    def time_shift(self, dtime):
        """Returns a copy of the observation object in which times have been
        shifted by a constant value.

        Input:
            dtime       the time offset to apply to the observation, in units of
                        seconds. A positive value shifts the observation later.

        Return:         a (shallow) copy of the object with a new time.
        """

        obs = Slit1D(self.axes, self.det_size, self.tstart + dtime, self.texp,
                     self.fov, self.path_id, self.frame_id)

        for key in self.subfields.keys():
            obs.insert_subfield(key, self.subfields[key])

        return obs

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Slit1D(unittest.TestCase):

    def runTest(self):

        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
