################################################################################
# oops/observation/slit1d.py: Subclass Slit1D of class Observation
################################################################################

import numpy as np
from polymath import Scalar, Pair, Vector

from oops.observation          import Observation
from oops.cadence.metronome    import Metronome
from oops.frame                import Frame
from oops.path                 import Path

class Slit1D(Observation):
    """A subclass of Observation consisting of a 1-D slit measurement with no
    time-dependence. However, it may still have additional axes (e.g., bands).
    """

    #===========================================================================
    def __init__(self, axes, tstart, texp, fov, path, frame, **subfields):
        """Constructor for a Slit1D observation.

        Input:
            axes        a list or tuple of strings, with one value for each axis
                        in the associated data array. A value of 'u' should
                        appear at the location of the array's u-axis if any;
                        'v' should appear at the location of the array's v-axis
                        if any. Only one of 'u' or 'v' can appear in a Slit1D.

            tstart      the start time of the observation in seconds TDB.

            texp        exposure duration of the observation in seconds.

            fov         a FOV (field-of-view) object, which describes the field
                        of view including any spatial distortion. It maps
                        between spatial coordinates (u,v) and instrument
                        coordinates (x,y). For a Slit1D object, one of the axes
                        of the FOV must have length 1.

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
        self.uv_shape = tuple(self.fov.uv_shape.vals)

        # Axes / Shape / Size
        self.axes = list(axes)
        assert (('u' in self.axes and 'v' not in self.axes) or
                ('v' in self.axes and 'u' not in self.axes))

        self.shape = len(axes) * [0]

        if 'u' in self.axes:
            self.u_axis = self.axes.index('u')
            self.v_axis = -1
            self.shape[self.u_axis] = self.uv_shape[0]
            self._along_slit_index = self.u_axis
            self._along_slit_uv_axis = 0
            self._cross_slit_uv_axis = 1
            self._along_slit_len = self.shape[self.u_axis]
        else:
            self.u_axis = -1
            self.v_axis = self.axes.index('v')
            self.shape[self.v_axis] = self.uv_shape[1]
            self._along_slit_index = self.v_axis
            self._along_slit_uv_axis = 1
            self._cross_slit_uv_axis = 0
            self._along_slit_len = self.shape[self.v_axis]

        self.swap_uv = False

        assert self.uv_shape[self._cross_slit_uv_axis] == 1
        self.t_axis = -1

        # Cadence
        self.cadence = Metronome.for_array0d(tstart, texp)

        # Timing
        self.tstart = self.cadence.tstart
        self.texp = self.cadence.texp

        self.time = self.cadence.time
        self.midtime = self.cadence.midtime

        self._scalar_time = (Scalar(self.time[0]), Scalar(self.time[1]))
        self._scalar_midtime = Scalar(self.midtime)

        # Optional subfields
        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

    def __getstate__(self):
        return (self.axes, self.tstart, self.texp, self.fov, self.path,
                self.frame, self.subfields)

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
        slit_coord = Observation.scalar_from_indices(indices,
                                                     self._along_slit_index,
                                                     derivs=derivs)
        slit_coord = self.scalar_from_indices(indices, self._along_slit_index)

        if remask:
            is_outside = ((slit_coord.vals < 0) |
                          (slit_coord.vals > self._along_slit_len))
            slit_coord = slit_coord.remask_or(is_outside)

        # Create the (u,v) Pair
        uv_vals = np.empty(slit_coord.shape + (2,))
        uv_vals[..., self._along_slit_uv_axis] = slit_coord.vals
        uv_vals[..., self._cross_slit_uv_axis] = 0.5
        uv = Pair(uv_vals, mask=slit_coord.mask)

        # Create time Scalar; shapeless is OK unless there's a mask
        time = self._scalar_midtime

        # Apply mask to time if necessary
        if remask and np.any(slit_coord.mask):
            time = Scalar.filled(uv.shape, self.midtime, mask=slit_coord.mask)

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

        # Interpret a 1-D index or a multi-D index
        slit_coord = Observation.scalar_from_indices(indices,
                                                     self._along_slit_index)

        slit_int = slit_coord.int(top=self._along_slit_len, remask=remask)

        # Create the (u,v) Pair
        uv_min_vals = np.zeros(slit_coord.shape + (2,), dtype='int')
        uv_min_vals[..., self._along_slit_uv_axis] = slit_int.vals
        uv_min = Pair(uv_min_vals, slit_int.mask)

        # Time
        time_min = Scalar.filled(uv_min.shape, self.time[0], mask=uv_min.mask)
        time_max = Scalar.filled(uv_min.shape, self.time[1], mask=uv_min.mask)

        return (uv_min, uv_min + Pair.INT11, time_min, time_max)

    #===========================================================================
    def time_range_at_uv(self, uv_pair, remask=False):
        """The start and stop times of the specified spatial pixel (u,v).

        For a 1-D slit, the index along the cross-slit axis is generally
        ignored, although values outside the range 0-1 will be masked if remask
        == True.

        Input:
            uv_pair     a Pair of spatial (u,v) data array coordinates,
                        truncated to integers if necessary.
            remask      True to mask values outside the field of view.

        Return:         a tuple containing Scalars of the start time and stop
                        time of each (u,v) pair, as seconds TDB.
        """

        return self.time_range_at_uv_0d(uv_pair, remask=remask)

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

        return Observation.uv_range_at_time_0d(self, time,
                                               uv_shape=self.fov.uv_shape,
                                               remask=remask)

    #===========================================================================
    def time_shift(self, dtime):
        """A copy of the observation object with a time-shift.

        Input:
            dtime       the time offset to apply to the observation, in units of
                        seconds. A positive value shifts the observation later.

        Return:         a (shallow) copy of the object with a new time.
        """

        obs = Slit1D(axes=self.axes, tstart=self.tstart + dtime, texp=self.texp,
                     fov=self.fov, path=self.path, frame=self.frame)

        for key in self.subfields.keys():
            obs.insert_subfield(key, self.subfields[key])

        return obs

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Slit1D(unittest.TestCase):

    def runTest(self):

        from oops.fov.flatfov import FlatFOV

        fov = FlatFOV((0.001,0.001), (20,1))
        obs = Slit1D(('u'), tstart=0., texp=10., fov=fov, path='SSB', frame='J2000')

        indices = Vector([(0,0),(1,0),(20,0),(21,0)])
        indices_ = indices.copy()       # clipped at 20
        indices_.vals[:,0][indices_.vals[:,0] == 20] -= 1

        # uvt() with remask == False
        (uv,time) = obs.uvt(indices)
        self.assertFalse(np.any(uv.mask))
        self.assertFalse(np.any(time.mask))
        self.assertEqual(time, 5.)
        self.assertEqual(uv.to_scalar(0), indices.to_scalar(0))
        self.assertEqual(uv.to_scalar(1), 0.5)

        # uvt() with remask == True
        (uv,time) = obs.uvt(indices, remask=True)

        self.assertTrue(np.all(uv.mask == np.array(3*[False] + [True])))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time[:3], (5,5,5))
        self.assertEqual(uv[:3].to_scalar(0), indices[:3].to_scalar(0))
        self.assertEqual(uv[:3].to_scalar(1), 0.5)

        # uvt_range() with remask == False
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertFalse(np.any(uv_min.mask))
        self.assertFalse(np.any(uv_max.mask))
        self.assertFalse(np.any(time_min.mask))
        self.assertFalse(np.any(time_max.mask))

        self.assertEqual(uv_min.to_scalar(0), indices_.to_scalar(0))
        self.assertEqual(uv_min.to_scalar(1), 0)
        self.assertEqual(uv_max.to_scalar(0), indices_.to_scalar(0) + 1)
        self.assertEqual(uv_max.to_scalar(1), 1)
        self.assertEqual(time_min, 0.)
        self.assertEqual(time_max, 10.)

        # uvt_range() with remask == True
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices,
                                                             remask=True)
        self.assertTrue(np.all(uv_min.mask == np.array(3*[False] + [True])))
        self.assertTrue(np.all(uv_max.mask == uv_min.mask))
        self.assertTrue(np.all(time_min.mask == uv_min.mask))
        self.assertTrue(np.all(time_max.mask == uv_min.mask))

        self.assertEqual(uv_min.to_scalar(0)[:2], indices.to_scalar(0)[:2])
        self.assertEqual(uv_min.to_scalar(1)[:2], 0)
        self.assertEqual(uv_max.to_scalar(0)[:2], indices.to_scalar(0)[:2] + 1)
        self.assertEqual(uv_max.to_scalar(1)[:2], 1)
        self.assertEqual(time_min[:2], 0.)
        self.assertEqual(time_max[:2], 10.)

        # time_range_at_uv() with remask == False
        uv = Pair([(0,0),(0,0.5),(0,1),(0,2),
                   (20,0),(20,0.5),(20,1),(20,2),
                   (21,0)])

        (time0, time1) = obs.time_range_at_uv(uv)

        self.assertEqual(time0, 0.)
        self.assertEqual(time1, 10.)

        # time_range_at_uv() with remask == True
        (time0, time1) = obs.time_range_at_uv(uv, remask=True)

        self.assertTrue(np.all(time0.mask == 3*[False] + [True] +
                                             3*[False] + 2*[True]))
        self.assertTrue(np.all(time1.mask == time0.mask))
        self.assertEqual(time0[:3], 0.)
        self.assertEqual(time1[:3], 10.)

        ####################################

        # Alternative axis order ('a','u','b')

        fov = FlatFOV((0.001,0.001), (20,1))
        obs = Slit1D(('a','u', 'b'), tstart=0., texp=10.,
                     fov=fov, path='SSB', frame='J2000')

        indices = Vector([(0,0,0),(0,1,99),(0,19,99),(10,20,99),(10,21,99)])
        indices_ = indices.copy()       # clipped at 20
        indices_.vals[:,1][indices_.vals[:,1] == 20] -= 1

        (uv,time) = obs.uvt(indices)

        self.assertEqual(uv.to_scalar(0), indices.to_scalar(1))
        self.assertEqual(uv.to_scalar(1), 0.5)
        self.assertEqual(time, 5.)

        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertEqual(uv_min.to_scalar(0), indices_.to_scalar(1))
        self.assertEqual(uv_max.to_scalar(0), indices_.to_scalar(1)+1)
        self.assertEqual(uv_min.to_scalar(1), 0.)
        self.assertEqual(uv_max.to_scalar(1), 1)
        self.assertEqual(time_min, 0.)
        self.assertEqual(time_max, 10.)

########################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
