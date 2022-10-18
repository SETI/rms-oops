################################################################################
# oops/observation/slit.py: Subclass Slit of class Observation
################################################################################

import numpy as np
from polymath import Scalar, Pair, Vector

from .                   import Observation
from .snapshot           import Snapshot
from ..cadence           import Cadence
from ..cadence.metronome import Metronome
from ..frame             import Frame
from ..path              import Path

class Slit(Observation):
    """A subclass of Observation consisting of a 2-D image constructed by
    rotating an instrument that has a 1-D array of sensors. The FOV describes
    the 1-D sensor array. The second axis of the image is simulated by sampling
    the slit according to the cadence as the instrument rotates.
    """

    #===========================================================================
    def __init__(self, axes, cadence, fov, path, frame, **subfields):
        """Constructor for a Slit observation.

        Input:
            axes        a list or tuple of strings, with one value for each axis
                        in the associated data array. A value of 'u' or 'ut'
                        should appear at the location of the array's u-axis;
                        'vt' or 'v' should appear at the location of the array's
                        v-axis. The 't' suffix is used for the one of these axes
                        that is emulated by time-sampling perpendicular to the
                        slit.

            cadence     a 1-D Cadence object defining the start time and
                        duration of each consecutive measurement. Alternatively,
                        a tuple or dictionary providing the input arguments to
                        the constructor Metronome.for_array1d():
                            (steps, tstart, texp, [interstep_delay])

            fov         a FOV (field-of-view) object, which describes the field
                        of view including any spatial distortion. It maps
                        between spatial coordinates (u,v) and instrument
                        coordinates (x,y). For a Slit object, one of the axes of
                        the FOV must have length 1.

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
        self._fov_uv_shape = tuple(self.fov.uv_shape.vals)

        # Cadence
        if isinstance(cadence, (tuple,list)):
            self.cadence = Metronome.for_array1d(*cadence)
        elif isinstance(cadence, dict):
            self.cadence = Metronome.for_array1d(**cadence)
        elif isinstance(cadence, Cadence):
            self.cadence = cadence
            assert len(self.cadence.shape) == 1
        else:
            raise TypeError('Invalid cadence class: ' + type(cadence).__name__)

        # Axes / Shape / Size
        self.axes = list(axes)
        assert (('u' in self.axes and 'vt' in self.axes) or
                ('v' in self.axes and 'ut' in self.axes))

        lines = self.cadence.shape[0]

        if 'ut' in self.axes:
            self.u_axis = self.axes.index('ut')
            self.v_axis = self.axes.index('v')
            self.t_axis = self.u_axis
            self._along_slit_axis = self.v_axis
            self._cross_slit_uv_axis = 0
            self._along_slit_uv_axis = 1
            self.uv_shape = (lines, self._fov_uv_shape[self._along_slit_axis])
        else:
            self.u_axis = self.axes.index('u')
            self.v_axis = self.axes.index('vt')
            self.t_axis = self.v_axis
            self._along_slit_axis = self.u_axis
            self._cross_slit_uv_axis = 1
            self._along_slit_uv_axis = 0
            self.uv_shape = (self._fov_uv_shape[self._along_slit_axis], lines)

        self.swap_uv = (self.u_axis > self.v_axis)

        assert self._fov_uv_shape[self._cross_slit_uv_axis] == 1
        self._along_slit_len = self._fov_uv_shape[self._along_slit_uv_axis]
        self._cross_slit_len = lines

        self.shape = len(axes) * [0]
        self.shape[self.u_axis] = self.uv_shape[0]
        self.shape[self.v_axis] = self.uv_shape[1]

        # Timing
        self.time = self.cadence.time
        self.midtime = self.cadence.midtime

        # Optional subfields
        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

        # Snapshot class proxy (for inventory)
        replacements = {
            'ut': 'u',
            'vt': 'v',
        }

        snapshot_axes = [replacements.get(axis, axis) for axis in axes]
        snapshot_tstart = self.cadence.time[0]
        snapshot_texp = self.cadence.time[1] - self.cadence.time[0]

        self.snapshot = Snapshot(snapshot_axes, snapshot_tstart, snapshot_texp,
                                 self.fov, self.path, self.frame, **subfields)

    def __getstate__(self):
        return (self.axes, self.cadence, self.fov, self.path, self.frame,
                self.subfields)

    def __setstate__(self, state):
        self.__init__(*state[:-1], **state[-1])

    #===========================================================================
    def uvt(self, indices, remask=False):
        """Coordinates (u,v) and time t for indices into the data array.

        This method supports non-integer index values.

        Input:
            indices     a Scalar or Vector of array indices.
            remask      True to mask values outside the field of view.

        Return:         (uv, time)
            uv          a Pair defining the values of (u,v) within the FOV that
                        are associated with the array indices.
            time        a Scalar defining the time in seconds TDB associated
                        with the array indices.
        """

        # Get the slit coordinate
        indices = Vector.as_vector(indices)
        slit_coord = indices.to_scalar(self._along_slit_axis)

        # Get the times
        tstep = indices.to_scalar(self.t_axis)
        time = self.cadence.time_at_tstep(tstep, remask=remask)

        # Re-mask the time if necessary
        if remask:
            is_outside = ((slit_coord.vals < 0) |
                          (slit_coord.vals > self._along_slit_len))
            time = time.mask_where(is_outside)

        # Create (u,v) Pair
        uv_vals = np.empty(indices.shape + (2,))
        uv_vals[..., self._along_slit_uv_axis] = slit_coord.vals
        uv_vals[..., self._cross_slit_uv_axis] = 0.5
        uv = Pair(uv_vals, time.mask)       # shared mask

        return (uv, time)

    #===========================================================================
    def uvt_range(self, indices, remask=False):
        """Ranges of FOV coordinates and time for integer array indices.

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

        # Get the slit coordinate
        indices = Vector.as_vector(indices)
        slit_coord = indices.to_scalar(self._along_slit_axis)
        slit_int = slit_coord.int(top=self._along_slit_len, remask=remask)

        # Get the times
        tstep = indices.to_scalar(self.t_axis)
        (time_min,
         time_max) = self.cadence.time_range_at_tstep(tstep, remask=remask)

        # Merge masks if necessary
        if remask:
            time_min = time_min.mask_where(slit_int.mask)
            time_max = time_max.mask_where(slit_int.mask)

        # Create (u,v) Pair
        uv_min_vals = np.empty(indices.shape + (2,), dtype='int')
        uv_min_vals[..., self._along_slit_uv_axis] = slit_int.vals
        uv_min_vals[..., self._cross_slit_uv_axis] = 0
        uv_min = Pair(uv_min_vals, time_min.mask)   # shared mask

        return (uv_min, uv_min + self._fov_uv_shape, time_min, time_max)

    #===========================================================================
    def uv_range_at_tstep(self, tstep, remask=False):
        """The range of integer spatial (u,v) coordinates active at the given
        time step.

        Input:
            tstep       a Scalar or Pair time step index.
            remask      True to mask values outside the time interval.

        Return:         a tuple (uv_min, uv_max)
            uv_min      an integer Pair defining the minimum integer values of
                        FOV (u,v)
                        coordinates active at this time step.
            uv_min      a Pair defining the maximum integer values of FOV (u,v)
                        coordinates active at this time step (exclusive).
        """

        tstep = Scalar.as_scalar(tstep)
        cross_slit_int = tstep.int(top=self._cross_slit_len, remask=remask)

        uv_min_vals = np.zeros(tstep.shape + (2,), dtype='int')
        uv_min_vals[..., self._cross_slit_axis] = cross_slit_int.vals
        uv_min = Pair(uv_min_vals, cross_slit_int.mask)

        return (uv_min, uv_min + self.fov.uv_shape)

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

        tstep = Pair.as_pair(uv_pair).to_scalar(self._cross_slit_uv_axis)
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

        return Observation.uv_range_at_time_1d(self, time, remask,
                                               self._along_slit_uv_index)

    #===========================================================================
    def time_shift(self, dtime):
        """A copy of the observation object with a time-shift.

        Input:
            dtime       the time offset to apply to the observation, in units of
                        seconds. A positive value shifts the observation later.

        Return:         a (shallow) copy of the object with a new time.
        """

        obs = Slit(self.axes, self.cadence.time_shift(dtime), self.fov,
                   self.path, self.frame)

        for key in self.subfields.keys():
            obs.insert_subfield(key, self.subfields[key])

        return obs

    #===========================================================================
    def inventory(self, *args, **kwargs):
        """Info about the bodies that appear unobscured inside the FOV. See
        Snapshot.inventory() for details.

        WARNING: Not properly updated for class Slit. Use at your own risk. This
        operates by returning every body that would have been inside the FOV of
        this observation if it were instead a Snapshot, evaluated at the given
        tfrac.
        """

        return self.snapshot.inventory(*args, **kwargs)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Slit(unittest.TestCase):

    def runTest(self):

        from ..cadence.metronome import Metronome
        from ..fov.flatfov import FlatFOV

        fov = FlatFOV((0.001,0.001), (10,1))
        cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        obs = Slit(axes=('u','vt'), cadence=cadence, fov=fov, path='SSB', frame='J2000')

        indices = Vector([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])
        indices_ = indices.copy()
        indices_.vals[:,0][indices_.vals[:,0] == 10] -= 1
        indices_.vals[:,1][indices_.vals[:,1] == 20] -= 1

        # uvt() with remask == False
        (uv, time) = obs.uvt(indices)

        self.assertFalse(np.any(uv.mask))
        self.assertFalse(np.any(time.mask))
        self.assertEqual(time, cadence.tstride * indices.to_scalar(1))
        self.assertEqual(uv.to_scalar(0), indices.to_scalar(0))
        self.assertEqual(uv.to_scalar(1), 0.5)

        # uvt() with remask == True
        (uv, time) = obs.uvt(indices, remask=True)

        self.assertTrue(np.all(uv.mask == np.array(6*[False] + [True])))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time[:6], cadence.tstride * indices.to_scalar(1)[:6])
        self.assertEqual(uv[:6].to_scalar(0), indices[:6].to_scalar(0))
        self.assertEqual(uv[:6].to_scalar(1), 0.5)

        # uvt_range() with remask == False
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertFalse(np.any(uv_min.mask))
        self.assertFalse(np.any(uv_max.mask))
        self.assertFalse(np.any(time_min.mask))
        self.assertFalse(np.any(time_max.mask))

        self.assertEqual(uv_min.to_scalar(0), indices_.to_scalar(0))
        self.assertEqual(uv_min.to_scalar(1), 0)
        self.assertEqual(uv_max.to_scalar(0), indices_.to_scalar(0) +
                                              obs.fov.uv_shape.vals[0])
        self.assertEqual(uv_max.to_scalar(1), 1)
        self.assertEqual(time_min, cadence.tstride * indices_.to_scalar(1))
        self.assertEqual(time_max, time_min + cadence.texp)

        # uvt_range() with remask == True
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices,
                                                             remask=True)

        self.assertTrue(np.all(uv_min.mask == np.array(6*[False] + [True])))
        self.assertTrue(np.all(uv_max.mask == uv_min.mask))
        self.assertTrue(np.all(time_min.mask == uv_min.mask))
        self.assertTrue(np.all(time_max.mask == uv_min.mask))

        self.assertEqual(uv_min.to_scalar(0)[:6], indices_.to_scalar(0)[:6])
        self.assertEqual(uv_min.to_scalar(1)[:6], 0)
        self.assertEqual(uv_max.to_scalar(0)[:6], indices_.to_scalar(0)[:6] +
                                                  obs.fov.uv_shape.vals[0])
        self.assertEqual(uv_max.to_scalar(1)[:6], 1)
        self.assertEqual(time_min[:6], cadence.tstride * indices_.to_scalar(1)[:6])
        self.assertEqual(time_max[:6], time_min[:6] + cadence.texp)

        # time_range_at_uv() with remask == False
        uv = Pair([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])

        (time0, time1) = obs.time_range_at_uv(uv)
        uv_ = uv.copy()
        uv_.vals[:,0][uv_.vals[:,0] == 10] -= 1
        uv_.vals[:,1][uv_.vals[:,1] == 20] -= 1

        self.assertEqual(time0, cadence.tstride * uv_.to_scalar(1))
        self.assertEqual(time1, time0 + cadence.texp)

        # time_range_at_uv() with remask == True
        (time0, time1) = obs.time_range_at_uv(uv, remask=True)

        self.assertTrue(np.all(time0.mask == 6*[False] + [True]))
        self.assertTrue(np.all(time1.mask == time0.mask))
        self.assertEqual(time0[:6], cadence.tstride * uv_.to_scalar(1)[:6])
        self.assertEqual(time1[:6], time0[:6] + cadence.texp)

        ####################################

        # Alternative axis order ('ut','v')

        fov = FlatFOV((0.001,0.001), (1,20))
        cadence = Metronome(tstart=0., tstride=10., texp=10., steps=10)
        obs = Slit(axes=('ut','v'), cadence=cadence, fov=fov, path='SSB', frame='J2000')

        indices = Vector([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])
        indices_ = indices.copy()
        indices_.vals[:,0][indices_.vals[:,0] == 10] -= 1
        indices_.vals[:,1][indices_.vals[:,1] == 20] -= 1

        (uv,time) = obs.uvt(indices)

        self.assertEqual(uv.to_scalar(0), 0.5)
        self.assertEqual(uv.to_scalar(1), indices.to_scalar(1))
        self.assertEqual(time, cadence.tstride * indices.to_scalar(0))

        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertEqual(uv_min.to_scalar(0), 0)
        self.assertEqual(uv_min.to_scalar(1), indices_.to_scalar(1))
        self.assertEqual(uv_max.to_scalar(0), 1)
        self.assertEqual(uv_max.to_scalar(1), indices_.to_scalar(1) + 20)
        self.assertEqual(time_min, cadence.tstride * indices_.to_scalar(0))
        self.assertEqual(time_max, time_min + cadence.texp)

        ####################################

        # Alternative texp for discontinuous indices

        fov = FlatFOV((0.001,0.001), (1,20))
        cadence = Metronome(tstart=0., tstride=10., texp=8., steps=10)
        obs = Slit(axes=('ut','v'), cadence=cadence, fov=fov, path='SSB', frame='J2000')

        self.assertEqual(obs.time[1], 98.)

        self.assertEqual(obs.uvt((0,0))[1],  0.)
        self.assertEqual(obs.uvt((5,0))[1], 50.)
        self.assertEqual(obs.uvt((5,5))[1], 50.)

        eps = 1.e-14
        delta = 1.e-13
        self.assertTrue(abs(obs.uvt((6      ,0))[1] - 60.) < delta)
        self.assertTrue(abs(obs.uvt((6.25   ,0))[1] - 62.) < delta)
        self.assertTrue(abs(obs.uvt((6.5    ,0))[1] - 64.) < delta)
        self.assertTrue(abs(obs.uvt((6.75   ,0))[1] - 66.) < delta)
        self.assertTrue(abs(obs.uvt((7 - eps,0))[1] - 68.) < delta)
        self.assertTrue(abs(obs.uvt((7.     ,0))[1] - 70.) < delta)

        self.assertEqual(obs.uvt((0,0))[0], (0.5,0.))
        self.assertEqual(obs.uvt((5,0))[0], (0.5,0.))
        self.assertEqual(obs.uvt((5,5))[0], (0.5,5.))

        self.assertTrue(abs(obs.uvt((6      ,0))[0] - (0.5,0.)) < delta)
        self.assertTrue(abs(obs.uvt((6.2    ,1))[0] - (0.5,1.)) < delta)
        self.assertTrue(abs(obs.uvt((6.4    ,2))[0] - (0.5,2.)) < delta)
        self.assertTrue(abs(obs.uvt((6.6    ,3))[0] - (0.5,3.)) < delta)
        self.assertTrue(abs(obs.uvt((6.8    ,4))[0] - (0.5,4.)) < delta)
        self.assertTrue(abs(obs.uvt((7 - eps,5))[0] - (0.5,5.)) < delta)
        self.assertTrue(abs(obs.uvt((7.     ,6))[0] - (0.5,6.)) < delta)

        # Test using scalar indices
        below = obs.uvt((0,20 - eps), remask=True)[0].to_scalar(1)
        exact = obs.uvt((0,20      ), remask=True)[0].to_scalar(1)
        above = obs.uvt((0,20 + eps), remask=True)[0].to_scalar(1)

        self.assertTrue(below < 20.)
        self.assertTrue(20. - below < delta)
        self.assertTrue(exact == 20.)
        self.assertTrue(above == Scalar.MASKED)
        self.assertTrue(above.mask)

        # Test using a Vector index
        indices = Vector([(0,20 - eps), (0,20), (0,20 + eps)])

        u = obs.uvt(indices, remask=True)[0].to_scalar(1)
        self.assertTrue(u == (below, exact, above))

        # Alternative texp and axes
        cadence = Metronome(tstart=0., tstride=10., texp=8., steps=10)
        obs = Slit(axes=('a','v','b','ut','c'), cadence=cadence, fov=fov, path='SSB', frame='J2000')

        self.assertEqual(obs.time[1], 98.)

        self.assertEqual(obs.uvt((1,0,3,0,4))[1],  0.)
        self.assertEqual(obs.uvt((1,0,3,5,4))[1], 50.)
        self.assertEqual(obs.uvt((1,0,3,5,4))[1], 50.)

        eps = 1.e-15
        delta = 1.e-13
        self.assertTrue(abs(obs.uvt((1,0,0,6      ,0))[1] - 60.) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,6.25   ,0))[1] - 62.) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,6.5    ,0))[1] - 64.) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,6.75   ,0))[1] - 66.) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,7 - eps,0))[1] - 68.) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,7.     ,0))[1] - 70.) < delta)

        self.assertEqual(obs.uvt((0,0,0,0,0))[0], (0.5,0.))
        self.assertEqual(obs.uvt((0,0,0,5,0))[0], (0.5,0.))
        self.assertEqual(obs.uvt((0,5,0,5,0))[0], (0.5,5.))

        self.assertTrue(abs(obs.uvt((1,0,4,6      ,7))[0] - (0.5,0.)) < delta)
        self.assertTrue(abs(obs.uvt((1,1,4,6.2    ,7))[0] - (0.5,1.)) < delta)
        self.assertTrue(abs(obs.uvt((1,2,4,6.4    ,7))[0] - (0.5,2.)) < delta)
        self.assertTrue(abs(obs.uvt((1,3,4,6.6    ,7))[0] - (0.5,3.)) < delta)
        self.assertTrue(abs(obs.uvt((1,4,4,6.8    ,7))[0] - (0.5,4.)) < delta)
        self.assertTrue(abs(obs.uvt((1,5,4,7 - eps,7))[0] - (0.5,5.)) < delta)
        self.assertTrue(abs(obs.uvt((1,6,4,7.     ,7))[0] - (0.5,6.)) < delta)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
