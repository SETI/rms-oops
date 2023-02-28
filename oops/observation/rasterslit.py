################################################################################
# oops/observation/rasterslit.py: Subclass RasterSlit of class Observation
################################################################################

import numpy as np
from polymath import Boolean, Pair, Vector

from oops.observation          import Observation
from oops.observation.snapshot import Snapshot
from oops.cadence              import Cadence
from oops.cadence.dualcadence  import DualCadence
from oops.frame                import Frame
from oops.path                 import Path

from oops.observation.timedimage import TimedImage

class RasterSlit(TimedImage):
    """A subclass of Observation consisting of a 2-D image in which one
    dimension is constructed by sweeping a single pixel along a slit, and the
    second dimension is simulated by rotation of the camera.

    The FOV describes the 1-D slit. This differs from a Slit subclass in
    that a single sensor is moving to emulate the 1-D slit. It differs from
    a RasterScan in that the FOV describes a 1-D slit and not a 2-D image.
    """

    pass

#     INVENTORY_IMPLEMENTED = True
#
#     # Relates these axes to Snapshot axes
#     AXIS_REPLACEMENTS = {
#         'ufast':  'u',
#         'uslow':  'u',
#         'vfast':  'v',
#         'vslow':  'v',
#     }
#
#     #===========================================================================
#     def __init__(self, axes, cadence, fov, path, frame, **subfields):
#         """Constructor for a RasterSlit observation.
#
#         Input:
#             axes        a list or tuple of strings, with one value for each axis
#                         in the associated data array. A value of 'ufast' or
#                         'uslow' should appear at the location of the array's
#                         u-axis; 'vslow' or 'vfast' should appear at the location
#                         of the array's v-axis. The 'fast' suffix identifies
#                         which of these is in the fast-scan direction; the 'slow'
#                         suffix identifies the slow-scan direction.
#
#             cadence     a 2-D Cadence object defining the start time and
#                         duration of each consecutive measurement. Alternatively,
#                         this input can be a tuple or dictionary providing input
#                         arguments to the constructor DualCadence.for_array2d()
#                         (excluding the number of samples, which is defined by
#                         the FOV):
#                             (lines, tstart, texp,
#                                 [intersample_delay [, interline_delay]])
#
#             fov         a FOV (field-of-view) object, which describes the field
#                         of view including any spatial distortion. It maps
#                         between spatial coordinates (u,v) and instrument
#                         coordinates (x,y). For a RasterSlit object, one of the
#                         axes of the FOV must have length 1, and the other must
#                         match the length of the cadence's fast axis.
#
#             path        the path waypoint co-located with the instrument.
#
#             frame       the wayframe of a coordinate frame fixed to the optics
#                         of the instrument. This frame should have its Z-axis
#                         pointing outward near the center of the line of sight,
#                         with the X-axis pointing rightward and the y-axis
#                         pointing downward.
#
#             subfields   a dictionary containing all of the optional attributes.
#                         Additional subfields may be included as needed.
#         """
#
#         # Basic properties
#         self.path = Path.as_waypoint(path)
#         self.frame = Frame.as_wayframe(frame)
#
#         # FOV
#         self.fov = fov
#         fov_uv_shape = tuple(self.fov.uv_shape.vals)
#
#         # Axes
#         self.axes = list(axes)
#         assert (('ufast' in self.axes and 'vslow' in self.axes) or
#                 ('vfast' in self.axes and 'uslow' in self.axes))
#
#         if 'ufast' in self.axes:
#             self.u_axis = self.axes.index('ufast')
#             self.v_axis = self.axes.index('vslow')
#             self._fast_axis = self.u_axis
#             self._slow_axis = self.v_axis
#             self._cross_slit_uv_axis = 1
#             self._along_slit_uv_axis = 0
#         else:
#             self.u_axis = self.axes.index('uslow')
#             self.v_axis = self.axes.index('vfast')
#             self._fast_axis = self.v_axis
#             self._slow_axis = self.u_axis
#             self._cross_slit_uv_axis = 0
#             self._along_slit_uv_axis = 1
#
#         self.swap_uv = (self.u_axis > self.v_axis)
#         self.t_axis = (self._slow_axis, self._fast_axis)
#
#         # Cadence
#         samples = fov_uv_shape[self._along_slit_uv_axis]
#
#         if isinstance(cadence, (tuple,list)):
#             self.cadence = DualCadence.for_array2d(samples, *cadence)
#         elif isinstance(cadence, dict):
#             self.cadence = DualCadence.for_array2d(samples, **cadence)
#         elif isinstance(cadence, Cadence):
#             self.cadence = cadence
#             assert len(self.cadence.shape) == 2
#             assert self.cadence.shape[1] == samples
#         else:
#             raise TypeError('Invalid cadence class: ' + type(cadence).__name__)
#
#         # Shape / Size
#         lines = self.cadence.shape[0]
#
#         uv_shape = [0, 0]
#         uv_shape[self._cross_slit_uv_axis] = lines
#         uv_shape[self._along_slit_uv_axis] = samples
#         self.uv_shape = tuple(uv_shape)
#
#         self._cross_slit_len = lines
#         self._along_slit_len = samples
#
#         assert fov_uv_shape[self._cross_slit_uv_axis] == 1
#         assert fov_uv_shape[self._along_slit_uv_axis] == samples
#
#         self.shape = len(axes) * [0]
#         self.shape[self.u_axis] = self.uv_shape[0]
#         self.shape[self.v_axis] = self.uv_shape[1]
#
#         # Timing
#         self.time = self.cadence.time
#         self.midtime = self.cadence.midtime
#
#         # Optional subfields
#         self.subfields = {}
#         for key in subfields.keys():
#             self.insert_subfield(key, subfields[key])
#
#         # Snapshot class proxy (for inventory)
#         snapshot_axes = [RasterSlit.AXIS_REPLACEMENTS.get(axis, axis)
#                          for axis in axes]
#         snapshot_tstart = self.cadence.time[0]
#         snapshot_texp = self.cadence.time[1] - self.cadence.time[0]
#
#         self.snapshot = Snapshot(snapshot_axes, snapshot_tstart, snapshot_texp,
#                                  self.fov, self.path, self.frame, **subfields)
#
#     def __getstate__(self):
#         return (self.axes, self.cadence, self.fov, self.path, self.frame,
#                 self.subfields)
#
#     def __setstate__(self, state):
#         self.__init__(*state[:-1], **state[-1])
#
#     #===========================================================================
#     def uvt(self, indices, remask=False, derivs=True):
#         """Coordinates (u,v) and time t for indices into the data array.
#
#         This method supports non-integer index values.
#
#         Input:
#             indices     a Scalar or Vector of array indices.
#             remask      True to mask values outside the field of view.
#             derivs      True to include derivatives in the returned values.
#
#         Return:         (uv, time)
#             uv          a Pair defining the values of (u,v) within the FOV that
#                         are associated with the array indices.
#             time        a Scalar defining the time in seconds TDB associated
#                         with the array indices.
#         """
#
#         indices = Vector.as_vector(indices, recursive=derivs)
#
#         # Create time Scalar
#         tstep = indices.to_pair(self.t_axis)
#         time = self.cadence.time_at_tstep(tstep, remask=remask)
#             # tstep is 2-D so this re-masks where either axis is out of range
#
#         # Interpret the slit coordinate as u or v
#         slit_coord = indices.to_scalar(self._fast_axis)
#
#         # Create (u,v) Pair
#         uv_vals = np.empty(indices.shape + (2,))
#         uv_vals[..., self._along_slit_uv_axis] = slit_coord.vals
#         uv_vals[..., self._cross_slit_uv_axis] = 0.5
#         uv = Pair(uv_vals, mask=time.mask)
#
#         return (uv, time)
#
#     #===========================================================================
#     def uvt_range(self, indices, remask=False):
#         """Ranges of (u,v) spatial coordinates and time for integer array
#         indices.
#
#         Input:
#             indices     a Scalar or Vector of array indices.
#             remask      True to mask values outside the field of view.
#
#         Return:         (uv_min, uv_max, time_min, time_max)
#             uv_min      a Pair defining the minimum values of FOV (u,v)
#                         associated the pixel.
#             uv_max      a Pair defining the maximum values of FOV (u,v)
#                         associated the pixel.
#             time_min    a Scalar defining the minimum time associated with the
#                         array indices. It is given in seconds TDB.
#             time_max    a Scalar defining the maximum time value.
#         """
#
#         indices = Vector.as_vector(indices, recursive=False)
#
#         # Get the time range
#         tstep = indices.to_pair(self.t_axis)
#         tstep_int = tstep.int(top=self.cadence.shape, remask=remask)
#             # tstep_int is 2-D so this re-masks for either axis out of range
#
#         (time_min,
#          time_max) = self.cadence.time_range_at_tstep(tstep_int, remask=False)
#
#         # Interpret the slit coordinate as u or v
#         slit_coord = indices.to_scalar(self._fast_axis)
#         slit_int = slit_coord.int(top=self._along_slit_len, remask=False,
#                                                             inclusive=True)
#
#         # Create (u,v) Pair
#         uv_min_vals = np.zeros(indices.shape + (2,))
#         uv_min_vals[..., self._along_slit_uv_axis] = slit_int.vals
#         uv_min = Pair(uv_min_vals, mask=tstep_int.mask)
#
#         return (uv_min, uv_min + Pair.INT11, time_min, time_max)
#
#     #===========================================================================
#     def uv_range_at_tstep(self, tstep, remask=False):
#         """A tuple defining the range of spatial (u,v) pixels active at a
#         particular time step.
#
#         Input:
#             tstep       a Pair time step index.
#             remask      True to mask values outside the time interval.
#
#         Return:         a tuple (uv_min, uv_max)
#             uv_min      a Pair defining the minimum values of FOV (u,v)
#                         coordinates active at this time step.
#             uv_min      a Pair defining the maximum values of FOV (u,v)
#                         coordinates active at this time step (exclusive).
#         """
#
#         tstep = Pair.as_pair(tstep)
#         tstep_int = tstep.int(top=self.cadence.shape, remask=remask)
#
#         if self._slow_uv_axis == 0:
#             uv_min = tstep_int
#         else:
#             uv_min = tstep_int.swapxy()
#
#         return (uv_min, uv_min + Pair.INT11)
#
#     #===========================================================================
#     def time_range_at_uv(self, uv_pair, remask=False):
#         """The start and stop times of the specified spatial pixel (u,v).
#
#         Input:
#             uv_pair     a Pair of spatial (u,v) data array coordinates,
#                         truncated to integers if necessary.
#             remask      True to mask values outside the field of view.
#
#         Return:         a tuple containing Scalars of the start time and stop
#                         time of each (u,v) pair, as seconds TDB.
#         """
#
#         uv_pair = Pair.as_pair(uv_pair)
#         uv_pair_int = uv_pair.int(shape=self.uv_shape, remask=remask)
#
#         if self._cross_slit_uv_axis == 0:
#             tstep_int = uv_pair_int
#         else:
#             tstep_int = uv_pair_int.swapxy()
#
#         return self.cadence.time_range_at_tstep(tstep_int, remask=remask)
#
#     #===========================================================================
#     def uv_range_at_time(self, time, remask=False):
#         """The (u,v) range of spatial pixels observed at the specified time.
#
#         Input:
#             time        a Scalar of time values in seconds TDB.
#             remask      True to mask values outside the time limits.
#
#         Return:         (uv_min, uv_max)
#             uv_min      the lower (u,v) corner Pair of the area observed at the
#                         specified time.
#             uv_max      the upper (u,v) corner Pair of the area observed at the
#                         specified time.
#         """
#
#         return Observation.uv_range_at_time_2d(self, time,
#                                                uv_shape=Pair.INT11,
#                                                slow=self._cross_slit_uv_axis,
#                                                fast=self._along_slit_uv_axis,
#                                                remask=remask)
#
#     #===========================================================================
#     def time_shift(self, dtime):
#         """A copy of the observation object with a time-shift.
#
#         Input:
#             dtime       the time offset to apply to the observation, in units of
#                         seconds. A positive value shifts the observation later.
#
#         Return:         a (shallow) copy of the object with a new time.
#         """
#
#         obs = RasterSlit(axes=self.axes, cadence=self.cadence.time_shift(dtime),
#                          fov=self.fov, path=self.path, frame=self.frame)
#
#         for key in self.subfields.keys():
#             obs.insert_subfield(key, self.subfields[key])
#
#         return obs
#
#     #===========================================================================
#     def inventory(self, *args, **kwargs):
#         """Info about the bodies that appear unobscured inside the FOV. See
#         Snapshot.inventory() for details.
#
#         WARNING: Not properly updated for class RasterSlit. Use at your own
#         risk. This operates by returning every body that would have been inside
#         the FOV of this observation if it were instead a Snapshot, evaluated at
#         the given tfrac.
#         """
#
#         return self.snapshot.inventory(*args, **kwargs)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_RasterSlit(unittest.TestCase):

    def runTest(self):

        from oops.cadence.metronome import Metronome
        from oops.cadence.dualcadence import DualCadence
        from oops.fov.flatfov import FlatFOV

        fov = FlatFOV((0.001,0.001), (10,1))
        slow_cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        fast_cadence = Metronome(tstart=0., tstride=1., texp=1., steps=10)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterSlit(axes=('ufast','vslow'), cadence=cadence, fov=fov,
                         path='SSB', frame='J2000')

        indices = Pair([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])
        indices_ = indices.copy()
        indices_.vals[:,0][indices.vals[:,0] == 10] -= 1
        indices_.vals[:,1][indices.vals[:,1] == 20] -= 1

        # uvt() with remask == False
        (uv, time) = obs.uvt(indices)

        self.assertFalse(np.any(uv.mask))
        self.assertFalse(np.any(time.mask))
        self.assertEqual(time, [0, 100, 190, 10, 110, 200, 200])
        self.assertEqual(uv.to_scalar(0), indices.to_scalar(0))
        self.assertEqual(uv.to_scalar(1), 0.5)

        # uvt() with remask == True
        (uv, time) = obs.uvt(indices, remask=True)

        self.assertTrue(np.all(uv.mask == np.array(6*[False] + [True])))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time[:6], [0, 100, 190, 10, 110, 200])
        self.assertEqual(uv[:6].to_scalar(0), indices[:6].to_scalar(0))
        self.assertEqual(uv[:6].to_scalar(1), 0.5)

        # uvt() with remask == True, new indices
        non_ints = indices + (0.2, 0.9)
        (uv, time) = obs.uvt(non_ints, remask=True)

        self.assertTrue(np.all(uv.mask == np.array(2*[False] + 5*[True])))
        self.assertTrue(np.all(time.mask == uv.mask))

        # uvt_range() with remask == False
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(non_ints)

        self.assertFalse(np.any(uv_min.mask))
        self.assertFalse(np.any(uv_max.mask))
        self.assertFalse(np.any(time_min.mask))
        self.assertFalse(np.any(time_max.mask))

        self.assertEqual(uv_min.to_scalar(0), indices.to_scalar(0))
        self.assertEqual(uv_min.to_scalar(1), 0)
        self.assertEqual(uv_max.to_scalar(0), indices.to_scalar(0) + 1)
        self.assertEqual(uv_max.to_scalar(1), 1)
        self.assertEqual(time_min, [0, 100, 190,  9, 109, 199, 199])
        self.assertEqual(time_max, time_min + fast_cadence.texp)

        # uvt_range() with remask == True
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(non_ints,
                                                             remask=True)

        self.assertEqual(Boolean(uv_min.mask), 2*[False] + 5*[True])
        self.assertEqual(Boolean(uv_max.mask), uv_min.mask)
        self.assertEqual(Boolean(time_min.mask), uv_min.mask)
        self.assertEqual(Boolean(time_max.mask), uv_min.mask)

        self.assertEqual(uv_min.to_scalar(0)[:2], indices.to_scalar(0)[:2])
        self.assertEqual(uv_min.to_scalar(1)[:2], 0)
        self.assertEqual(uv_max.to_scalar(0)[:2], indices.to_scalar(0)[:2] + 1)
        self.assertEqual(uv_max.to_scalar(1)[:2], 1)
        self.assertEqual(time_min[:2],
                         (slow_cadence.tstride * indices.to_scalar(1) +
                          fast_cadence.tstride * indices.to_scalar(0))[:2])
        self.assertEqual(time_max[:2], time_min[:2] + fast_cadence.texp)

        # time_range_at_uv() with remask == False
        uv = Pair([(0,0),(0,20),(10,0),(10,20),(10,21)])

        (time0, time1) = obs.time_range_at_uv(uv)

        self.assertEqual(time0, [0, 190, 9, 199, 199])
        self.assertEqual(time1, time0 + fast_cadence.texp)

        # time_range_at_uv() with remask == True
        (time0, time1) = obs.time_range_at_uv(uv, remask=True)

        self.assertTrue(np.all(time0.mask == 4*[False] + [True]))
        self.assertTrue(np.all(time1.mask == 4*[False] + [True]))
        self.assertEqual(time0[:4], [0, 190, 9, 199])
        self.assertEqual(time1[:4], time0[:4] + fast_cadence.texp)

        ####################################
        # Alternative axis order ('uslow','vfast')
        ####################################

        fov = FlatFOV((0.001,0.001), (1,20))
        slow_cadence = Metronome(tstart=0., tstride=10., texp=10., steps=10)
        fast_cadence = Metronome(tstart=0., tstride=0.5, texp=0.5, steps=20)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterSlit(axes=('uslow','vfast'), cadence=cadence, fov=fov, path='SSB', frame='J2000')

        indices = Pair([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])
        indices_ = indices.copy()
        indices_.vals[:,0][indices.vals[:,0] == 10] -= 1
        indices_.vals[:,1][indices.vals[:,1] == 20] -= 1

        (uv, time) = obs.uvt(indices)

        self.assertEqual(uv.to_scalar(0), 0.5)
        self.assertEqual(uv.to_scalar(1), indices.to_scalar(1))
        self.assertEqual(time, [0, 5, 10, 90, 95, 100, 100])

        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertEqual(uv_min.to_scalar(0), 0)
        self.assertEqual(uv_min.to_scalar(1), indices_.to_scalar(1))
        self.assertEqual(uv_max.to_scalar(0), 1)
        self.assertEqual(uv_max.to_scalar(1), indices_.to_scalar(1) + 1)
        self.assertEqual(time_min, [0, 5, 9.5, 90, 95, 99.5, 99.5])
        self.assertEqual(time_max, time_min + fast_cadence.texp)

        (time0, time1) = obs.time_range_at_uv(indices)

        self.assertEqual(time0, time_min)
        self.assertEqual(time1, time0 + fast_cadence.texp)

        ################################################
        # Alternative texp for discontinuous indices
        ################################################

        fov = FlatFOV((0.001,0.001), (10,1))
        slow_cadence = Metronome(tstart=0., tstride=10., texp=8., steps=20)
        fast_cadence = Metronome(tstart=0., tstride=1., texp=0.5, steps=10)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterSlit(axes=('ufast','vslow'), cadence=cadence, fov=fov,
                                                 path='SSB', frame='J2000')

        self.assertEqual(obs.time[1], 199.5)

        self.assertEqual(obs.uvt((0,0))[1],  0.)
        self.assertEqual(obs.uvt((5,0))[1],  5.)
        self.assertEqual(obs.uvt((5,5))[1], 55.)
        self.assertEqual(obs.uvt((5.0, 5.5))[1], 55.)
        self.assertEqual(obs.uvt((5.5, 5.0))[1], 55.25)

        eps = 1.e-15
        delta = 1.e-13
        self.assertTrue(abs(obs.uvt((6.     ,0))[1] - 6.000) < delta)
        self.assertTrue(abs(obs.uvt((6.25   ,0))[1] - 6.125) < delta)
        self.assertTrue(abs(obs.uvt((6.5    ,0))[1] - 6.250) < delta)
        self.assertTrue(abs(obs.uvt((6.75   ,0))[1] - 6.375) < delta)
        self.assertTrue(abs(obs.uvt((7 - eps,0))[1] - 6.500) < delta)
        self.assertTrue(abs(obs.uvt((7.     ,0))[1] - 7.000) < delta)

        self.assertEqual(obs.uvt((0,0))[0], (0.,0.5))
        self.assertEqual(obs.uvt((5,0))[0], (5.,0.5))
        self.assertEqual(obs.uvt((5,5))[0], (5.,0.5))

        self.assertTrue(abs(obs.uvt((6.     ,0))[0] - (6.0,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((6.2    ,1))[0] - (6.2,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((6.4    ,2))[0] - (6.4,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((6.6    ,3))[0] - (6.6,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((6.8    ,4))[0] - (6.8,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((7.     ,6))[0] - (7.0,0.5)) < delta)

        self.assertTrue(abs(obs.uvt((1, 0      ))[0] - (1.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((2, 1.25   ))[0] - (2.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((3, 2.5    ))[0] - (3.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((4, 3.75   ))[0] - (4.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((5, 5 - eps))[0] - (5.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((5, 5.     ))[0] - (5.,0.5)) < delta)

        ################################################
        # Alternative tstride for even more discontinuous indices
        ################################################

        fov = FlatFOV((0.001,0.001), (10,1))
        slow_cadence = Metronome(tstart=0., tstride=11., texp=10., steps=20)
        fast_cadence = Metronome(tstart=0., tstride=1., texp=0.8, steps=10)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterSlit(axes=('ufast','vslow'), cadence=cadence, fov=fov,
                                                 path='SSB', frame='J2000')

        self.assertEqual(obs.time[1], 218.8)

        self.assertEqual(obs.uvt((0,0))[1],  0.)
        self.assertEqual(obs.uvt((5,0))[1],  5.)
        self.assertEqual(obs.uvt((5,5))[1], 60.)
        self.assertEqual(obs.uvt((5.0, 5.5))[1], 60.)
        self.assertEqual(obs.uvt((5.5, 5.0))[1], 60.4)
        self.assertEqual(obs.uvt((5.5, 5.5))[1], 60.4)

        eps = 1.e-14
        delta = 1.e-13
        self.assertTrue(abs(obs.uvt((6      ,0))[1] - 6. ) < delta)
        self.assertTrue(abs(obs.uvt((6.25   ,0))[1] - 6.2) < delta)
        self.assertTrue(abs(obs.uvt((6.5    ,0))[1] - 6.4) < delta)
        self.assertTrue(abs(obs.uvt((6.75   ,0))[1] - 6.6) < delta)
        self.assertTrue(abs(obs.uvt((7 - eps,0))[1] - 6.8) < delta)
        self.assertTrue(abs(obs.uvt((7.     ,0))[1] - 7.0) < delta)

        self.assertTrue(abs(obs.uvt((9       ,0))[1] -  9. ) < delta)
        self.assertTrue(abs(obs.uvt((9.25    ,0))[1] -  9.2) < delta)
        self.assertTrue(abs(obs.uvt((9.5     ,0))[1] -  9.4) < delta)
        self.assertTrue(abs(obs.uvt((9.75    ,0))[1] -  9.6) < delta)
        self.assertTrue(abs(obs.uvt((10 - eps,0))[1] -  9.8) < delta)
        self.assertTrue(abs(obs.uvt((0.      ,1))[1] - 11. ) < delta)

        self.assertTrue(abs(obs.uvt((6.00, 0.   ))[1] -  6. ) < delta)
        self.assertTrue(abs(obs.uvt((6.25, 0.   ))[1] -  6.2) < delta)
        self.assertTrue(abs(obs.uvt((6.25, 1.   ))[1] - 17.2) < delta)
        self.assertTrue(abs(obs.uvt((6.25, 2-eps))[1] - 17.2) < delta)
        self.assertTrue(abs(obs.uvt((6.25, 2    ))[1] - 28.2) < delta)

        # Test the upper edge
        pair = (10-eps, 0)
        self.assertTrue((obs.uvt(pair, True)[0] - (10, 0.5)).rms() < delta)
        self.assertTrue((obs.uvt(pair, True)[1] - 9.8).abs() < delta)
        self.assertFalse(obs.uvt(pair, True)[0].mask)

        pair = (10, 0)
        self.assertTrue((obs.uvt(pair, True)[0] - (10, 0.5)).rms() < delta)
        self.assertTrue((obs.uvt(pair, True)[1] - 9.8).abs() < delta)
        self.assertFalse(obs.uvt(pair, True)[0].mask)

        pair = (10+eps, 0)
        self.assertTrue(obs.uvt(pair, True)[0].mask)

        pair = (10-eps, 1-eps)
        self.assertTrue((obs.uvt(pair, True)[0] - (10, 0.5)).rms() < delta)
        self.assertTrue((obs.uvt(pair, True)[1] - 9.8).abs() < delta)
        self.assertFalse(obs.uvt(pair, True)[0].mask)

        pair = (10, 1)
        self.assertTrue((obs.uvt(pair, True)[0] - (10, 0.5)).rms() < delta)
        self.assertTrue((obs.uvt(pair, True)[1] - 20.8).abs() < delta)
        self.assertFalse(obs.uvt(pair, True)[0].mask)

        pair = (10, 20)
        self.assertTrue((obs.uvt(pair, True)[0] - (10, 0.5)).rms() < delta)
        self.assertTrue((obs.uvt(pair, True)[1] - 218.8).abs() < delta)
        self.assertFalse(obs.uvt(pair, True)[0].mask)

        pair = (10, 20+eps)
        self.assertTrue(obs.uvt(pair, True)[0].mask)

        ################################################
        # Alternative, discontinuous and weird axes
        ################################################

        fov = FlatFOV((0.001,0.001), (10,1))
        slow_cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        fast_cadence = Metronome(tstart=0., tstride=1., texp=0.8, steps=10)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterSlit(axes=('a','vslow','b','ufast','c'), cadence=cadence, fov=fov, path='SSB', frame='J2000')

        self.assertEqual(obs.time[1], 199.8)

        self.assertEqual(obs.uvt((1,0,3,0,4))[1],   0.)
        self.assertEqual(obs.uvt((1,0,3,5,4))[1],   5.)
        self.assertEqual(obs.uvt((1,0,3,5.5,4))[1], 5.4)

        eps = 1.e-15
        delta = 1.e-13
        self.assertTrue(abs(obs.uvt((1,0,0,6      ,0))[1] - 6. ) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,6.25   ,0))[1] - 6.2) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,6.5    ,0))[1] - 6.4) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,6.75   ,0))[1] - 6.6) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,7 - eps,0))[1] - 6.8) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,7.     ,0))[1] - 7.0) < delta)

        self.assertEqual(obs.uvt((0,0,0,0,0))[0], (0.,0.5))
        self.assertEqual(obs.uvt((0,0,0,5,0))[0], (5.,0.5))
        self.assertEqual(obs.uvt((0,5,0,5,0))[0], (5.,0.5))

        self.assertTrue(abs(obs.uvt((1,0,4,6      ,7))[0] - (6.0,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1,1,4,6.2    ,7))[0] - (6.2,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1,2,4,6.4    ,7))[0] - (6.4,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1,3,4,6.6    ,7))[0] - (6.6,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1,4,4,6.8    ,7))[0] - (6.8,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1,6,4,7.     ,7))[0] - (7.0,0.5)) < delta)

        self.assertTrue(abs(obs.uvt((1, 0      ,4,1,7))[0] - (1.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1, 1.25   ,4,2,7))[0] - (2.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1, 2.5    ,4,3,7))[0] - (3.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1, 3.75   ,4,4,7))[0] - (4.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1, 5 - eps,4,5,7))[0] - (5.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1, 5.     ,4,5,7))[0] - (5.,0.5)) < delta)

########################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
