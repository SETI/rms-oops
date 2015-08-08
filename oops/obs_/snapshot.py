################################################################################
# oops/obs_/snapshot.py: Subclass Snapshot of class Observation
################################################################################

import numpy as np
from polymath import *

from oops.obs_.observation   import Observation
from oops.cadence_.metronome import Metronome
from oops.path_.path         import Path
from oops.frame_.frame       import Frame
from oops.event              import Event

class Snapshot(Observation):
    """A Snapshot is an Observation consisting of a 2-D image made up of pixels
    all exposed at the same time."""

    def __init__(self, axes, tstart, texp, fov, path, frame, **subfields):
        """Constructor for a Snapshot.

        Input:
            axes        a list or tuple of strings, with one value for each axis
                        in the associated data array. A value of 'u' should
                        appear at the location of the array's u-axis; 'v' should
                        appear at the location of the array's v-axis. For
                        example, ('v','u'), is correct for a 2-D array read from
                        an image file in FITS or VICAR format.
            tstart      the start time of the observation in seconds TDB.
            texp        exposure time of the observation in seconds.

            fov         a FOV (field-of-view) object, which describes the field
                        of view including any spatial distortion. It maps
                        between spatial coordinates (u,v) and instrument
                        coordinates (x,y).
            path        the path waypoint co-located with the instrument.
            frame       the wayframe of a coordinate frame fixed to the optics
                        of the instrument. This frame should have its Z-axis
                        pointing outward near the center of the line of sight,
                        with the X-axis pointing rightward and the y-axis
                        pointing downward.
            subfields   a dictionary containing all of the optional attributes.
                        Additional subfields may be included as needed.
        """

        self.cadence = Metronome(tstart, texp, texp, 1)
        self.fov = fov
        self.path = Path.as_waypoint(path)
        self.frame = Frame.as_wayframe(frame)

        self.axes = list(axes)
        self.u_axis = self.axes.index('u')
        self.v_axis = self.axes.index('v')
        self.uv_shape = list(self.fov.uv_shape.vals)

        self.t_axis = -1
        self.texp = texp
        self.time = self.cadence.time
        self.midtime = self.cadence.midtime

        self.scalar_time = (Scalar(self.time[0]), Scalar(self.time[1]))
        self.scalar_midtime = Scalar(self.midtime)

        self.shape = len(axes) * [0]
        self.shape[self.u_axis] = self.uv_shape[0]
        self.shape[self.v_axis] = self.uv_shape[1]

        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

    def uvt(self, indices, fovmask=False):
        """Return coordinates (u,v) and time t for indices into the data array.

        This method supports non-integer index values.

        Input:
            indices     a Tuple of array indices.
            fovmask     True to mask values outside the field of view.

        Return:         (uv, time)
            uv          a Pair defining the values of (u,v) associated with the
                        array indices.
            time        a Scalar defining the time in seconds TDB associated
                        with the array indices.
        """

        indices = Vector.as_vector(indices)

        uv = indices.to_pair((self.u_axis,self.v_axis))
        time = self.scalar_midtime

        if fovmask:
            is_outside = self.uv_is_outside(uv, inclusive=True)
            if np.any(is_outside):
                mask = uv.mask | is_outside
                if np.any(mask != uv.mask):
                    uv = uv.remask(mask)

                    time_values = np.empty(uv.shape)
                    time_values[...] = self.midtime
                    time = Scalar(time_values, mask)

        return (uv, time)

    def uvt_range(self, indices, fovmask=False):
        """Return ranges of coordinates and time for integer array indices.

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

        indices = Vector.as_int(indices)

        uv_min = indices.to_pair((self.u_axis,self.v_axis))
        uv_max = uv_min + Pair.ONES

        time_min = self.scalar_time[0]
        time_max = self.scalar_time[1]

        if fovmask:
            is_outside = self.uv_is_outside(uv_min, inclusive=False)
            if np.any(is_outside):
                mask = uv_min.mask | is_outside
                if np.any(mask != uv_min.mask):
                    uv_min = uv_min.remask(mask)
                    uv_max = uv_max.remask(mask)

                    time_min_vals = np.empty(uv_min.shape)
                    time_max_vals = np.empty(uv_min.shape)

                    time_min_vals[...] = self.time[0]
                    time_max_vals[...] = self.time[1]

                    time_min = Scalar(time_min_vals, mask)
                    time_max = Scalar(time_max_vals, mask)

        return (uv_min, uv_max, time_min, time_max)

# Untested...
#     def indices_at_uvt(self, uv_pair, time, fovmask=False):
#         """Returns a Tuple of indices corresponding to a given spatial location
#         and time. This method supports non-integer positions and time steps, and
#         returns fractional indices.
# 
#         Input:
#             uv_pair     a Pair of spatial (u,v) coordinates in or near the field
#                         of view.
#             time        a Scalar of times in seconds TDB.
#             fovmask     True to mask values outside the field of view.
# 
#         Return:
#             indices     a Tuple of array indices. Any array indices not
#                         constrained by (u,v) or time are returned with value 0.
#                         Note that returned indices can fall outside the nominal
#                         limits of the data object.
#         """
# 
#         uv_pair = Pair.as_pair(uv_pair)
#         time = Scalar.as_scalar(time)
#         (uv_pair, time) = Array_.broadcast_arrays(uv_pair, time)
# 
#         (u,v) = uv_pair.to_scalars()
# 
#         index_vals = np.zeros(uv_pair.shape + [len(self.axes)])
#         index_vals[..., self.u_axis] = u.vals
#         index_vals[..., self.v_axis] = v.vals
# 
#         if fovmask:
#             is_inside = ((self.uv_is_inside(uv_pair, inclusive=True) &
#                          (time >= self.time[0]) & (time <= self.time[1]))
#             if not np.all(is_inside):
#                 mask = uv_pair.mask | np.logical_not(is_inside)
#             else:
#                 mask = uv_pair.mask
# 
#         return Vector(index_vals, mask)

    def times_at_uv(self, uv_pair, fovmask=False):
        """Return start and stop times of the specified spatial pixel (u,v).

        Input:
            uv_pair     a Pair of spatial (u,v) coordinates in and observation's
                        field of view. The coordinates need not be integers, but
                        any fractional part is truncated.
            fovmask     True to mask values outside the field of view.

        Return:         a tuple containing Scalars of the start time and stop
                        time of each (u,v) pair, as seconds TDB.
        """

        if fovmask:
            is_outside = self.uv_is_outside(uv_pair, inclusive=True)
            if np.any(is_outside):
                time_min_vals = np.empty(uv_pair.shape)
                time_max_vals = np.empty(uv_pair.shape)

                time_min_vals[...] = self.time[0]
                time_max_vals[...] = self.time[1]

                time_min = Scalar(time_min_vals, is_outside)
                time_max = Scalar(time_max_vals, is_outside)

                return (time_min, time_max)

        return self.scalar_time

# Untested but not needed as of 7/7/12...
#     def uv_at_time(self, time, fovmask=False):
#         """Returns the (u,v) ranges of spatial pixel observed at the specified
#         time.
# 
#         Input:
#             uv_pair     a Scalar of time values in seconds TDB.
#             fovmask     True to mask values outside the time limits and/or the
#                         field of view.
# 
#         Return:         (uv_min, uv_max)
#             uv_min      the lower (u,v) corner of the area observed at the
#                         specified time.
#             uv_max      the upper (u,v) corner of the area observed at the
#                         specified time.
#         """
# 
#         uv_min = Scalar((0,0))
#         uv_max = Scalar(self.uv_shape)
# 
#         if fovmask or np.any(time.mask):
#             is_inside = (time >= self.time[0]) & (time <= self.time[1])
#             if not np.all(is_inside):
#                 uv_min_vals = np.zeros(time.shape + (2,))
# 
#                 uv_max_vals = np.empty(time.shape + (2,))
#                 uv_max_vals[...,0] = self.uv_shape[0]
#                 uv_max_vals[...,1] = self.uv_shape[1]
# 
#                 uv_min = Pair(uv_min_vals, time.mask |
#                                            np.logical_not(is_inside))
#                 uv_max = Pair(uv_max_vals, mask)
# 
#         return (uv_min, uv_max)

    def sweep_duv_dt(self, uv_pair):
        """Return the mean local sweep speed of the instrument along (u,v) axes.

        Input:
            uv_pair     a Pair of spatial indices (u,v).

        Return:         a Pair containing the local sweep speed in units of
                        pixels per second in the (u,v) directions.
        """

        return Pair.ZERO

    def time_shift(self, dtime):
        """Return a copy of the observation object with a time-shift.

        Input:
            dtime       the time offset to apply to the observation, in units of
                        seconds. A positive value shifts the observation later.

        Return:         a (shallow) copy of the object with a new time.
        """

        obs = Snapshot(self.axes, self.time[0] + dtime, self.texp,
                       self.fov, self.path, self.frame)

        for key in self.subfields.keys():
            obs.insert_subfield(key, self.subfields[key])

        return obs

################################################################################
# Overrides of Observation methods
################################################################################

    def uv_from_ra_and_dec(self, ra, dec, derivs=False, iters=1, quick={},
                           apparent=False):
        """Convert arbitrary scalars of RA and dec to FOV (u,v) coordinates.

        Input:
            ra          a Scalar of J2000 right ascensions.
            dec         a Scalar of J2000 declinations.
            derivs      True to propagate derivatives of ra and dec through to
                        derivatives of the returned (u,v) Pairs.
            iters       the number of iterations to perform until convergence
                        is reached. Two is the most that should ever be needed;
                        Snapshot should override to one.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
            apparent    True to interpret the (RA,dec) values as apparent
                        coordinates; False to interpret them as actual
                        coordinates. Default is False.

        Return:         a Pair of (u,v) coordinates.

        Note: The only reasons for iteration are that the C-matrix and the
        velocity WRT the SSB could vary during the observation. This can be
        neglected for a Snapshot.
        """

        return Observation.uv_from_ra_and_dec(self, ra, dec, derivs=derivs,
                                              iters=1, quick=quick,
                                              apparent=apparent)
        
    def uv_from_path(self, path, derivs=False, quick={}, converge={}):
        """Return the (u,v) indices of an object in the FOV, given its path.

        Note: This procedure assumes that movement along a path is very limited
        during the exposure time of an individual pixel. It could fail to
        converge if there is a large gap in timing between adjacent pixels at a
        time when the object is crossing that gap. However, even then, it should
        select roughly the correct location. It could also fail to converge
        during a fast slew.

        Input:
            path        a Path object.
            derivs      True to propagate derivatives of the link time and
                        position into the returned event.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
            converge    an optional dictionary of parameters to override the
                        configured default convergence parameters. The default
                        configuration is defined in config.py.

        Return:         the (u,v) indices of the pixel in which the point was
                        found. The path is evaluated at the mid-time of this
                        pixel.
        """

        obs_event = Event(self.midtime, Vector3.ZERO, self.path, self.frame)
        (path_event, obs_event) = path.photon_to_event(obs_event,
                                    derivs=False, guess=guess,
                                    quick=quick, converge=converge)

        return self.fov.uv_from_los(obs_event.neg_arr_ap, derivs=derivs)

    def uv_from_coords(self, surface, coords, underside=False, derivs=False,
                             quick={}, converge={}):
        """The (u,v) indices of a surface point, given its coordinates.

        Input:
            surface     a Surface object.
            coords      a tuple containing two or three Scalars of surface
                        coordinates. The Scalars need not be the same shape,
                        but must broadcast to the same shape.
            underside   True for the underside of the surface (emission > 90
                        degrees) to be unmasked.
            derivs      True to propagate derivatives of the link time and
                        position into the returned event.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
            converge    an optional dictionary of parameters to override the
                        configured default convergence parameters. The default
                        configuration is defined in config.py.

        Return:         the (u,v) indices of the pixel in which the point was
                        found. The path is evaluated at the mid-time of this
                        pixel.
        """

        obs_event = Event(self.midtime, Vector3.ZERO, self.path, self.frame)
        (surface_event,
         obs_event) = surface.photon_to_event_by_coords(obs_event, coords,
                                derivs=derivs, quick=quick, converge=converge)

        neg_arr_ap = obs_event.neg_arr_ap
        if not underside:
            normal = surface.normal(surface_event.pos)
            neg_arr_ap = neg_arr_ap.mask_where(normal.dot(surface_event.dep_ap)
                                               > 0)

        return self.fov.uv_from_los(neg_arr_ap, derivs=derivs)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Snapshot(unittest.TestCase):

    def runTest(self):

        from oops.fov_.flatfov import FlatFOV

        fov = FlatFOV((0.001,0.001), (10,20))
        obs = Snapshot(axes=('u','v'), tstart=98., texp=2.,
                       fov=fov, path='SSB', frame='J2000')

        indices = Vector([(0.,0.),(0.,20.),(10.,0.),(10.,20.),(10.,21.)])

        # uvt() with fovmask == False
        (uv,time) = obs.uvt(indices)

        self.assertFalse(uv.mask)
        self.assertFalse(time.mask)
        self.assertEqual(time, 99.)
        self.assertEqual(uv, Pair.as_pair(indices))

        # uvt() with fovmask == True
        (uv,time) = obs.uvt(indices, fovmask=True)

        self.assertTrue(np.all(uv.mask == np.array(4*[False] + [True])))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time[:4], 99.)
        self.assertEqual(uv[:4], Pair.as_pair(indices)[:4])

        # uvt_range() with fovmask == False
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertFalse(uv_min.mask)
        self.assertFalse(uv_max.mask)
        self.assertFalse(time_min.mask)
        self.assertFalse(time_max.mask)

        self.assertEqual(uv_min, Pair.as_pair(indices))
        self.assertEqual(uv_max, Pair.as_pair(indices) + (1,1))
        self.assertEqual(time_min,  98.)
        self.assertEqual(time_max, 100.)

        # uvt_range() with fovmask == False, new indices
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices+(0.2,0.9))

        self.assertFalse(uv_min.mask)
        self.assertFalse(uv_max.mask)
        self.assertFalse(time_min.mask)
        self.assertFalse(time_max.mask)

        self.assertEqual(uv_min, Pair.as_pair(indices))
        self.assertEqual(uv_max, Pair.as_pair(indices) + (1,1))
        self.assertEqual(time_min,  98.)
        self.assertEqual(time_max, 100.)

        # uvt_range() with fovmask == True, new indices
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices+(0.2,0.9),
                                                             fovmask=True)
        self.assertTrue(np.all(uv_min.mask == [False] + 4*[True]))
        self.assertTrue(np.all(uv_min.mask == uv_max.mask))
        self.assertTrue(np.all(uv_min.mask == time_min.mask))
        self.assertTrue(np.all(uv_min.mask == time_max.mask))

        self.assertEqual(uv_min[0], Pair.as_pair(indices)[0])
        self.assertEqual(uv_max[0], (Pair.as_pair(indices) + (1,1))[0])
        self.assertEqual(time_min[0],  98.)
        self.assertEqual(time_max[0], 100.)

        # times_at_uv() with fovmask == False
        uv_pair = Pair([(0.,0.),(0.,20.),(10.,0.),(10.,20.),(10.,21.)])

        (time0, time1) = obs.times_at_uv(uv_pair)

        self.assertEqual(time0,  98.)
        self.assertEqual(time1, 100.)

        # times_at_uv() with fovmask == True
        (time0, time1) = obs.times_at_uv(uv_pair, fovmask=True)

        self.assertTrue(np.all(time0.mask == 4*[False] + [True]))
        self.assertTrue(np.all(time1.mask == 4*[False] + [True]))
        self.assertEqual(time0[:4],  98.)
        self.assertEqual(time1[:4], 100.)

        # Alternative axis order ('v','u')
        obs = Snapshot(axes=('v','u'), tstart=98., texp=2.,
                       fov=fov, path='SSB', frame='J2000')
        indices = Pair([(0,0),(0,10),(20,0),(20,10),(20,11)])

        (uv,time) = obs.uvt(indices)

        self.assertEqual(uv, indices.to_pair((1,0)))

        (uv,time) = obs.uvt(indices, fovmask=True)

        self.assertEqual(uv[:4], indices.to_pair((1,0))[:4])
        self.assertTrue(np.all(uv.mask == 4*[False] + [True]))

        # Alternative axis order ('v', 'a', 'u')
        obs = Snapshot(axes=('v','a','u'), tstart=98., texp=2.,
                       fov=fov, path='SSB', frame='J2000')
        indices = Vector([(0,-1,0),(0,99,10),(20,-9,0),(20,77,10),(20,44,11)])
        (uv,time) = obs.uvt(indices)

        self.assertEqual(uv, indices.to_pair((2,0)))

        (uv,time) = obs.uvt(indices, fovmask=True)

        self.assertEqual(uv[:4], indices.to_pair((2,0))[:4])
        self.assertTrue(np.all(uv.mask == 4*[False] + [True]))

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
