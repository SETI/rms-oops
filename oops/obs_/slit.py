################################################################################
# oops/obs_/slit.py: Subclass Slit of class Observation
################################################################################

import numpy as np
from polymath import *

from oops.obs_.observation   import Observation
from oops.cadence_.cadence   import Cadence
from oops.path_.path         import Path
from oops.frame_.frame       import Frame
from oops.event              import Event

#*******************************************************************************
# Slit
#*******************************************************************************
class Slit(Observation):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    A Slit is subclass of Observation consisting of a 2-D image constructed
    by rotating an instrument that has a 1-D array of sensors. The FOV
    describes the 1-D sensor array. The second axis of the pixel array is
    simulated by sampling the slit according to the cadence as the instrument
    rotates.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    PACKRAT_ARGS = ['axes', 'det_size', 'cadence', 'fov', 'path', 'frame',
                    '**subfields']

    #===========================================================================
    # __init__
    #===========================================================================
    def __init__(self, axes, det_size, cadence, fov, path, frame, **subfields):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Constructor for a Slit observation.

        Input:
            axes        a list or tuple of strings, with one value for each axis
                        in the associated data array. A value of 'u' or 'ut'
                        should appear at the location of the array's u-axis;
                        'vt' or 'v' should appear at the location of the array's
                        v-axis. The 't' suffix is used for the one of these axes
                        that is emulated by time-sampling perpendicular to the
                        slit.

            det_size    the size of the detectors in FOV units parallel to the
                        slit. It will be < 1 if there are gaps between the
                        detectors.

            cadence     a Cadence object defining the start time and duration of
                        each consecutive measurement.  Alternatively, a 
                        dictionary containing the following entries, from 
                        which a cadence object is constructed:

                         times: a list or 1-D array of times in seconds.
                         texp:  exposure time in seconds for each step.


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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #--------------------------------------------------
        # Basic properties
        #--------------------------------------------------
        self.fov = fov
        self.path = Path.as_waypoint(path)
        self.frame = Frame.as_wayframe(frame)

        #--------------------------------------------------
        # Cadence
        #--------------------------------------------------
        if isinstance(cadence, Cadence): self.cadence = cadence
        else: self.cadence = self._default_cadence(cadence)

        #--------------------------------------------------
        # Axes
        #--------------------------------------------------
        self.axes = list(axes)
        assert (('u' in self.axes and 'vt' in self.axes) or
                ('v' in self.axes and 'ut' in self.axes))

        if 'ut' in self.axes:
            self.u_axis = self.axes.index('ut')
            self.v_axis = self.axes.index('v')
            self.t_axis = self.u_axis
            self.along_slit_axis = self.v_axis
            self.cross_slit_uv_axis = 0
            self.along_slit_uv_axis = 1
            self.uv_shape = [self.cadence.shape[0],
                             self.fov.uv_shape.vals[self.along_slit_axis]]
        else:
            self.u_axis = self.axes.index('u')
            self.v_axis = self.axes.index('vt')
            self.t_axis = self.v_axis
            self.along_slit_axis = self.u_axis
            self.cross_slit_uv_axis = 1
            self.along_slit_uv_axis = 0
            self.uv_shape = [self.fov.uv_shape.vals[self.along_slit_axis],
                             self.cadence.shape[0]]

        self.swap_uv = (self.u_axis > self.v_axis)

        #--------------------------------------------------
        # Timing
        #--------------------------------------------------
        self.time = self.cadence.time
        self.midtime = self.cadence.midtime

        assert len(self.cadence.shape) == 1
        assert self.fov.uv_shape.vals[self.cross_slit_uv_axis] == 1

        #--------------------------------------------------
        # Shape / Size
        #--------------------------------------------------
        self.along_slit_shape = self.uv_shape[self.along_slit_uv_axis]

        self.det_size = det_size
        self.slit_is_discontinuous = (self.det_size < 1)

        self.shape = len(axes) * [0]
        self.shape[self.u_axis] = self.uv_shape[0]
        self.shape[self.v_axis] = self.uv_shape[1]

        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

        return
    #===========================================================================



    #===========================================================================
    # _default_cadence
    #===========================================================================
    def _default_cadence(self, dict):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return a cadence object a dictionary of parameters.

        Input:
            dict        Dictionary containing the following entries:

                         TBD

        Return:         Cadence object.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        times = dict['times']
        texp = dict['texp']

        return Sequence(times, texp)
    #===========================================================================



    #===========================================================================
    # uvt
    #===========================================================================
    def uvt(self, indices, fovmask=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return coordinates (u,v) and time t for indices into the data array.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        indices = Vector.as_vector(indices)
        slit_coord = indices.to_scalar(self.along_slit_axis)

        #---------------------------------
        # Handle discontinuous detectors
        #---------------------------------
        if self.slit_is_discontinuous:

	    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            # Identify indices at exact upper limit; treat these as inside
	    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            at_upper_limit = (slit_coord == self.along_slit_shape)

	    #- - - - - - - - - - - - - - - - - - - - - - - - 
            # Map continuous index to discontinuous (u,v)
	    #- - - - - - - - - - - - - - - - - - - - - - - - 
            slit_int = slit_coord.int()
            slit_coord = slit_int + (slit_coord - slit_int) * self.det_size

	    #- - - - - - - - - - - - - - - -
            # Adjust values at upper limit
	    #- - - - - - - - - - - - - - - -
            slit_coord = slit_coord.mask_where(at_upper_limit,
                            replace = self.along_slit_shape + self.det_size - 1,
                            remask = False)

        #---------------------
        # Create (u,v) Pair
        #---------------------
        uv_vals = np.empty(indices.shape + (2,))
        uv_vals[..., self.along_slit_uv_axis] = slit_coord.vals
        uv_vals[..., self.cross_slit_uv_axis] = 0.5
        uv = Pair(uv_vals, indices.mask)

        #----------------------
        # Create time Scalar
        #----------------------
        tstep = indices.to_scalar(self.t_axis)
        time = self.cadence.time_at_tstep(tstep, mask=fovmask)

        #---------------------------
        # Apply mask if necessary
        #---------------------------
        if fovmask:
            u_index = indices.vals[..., self.u_axis]
            v_index = indices.vals[..., self.v_axis]
            is_outside = ((u_index < 0) |
                          (v_index < 0) |
                          (u_index > self.uv_shape[0]) |
                          (v_index > self.uv_shape[1]))
            if np.any(is_outside):
                uv = uv.mask_where(is_outside)
                time = time.mask_where(is_outside)

        return (uv, time)
    #===========================================================================



    #===========================================================================
    # uvt_range
    #===========================================================================
    def uvt_range(self, indices, fovmask=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return ranges of coordinates and time for integer array indices.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        indices = Vector.as_vector(indices).as_int()

        slit_coord = indices.to_scalar(self.along_slit_axis)

        uv_vals = np.empty(indices.shape + (2,), dtype='int')
        uv_vals[..., self.along_slit_uv_axis] = slit_coord.vals
        uv_vals[..., self.cross_slit_uv_axis] = 0
        uv_min = Pair(uv_vals, indices.mask)
        uv_max = uv_min + Pair.ONES

        tstep = indices.to_scalar(self.t_axis)
        (time_min,
         time_max) = self.cadence.time_range_at_tstep(tstep, mask=fovmask)

        if fovmask:
            u_index = indices.vals[..., self.u_axis]
            v_index = indices.vals[..., self.v_axis]
            is_outside = ((u_index < 0) |
                          (v_index < 0) |
                          (u_index >= self.uv_shape[0]) |
                          (v_index >= self.uv_shape[1]))
            if np.any(is_outside):
                uv_min = uv_min.mask_where(is_outside)
                uv_max = uv_max.mask_where(is_outside)
                time_min = time_min.mask_where(is_outside)
                time_max = time_max.mask_where(is_outside)

        return (uv_min, uv_max, time_min, time_max)
    #===========================================================================



    #===========================================================================
    # uv_range_at_tstep
    #===========================================================================
    def uv_range_at_tstep(self, tstep):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return a tuple defining the range of (u,v) active at a particular
        integer time step.

        Input:
            tstep       a time step index (integer or tuple).

        Return:         a tuple (uv_min, uv_max)
            uv_min      a Pair defining the minimum values of (u,v) coordinates
                        active at this time step.
            uv_min      a Pair defining the maximum values of (u,v) coordinates
                        active at this time step (exclusive)
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return (Pair.ZERO, self.fov.uv_shape)
    #===========================================================================



    #===========================================================================
    # times_at_uv
    #===========================================================================
    def times_at_uv(self, uv_pair, fovmask=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return start and stop times of the specified spatial pixel (u,v).

        Input:
            uv_pair     a Pair of spatial (u,v) coordinates in and observation's
                        field of view. The coordinates need not be integers, but
                        any fractional part is truncated.
            fovmask     True to mask values outside the field of view.

        Return:         a tuple containing Scalars of the start time and stop
                        time of each (u,v) pair, as seconds TDB.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        uv_pair = Pair.as_int(uv_pair)
        tstep = uv_pair.to_scalar(self.cross_slit_uv_axis)
        (time0, time1) = self.cadence.time_range_at_tstep(tstep, mask=fovmask)

        if fovmask:
            u_index = uv_pair.vals[..., 0]
            v_index = uv_pair.vals[..., 1]
            is_outside = ((u_index < 0) |
                          (v_index < 0) |
                          (u_index > self.uv_shape[0]) |
                          (v_index > self.uv_shape[1]))
            if np.any(is_outside):
                time0 = time0.mask_where(is_outside)
                time1 = time1.mask_where(is_outside)

        return (time0, time1)
    #===========================================================================



    #===========================================================================
    # sweep_duv_dt
    #===========================================================================
    def sweep_duv_dt(self, uv_pair):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return the mean local sweep speed of the instrument along (u,v) axes.

        Input:
            uv_pair     a Pair of spatial indices (u,v).

        Return:         a Pair containing the local sweep speed in units of
                        pixels per second in the (u,v) directions.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return Pair.ZERO
    #===========================================================================



    #===========================================================================
    # time_shift
    #===========================================================================
    def time_shift(self, dtime):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return a copy of the observation object with a time-shift.

        Input:
            dtime       the time offset to apply to the observation, in units of
                        seconds. A positive value shifts the observation later.

        Return:         a (shallow) copy of the object with a new time.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        obs = Slit(self.axes, self.uv_size,
                   self.cadence.time_shift(dtime),
                   self.fov, self.path, self.frame)

        for key in self.subfields.keys():
            obs.insert_subfield(key, self.subfields[key])

        return obs
    #===========================================================================


#*******************************************************************************

################################################################################
# UNIT TESTS
################################################################################

import unittest

#*******************************************************************************
# Test_Slit
#*******************************************************************************
class Test_Slit(unittest.TestCase):

    #===========================================================================
    # runTest
    #===========================================================================
    def runTest(self):

        from oops.cadence_.metronome import Metronome
        from oops.fov_.flatfov import FlatFOV

        fov = FlatFOV((0.001,0.001), (10,1))
        cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        obs = Slit(axes=('u','vt'), det_size=1,
                   cadence=cadence, fov=fov, path='SSB', frame='J2000')

        indices = Vector([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])

        #-------------------------------
        # uvt() with fovmask == False
        #-------------------------------
        (uv,time) = obs.uvt(indices)

        self.assertFalse(np.any(uv.mask))
        self.assertFalse(np.any(time.mask))
        self.assertEqual(time, cadence.tstride * indices.to_scalar(1))
        self.assertEqual(uv.to_scalar(0), indices.to_scalar(0))
        self.assertEqual(uv.to_scalar(1), 0.5)

        #------------------------------
        # uvt() with fovmask == True
        #------------------------------
        (uv,time) = obs.uvt(indices, fovmask=True)

        self.assertTrue(np.all(uv.mask == np.array(6*[False] + [True])))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time[:6], cadence.tstride * indices.to_scalar(1)[:6])
        self.assertEqual(uv[:6].to_scalar(0), indices[:6].to_scalar(0))
        self.assertEqual(uv[:6].to_scalar(1), 0.5)

        #------------------------------------
        # uvt_range() with fovmask == False
        #------------------------------------
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertFalse(np.any(uv_min.mask))
        self.assertFalse(np.any(uv_max.mask))
        self.assertFalse(np.any(time_min.mask))
        self.assertFalse(np.any(time_max.mask))

        self.assertEqual(uv_min.to_scalar(0), indices.to_scalar(0))
        self.assertEqual(uv_min.to_scalar(1), 0)
        self.assertEqual(uv_max.to_scalar(0), indices.to_scalar(0) + 1)
        self.assertEqual(uv_max.to_scalar(1), 1)
        self.assertEqual(time_min, cadence.tstride * indices.to_scalar(1))
        self.assertEqual(time_max, time_min + cadence.texp)

        #------------------------------------
        # uvt_range() with fovmask == True
        #------------------------------------
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices,
                                                             fovmask=True)

        self.assertTrue(np.all(uv_min.mask == np.array(2*[False] + 5*[True])))
        self.assertTrue(np.all(uv_max.mask == uv_min.mask))
        self.assertTrue(np.all(time_min.mask == uv_min.mask))
        self.assertTrue(np.all(time_max.mask == uv_min.mask))

        self.assertEqual(uv_min.to_scalar(0)[:2], indices.to_scalar(0)[:2])
        self.assertEqual(uv_min.to_scalar(1)[:2], 0)
        self.assertEqual(uv_max.to_scalar(0)[:2], indices.to_scalar(0)[:2] + 1)
        self.assertEqual(uv_max.to_scalar(1)[:2], 1)
        self.assertEqual(time_min[:2], cadence.tstride *
                                       indices.to_scalar(1)[:2])
        self.assertEqual(time_max[:2], time_min[:2] + cadence.texp)

        #--------------------------------------
        # times_at_uv() with fovmask == False
        #--------------------------------------
        uv = Pair([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])

        (time0, time1) = obs.times_at_uv(uv)

        self.assertEqual(time0, cadence.tstride * uv.to_scalar(1))
        self.assertEqual(time1, time0 + cadence.texp)

        #--------------------------------------
        # times_at_uv() with fovmask == True
        #--------------------------------------
        (time0, time1) = obs.times_at_uv(uv, fovmask=True)

        self.assertTrue(np.all(time0.mask == 6*[False] + [True]))
        self.assertTrue(np.all(time1.mask == time0.mask))
        self.assertEqual(time0[:6], cadence.tstride * uv.to_scalar(1)[:6])
        self.assertEqual(time1[:6], time0[:6] + cadence.texp)

        ####################################


        #--------------------------------------
        # Alternative axis order ('ut','v')
        #--------------------------------------

        fov = FlatFOV((0.001,0.001), (1,20))
        cadence = Metronome(tstart=0., tstride=10., texp=10., steps=10)
        obs = Slit(axes=('ut','v'), det_size=1,
                   cadence=cadence, fov=fov, path='SSB', frame='J2000')

        indices = Vector([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])

        (uv,time) = obs.uvt(indices)

        self.assertEqual(uv.to_scalar(0), 0.5)
        self.assertEqual(uv.to_scalar(1), indices.to_scalar(1))
        self.assertEqual(time, cadence.tstride * indices.to_scalar(0))

        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertEqual(uv_min.to_scalar(0), 0)
        self.assertEqual(uv_min.to_scalar(1), indices.to_scalar(1))
        self.assertEqual(uv_max.to_scalar(0), 1)
        self.assertEqual(uv_max.to_scalar(1), indices.to_scalar(1) + 1)
        self.assertEqual(time_min, cadence.tstride * indices.to_scalar(0))
        self.assertEqual(time_max, time_min + cadence.texp)

        ####################################


        #-----------------------------------------------------------
        # Alternative det_size and texp for discontinuous indices
        #-----------------------------------------------------------

        fov = FlatFOV((0.001,0.001), (1,20))
        cadence = Metronome(tstart=0., tstride=10., texp=8., steps=10)
        obs = Slit(axes=('ut','v'), det_size=0.8,
                   cadence=cadence, fov=fov, path='SSB', frame='J2000')

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

        self.assertTrue(abs(obs.uvt((1, 0      ))[0] - (0.5,0.0)) < delta)
        self.assertTrue(abs(obs.uvt((2, 1.25   ))[0] - (0.5,1.2)) < delta)
        self.assertTrue(abs(obs.uvt((3, 2.5    ))[0] - (0.5,2.4)) < delta)
        self.assertTrue(abs(obs.uvt((4, 3.75   ))[0] - (0.5,3.6)) < delta)
        self.assertTrue(abs(obs.uvt((5, 5 - eps))[0] - (0.5,4.8)) < delta)
        self.assertTrue(abs(obs.uvt((5, 5.     ))[0] - (0.5,5.0)) < delta)

        #------------------------------
        # Test using scalar indices
        #------------------------------
        below = obs.uvt((0,20 - eps), fovmask=True)[0].to_scalar(1)
        exact = obs.uvt((0,20      ), fovmask=True)[0].to_scalar(1)
        above = obs.uvt((0,20 + eps), fovmask=True)[0].to_scalar(1)

        self.assertTrue(abs(below - 19.8) < delta)
        self.assertTrue(abs(exact - 19.8) < delta)
        self.assertTrue(abs(above.values - 20.0) < delta)
        self.assertTrue(above.mask)

        #-----------------------------
        # Test using a Vector index
        #-----------------------------
        indices = Vector([(0,20 - eps), (0,20), (0,20 + eps)])

        u = obs.uvt(indices, fovmask=True)[0].to_scalar(1)
        self.assertTrue(abs(u[0] - 19.8) < delta)
        self.assertTrue(abs(u[1] - 19.8) < delta)
        self.assertTrue(abs(u[2].values - 20.0) < delta)
        self.assertTrue(u.mask[2])

        #----------------------------------------------
        # Alternative with uv_size and texp and axes
        #----------------------------------------------
        cadence = Metronome(tstart=0., tstride=10., texp=8., steps=10)
        obs = Slit(axes=('a','v','b','ut','c'), det_size=0.8,
                   cadence=cadence, fov=fov, path='SSB', frame='J2000')

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

        self.assertTrue(abs(obs.uvt((1, 0      ,4,1,7))[0] - (0.5,0.0)) < delta)
        self.assertTrue(abs(obs.uvt((1, 1.25   ,4,2,7))[0] - (0.5,1.2)) < delta)
        self.assertTrue(abs(obs.uvt((1, 2.5    ,4,3,7))[0] - (0.5,2.4)) < delta)
        self.assertTrue(abs(obs.uvt((1, 3.75   ,4,4,7))[0] - (0.5,3.6)) < delta)
        self.assertTrue(abs(obs.uvt((1, 5 - eps,4,5,7))[0] - (0.5,4.8)) < delta)
        self.assertTrue(abs(obs.uvt((1, 5.     ,4,5,7))[0] - (0.5,5.0)) < delta)
    #===========================================================================


#*******************************************************************************


########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
