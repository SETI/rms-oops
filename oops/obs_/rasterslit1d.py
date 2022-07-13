################################################################################
# oops/obs_/rasterslit1d.py: Subclass RasterSlit1D of class Observation
################################################################################

import numpy as np
from polymath import *

from oops.obs_.observation   import Observation
from oops.cadence_.cadence   import Cadence
from oops.path_.path         import Path
from oops.frame_.frame       import Frame
from oops.event              import Event

#*******************************************************************************
# RasterSlit1D
#*******************************************************************************
class RasterSlit1D(Observation):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    A RasterSlit1D is subclass of Observation consisting of a 1-D observation
    in which the one dimension is constructed by sweeping a single pixel along a
    slit. The FOV describes the single pixel; the slit is simulated by rotating
    the camera.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    PACKRAT_ARGS = ['axes', 'det_size', 'cadence', 'fov', 'path', 'frame',
                    '**subfields']

    #===========================================================================
    # __init__
    #===========================================================================
    def __init__(self, axes, det_size, cadence, fov, path, frame, **subfields):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Constructor for a RasterSlit observation.

        Input:

            axes        a list or tuple of strings, with one value for each axis
                        in the associated data array. A value of 'ut' should
                        appear at the location of the array's u-axis if any;
                        'vt' should appear at the location of the array's v-axis
                        if any. Only one of 'ut' or 'vt' can appear.

            det_size    the size of the detector in FOV units parallel to the
                        slit. It will be < 1 if there are gaps between the
                        samples, or > 1 if the detector moves by less than its
                        full size within the fast time step.


            cadence     a 1-D Cadence object defining the timing of each
                        consecutive measurement along the slit.  Alternatively, 
                        a tuple of the form:

                          (tbd)

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
        else: self.cadence = self._default_cadence(*cadence)

        assert len(self.cadence.shape) == 1

        #--------------------------------------------------
        # Axes
        #--------------------------------------------------
        self.axes = list(axes)
        assert (('ut' in self.axes and 'vt' not in self.axes) or
                ('vt' in self.axes and 'ut' not in self.axes))

        self.shape = len(axes) * [0]

        if 'ut' in self.axes:
            self.u_axis = self.axes.index('ut')
            self.v_axis = -1
            self.t_axis = self.u_axis
            self.along_slit_uv_index = 0
            self.cross_slit_uv_index = 1
            self.shape[self.u_axis] = self.fov.uv_shape.vals[0]
            self.along_slit_shape = self.shape[self.u_axis]
        else:
            self.u_axis = -1
            self.v_axis = self.axes.index('vt')
            self.t_axis = self.v_axis
            self.along_slit_uv_index = 1
            self.cross_slit_uv_index = 0
            self.shape[self.v_axis] = self.fov.uv_shape.vals[1]
            self.along_slit_shape = self.shape[self.v_axis]

        self.swap_uv = False

        #--------------------------------------------------
        # Shape / Size
        #--------------------------------------------------
        self.det_size = det_size
        self.slit_is_discontinuous = (self.det_size != 1)

        self.uv_shape = self.fov.uv_shape.vals
        assert self.fov.uv_shape.vals[self.cross_slit_uv_index] == 1

        #--------------------------------------------------
        # Timing
        #--------------------------------------------------
        self.time = self.cadence.time
        self.midtime = self.cadence.midtime

        self.scalar_time = (Scalar(self.time[0]), Scalar(self.time[1]))
        self.scalar_midtime = Scalar(self.midtime)

        duv_dt_basis_vals = np.zeros(2)
        duv_dt_basis_vals[self.along_slit_uv_index] = 1.
        self.duv_dt_basis = Pair(duv_dt_basis_vals)

        #--------------------------------------------------
        # Optional subfields
        #--------------------------------------------------
        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])
    #===========================================================================



    #===========================================================================
    # _default_cadence
    #===========================================================================
    def _default_cadence(self, tstart, tstride, texp, steps):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return a cadence object a dictionary of parameters.

        Input:
            dict        Dictionary containing the following entries:

                         TBD

        Return:         Cadence object.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return Metronome(tstart, tstride, texp, steps)
    #===========================================================================



    #===========================================================================
    # uvt
    #===========================================================================
    def uvt(self, indices, fovmask=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return coordinates (u,v) and time t for indices into the data array.

        This method supports non-integer index values.

        Input:
            indices     a Vector (or subclass) of array indices.
            fovmask     True to mask values outside the field of view.

        Return:         (uv, time)
            uv          a Pair defining the values of (u,v) associated with the
                        array indices.
            time        a Scalar defining the time in seconds TDB associated
                        with the array indices.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        indices = Vector.as_vector(indices)
        slit_coord = indices.to_scalar(self.t_axis)

        #-----------------------------------
        # Handle discontinuous detectors
        #-----------------------------------
        if self.slit_is_discontinuous:

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Identify indices at exact upper limit; treat these as inside
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            at_upper_limit = (slit_coord == self.along_slit_shape)

            #- - - - - - - - - - - - - - - - - - - - - - - - 
            # Map continuous index to discontinuous (u,v)
            #- - - - - - - - - - - - - - - - - - - - - - - - 
            slit_int = slit_coord.int()
            slit_coord = slit_int + (slit_coord - slit_int) * self.det_size

            #- - - - - - - - - - - - - - - - - 
            # Adjust values at upper limit
            #- - - - - - - - - - - - - - - - - 
            slit_coord = slit_coord.mask_where(at_upper_limit,
                            replace = self.along_slit_shape + self.det_size - 1,
                            remask = False)

        #----------------------
        # Create (u,v) Pair
        #----------------------
        uv_vals = np.empty(indices.shape + (2,))
        uv_vals[..., self.along_slit_uv_index] = slit_coord.vals
        uv_vals[..., self.cross_slit_uv_index] = 0.5
        uv = Pair(uv_vals, indices.mask)

        #-----------------------
        # Create time Scalar
        #-----------------------
        tstep = indices.to_scalar(self.t_axis)
        time = self.cadence.time_at_tstep(tstep, mask=fovmask)

        #-----------------------------
        # Apply mask if necessary
        #-----------------------------
        if fovmask:
            is_outside = self.uv_is_outside(uv, inclusive=True)
            if np.any(is_outside):
                uv = uv.mask_where(is_outside)
                time = time.mask_where(is_outside)

        return (uv, time)
    #===========================================================================



    #===========================================================================
    # uvt_range
    #===========================================================================
    def uvt_range(self, indices, fovmask=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return ranges of coordinates and time for integer array indices.

        Input:
            indices     a Vector (or subclass) of integer array indices.
            fovmask     True to mask values outside the field of view.

        Return:         (uv_min, uv_max, time_min, time_max)
            uv_min      a Pair defining the minimum values of (u,v) associated
                        the pixel.
            uv_max      a Pair defining the maximum values of (u,v).
            time_min    a Scalar defining the minimum time associated with the
                        pixel. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time value.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        indices = Vector.as_vector(indices).as_int()
        slit_coord = indices.to_scalar(self.t_axis)

        uv_vals = np.empty(indices.shape + (2,), dtype='int')
        uv_vals[..., self.along_slit_uv_index] = slit_coord.vals
        uv_vals[..., self.cross_slit_uv_index] = 0
        uv_min = Pair(uv_vals, indices.mask)
        uv_max = uv_min + Pair.ONES

        tstep = indices.to_scalar(self.t_axis)
        (time_min,
         time_max) = self.cadence.time_range_at_tstep(tstep, mask=fovmask)

        if fovmask:
            is_outside = self.uv_is_outside(uv_max, inclusive=True)
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
    def uv_range_at_tstep(self, *tstep):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return a tuple defining the range of (u,v) coordinates active at a
        particular time step.

        Input:
            tstep       a time step index (one or two integers). Not checked for
                        out-of-range errors.

        Return:         a tuple (uv_min, uv_max)
            uv_min      a Pair defining the minimum values of (u,v) coordinates
                        active at this time step.
            uv_min      a Pair defining the maximum values of (u,v) coordinates
                        active at this time step (exclusive).
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return (Pair.ZEROS, Pair.ONES)
    #===========================================================================



    #===========================================================================
    # times_at_uv
    #===========================================================================
    def times_at_uv(self, uv_pair, fovmask=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        uv_pair = Pair.as_pair(uv_pair).as_int()
        tstep = uv_pair.to_scalar(self.along_slit_uv_index)

        return self.cadence.time_range_at_tstep(tstep, mask=fovmask)
    #===========================================================================



    #===========================================================================
    # sweep_duv_dt
    #===========================================================================
    def sweep_duv_dt(self, uv_pair):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return the mean local sweep speed of the instrument along (u,v) axes.

        Input:
            uv_pair     a Pair of spatial indices (u,v).

        Return:         a Pair containing the local sweep speed in units of
                        pixels per second in the (u,v) directions.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        uv_pair = Pair.as_pair(uv_pair)
        tstep = uv_pair.to_scalar(self.along_slit_uv_index)

        return self.duv_dt_basis / self.cadence.tstride_at_tstep(tstep)
    #===========================================================================



    #===========================================================================
    # time_shift
    #===========================================================================
    def time_shift(self, dtime):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return a copy of the observation object with a time-shift.

        Input:
            dtime       the time offset to apply to the observation, in units of
                        seconds. A positive value shifts the observation later.

        Return:         a (shallow) copy of the object with a new time.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        obs = RasterSlit1D(self.axes, self.det_size,
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
# Test_RasterSlit1D
#*******************************************************************************
class Test_RasterSlit1D(unittest.TestCase):

    #===========================================================================
    # runTest
    #===========================================================================
    def runTest(self):

        from oops.cadence_.metronome import Metronome
        from oops.fov_.flatfov import FlatFOV

        fov = FlatFOV((0.001,0.001), (10,1))
        cadence = Metronome(tstart=0., tstride=10., texp=10., steps=10)
        obs = RasterSlit1D(axes=('ut','a'), det_size=1, cadence=cadence,
                           fov=fov, path='SSB', frame='J2000')

        indices = Pair([(0,0),(10,0),(11,0)])

        #-------------------------------
        # uvt() with fovmask == False
        #-------------------------------
        (uv, time) = obs.uvt(indices)

        self.assertFalse(np.any(uv.mask))
        self.assertFalse(np.any(time.mask))
        self.assertEqual(time, cadence.tstride * indices.to_scalar(0))
        self.assertEqual(uv.to_scalar(0), indices.to_scalar(0))
        self.assertEqual(uv.to_scalar(1), 0.5)

        #------------------------------
        # uvt() with fovmask == True
        #------------------------------
        (uv, time) = obs.uvt(indices, fovmask=True)

        self.assertTrue(np.all(uv.mask == np.array(2*[False] + [True])))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time[:2], cadence.tstride * indices.to_scalar(0)[:2])
        self.assertEqual(uv[:2].to_scalar(0), indices[:2].to_scalar(0))
        self.assertEqual(uv[:2].to_scalar(1), 0.5)

        #------------------------------------------
        # uvt() with fovmask == True, new indices
        #------------------------------------------
        (uv, time) = obs.uvt(indices+(0.2,0.9), fovmask=True)

        self.assertTrue(np.all(uv.mask == np.array([False] + 2*[True])))
        self.assertTrue(np.all(time.mask == uv.mask))

        #-------------------------------------
        # uvt_range() with fovmask == False
        #-------------------------------------
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertFalse(np.any(uv_min.mask))
        self.assertFalse(np.any(uv_max.mask))
        self.assertFalse(np.any(time_min.mask))
        self.assertFalse(np.any(time_max.mask))

        self.assertEqual(uv_min.to_scalar(0), indices.to_scalar(0))
        self.assertEqual(uv_min.to_scalar(1), 0)
        self.assertEqual(uv_max.to_scalar(0), indices.to_scalar(0) + 1)
        self.assertEqual(uv_max.to_scalar(1), 1)
        self.assertEqual(time_min, cadence.tstride * indices.to_scalar(0))
        self.assertEqual(time_max, time_min + 10.)

        #------------------------------------
        # uvt_range() with fovmask == True
        #------------------------------------
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices,
                                                             fovmask=True)

        self.assertTrue(np.all(uv_min.mask == np.array([False] + 2*[True])))
        self.assertTrue(np.all(uv_max.mask == uv_min.mask))
        self.assertTrue(np.all(time_min.mask == uv_min.mask))
        self.assertTrue(np.all(time_max.mask == uv_min.mask))

        self.assertEqual(uv_min.to_scalar(0)[:1], indices.to_scalar(0)[:1])
        self.assertEqual(uv_min.to_scalar(1)[:1], 0)
        self.assertEqual(uv_max.to_scalar(0)[:1], indices.to_scalar(0)[:1] + 1)
        self.assertEqual(uv_max.to_scalar(1)[:1], 1)
        self.assertEqual(time_min[:1], cadence.tstride*indices.to_scalar(0)[:1])
        self.assertEqual(time_max[:1], time_min[:1] + cadence.texp)

        #--------------------------------------
        # times_at_uv() with fovmask == False
        #--------------------------------------
        uv = Pair([(0,0),(0,20),(10,0),(10,20),(11,21)])

        (time0, time1) = obs.times_at_uv(uv)

        self.assertEqual(time0, cadence.tstride * uv.to_scalar(0))
        self.assertEqual(time1, time0 + cadence.texp)

        #--------------------------------------
        # times_at_uv() with fovmask == True
        #--------------------------------------
        (time0, time1) = obs.times_at_uv(uv, fovmask=True)

        self.assertTrue(np.all(time0.mask == 4*[False] + [True]))
        self.assertTrue(np.all(time1.mask == 4*[False] + [True]))
        self.assertEqual(time0[:4], cadence.tstride * uv.to_scalar(0)[:4])
        self.assertEqual(time1[:4], time0[:4] + cadence.texp)

        ####################################


        #----------------------------------------------
        # Alternative axis order ('uslow','vfast')
        #----------------------------------------------

        fov = FlatFOV((0.001,0.001), (1,10))
        cadence = Metronome(tstart=0., tstride=10., texp=10., steps=10)
        obs = RasterSlit1D(axes=('a','vt'), det_size=1, cadence=cadence,
                           fov=fov, path='SSB', frame='J2000')

        indices = Pair([(0,0),(0,9),(0,10),(0,11)])

        (uv,time) = obs.uvt(indices)

        self.assertEqual(uv.to_scalar(0), 0.5)
        self.assertEqual(uv.to_scalar(1), indices.to_scalar(1))
        self.assertEqual(time, cadence.tstride * indices.to_scalar(1))

        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertEqual(uv_min.to_scalar(0), 0)
        self.assertEqual(uv_min.to_scalar(1), indices.to_scalar(1))
        self.assertEqual(uv_max.to_scalar(0), 1)
        self.assertEqual(uv_max.to_scalar(1), indices.to_scalar(1) + 1)
        self.assertEqual(time_min, cadence.tstride * indices.to_scalar(1))
        self.assertEqual(time_max, time_min + cadence.texp)

        (time0,time1) = obs.times_at_uv(indices)

        self.assertEqual(time0, cadence.tstride * indices.to_scalar(1))
        self.assertEqual(time1, time0 + cadence.texp)

        ####################################


        #-----------------------------------------------------------
        # Alternative det_size and texp for discontinuous indices
        #-----------------------------------------------------------

        fov = FlatFOV((0.001,0.001), (10,1))
        cadence = Metronome(tstart=0., tstride=10., texp=8., steps=10)
        obs = RasterSlit1D(axes=('ut','a'), det_size=0.5, cadence=cadence,
                           fov=fov, path='SSB', frame='J2000')

        self.assertEqual(obs.time[1], 98.)

        self.assertEqual(obs.uvt((0,0),True)[1],    0.)
        self.assertEqual(obs.uvt((5,0),True)[1],   50.)
        self.assertEqual(obs.uvt((5.5,0),True)[1], 54.)
        self.assertEqual(obs.uvt((9.5,0),True)[1], 94.)
        self.assertEqual(obs.uvt((10.,0),True)[1], 98.)
        self.assertTrue(obs.uvt((10.001,0),True)[1].mask)

        eps = 1.e-14
        delta = 1.e-13
        self.assertTrue(abs(obs.uvt((6.     ,0),True)[0] - (6.0,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((6.2    ,1),True)[0] - (6.1,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((6.4    ,2),True)[0] - (6.2,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((6.6    ,3),True)[0] - (6.3,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((6.8    ,4),True)[0] - (6.4,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((7. -eps,5),True)[0] - (6.5,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((7.     ,6),True)[0] - (7.0,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((10.-eps,7),True)[0] - (9.5,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((10     ,8),True)[0] - (9.5,0.5)) < delta)
        self.assertTrue(obs.uvt((10.+eps,8),True)[0].mask)

        indices = Pair([(10-eps,0), (10,0), (10+eps,0)])

        (uv,t) = obs.uvt(indices, fovmask=True)
        self.assertTrue(np.all(t.mask == np.array(2*[False] + [True])))
    #===========================================================================


#*******************************************************************************



########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
