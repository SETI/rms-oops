################################################################################
# oops/frame_/poleframe.py: Subclass PoleFrame of class Frame
################################################################################

import numpy as np
from polymath import *

from oops.frame_.frame   import Frame
from oops.frame_.cmatrix import Cmatrix
from oops.transform      import Transform
from oops.constants      import *

#*******************************************************************************
# PoleFrame
#*******************************************************************************
class PoleFrame(Frame):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    PoleFrame is a Frame subclass describing a non-rotating frame centered on
    the Z-axis of a body's pole vector. This differs from RingFrame in that the
    pole may precess around a separate, invariable pole for the system. Because
    of this behavior, the reference longitude is defined as the ascending node
    of the invariable plane rather than that of the ring plane. This frame is
    recommended for Neptune in particular.

    This frame also supports Laplace Planes. A Laplace Plane can have a
    specified tilt angle, which is a rotation of the planet's pole toward the
    invariable pole.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    PACKRAT_ARGS = ['planet_frame', 'invariable_pole', 'tilt', 'retrograde',
                    'frame_id', 'given_cache_size']

    #===========================================================================
    # __init__
    #===========================================================================
    def __init__(self, frame, pole, tilt=0., retrograde=False, id='+',
                       cache_size=1000):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Constructor for a PoleFrame.

        Input:
            frame       a (possibly) rotating frame, or its ID, describing the
                        central planet relative to J2000. This is typically a
                        body's rotating SpiceFrame.

            pole        The pole of the invariable plane, about which planet's
                        pole precesses. This enables the reference longitude to
                        be defined properly. Defined in J2000 coordinates.

            tilt        The tilt of the Laplace Plane from the planet's pole
                        toward or beyond the invariable pole. Default is zero.

            retrograde  True to flip the sign of the Z-axis. Necessary for
                        retrograde systems like Uranus.

            id          the ID under which the frame will be registered. None to
                        leave the frame unregistered. If the value is "+", then
                        the registered name is the planet frame's name with the
                        suffix "_POLE". Note that this default ID will not be
                        unique if frames are defined for multiple Laplace Planes
                        around the same planet.

            cache_size  number of transforms to cache. This can be useful
                        because it avoids unnecessary SPICE calls when the frame
                        is being used repeatedly at a finite set of times.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #-------------------------------------------------
        # Rotates from J2000 to the invariable frame
        #-------------------------------------------------
        (ra,dec,_) = Vector3.as_vector3(pole).to_ra_dec_length(recursive=False)
        self.invariable_matrix = Matrix3.pole_rotation(ra,dec)
        self.invariable_pole = pole

        self.tilt = tilt
        self.cos_tilt = np.cos(tilt)
        self.sin_tilt = np.sin(tilt)

        self.planet_frame = Frame.as_frame(frame).wrt(Frame.J2000)
        self.origin = self.planet_frame.origin
        self.retrograde = retrograde
        self.shape = ()
        self.keys = set()
        self.reference = Frame.J2000

        #--------------------------
        # Fill in the frame ID
        #--------------------------
        if id is None:
            self.frame_id = Frame.temporary_frame_id()
        elif id == '+':
            self.frame_id = self.planet_frame.frame_id + '_POLE'
        else:
            self.frame_id = id

        #--------------------------
        # Register if necessary
        #--------------------------
        if id:
            self.register()
        else:
            self.wayframe = self

        self.cache = {}
        self.trim_size = max(cache_size//10, 1)
        self.given_cache_size = cache_size
        self.cache_size = cache_size + self.trim_size
        self.cache_counter = 0
        self.cached_value_returned = False          # Just used for debugging
    #===========================================================================



    #===========================================================================
    # transform_at_time
    #===========================================================================
    def transform_at_time(self, time, quick={}):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The Transform into the this Frame at a Scalar of times.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------
        # Check cache first if time is a Scalar
        #------------------------------------------
        time = Scalar.as_scalar(time)
        if time.shape == ():
            key = time.values

            if key in self.cache:
                self.cached_value_returned = True
                (count, key, xform) = self.cache[key]
                self.cache_counter += 1
                count[0] = self.cache_counter
                return xform

        self.cached_value_returned = False

        #------------------------------------------------------------
        # Calculate the planet frame for the current time in J2000
        # Note that all that really matters is the planet's pole
        #------------------------------------------------------------
        xform = self.planet_frame.transform_at_time(time, quick=quick)

        #-----------------------------------------------------
        # For a retrograde ring, the X- and Z-axes reverse
        #-----------------------------------------------------
        if self.retrograde:
            xform = xform.rotate_transform(oops.Matrix([[-1., 0., 0.],
                                                        [ 0., 1., 0.],
                                                        [ 0., 0.,-1.]]))

        #----------------------------------------------------------------------
        # The bottom row of the matrix is the Z-axis of the ring frame in J2000
        #----------------------------------------------------------------------
        z_axis = xform.matrix.row_vector(2)

        matrix = xform.matrix.vals

        #-----------------------------------------------------------------
        # If the tilt is nonzero, rotate toward the invariable pole now
        #-----------------------------------------------------------------
        if self.tilt != 0.:

            # This is the axis to rotate around by angle self.tilt
            rotation_pole = z_axis.ucross(self.invariable_pole)

            # This is the vector perpendicular to both z_axis and perp
            # It is where the Z-axis ends up if the tilt is 90 degrees
            target_pole = rotation_pole.ucross(z_axis)

            z_axis = z_axis * self.cos_tilt + target_pole * self.sin_tilt

        matrix = Matrix3.twovec(z_axis, 2, Vector3.ZAXIS.cross(z_axis), 0)
        planet_matrix = Matrix3(matrix)

        #-----------------------------------------------------------------------
        # This is the RingFrame matrix. It rotates from J2000 to the frame where
        # the pole at epoch is along the Z-axis and the ascending node relative
        # to the J2000 equator is along the X-axis.
        #-----------------------------------------------------------------------
        if self.tilt != 0.:
            test = planet_matrix.unrotate(rotation_pole)
            print 11111, time, np.arctan2(test.vals[1], test.vals[0]) * 180/np.pi

        #---------------------------------------------------------------------
        # Locate the J2000 ascending node of the RingFrame on the invariable
        # plane.
        #---------------------------------------------------------------------
        planet_pole_j2000 = planet_matrix.inverse() * Vector3.ZAXIS
        joint_node_j2000 = self.invariable_pole.cross(planet_pole_j2000)

        joint_node_wrt_planet = planet_matrix * joint_node_j2000
        joint_node_wrt_frame = self.invariable_matrix * joint_node_j2000

        node_lon_wrt_planet = joint_node_wrt_planet.to_ra_dec_length()[0]
        node_lon_wrt_frame = joint_node_wrt_frame.to_ra_dec_length()[0]

        # Align the X-axis with the node of the invariable plane
        matrix = Matrix3.z_rotation(node_lon_wrt_planet -
                                    node_lon_wrt_frame) * planet_matrix

        #------------------------
        # Create the transform
        #------------------------
        xform = Transform(Matrix3(matrix, xform.matrix.mask), Vector3.ZERO,
                          self.wayframe, self.reference)

        if self.tilt != 0.:
            print 22222, time, (NODE0 + time * DNODE_DT) % (2*np.pi) * 180/np.pi

        #--------------------------------------
        # Cache the transform if necessary
        #--------------------------------------
        if time.shape == () and self.given_cache_size > 0:

            # Trim the cache, removing the values used least recently
            if len(self.cache) >= self.cache_size:
                all_keys = self.cache.values()
                all_keys.sort()
                for (_, old_key, _) in all_keys[:self.trim_size]:
                    del self.cache[old_key]

            # Insert into the cache
            key = time.values
            self.cache_counter += 1
            count = np.array([self.cache_counter])
            self.cache[key] = (count, key, xform)

        return xform
    #===========================================================================



    #===========================================================================
    # node_at_time
    #===========================================================================
    def node_at_time(self, time, quick={}):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Angle from the original X-axis to the invariable plane's ascending
        node.

        This serves as the X-axis of this frame.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return 0.
    #===========================================================================


#*******************************************************************************


################################################################################
# UNIT TESTS
################################################################################

import unittest

#*******************************************************************************
# Test_PoleFrame
#*******************************************************************************
class Test_PoleFrame(unittest.TestCase):

    #===========================================================================
    # runTest
    #===========================================================================
    def runTest(self):

        #-----------------------------------------
        # Imports are here to reduce conflicts
        #-----------------------------------------
        import os
        import cspyce
        from oops.event import Event
        from oops.frame_.ringframe   import RingFrame
        from oops.frame_.spiceframe  import SpiceFrame
        from oops.path_.spicepath    import SpicePath
        from oops.path_.path         import Path
        from oops.unittester_support import TESTDATA_PARENT_DIRECTORY

        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, 'SPICE/naif0009.tls'))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, 'SPICE/pck00010.tpc'))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, 'SPICE/de421.bsp'))

        Path.reset_registry()
        Frame.reset_registry()

        center = SpicePath('MARS', 'SSB')
        planet = SpiceFrame('IAU_MARS', 'J2000')
        self.assertEqual(Frame.as_wayframe('IAU_MARS'), planet.wayframe)

        #-----------------------------------------------------------------------
        # This invariable pole is aligned with the planet's pole, so this should
        # behave just like a RingFrame
        #-----------------------------------------------------------------------
        pole = planet.transform_at_time(0.).matrix.inverse() * Vector3.ZAXIS
        poleframe = PoleFrame(planet, pole, cache_size=0)
        ringframe = RingFrame(planet, epoch=0.)
        self.assertEqual(Frame.as_wayframe('IAU_MARS_POLE'), poleframe.wayframe)

        vectors = Vector3(np.random.rand(3,4,2,3)).unit()

        ring_vecs = ringframe.transform_at_time(0.).rotate(vectors)
        pole_vecs = poleframe.transform_at_time(0.).rotate(vectors)
        diffs = ring_vecs - pole_vecs
        self.assertTrue(np.max(np.abs(diffs.values)) < 1.e-15)

        posvel = np.random.rand(3,4,2,6)
        event = Event(0., (posvel[...,0:3], posvel[...,3:6]), 'SSB', 'J2000')
        rotated = event.wrt_frame('IAU_MARS')
        fixed   = event.wrt_frame(poleframe)

        #---------------------------------------------
        # Confirm Z axis is tied to planet's pole
        #---------------------------------------------
        diffs = Scalar(rotated.pos.vals[...,2]) - Scalar(fixed.pos.vals[...,2])
        self.assertTrue(np.max(np.abs(diffs.values)) < 1.e-15)

        #------------------------------------------------
        # Confirm X-axis is tied to the J2000 equator
        #------------------------------------------------
        xaxis = Event(0., Vector3.XAXIS, 'SSB', poleframe)
        test = xaxis.wrt_frame('J2000').pos
        self.assertTrue(abs(test.values[2]) < 1.e-15)

        #--------------------------------------
        # Confirm it's the ascending node
        #--------------------------------------
        xaxis = Event(0., (1,1.e-8,0), 'SSB', poleframe)
        test = xaxis.wrt_frame('J2000').pos
        self.assertTrue(test.values[2] > 0.)

        #### Now try for Neptune

        center = SpicePath('NEPTUNE', 'SSB')
        planet = SpiceFrame('IAU_NEPTUNE', 'J2000')

        #-----------------------------------------------------------------------
        # This invariable pole is aligned with the planet's pole, so this should
        #-----------------------------------------------------------------------
        # behave just like a RingFrame
        pole = planet.transform_at_time(0.).matrix.inverse() * Vector3.ZAXIS
        poleframe = PoleFrame(planet, pole, cache_size=0)
        ringframe = RingFrame(planet, epoch=0.)

        vectors = Vector3(np.random.rand(3,4,2,3)).unit()

        ring_vecs = ringframe.transform_at_time(0.).rotate(vectors)
        pole_vecs = poleframe.transform_at_time(0.).rotate(vectors)
        diffs = ring_vecs - pole_vecs
        self.assertTrue(np.max(np.abs(diffs.values)) < 1.e-15)

        posvel = np.random.rand(3,4,2,6)
        event = Event(0., (posvel[...,0:3], posvel[...,3:6]), 'SSB', 'J2000')
        rotated = event.wrt_frame('IAU_NEPTUNE')
        fixed   = event.wrt_frame(poleframe)

        #--------------------------------------------
        # Confirm Z axis is tied to planet's pole
        #--------------------------------------------
        diffs = Scalar(rotated.pos.vals[...,2]) - Scalar(fixed.pos.vals[...,2])
        self.assertTrue(np.max(np.abs(diffs.values)) < 1.e-15)

        #------------------------------------------------
        # Confirm X-axis is tied to the J2000 equator
        #------------------------------------------------
        xaxis = Event(0., Vector3.XAXIS, 'SSB', poleframe)
        test = xaxis.wrt_frame('J2000').pos
        self.assertTrue(abs(test.values[2]) < 1.e-15)

        #-------------------------------------
        # Confirm it's the ascending node
        #-------------------------------------
        xaxis = Event(0., (1,1.e-8,0), 'SSB', poleframe)
        test = xaxis.wrt_frame('J2000').pos
        self.assertTrue(test.values[2] > 0.)

        #### Neptune at multiple times, with actual polar precession

        times = Scalar(np.arange(1000) * 86400. * 365.)     # 1000 years

        ra  = cspyce.bodvrd('NEPTUNE', 'POLE_RA')[0]  * np.pi/180
        dec = cspyce.bodvrd('NEPTUNE', 'POLE_DEC')[0] * np.pi/180
        pole = Vector3.from_ra_dec_length(ra,dec)
        poleframe = PoleFrame(planet, pole, cache_size=0)

        #----------------------------------------
        # Make sure Z-axis tracks Neptune pole
        #----------------------------------------
        pole_vecs = poleframe.transform_at_time(times).unrotate(Vector3.ZAXIS)
        test_vecs = planet.transform_at_time(times).unrotate(Vector3.ZAXIS)
        diffs = pole_vecs - test_vecs
        self.assertTrue(np.max(np.abs(diffs.values)) < 1.e-15)

        #---------------------------------------------------------
        # Make sure Z-axis circles the pole at uniform distance
        #---------------------------------------------------------
        seps = pole_vecs.sep(pole)
        sep_mean = seps.mean()
        self.assertTrue(np.max(np.abs(seps.values - sep_mean)) < 3.e-5)

        #------------------------------------------------------
        # Make sure the X-axis stays close to the ecliptic
        #------------------------------------------------------
        node_vecs = poleframe.transform_at_time(times).unrotate(Vector3.XAXIS)
        min_node_z = np.min(node_vecs.values[:,2])
        max_node_z = np.max(node_vecs.values[:,2])
        self.assertTrue(min_node_z > -0.0062)
        self.assertTrue(max_node_z <  0.0062)
        self.assertTrue(abs(min_node_z + max_node_z) < 1.e-8)

        #-------------------------------------------------------------
        # Make sure the X-axis stays in a generally fixed direction
        #-------------------------------------------------------------
        diffs = node_vecs - node_vecs[0]
        self.assertTrue(diffs.norm().max() < 0.02)

        #### Test cache

        poleframe = PoleFrame(planet, pole, cache_size=3)
        self.assertTrue(poleframe.cache_size == 4)
        self.assertTrue(poleframe.trim_size == 1)
        self.assertTrue(len(poleframe.cache) == 0)

        pole_vecs = poleframe.transform_at_time(times).unrotate(Vector3.ZAXIS)
        self.assertTrue(len(poleframe.cache) == 0)  # don't cache vectors
        self.assertFalse(poleframe.cached_value_returned)

        pole_vecs = poleframe.transform_at_time(100.).unrotate(Vector3.ZAXIS)
        self.assertTrue(len(poleframe.cache) == 1)
        self.assertTrue(100. in poleframe.cache)
        self.assertFalse(poleframe.cached_value_returned)

        pole_vecs = poleframe.transform_at_time(100.).unrotate(Vector3.ZAXIS)
        self.assertTrue(len(poleframe.cache) == 1)
        self.assertTrue(poleframe.cached_value_returned)

        pole_vecs = poleframe.transform_at_time(200.).unrotate(Vector3.ZAXIS)
        self.assertTrue(len(poleframe.cache) == 2)

        pole_vecs = poleframe.transform_at_time(300.).unrotate(Vector3.ZAXIS)
        self.assertTrue(len(poleframe.cache) == 3)

        pole_vecs = poleframe.transform_at_time(400.).unrotate(Vector3.ZAXIS)
        self.assertTrue(len(poleframe.cache) == 4)

        pole_vecs = poleframe.transform_at_time(500.).unrotate(Vector3.ZAXIS)
        self.assertTrue(len(poleframe.cache) == 4)
        self.assertTrue(100. not in poleframe.cache)

        pole_vecs = poleframe.transform_at_time(200.).unrotate(Vector3.ZAXIS)
        self.assertTrue(len(poleframe.cache) == 4)
        self.assertTrue(poleframe.cached_value_returned)

        pole_vecs = poleframe.transform_at_time(100.).unrotate(Vector3.ZAXIS)
        self.assertTrue(len(poleframe.cache) == 4)
        self.assertFalse(poleframe.cached_value_returned)
        self.assertTrue(300. not in poleframe.cache)

        Path.reset_registry()
        Frame.reset_registry()
    #===========================================================================


#*******************************************************************************



########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
