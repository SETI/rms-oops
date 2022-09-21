################################################################################
# oops/frame/spiceframe.py: Subclass SpiceFrame of class Frame
################################################################################

import numpy as np
import cspyce

from polymath import Scalar, Vector3, Matrix3, Quaternion

from scipy.interpolate import UnivariateSpline

from .                import Frame
from ..config         import QUICK
from ..path           import Path
from ..path.spicepath import SpicePath
from ..transform      import Transform
from ..constants      import DPR
import oops.spice_support as spice

class SpiceFrame(Frame):
    """A Frame defined within the SPICE toolkit."""

    #===========================================================================
    def __init__(self, spice_frame, spice_reference='J2000', frame_id=None,
                       omega_type='tabulated', omega_dt=1., unpickled=False):
        """Constructor for a SpiceFrame.

        Input:
            spice_frame     the name or integer ID of the destination frame
                            or of the central body as used in the SPICE toolkit.

            spice_reference the name or integer ID of the reference frame as
                            used in the SPICE toolkit; 'J2000' by default.

            frame_id        the string ID under which the frame will be
                            registered. By default, this will be the name as
                            used by the SPICE toolkit.

            omega_type      'tabulated' to take omega directly from the kernel;
                            'numerical' to return numerical derivatives.
                            'zero'      to ignore omega vectors.

            omega_dt        default time step in seconds to use for spline-based
                            numerical derivatives of omega.

            unpickled       True if this object was read from a pickle file. If
                            so, then it will be treated as a duplicate of a
                            pre-existing SpicePath for the same SPICE ID.
        """

        # Preserve the inputs
        self.spice_frame = spice_frame
        self.spice_reference = spice_reference
        self.omega_type = omega_type
        self.omega_dt = omega_dt

        # Interpret the SPICE frame and reference IDs
        (self.spice_frame_id,
         self.spice_frame_name) = spice.frame_id_and_name(spice_frame)

        (self.spice_reference_id,
         self.spice_reference_name) = spice.frame_id_and_name(spice_reference)

        # Fill in the Frame ID and save it in the SPICE global dictionary
        self.frame_id = frame_id or self.spice_frame_name
        spice.FRAME_TRANSLATION[self.spice_frame_id]   = self.frame_id
        spice.FRAME_TRANSLATION[self.spice_frame_name] = self.frame_id

        # Fill in the reference wayframe
        reference_id = spice.FRAME_TRANSLATION[self.spice_reference_id]
        self.reference = Frame.as_wayframe(reference_id)

        # Fill in the origin waypoint
        self.spice_origin_id = cspyce.frinfo(self.spice_frame_id)[0]
        self.spice_origin_name = cspyce.bodc2n(self.spice_origin_id)
        origin_id = spice.PATH_TRANSLATION[self.spice_origin_id]

        try:
            self.origin = Path.as_waypoint(origin_id)
        except KeyError:
            # If the origin path was never defined, define it now
            origin_path = SpicePath(origin_id)
            self.origin = origin_path.waypoint

        # No shape, no keys
        self.shape = ()
        self.keys = set()

        # Save interpolation method
        assert omega_type in ('tabulated', 'numerical', 'zero')
        self.omega_tabulated = (omega_type == 'tabulated')
        self.omega_numerical = (omega_type == 'numerical')
        self.omega_zero = (omega_type == 'zero')

        # Always register a SpiceFrame
        # This also fills in the waypoint
        self.register(unpickled=unpickled)

    # Unpickled frames will always have temporary IDs to avoid conflicts
    def __getstate__(self):
        return (self.spice_frame_name, self.spice_reference, self.omega_type,
                self.omega_dt)

    def __setstate__(self, state):

        (spice_frame_name, spice_reference, omega_type, omega_dt) = state

        # If this is a duplicate of a pre-existing SpiceFrame, make sure it gets
        # assigned the pre-existing frame ID and Wayframe.
        frame_id = spice.FRAME_TRANSLATION.get(spice_frame_name, None)
        self.__init__(spice_frame_name, spice_reference, frame_id=frame_id,
                      omega_type=omega_type, omega_dt=omega_dt, unpickled=True)

    #===========================================================================
    def transform_at_time(self, time, quick={}):
        """A Transform that rotates from the reference frame into this frame.

        Input:
            time            a Scalar time.
            quick           an optional dictionary of parameter values to use as
                            overrides to the configured default QuickPath and
                            QuickFrame parameters; use False to disable the use
                            of QuickPaths and QuickFrames.

        Return:             the corresponding Tranform applicable at the
                            specified time(s).
        """

        time = Scalar.as_scalar(time).as_float()

        ######## Handle a single time
        if time.shape == ():

            # Case 1: omega_type = tabulated
            if self.omega_tabulated:
                matrix6 = cspyce.sxform(self.spice_reference_name,
                                        self.spice_frame_name,
                                        time.values)
                (matrix, omega) = cspyce.xf2rav(matrix6)

                return Transform(matrix, omega, self, self.reference)

            # Case 2: omega_type = zero
            elif self.omega_zero:
                matrix = cspyce.pxform(self.spice_reference_name,
                                       self.spice_frame_name,
                                       time.values)

                return Transform(matrix, Vector3.ZERO, self, self.reference)

            # Case 3: omega_type = numerical
            else:
                et = time.vals
                times = np.array((et - self.omega_dt, et, et + self.omega_dt))
                mats = np.empty((3,3,3))

                for j in range(len(times)):
                    mats[j] = cspyce.pxform(self.spice_reference_name,
                                            self.spice_frame_name,
                                            times[j])

                # Convert three matrices to quaternions
                quats = Quaternion.as_quaternion(Matrix3(mats))

                # Use a Univariate spline to get components of the derivative
                qdot = np.empty(4)
                for j in range(4):
                    spline = UnivariateSpline(times, quats.values[:,j], k=2,
                                                                        s=0)
                    qdot[j] = spline.derivative(1)(et)

                omega = 2. * (Quaternion(qdot) / quats[1]).values[1:4]
                return Transform(mats[1], omega, self, self.reference)

        ######## Apply the quick_frame if requested

        if isinstance(quick, dict):
            quick = quick.copy()
            quick['quickframe_numerical_omega'] = self.omega_numerical
            quick['ignore_quickframe_omega'] = self.omega_zero

            if self.omega_numerical:
                quick['frame_time_step'] = min(quick['frame_time_step'],
                                               self.omega_dt)

            frame = self.quick_frame(time, quick)
            return frame.transform_at_time(time, quick=False)

        ######## Handle multiple times

        # Case 1: omega_type = tabulated
        if self.omega_tabulated:
            matrix = np.empty(time.shape + (3,3))
            omega  = np.empty(time.shape + (3,))

            for i,t in np.ndenumerate(time.values):
                matrix6 = cspyce.sxform(self.spice_reference_name,
                                        self.spice_frame_name,
                                        t)
                (matrix[i], omega[i]) = cspyce.xf2rav(matrix6)

        # Case 2: omega_type = zero
        elif self.omega_zero:
            matrix = np.empty(time.shape + (3,3))
            omega  = np.zeros(time.shape + (3,))

            for i,t in np.ndenumerate(time.values):
                matrix[i] = cspyce.pxform(self.spice_reference_name,
                                          self.spice_frame_name,
                                          t)

        # Case 3: omega_type = numerical
        # This procedure calculates each omega using its own UnivariateSpline;
        # it could be very slow. A QuickFrame is recommended as it would
        # accomplish the same goals much faster.
        else:
            matrix = np.empty(time.shape + (3,3))
            omega  = np.empty(time.shape + (3,))

            for i,t in np.ndenumerate(time.values):

                # Define a set of three times centered on given time
                times = np.array((t - self.omega_dt, t, t + self.omega_dt))

                # Generate the rotation matrix at each time
                mats = np.empty((3,3,3))
                for j in range(len(times)):
                    mats[j] = cspyce.pxform(self.spice_reference_name,
                                            self.spice_frame_name,
                                            times[j])

                # Convert these three matrices to quaternions
                quats = Quaternion.as_quaternion(Matrix3(mats))

                # Use a Univariate spline to get components of the derivative
                qdot = np.empty(4)
                for j in range(4):
                    spline = UnivariateSpline(times, quats.values[:,j], k=2,
                                                                        s=0)
                    qdot[j] = spline.derivative(1)(t)

                omega[i] = 2. * (Quaternion(qdot) / quats[1]).values[1:4]
                matrix[i] = mats[1]

        return Transform(matrix, omega, self, self.reference)

    #===========================================================================
    def transform_at_time_if_possible(self, time, quick={}):
        """A Transform that rotates from the reference frame into this frame.

        Unlike method transform_at_time(), this variant tolerates times that
        raise cspyce errors. It returns a new time Scalar along with the new
        Transform, where both objects skip over the times at which the transform
        could not be evaluated.

        Input:
            time            a Scalar time, which must be 0-D or 1-D.
            quick           an optional dictionary of parameter values to use as
                            overrides to the configured default QuickPath and
                            QuickFrame parameters; use False to disable the use
                            of QuickPaths and QuickFrames.

        Return:             (newtimes, transform)
            newtimes        a Scalar time, possibly containing a subset of the
                            times given.
            transform       the corresponding Tranform applicable at the new
                            time(s).
        """

        time = Scalar.as_scalar(time).as_float()

        # A single input time can be handled via the previous method
        if time.shape == ():
            return (time, self.transform_at_time(time, quick))

        ######## Apply the quick_frame if requested

        if isinstance(quick, dict):
            quick = quick.copy()
            quick['quickframe_numerical_omega'] = self.omega_numerical
            quick['ignore_quickframe_omega'] = self.omega_zero

            if self.omega_numerical:
                quick['frame_time_step'] = min(quick['frame_time_step'],
                                               self.omega_dt)

            frame = self.quick_frame(time, quick)
            return frame.transform_at_time_if_possible(time, quick=False)

        ######## Handle multiple times

        # Lists used in case of error
        new_time = []
        matrix_list = []
        omega_list = []

        error_found = False

        # Case 1: omega_type = tabulated
        if self.omega_tabulated:
            matrix = np.empty(time.shape + (3,3))
            omega  = np.empty(time.shape + (3,))

            for i,t in np.ndenumerate(time.values):
                try:
                    matrix6 = cspyce.sxform(self.spice_reference_name,
                                            self.spice_frame_name,
                                            t)
                    (matrix[i], omega[i]) = cspyce.xf2rav(matrix6)

                    new_time.append(t)
                    matrix_list.append(matrix[i])
                    omega_list.append(omega[i])

                except (RuntimeError, ValueError, IOError) as e:
                    if len(time.shape) > 1:
                        raise e
                    error_found = True

        # Case 2: omega_type = zero
        elif self.omega_zero:
            matrix = np.empty(time.shape + (3,3))
            omega  = np.zeros(time.shape + (3,))

            for i,t in np.ndenumerate(time.values):
                try:
                    matrix[i] = cspyce.pxform(self.spice_reference_name,
                                              self.spice_frame_name,
                                              t)

                    new_time.append(t)
                    matrix_list.append(matrix[i])
                    omega_list.append((0.,0.,0.))

                except (RuntimeError, ValueError, IOError) as e:
                    if len(time.shape) > 1:
                        raise e
                    error_found = True

        # Case 3: omega_type = numerical
        # This procedure calculates each omega using its own UnivariateSpline;
        # it could be very slow. A QuickFrame is recommended as it would
        # accomplish the same goals much faster.
        else:
            matrix = np.empty(time.shape + (3,3))
            omega  = np.empty(time.shape + (3,))

            for i,t in np.ndenumerate(time.values):
              try:
                times = np.array((t - self.omega_dt, t, t + self.omega_dt))
                mats = np.empty((3,3,3))

                for j in range(len(times)):
                    mats[j] = cspyce.pxform(self.spice_reference_name,
                                            self.spice_frame_name,
                                            times[j])

                # Convert three matrices to quaternions
                quats = Quaternion.as_quaternion(Matrix3(mats))

                # Use a Univariate spline to get components of the derivative
                qdot = np.empty(4)
                for j in range(4):
                    spline = UnivariateSpline(times, quats.values[:,j], k=2,
                                                                        s=0)
                    qdot[j] = spline.derivative(1)(t)

                omega[i] = 2. * (Quaternion(qdot) / quats[1]).values[1:4]
                matrix[i] = mats[1]

                new_time.append(t)
                matrix_list.append(matrix[i])
                omega_list.append(omega[i])

              except (RuntimeError, ValueError, IOError) as e:
                if len(time.shape) > 1:
                    raise e
                error_found = True

        if error_found:
            if len(new_time) == 0:
                raise e

            time = Scalar(new_time)
            matrix = Matrix3(matrix_list)
            omega = Vector3(omega_list)
        else:
            matrix = Matrix3(matrix)
            omega = Vector3(omega)

        return (time, Transform(matrix, omega, self, self.reference))

################################################################################
# UNIT TESTS
################################################################################

# Here we also test many of the overall Frame operations, because we can be
# confident that cspyce produces valid results.

import unittest

class Test_SpiceFrame(unittest.TestCase):

    def runTest(self):

        np.random.seed(6242)

        # Imports are here to avoid conflicts
        import os.path
        from ..path           import Path
        from ..path.spicepath import SpicePath
        from ..frame          import QuickFrame
        from ..event          import Event

        from oops.unittester_support import TESTDATA_PARENT_DIRECTORY
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, 'SPICE/naif0009.tls'))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, 'SPICE/pck00010.tpc'))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, 'SPICE/de421.bsp'))

        Path.USE_QUICKPATHS = False
        Frame.USE_QUICKFRAMES = False

        Path.reset_registry()
        Frame.reset_registry()

        ignore = SpicePath('EARTH', 'SSB')

        earth = SpiceFrame('IAU_EARTH', 'J2000')
        time  = Scalar(np.random.rand(3,4,2) * 1.e8)
        posvel = np.random.rand(3,4,2,6,1)
        event = Event(time, (posvel[...,0:3,0],posvel[...,3:6,0]), 'SSB',
                                                                   'J2000')
        rotated = event.wrt_frame('IAU_EARTH')

        for i,t in np.ndenumerate(time.vals):
            matrix6 = cspyce.sxform('J2000', 'IAU_EARTH', t)
# The matrix subclass is to be deprecated? Really?
#             spiceval = np.matrix(matrix6) * np.matrix(posvel[i])
            spiceval = np.matmul(matrix6, posvel[i])[...,np.newaxis]

            dpos = rotated.pos[i].vals[...,np.newaxis] - spiceval[0:3,0]
            dvel = rotated.vel[i].vals[...,np.newaxis] - spiceval[3:6,0]

            self.assertTrue(np.all(np.abs(dpos) < 1.e-15))
            self.assertTrue(np.all(np.abs(dvel) < 1.e-15))

        # Tests of combined frames
        Path.reset_registry()
        Frame.reset_registry()

        ignore = SpicePath('EARTH', 'SSB')
        ignore = SpicePath('VENUS', 'EARTH')
        ignore = SpicePath('MARS', 'VENUS')
        ignore = SpicePath('MOON', 'VENUS')

        earth  = SpiceFrame('IAU_EARTH', 'J2000')
        b1950  = SpiceFrame('B1950', 'IAU_EARTH')
        venus  = SpiceFrame('IAU_VENUS', 'B1950')
        mars   = SpiceFrame('IAU_MARS',  'J2000')
        mars   = SpiceFrame('IAU_MOON',  'B1950')

        times = Scalar(np.arange(-3.e8, 3.01e8, 0.5e7))

        frame = Frame.as_frame('IAU_EARTH').wrt('J2000')
        transform = frame.transform_at_time(times)
        for i in range(times.vals.size):
            matrix6 = cspyce.sxform('J2000', 'IAU_EARTH', times[i].vals)
            (matrix, omega) = cspyce.xf2rav(matrix6)

            dmatrix = transform.matrix[i].vals - matrix
            domega  = transform.omega[i].vals  - omega

            self.assertTrue(np.all(np.abs(dmatrix) < 1.e-14))
            self.assertTrue(np.all(np.abs(domega)  < 1.e-14))

        frame = Frame.as_frame('J2000').wrt('IAU_EARTH')
        transform = frame.transform_at_time(times)
        for i in range(times.vals.size):
            matrix6 = cspyce.sxform('IAU_EARTH', 'J2000', times[i].vals)
            (matrix, omega) = cspyce.xf2rav(matrix6)

            dmatrix = transform.matrix[i].vals - matrix
            domega  = transform.omega[i].vals  - omega

            self.assertTrue(np.all(np.abs(dmatrix) < 1.e-14))
            self.assertTrue(np.all(np.abs(domega)  < 1.e-14))

        frame = Frame.as_frame('B1950').wrt('J2000')
        transform = frame.transform_at_time(times)
        for i in range(times.vals.size):
            matrix6 = cspyce.sxform('J2000', 'B1950', times[i].vals)
            (matrix, omega) = cspyce.xf2rav(matrix6)

            dmatrix = transform.matrix[i].vals - matrix
            domega  = transform.omega[i].vals  - omega

            self.assertTrue(np.all(np.abs(dmatrix) < 1.e-14))
            self.assertTrue(np.all(np.abs(domega)  < 1.e-14))

        frame = Frame.as_frame('J2000').wrt('B1950')
        transform = frame.transform_at_time(times)
        for i in range(times.vals.size):
            matrix6 = cspyce.sxform('B1950', 'J2000', times[i].vals)
            (matrix, omega) = cspyce.xf2rav(matrix6)

            dmatrix = transform.matrix[i].vals - matrix
            domega  = transform.omega[i].vals  - omega

            self.assertTrue(np.all(np.abs(dmatrix) < 1.e-14))
            self.assertTrue(np.all(np.abs(domega)  < 1.e-14))

        Path.reset_registry()
        Frame.reset_registry()

        ########################################
        # Test for a Cassini C kernel
        ########################################

        # Load all the required kernels for Cassini ISS on 2007-312
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY,
                                  'SPICE', 'naif0009.tls'))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY,
                                  'SPICE', 'cas00149.tsc'))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY,
                                  'SPICE', 'cas_v40.tf'))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY,
                                  'SPICE', 'cas_status_v04.tf'))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY,
                                  'SPICE', 'cas_iss_v10.ti'))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY,
                                  'SPICE', 'pck00010.tpc'))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY,
                                  'SPICE', 'cpck14Oct2011.tpc'))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY,
                                  'SPICE', 'de421.bsp'))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY,
                                  'SPICE', 'sat052.bsp'))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY,
                                  'SPICE', 'sat083.bsp'))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY,
                                  'SPICE', 'sat125.bsp'))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY,
                                  'SPICE', 'sat128.bsp'))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY,
                                  'SPICE', 'sat164.bsp'))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY,
                                  'SPICE', '07312_07317ra.bc'))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY,
                                  'SPICE', '080123R_SCPSE_07309_07329.bsp'))

        ignore = SpicePath('CASSINI', 'SSB')
        ignore = SpiceFrame('CASSINI_ISS_NAC')
        ignore = SpiceFrame('CASSINI_ISS_WAC')

        # Look up N1573186009_1.IMG from COISS_2039/data/1573186009_1573197826/
        timestring = '2007-312T03:34:16.391'
        TDB = cspyce.str2et(timestring)

        nacframe = Frame.J2000.wrt('CASSINI_ISS_NAC')
        matrix = nacframe.transform_at_time(TDB).matrix
        optic_axis = (matrix * Vector3((0,0,1))).vals

        test_ra  = (np.arctan2(optic_axis[1], optic_axis[0]) * DPR) % 360
        test_dec = np.arcsin(optic_axis[2]) * DPR

        right_ascension = 194.30861     # from the index table
        declination = 3.142808

        self.assertTrue(np.all(np.abs(test_ra - right_ascension) < 0.5))
        self.assertTrue(np.all(np.abs(test_dec - declination) < 0.5))

        ########################################
        # Test of various omega methods
        ########################################

        wac1 = SpiceFrame('CASSINI_ISS_WAC', omega_type='tabulated', frame_id='wac1')
        wac2 = SpiceFrame('CASSINI_ISS_WAC', omega_type='numerical', frame_id='wac2')
        wac3 = SpiceFrame('CASSINI_ISS_WAC', omega_type='zero',      frame_id='wac3')

        # Test a single time
        xform1 = wac1.transform_at_time(TDB, quick=None)
        xform2 = wac2.transform_at_time(TDB, quick=None)
        xform3 = wac3.transform_at_time(TDB, quick=None)
        self.assertEqual(xform1.matrix, xform2.matrix)
        self.assertEqual(xform1.matrix, xform3.matrix)
        self.assertEqual(xform3.omega, Vector3.ZERO)

        DT = 0.2
        xform2a = wac2.transform_at_time(TDB-DT, quick=None)
        xform2b = wac2.transform_at_time(TDB+DT, quick=None)

        axes = (Vector3.XAXIS, Vector3.YAXIS, Vector3.ZAXIS)
        for i in range(3):
            v = xform2.omega.ucross(axes[i])
            rotated_time0 = xform2a.matrix.unrotate(v)
            rotated_time1 = xform2b.matrix.unrotate(v)
            angle = rotated_time1.sep(rotated_time0)
            ratio = (angle/(2.*DT) / xform2.omega.norm()).vals
            self.assertAlmostEqual(ratio, 1., places=9)

        # Test an array of times
        times = Scalar((TDB-5., TDB+2., TDB-9.5, TDB+7.7))

        xform1 = wac1.transform_at_time(times, quick=None)
        xform2 = wac2.transform_at_time(times, quick=None)
        xform3 = wac3.transform_at_time(times, quick=None)
        self.assertEqual(xform1.matrix, xform2.matrix)
        self.assertEqual(xform1.matrix, xform3.matrix)
        self.assertEqual(xform3.omega, Vector3.ZERO)

        DT = 1.
        xform2a = wac2.transform_at_time(times-DT, quick=None)
        xform2b = wac2.transform_at_time(times+DT, quick=None)

        axes = (Vector3.XAXIS, Vector3.YAXIS, Vector3.ZAXIS)
        for i in range(3):
            v = xform2.omega.ucross(axes[i])
            rotated_time0 = xform2a.matrix.unrotate(v)
            rotated_time1 = xform2b.matrix.unrotate(v)
            angle = rotated_time1.sep(rotated_time0)
            ratio = angle/(2.*DT) / xform2.omega.norm()
            mask = (angle.vals != 0.)
            self.assertTrue(abs(ratio[mask]).max() - 1. < 1.e-9)

        # Test a single value using transform_at_time_if_possible
        xform1 = wac1.transform_at_time_if_possible(TDB, quick=None)[1]
        xform2 = wac2.transform_at_time_if_possible(TDB, quick=None)[1]
        self.assertEqual(xform1.matrix, xform2.matrix)

        DT = 0.2
        xform2a = wac2.transform_at_time_if_possible(TDB-DT, quick=None)[1]
        xform2b = wac2.transform_at_time_if_possible(TDB+DT, quick=None)[1]

        axes = (Vector3.XAXIS, Vector3.YAXIS, Vector3.ZAXIS)
        for i in range(3):
            v = xform2.omega.ucross(axes[i])
            rotated_time0 = xform2a.matrix.unrotate(v)
            rotated_time1 = xform2b.matrix.unrotate(v)
            angle = rotated_time1.sep(rotated_time0)
            ratio = (angle/(2.*DT) / xform2.omega.norm()).vals
            self.assertAlmostEqual(ratio, 1., places=9)

        # Test an array of times using transform_at_time_if_possible
        times = Scalar((TDB-5., TDB+2., TDB-9.5, TDB+7.7))

        xform1 = wac1.transform_at_time_if_possible(times, quick=None)[1]
        xform2 = wac2.transform_at_time_if_possible(times, quick=None)[1]
        xform3 = wac3.transform_at_time_if_possible(times, quick=None)[1]
        self.assertEqual(xform1.matrix, xform2.matrix)
        self.assertEqual(xform1.matrix, xform3.matrix)
        self.assertEqual(xform3.omega, Vector3.ZERO)

        DT = 1.
        xform2a = wac2.transform_at_time_if_possible(times-DT, quick=None)[1]
        xform2b = wac2.transform_at_time_if_possible(times+DT, quick=None)[1]

        axes = (Vector3.XAXIS, Vector3.YAXIS, Vector3.ZAXIS)
        for i in range(3):
            v = xform2.omega.ucross(axes[i])
            rotated_time0 = xform2a.matrix.unrotate(v)
            rotated_time1 = xform2b.matrix.unrotate(v)
            angle = rotated_time1.sep(rotated_time0)
            ratio = angle/(2.*DT) / xform2.omega.norm()
            mask = (angle.vals != 0.)
            self.assertTrue(abs(ratio[mask]).max() - 1. < 1.e-9)

        # Test an array of times using transform_at_time_if_possible
        # In this run, several times will fail.
        times = Scalar((TDB-5., TDB-1.e20, TDB+2., TDB-9.5, TDB+7.7, TDB+1.e10))

        xform1 = wac1.transform_at_time_if_possible(times, quick=None)[1]
        xform2 = wac2.transform_at_time_if_possible(times, quick=None)[1]
        xform3 = wac3.transform_at_time_if_possible(times, quick=None)[1]
        self.assertEqual(xform1.matrix, xform2.matrix)
        self.assertEqual(xform1.matrix, xform3.matrix)
        self.assertEqual(xform3.omega, Vector3.ZERO)

        self.assertEqual(xform1.shape, (4,))    # two of the times are invalid
        self.assertEqual(xform2.shape, (4,))

        DT = 1.
        xform2a = wac2.transform_at_time_if_possible(times-DT, quick=None)[1]
        xform2b = wac2.transform_at_time_if_possible(times+DT, quick=None)[1]

        axes = (Vector3.XAXIS, Vector3.YAXIS, Vector3.ZAXIS)
        for i in range(3):
            v = xform2.omega.ucross(axes[i])
            rotated_time0 = xform2a.matrix.unrotate(v)
            rotated_time1 = xform2b.matrix.unrotate(v)
            angle = rotated_time1.sep(rotated_time0)
            ratio = angle/(2.*DT) / xform2.omega.norm()
            mask = (angle.vals != 0.)
            self.assertTrue(abs(ratio[mask]).max() - 1. < 1.e-9)

        ########################################
        # Tests of QuickFrame interpolation
        ########################################

        wac1 = SpiceFrame('CASSINI_ISS_WAC', omega_type='tabulated',)
        wac2 = SpiceFrame('CASSINI_ISS_WAC', omega_type='numerical',
                                             omega_dt=0.1)
        wac3 = SpiceFrame('CASSINI_ISS_WAC', omega_type='zero')

        quickdict = QUICK.dictionary.copy()
        quickdict['quickframe_numerical_omega'] = False
        quickdict['frame_time_step'] = 0.01
        wac1a = QuickFrame(wac1, (TDB-100.,TDB+100.), quickdict)

        quickdict['quickframe_numerical_omega'] = True
        wac1b = QuickFrame(wac1, (TDB-100.,TDB+100.), quickdict)

        wac3a = QuickFrame(wac3, (TDB-100.,TDB+100.), quickdict)

        # Test a single time
        time = TDB - 44.
        xform1 = wac1a.transform_at_time(time, quick=None)
        xform2 = wac1b.transform_at_time(time, quick=None)
        xform3 = wac2.transform_at_time(time, quick=None)
        xform4 = wac3.transform_at_time(time, quick=None)
        self.assertEqual(xform4.omega, Vector3.ZERO)

        self.assertEqual(xform1.matrix, xform2.matrix)

        diff = xform3.matrix.values - xform1.matrix.values
        self.assertTrue(np.max(abs(diff)) < 1.e-11)

        diff = xform4.matrix.values - xform1.matrix.values
        self.assertTrue(np.max(abs(diff)) < 1.e-11)

        diff = (xform3.omega - xform2.omega).norm()
        self.assertTrue(diff.vals < 1.e-7)

        diff = (xform3.omega - xform2.omega).norm() / xform2.omega.norm()
        self.assertTrue(diff.vals < 1.e-3)

        # Test the linear interpolation limit where delta-time < 1 sec
        time = Scalar((TDB - 41.0123, TDB - 41.1357, TDB - 41.6543))
        xform2 = wac1b.transform_at_time(time, quick=None)
        xform3 = wac2.transform_at_time(time, quick=None)
        xform4 = wac3.transform_at_time(time, quick=None)
        self.assertEqual(xform4.omega, Vector3.ZERO)

        diff = xform3.matrix.values - xform2.matrix.values
        self.assertTrue(np.max(abs(diff)) < 3.e-9)

        diff = xform4.matrix.values - xform2.matrix.values
        self.assertTrue(np.max(abs(diff)) < 3.e-9)

        diff = (xform3.omega - xform2.omega).norm()
        self.assertTrue(diff.max() < 1.e-7)

        # Test the linear interpolation limit where delta-time > 1 sec
        time = Scalar((TDB - 40.0123, TDB + 41.1357, TDB - 1.6543))
        xform2 = wac1b.transform_at_time(time, quick=None)
        xform3 = wac2.transform_at_time(time, quick=None)
        xform4 = wac3.transform_at_time(time, quick=None)
        self.assertEqual(xform4.omega, Vector3.ZERO)

        diff = xform3.matrix.values - xform2.matrix.values
        self.assertTrue(np.max(abs(diff)) < 1.e-9)

        diff = xform4.matrix.values - xform2.matrix.values
        self.assertTrue(np.max(abs(diff)) < 1.e-9)

        diff = (xform3.omega - xform2.omega).norm()
        self.assertTrue(diff.max() < 1.e-7)

        Path.reset_registry()
        Frame.reset_registry()

        Path.USE_QUICKPATHS = True
        Frame.USE_QUICKFRAMES = True

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
