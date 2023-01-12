################################################################################
# oops/fov/TDIFOV.py: TDIFOV subclass of class FOV
################################################################################

from polymath import Scalar, Pair
from oops.fov import FOV

class TDIFOV(FOV):
    """FOV subclass to apply TDI timing to another FOV."""

    IS_TIME_INDEPENDENT = False

    #===========================================================================
    def __init__(self, fov, tstop, tdi_texp, tdi_axis):
        """Constructor for a TDIFOV.

        Input:
            fov         the time-independent FOV to which this TDI timing is
                        applied.
            tstop       end time of the observation.
            tdi_texp    time interval between TDI shifts.
            tdi_axis    "u", "v", "-u", or "-v", the FOV axis and along which
                        the "Time Delay and Integration" applies, with the sign
                        of the direction.
        """

        self.fov = fov
        self.tstop = float(tstop)
        self.tdi_texp = float(tdi_texp)
        self.tdi_axis = tdi_axis[-1]
        self.tdi_sign = -1 if '-' in tdi_axis else 1

        # Validation
        assert tdi_axis in {'u', 'v', '-u', '-v', '+u', '+v'}

        # Interpret the axis
        if self.tdi_axis == 'u':
            self._duv_dshift = Pair((self.tdi_sign, 0))
            self._uv_line_index = 0
        else:
            self._duv_dshift = Pair((0, self.tdi_sign))
            self._uv_line_index = 1

        self._duv_dt = self._duv_dshift / self.tdi_texp

        # Required attributes
        self.uv_los   = self.fov.uv_los
        self.uv_scale = self.fov.uv_scale
        self.uv_shape = self.fov.uv_shape
        self.uv_area  = self.fov.uv_area

    def __getstate__(self):
        return (self.fov, self.tdi_axis, self.cadence)

    def __setstate__(self, *state):
        self.__init__(*state)

    #===========================================================================
    def xy_from_uvt(self, uv_pair, time=None, derivs=False, remask=False,
                                                            **keywords):
        """The (x,y) camera frame coordinates given the FOV coordinates (u,v) at
        the specified time.

        Input:
            uv_pair     (u,v) coordinate Pair in the FOV.
            time        Scalar of optional absolute time in seconds.
            derivs      If True, any derivatives in (u,v) get propagated into
                        the returned (x,y) Pair.
            remask      True to mask (u,v) coordinates outside the field of
                        view; False to leave them unmasked.
            **keywords  Additional keywords arguments are passed directly to the
                        reference FOV.

        Return:         Pair of same shape as uv_pair, giving the transformed
                        (x,y) coordinates in the camera's frame.
        """

        # Update (u,v) based on the line and the number of TDI stages
        uv = Pair.as_pair(uv_pair, recursive=derivs).copy(recursive=False)
        line = uv.to_scalar(self._uv_line_index, recursive=False)
            # uv and line share memory, so updating line also updates uv.

        # Determine the number of TDI shifts
        time = Scalar.as_scalar(time, recursive=False)
        shifts = -1 - ((time - self.tstop) // self.tdi_texp).as_int()
        shifts[time == self.tstop] = 0

        # Apply the line shift to our copy of uv
        line -= self.tdi_sign * shifts

        # If a time derivative is present, we need to compensate for the TDI
        # readout
        if derivs:
            uv.insert_derivs(uv_pair.derivs.copy())  # copy dict but not derivs
            if 't' in uv.derivs:
                uv.derivs['t'] = uv.derivs['t'] - self._duv_dt

        return self.fov.xy_from_uvt(uv, derivs=derivs, remask=remask,
                                                       **keywords)

    #===========================================================================
    def uv_from_xyt(self, xy_pair, time=None, derivs=False, remask=False,
                                                            **keywords):
        """The (u,v) FOV coordinates given the (x,y) camera frame coordinates at
        the specified time.

        Input:
            xy_pair     (x,y) Pair in FOV coordinates.
            time        Scalar of optional absolute time in seconds.
            derivs      If True, any derivatives in (x,y) get propagated into
                        the returned (u,v) Pair.
            remask      True to mask (u,v) coordinates outside the field of
                        view; False to leave them unmasked.
            **keywords  Additional keywords arguments are passed directly to the
                        reference FOV.

        Return:         Pair of same shape as xy_pair, giving the computed (u,v)
                        FOV coordinates.
        """

        # Apply the conversion for tstop
        uv = self.fov.uv_from_xyt(xy_pair, derivs=derivs, remask=remask)

        # Extract the line index from uv, sharing memory
        line = uv.to_scalar(self._uv_line_index, recursive=False)

        # Determine the number of TDI shifts
        time = Scalar.as_scalar(time, recursive=False)
        shifts = -1 - ((time - self.tstop) // self.tdi_texp).as_int()
        shifts[time == self.tstop] = 0

        # Apply the line shift to uv
        line += self.tdi_sign * shifts

        # If a time derivative is present, we need to compensate for the TDI
        # readout
        if 't' in uv.derivs:
            uv.derivs['t'] += self._duv_dt

        return uv

################################################################################
# UNIT TESTS
################################################################################

import unittest
import numpy as np

class Test_TDIFOV(unittest.TestCase):

    def runTest(self):

        np.random.seed(9816)
        from oops.fov.flatfov import FlatFOV

        ################################################
        # 10 lines, TDI -v, 8 sec/shift, tstop=100
        ################################################

        staticfov = FlatFOV((1/2048.,-1/2048.), (100,10))
        fov = TDIFOV(staticfov, 100, 8., '-v')

        uv = Pair.combos(np.arange(0,101,50), np.arange(11))
        xy0 = staticfov.xy_from_uvt(uv)

        self.assertEqual(fov.xy_from_uvt(uv, time=100), xy0)
        self.assertEqual(fov.xy_from_uvt(uv, time=92), xy0)
        self.assertEqual(fov.xy_from_uvt(uv, time=84)[:,:-1], xy0[:,1:])
        self.assertEqual(fov.xy_from_uvt(uv, time=83)[:,:-2], xy0[:,2:])
        self.assertEqual(fov.xy_from_uvt(uv, time=101)[:,1:], xy0[:,:-1])

        self.assertEqual(fov.uv_from_xyt(xy0, time=100), uv)
        self.assertEqual(fov.uv_from_xyt(xy0, time=92), uv)
        self.assertEqual(fov.uv_from_xyt(xy0, time=84)[:,1:], uv[:,:-1])
        self.assertEqual(fov.uv_from_xyt(xy0, time=83)[:,2:], uv[:,:-2])
        self.assertEqual(fov.uv_from_xyt(xy0, time=101)[:,:-1], uv[:,1:])

        # with derivs
        N = 100
        uv = Pair.combos(50 + 20. * np.random.randn(N),
                          5 +  3. * np.random.randn(N))
        time = Scalar(90 + 20 * np.random.randn(N))
        uv.insert_deriv('rs', Pair(np.random.randn(N,2,2), drank=1))
        uv.insert_deriv('q' , Pair(np.random.randn(N,2)))
        uv.insert_deriv('t' , Pair(np.random.randn(N,2)))

        xy0 = staticfov.xy_from_uvt(uv, derivs=True)
        xy  = fov.xy_from_uvt(uv, time=time, derivs=True)
        self.assertEqual(xy0.d_drs, xy.d_drs)
        self.assertEqual(xy0.d_dq,  xy.d_dq)

        diffs = xy0.d_dt - xy.d_dt
        self.assertTrue(np.all(diffs.vals[...,0] == 0))
        self.assertTrue(np.all(abs(diffs.vals[...,1] - 1/2048./8.) < 1.e-14))

        ################################################
        # 10 lines, TDI +v, 8 sec/shift, tstop=100
        ################################################

        staticfov = FlatFOV((1/2048.,-1/2048.), (100,10))
        fov = TDIFOV(staticfov, 100, 8., '+v')

        uv = Pair.combos(np.arange(0,101,50), np.arange(11))
        xy0 = staticfov.xy_from_uvt(uv)

        self.assertEqual(fov.xy_from_uvt(uv, time=100), xy0)
        self.assertEqual(fov.xy_from_uvt(uv, time=92), xy0)
        self.assertEqual(fov.xy_from_uvt(uv, time=84)[:,1:], xy0[:,:-1])
        self.assertEqual(fov.xy_from_uvt(uv, time=83)[:,2:], xy0[:,:-2])
        self.assertEqual(fov.xy_from_uvt(uv, time=101)[:,:-1], xy0[:,1:])

        self.assertEqual(fov.uv_from_xyt(xy0, time=100), uv)
        self.assertEqual(fov.uv_from_xyt(xy0, time=92), uv)
        self.assertEqual(fov.uv_from_xyt(xy0, time=84)[:,:-1], uv[:,1:])
        self.assertEqual(fov.uv_from_xyt(xy0, time=83)[:,:-2], uv[:,2:])
        self.assertEqual(fov.uv_from_xyt(xy0, time=101)[:,1:], uv[:,:-1])

        # with derivs
        N = 100
        uv = Pair.combos(50 + 20. * np.random.randn(N),
                          5 +  3. * np.random.randn(N))
        time = Scalar(90 + 20 * np.random.randn(N))
        uv.insert_deriv('rs', Pair(np.random.randn(N,2,2), drank=1))
        uv.insert_deriv('q' , Pair(np.random.randn(N,2)))
        uv.insert_deriv('t' , Pair(np.random.randn(N,2)))

        xy0 = staticfov.xy_from_uvt(uv, derivs=True)
        xy  = fov.xy_from_uvt(uv, time=time, derivs=True)
        self.assertEqual(xy0.d_drs, xy.d_drs)
        self.assertEqual(xy0.d_dq,  xy.d_dq)

        diffs = xy0.d_dt - xy.d_dt
        self.assertTrue(np.all(diffs.vals[...,0] == 0))
        self.assertTrue(np.all(abs(diffs.vals[...,1] + 1/2048./8.) < 1.e-14))

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
