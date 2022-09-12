################################################################################
# oops/fov/TDIFOV.py: TDIFOV subclass of class FOV
################################################################################

import numpy as np
from polymath import Qube, Boolean, Scalar, Pair, Vector
from polymath import Vector3, Matrix3, Quaternion

from ..fov import FOV
from ..cadence.tdicadence import TDICadence

class TDIFOV(FOV):
    """FOV subclass to apply TDI timing to another FOV."""

    IS_TIME_INDEPENDENT = False         # updated below if necessary

    #===========================================================================
    def __init__(self, fov, tdi_axis, cadence):
        """Constructor for a TDIFOV.

        Input:
            fov         the time-independent FOV to which this TDI timing is
                        applied.

            tdi_axis    'u' or 'v', the FOV axis along which "Time Delay and
                        Integration" applies.

            cadence     the TDICadence for this FOV.
        """

        self.fov = fov
        self.tdi_axis = tdi_axis
        self.cadence = cadence

        # Copy key attributes from the cadence
        self.lines = self.cadence.lines
        self.tstart = self.cadence.tstart
        self.tdi_texp = self.cadence.tdi_texp
        self.tdi_stages = self.cadence.tdi_stages
        self.tdi_sign = self.cadence.tdi_sign

        # Validation
        assert self.tdi_axis in ('u', 'v')
        assert self.fov.IS_TIME_INDEPENDENT

        # Interpret the axis
        if self.tdi_axis == 'u':
            self._duv_dshift = Pair((self.tdi_sign, 0))
            self._uv_line_index = 0
        else:
            self._duv_dshift = Pair((0, self.tdi_sign))
            self._uv_line_index = 1

        self._duv = self._duv_dshift * (self.tdi_stages - 1)
        self._duv_dt = self._duv_dshift * self.tdi_texp

        assert self.lines == self.fov.uv_shape.vals[self._uv_line_index]

        self._max_line = self.lines - 1

        # Required attributes
        self.uv_los  = self.fov.uv_los
        self.uv_scale = self.fov.uv_scale
        self.uv_shape = self.fov.uv_shape
        self.uv_area  = self.fov.uv_area

        self.IS_TIME_INDEPENDENT = (self.cadence.tdi_stages <= 1)

    def __getstate__(self):
        return (self.fov, self.tdi_axis, self.cadence)

    def __setstate__(self):
        self.__init__(*state)

    #===========================================================================
    def xy_from_uvt(self, uv_pair, tfrac=0.5, time=None, derivs=False,
                          remask=False, **keywords):
        """The (x,y) camera frame coordinates given the FOV coordinates (u,v) at
        the specified time.

        Input:
            uv_pair     (u,v) coordinate Pair in the FOV.
            tfrac       Scalar of fractional times during the exposure, where
                        tfrac=0 at the beginning and 1 at the end. Default is
                        0.5.
            time        Scalar of optional absolute time in seconds. Only one of
                        tfrac and time can be specified; the other must be None.
            derivs      If True, any derivatives in (u,v) get propagated into
                        the returned (x,y) Pair.
            remask      True to mask times that are outside the exposure
                        interval.
            **keywords  Additional keywords arguments are passed directly to the
                        reference FOV.

        Return:         Pair of same shape as uv_pair, giving the transformed
                        (x,y) coordinates in the camera's frame.
        """

        # If stages == 1, this is easy
        if self.IS_TIME_INDEPENDENT:
            return self.fov.xy_from_uvt(uv_pair, derivs=derivs, **keywords)

        # Update (u,v) based on the line and the number of TDI stages
        uv = Pair.as_pair(uv_pair, recursive=derivs).copy()
        line = uv.to_scalar(self._uv_line_index, recursive=False)

        # Note that uv and line share memory, so updating line also updates uv.

        # Shift each (u,v) based on tfrac or time.
        if tfrac is not None:
            if time is not None:
                raise ValueError('One of tfrac and time must be None')

            tfrac = Scalar.as_scalar(tfrac, recursive=False)
            if remask:
                tfrac = tfrac.clip(0., 1., remask=True)

            shifts_at_time = tfrac * self._cadence.tdi_shifts_at_line(line)

        else:
            time = Scalar.as_scalar(time, recursive=False)
            if remask:
                time = time.clip(self.cadence.time[0], self.cadence.time[1],
                                 remask=True)

            shifts_at_time = (self.tstop - time) / self.tdi_texp

        line -= self.tdi_sign * shifts_at_time.int(top=self.tdi_stages)

        # If a time derivative is present, we need to compensate for the TDI
        # readout
        if derivs and 't' in uv.derivs:
            uv.derivs['t'] -= self._duv_dt

        return self.fov.xy_from_uvt(uv, derivs=derivs, **keywords)

    #===========================================================================
    def uv_from_xyt(self, xy_pair, tfrac=0.5, time=None, derivs=False,
                          remask=False, **keywords):
        """The (u,v) FOV coordinates given the (x,y) camera frame coordinates at
        the specified time.

        Input:
            xy_pair     (x,y) Pair in FOV coordinates.
            tfrac       Scalar of fractional times during the exposure, where
                        tfrac=0 at the beginning and 1 at the end. Default is
                        0.5.
            time        Scalar of optional absolute time in seconds. Only one of
                        tfrac and time can be specified; the other must be None.
            derivs      If True, any derivatives in (x,y) get propagated into
                        the returned (u,v) Pair.
            remask      True to mask out times that are outside the exposure
                        interval.
            **keywords  Additional keywords arguments are passed directly to the
                        reference FOV.

        Return:         Pair of same shape as xy_pair, giving the computed (u,v)
                        FOV coordinates.
        """

        # Apply the conversion for the end time
        xy_pair = Pair.as_pair(xy_pair, recursive=derivs)
        uv_at_t1 = self.fov.uv_from_xyt(xy_pair, derivs=derivs)

        # If stages == 1, we're done
        if self.IS_TIME_INDEPENDENT:
            return uv_at_t1

        # If a time, not a tfrac, has been provided, this is relatively easy
        if time is not None:
            time = Scalar.as_scalar(time, derivs=False)
            if remask:
                time = time.clip(self.cadence.time[0], self.cadence.time[1],
                                 remask=True)

            nshifts = (self.tstop - time.wod) / self.tdi_stages
            nshifts = nshifts.int(top=self.tdi_stages)
            uv = uv_at_t1 + nshifts * self._duv_dshift

            # If a time derivative is present, we need to compensate for the TDI
            # readout
            if derivs and 't' in uv.derivs:
                uv.derivs['t'] += self._duv_dt

            return uv

        # The tfrac case is trickier, and there are potential issues.
        #
        # Given (u,v,tfrac), we need to solve for the associated (u,v) at the
        # end time, where tfrac == 1. This is the inverse of the problem in
        # xy_from_uvt, where we can easily determine (u,v) at tfrac == 1, and
        # then use the TDI line number to shift the (u,v) coordinates.
        #
        # This inversion is not unique! Consider an image with 4 lines and 3 TDI
        # stages. This is the mapping:
        #
        # u = 0; tfrac in [2/3 --   1]; x = 0
        # u = 0; tfrac in [1/3 -- 2/3]; x = 1
        # u = 0; tfrac in [  0 -- 1/3]; x = 2
        # u = 1; tfrac in [2/3 --   1]; x = 1
        # u = 1; tfrac in [1/3 -- 2/3]; x = 2
        # u = 1; tfrac in [  0 -- 1/3]; x = 3
        # u = 2; tfrac in [1/2 --   1]; x = 2
        # u = 2; tfrac in [  0 -- 1/2]; x = 3
        # u = 3; tfrac in [  0 --   1]; x = 3
        #
        # The inverse of this function does not exist. For example, when tfrac
        # < 1/3, x = 3 should map to u = 1, 2, and 3. This ambiguity does not
        # arise when time is expressed in absolute units rather than as tfrac.
        #
        # We choose to have x(tfrac) always return the largest of its possible
        # values of u, because that is the only way for this function ever to
        # return a value of 3 in this example.
        #
        # x = 0; tfrac in [2/3 --   1]; u =  0
        # x = 0; tfrac in [1/3 -- 2/3]; u = -1
        # x = 0; tfrac in [  0 -- 1/3]; u = -2
        #
        # x = 1; tfrac in [2/3 --   1]; u =  1
        # x = 1; tfrac in [1/3 -- 2/3]; u =  0
        # x = 1; tfrac in [  0 -- 1/3]; u = -1
        #
        # x = 2; tfrac in [1/2 --   1]; u = 2
        # x = 2; tfrac in [1/3 -- 1/2]; u = 1
        # x = 2; tfrac in [  0 -- 1/3]; u = 0
        #
        # x = 3; tfrac in [  0 --   1]; u = 3
        #
        # More generally, for x <= tfrac * (lines - stages)
        #
        #       u = u(x) - floor((1 - tfrac) * stages)
        #
        # For other x, we need to iterate through the possible values of u and
        # find the one where result is self-consistent. We start with the
        # value of u with the _least_ number of possible TDI stages.
        #
        #   for utest in range(u(x), u(x) - stages, -1):
        #       utest_stages = 1 + cadence.tdi_shifts_at_line(u(x))
        #       u = u(x) - floor((1. - tfrac) * utest_stages)
        #       if u == utest:
        #           return u
        #
        # A self-consistent solution is one where the TDI-shifted value of u
        # has the same number of TDI integration steps as was assumed to
        # determine the number of steps.

        tfrac = Scalar.as_scalar(tfrac, recursive=False)
        tfrac = tfrac.clip(0., 1., remask=remask)
        one_minus_tfrac = 1. - tfrac
        line_at_t1 = uv_at_t1.to_scalar(self._uv_line_index, recursive=False)

        for k in range(self.tdi_stages):
            nshift = -self.tdi_sign * k
            ltest = (line_at_t1 - nshift).clip(0, self._max_line, remask=False)
            stages = self.cadence.tdi_shifts_at_line(ltest) + 1
            nshifts = (one_minus_tfrac * stages).int(top=stages)
            test_ = line + self.tdi_sign * nshifts

        #### IN PROGRESS ####

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_TDIFOV(unittest.TestCase):

    def runTest(self):

        #### TBD
        print('TDIFOV unit tests are needed!')

        # not tested at all. Make sure the two functions work
        # and are accurately inverses of one another.

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
