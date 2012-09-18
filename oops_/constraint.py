################################################################################
# oops_/fitting/constraint.py: Constraint class
#
# 5/18/12 MRS - Created.
################################################################################

import numpy as np

class Constraint(object):
    """The Constraint class provides a mechanism for replacing one set of
    fittable parameters with another. It supports parameter that are held fixed,
    confined within specified boundaries, or coupled to other parameters.
    """

    def __init__(self, fittable, constraints):
        """Constructor for a Constraint object.

        Input:
            obj             the fittable object for which these constraints are
                            to be applied.
            constraint      a list or tuple of the same length as the parameter
                            set that the fittable object requires. Each item in
                            the list defines the constraint being applied.
                None        the parameter remains unconstrained.
                value       the parameter will be held fixed at the given
                            floating-point value.
                (min,max)   the parameter is constrained to fall between the
                            given min and max values (inclusive).
                (min,None)  the parameter is constrained to be greater than or
                            equal to the given minimum.
                (None,max)  the parameter is constrained to be less than or
                            equal to the given maximum.
                (func(),n)  the parameter's value will equal the result of
                            evaluating the given function on the nth parameter
                            of the set passed to self.fittable. The function
                            will be called via:
                                func(fittable, params[n], derivs)
                            If derivs is True, then the function must return a
                            tuple containing the value and also the local
                            derivative.
        """

        self.fittable = fittable

        self.types = []
        self.info  = []
        self.old_index = []
        self.new_index = []
        self.fixed_params = []
    
        o = -1
        n = -1

        for c in constraints:
            o += 1
            if c is None:
                n += 1
                self.types.append("FLOAT")
                self.info.append(None)
                self.old_index.append(o)    # returns old index given new
                self.new_index.append(n)    # returns new index given old
                self.fixed_params.append(0.)
            elif type(c) is not type(()) and type(c) is not type([]):
                self.types.append("FIXED")
                self.info.append(c)
                self.new_index.append(None)
                self.fixed_params.append(c)
            elif c[0] is None:
                n += 1
                self.types.append("MAX")
                self.info.append(c[1])
                self.old_index.append(o)
                self.new_index.append(n)
                self.fixed_params.append(0.)
            elif c[1] is None:
                n += 1
                self.types.append("MIN")
                self.info.append(c[0])
                self.old_index.append(o)
                self.new_index.append(n)
                self.fixed_params.append(0.)
            elif "func" in type(c[0]).__name__:
                self.types.append("FUNC")
                self.info.append(c)
                self.new_index.append(None)
                self.fixed_params.append(0.)
            else:
                n += 1
                self.types.append("MINMAX")
                self.info.append(c)
                self.old_index.append(o)
                self.new_index.append(n)
                self.fixed_params.append(0.)

        self.nparams = len(self.old_index)
        self.fixed_params = np.asfarray(self.fixed_params)

    ########################################

    def new_params(self, oldpar, partials=False):
        """Returns a 1-D array containing the new parameters used by this
        Constraint, given the old parameters used by the fittable.
        """

        # Causes an error if we are still constructing the fittable
        # assert np.shape(oldpar) == (self.fittable.nparams,)

        newpar = np.zeros((self.nparams,))

        if partials:
            dnew_dold = np.zeros((self.nparams, self.fittable.nparams))

        for i in range(self.nparams):
            j = self.old_index[i]
            oldval = oldpar[j]
            info = self.info[j]

            if self.types[j] == "FLOAT":
                newval = oldval
                deriv = 1.
            elif self.types[j] == "MINMAX":
                # oldval = info[0] + info[1] * np.sin(newval)
                arg = (oldval - info[0]) / info[1]
                newval = np.arcsin(arg)
                if partials: deriv = 1. / np.sqrt(1. - arg**2) / info[1]
            elif self.types[j] == "MIN":
                # oldval = info + newval**2
                newval = np.sqrt(oldval - info)
                if partials: deriv = 0.5 / newval
            elif self.types[j] == "MAX":
                # oldval = info - newval**2
                newval = np.sqrt(info - oldval)
                if partials: deriv = -0.5 / newval

            newpar[i] = newval
            if partials: dnew_dold[i,j] = deriv

        # Return the results
        if partials:
            return (newpar, dnew_dold)
        else:
            return newpar

    ########################################

    def old_params(self, newpar, partials=False):
        """Returns a 1-D array containing the old parameters used by
        self.fittable.set_params(), given the new parameters used by the
        ConstrainedFittable object. Optionally, also returns the matrix of
        partial derivatives."""

        assert np.shape(newpar) == (self.nparams,)

        # Copy the fixed parameters
        oldpar = self.fixed_params.copy()

        if partials:
            dold_dnew = np.zeros((self.fittable.nparams, self.nparams))

        # Translate the new parameters into the corresponding old parameters
        for i in range(self.nparams):
            newval = newpar[i]
            j = self.old_index[i]
            info = self.info[j]

            if self.types[j] == "FLOAT":
                oldval = newval
                deriv = 1.
            elif self.types[j] == "MINMAX":
                oldval = info[0] + info[1] * np.sin(newval)
                deriv = info[1] * np.cos(newval)
            elif self.types[j] == "MIN":
                oldval = info + newval**2
                deriv = 2. * newval
            elif self.types[j] == "MAX":
                oldval = info - newval**2
                deriv = -2. * newval

            oldpar[j] = oldval
            if partials: dold_dnew[j,i] = deriv

        # Fill in any parameters derived from functions
        for j in range(self.fittable.nparams):
            info = self.info[j]

            if self.types[j] == "FUNC":
                (func, k) = info
                result = func(self.fittable, oldpar[k], derivs=partials)
                if partials:
                    (oldpar[j], dold_j_dold_k) = result
                    i = self.new_index[k]
                    dold_dnew[j,i] = dold_j_dold_k * dold_dnew[k,i]
                else:
                    oldpar[j] = result

        # Return the results
        if partials:
            return (oldpar, dold_dnew)
        else:
            return oldpar

    ########################################

    def partials_wrt_new(self, new_params, partials_wrt_old):
        """Returns an arbitrary array of derivatives with respect to the new
        parameters, given an array of derivatives with respect to the old
        parameters.
        """

        assert partials_wrt_old.shape[-1] == self.fittable.nparams

        dold_dnew = self.old_params(new_params, partials=True)[1]

        return np.sum(partials_wrt_old[..., np.newaxis] * dold_dnew, axis=-2)

    ########################################

    def partials_wrt_old(self, new_params, partials_wrt_new):
        """Returns an arbitrary array of derivatives with respect to the old
        parameters, given an array of derivatives with respect to the new
        parameters.

        Note that this cannot recover the correct derivatives for quantities
        coupled via a functional relationship.
        """

        assert partials_wrt_new.shape[-1] == self.nparams

        old_params = self.old_params(new_params)
        dnew_dold = self.new_params(old_params, partials=True)[1]

        return np.sum(partials_wrt_new[..., np.newaxis] * dnew_dold, axis=-2)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Dummy_Fittable(object):
    def __init__(self, nparams):
        self.nparams = nparams

def Dummy_Func(obj, x, derivs=False):
    if derivs:
        return (x*x*x, 3*x*x)
    else:
        return x*x*x

class Test_Constraint(unittest.TestCase):

    def runTest(self):
        fittable = Dummy_Fittable(2)

        # (None, None)

        fittable = Dummy_Fittable(2)
        test = Constraint(fittable, (None, None))
        self.assertEqual(test.nparams, 2)

        params = (1., 2.)
        self.assertEqual(test.old_params(params)[0], params[0])
        self.assertEqual(test.old_params(params)[1], params[1])
        self.assertEqual(test.new_params(params)[0], params[0])
        self.assertEqual(test.new_params(params)[1], params[1])

        partials_wrt_old = np.arange(10).reshape(5,2)
        partials_wrt_new = test.partials_wrt_new(params, partials_wrt_old)
        self.assertTrue(np.all(partials_wrt_old == partials_wrt_new))

        # (None, 7.)

        fittable = Dummy_Fittable(2)
        test = Constraint(fittable, (None, 7.))
        self.assertEqual(test.nparams, 1)

        old_params = (1.,7.)
        self.assertEqual(test.new_params(old_params)[0], 1.)

        new_params = (2.,)
        self.assertEqual(test.old_params(new_params)[0], 2.)
        self.assertEqual(test.old_params(new_params)[1], 7.)

        partials_wrt_old = np.arange(10).reshape(5,2)
        partials_wrt_new = test.partials_wrt_new(new_params, partials_wrt_old)

        self.assertTrue(np.all(partials_wrt_new[...,0] ==
                               partials_wrt_old[...,0]))

        # ((1.,3.), (2.,None), (None,4.))

        fittable = Dummy_Fittable(3)
        test = Constraint(fittable, ((1.,3.), (2.,None), (None,4.)))
        self.assertEqual(test.nparams, 3)

        old_params = (2., 3., 4.)
        new_params = test.new_params(old_params)
        test_params = test.old_params(new_params)
        self.assertEqual(test_params[0], old_params[0])
        self.assertEqual(test_params[1], old_params[1])
        self.assertEqual(test_params[2], old_params[2])

        new_params = (1., 3., 4.)
        old_params = test.old_params(new_params)
        test_params = test.new_params(old_params)
        self.assertTrue(abs(test_params[0] - new_params[0]) < 1.e-14)
        self.assertTrue(abs(test_params[1] - new_params[1]) < 1.e-14)
        self.assertTrue(abs(test_params[2] - new_params[2]) < 1.e-14)

        partials_wrt_old = np.arange(12).reshape(4,3)
        partials_wrt_new = test.partials_wrt_new(new_params, partials_wrt_old)
        test_old = test.partials_wrt_old(new_params, partials_wrt_new)
        self.assertTrue(np.all(np.abs(test_old - partials_wrt_old) < 1.e-14))

        partials_wrt_new = np.arange(12).reshape(4,3)
        partials_wrt_old = test.partials_wrt_old(new_params, partials_wrt_new)
        test_new = test.partials_wrt_new(new_params, partials_wrt_old)
        self.assertTrue(np.all(np.abs(test_new - partials_wrt_new) < 1.e-14))

        # (None, (Dummy_Func,0))

        fittable = Dummy_Fittable(2)
        test = Constraint(fittable, (None, (Dummy_Func, 0)))
        self.assertEqual(test.nparams, 1)

        old_params = (2., 8.)
        new_params = test.new_params(old_params)
        test_params = test.old_params(new_params)
        self.assertEqual(test_params[0], old_params[0])

        new_params = (2.,)
        old_params = test.old_params(new_params)
        self.assertEqual(old_params[0], new_params[0])
        self.assertEqual(old_params[1], new_params[0]**3)

        partials_wrt_old = np.arange(12).reshape(6,2)
        partials_wrt_new = test.partials_wrt_new(new_params, partials_wrt_old)
        test_old = test.partials_wrt_old(new_params, partials_wrt_new)
        # This cannot work!
        # self.assertTrue(np.all(np.abs(test_old[...,0] -
        #                               partials_wrt_old[...,0]) < 1.e-14))
        self.assertTrue(np.all(test_old[...,1] == 0.))
        self.assertTrue(np.all(partials_wrt_new[...,0] ==
                partials_wrt_old[...,0] +                       # partial = 1.
                partials_wrt_old[...,1] * 3*new_params[0]**2))  # partial = 12.

        partials_wrt_new = np.arange(4).reshape(4,1)
        partials_wrt_old = test.partials_wrt_old(new_params, partials_wrt_new)
        test_new = test.partials_wrt_new(new_params, partials_wrt_old)
        self.assertTrue(np.all(np.abs(test_new - partials_wrt_new) < 1.e-14))

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
