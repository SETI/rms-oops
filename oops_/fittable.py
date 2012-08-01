################################################################################
# oops_/fitting/fittable.py: Fittable interface
#
# 6/11/12 MRS - Created.
################################################################################

class Fittable(object):
    """The Fittable interface enables an oops class to be defined within a
    least-squares fitting procedure.

    Every Fittable object must have an attribute nparams, defining the number of
    parameters that it requires.
    """

    def set_params(self, params):
        """Redefines the Fittable object, using this set of parameters.

        Input:
            params      a list, tuple or 1-D Numpy array of floating-point
                        numbers, defining the parameters to be used in the
                        object returned.
        """

        pass

    def get_params(self):
        """Returns the current set of parameters defining this fittable object.

        Return:         a Numpy 1-D array of floating-point numbers containing
                        the parameter values defining this object.
        """

        pass

    def copy(self):
        """Returns a deep copy of the given object. The copy can be safely
        modified without affecting the original."""

        pass

################################################################################
