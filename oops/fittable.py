################################################################################
# oops/fittable.py: Fittable interface
#
# 6/11/12 MRS - Created.
# 9/29/12 MRS - Implemented caching.
################################################################################

class Fittable(object):
    """The Fittable interface enables an oops class to be used within a
    least-squares fitting procedure.

    Every Fittable object has these attributes:
        nparams         the number of parameters required.
        param_name      the name of the attribute holding the parameters.
        cache           a dictionary containing prior values of the object,
                        keyed by the parameter set as a tuple.
    """

    def set_params(self, params):
        """Redefines the Fittable object, using this set of parameters. It does
        not refer to the cache in advance. Before calling the class's
        method set_params_new(), it checks an internal cache and returns a
        cached version of the object if it exists. Override this method if the
        subclass should not use a cache.

        Input:
            params      a list, tuple or 1-D Numpy array of floating-point
                        numbers, defining the parameters to be used in the
                        object returned.
        """

        key = tuple(params)
        if self.cache.has_key(key):
            return self.cache[key]

        result = self.set_params_new(params)
        self.cache[key] = result

        return result

    def set_params_new(self, params):
        """Redefines the Fittable object, using this set of parameters. Unlike
        method set_params(), this method does not check the cache first.
        Override this method if the subclass should use a cache.

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

        return self.__dict__[self.param_name]

    def copy(self):
        """Returns a deep copy of the given object. The copy can be safely
        modified without affecting the original."""

        pass

    def clear_cache(self):
        """Clears the current cache."""

        self.cache = {}

################################################################################
