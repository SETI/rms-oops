################################################################################
# oops/fittable.py: Fittable interface
################################################################################

#*******************************************************************************
# Fittable
#*******************************************************************************
class Fittable(object):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    The Fittable interface enables any class to be used within a
    least-squares fitting procedure.

    Every Fittable object has these attributes:
        nparams         the number of parameters required.
        param_name      the name of the attribute holding the parameters.
        cache           a dictionary containing prior values of the object,
                        keyed by the parameter set as a tuple.

    It is also necessary to define these methods:
        set_params() or set_params_new()
        copy()
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #===========================================================================
    # set_params
    #===========================================================================
    def set_params(self, params):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
	Redefine the object using this set of parameters.

        This implementation checks the cache first, and then calls
        set_params_new() if the instance is not cached. Override this method
        if the Fittable object does not need a cache.

        Input:
            params      a list, tuple or 1-D Numpy array of floating-point
                        numbers, defining the parameters to be used in the
                        object returned.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        key = tuple(params)
        if key in self.cache:
            return self.cache[key]

        result = self.set_params_new(params)
        self.cache[key] = result

        return result
    #===========================================================================



    #===========================================================================
    # set_params_new
    #===========================================================================
    def set_params_new(self, params):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
	Redefine using this set of parameters. Do not check the cache first.

        Override this method if the subclass uses a cache. Then calls to
        set_params() will check the cache and only invoke this function when
        needed.

        Input:
            params      a list, tuple or 1-D Numpy array of floating-point
                        numbers, defining the parameters to be used in the
                        object returned.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pass
    #===========================================================================



    #===========================================================================
    # get_params
    #===========================================================================
    def get_params(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
	Return the current set of parameters defining this fittable object.

        This method normally does not need to be overridden.

        Return:         a Numpy 1-D array of floating-point numbers containing
                        the parameter values defining this object.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return self.__dict__[self.param_name]
    #===========================================================================



    #===========================================================================
    # copy
    #===========================================================================
    def copy(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
	Return a deep copy of this object.

        The copy can be safely modified without affecting the original.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pass
    #===========================================================================



    #===========================================================================
    # clear_cache
    #===========================================================================
    def clear_cache(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
	Clear the current cache.
	"""
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.cache = {}
    #===========================================================================



#*******************************************************************************



################################################################################
