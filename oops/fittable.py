##########################################################################################
# oops/fittable.py: Fittable interface
##########################################################################################
"""Support for Fittable (mutable) OOPS objects."""


class Fittable(object):
    """The Fittable interface enables any class to be used within a fitting procedure.

    Most OOPS objects are static, but objects that subclass Fittable can be modified
    in-place via a call to the method `set_params`. This is primarily used when fitting
    unknown values such as pointing corrections, time shifts, plate scales, or orbital
    elements.

    The following methods are defined for all Fittable objects:

    * `set_params`: Updates the parameter values for this object and, optionally any of
      its sub-objects.
    * `get_params`: Retrieves the current parameter values of this object and, optionally
      those of any of its sub-objects.
    * `refresh`: Makes sure this object is internally consistent. Always call this method
      after an object or any of the sub-objects might have been modified.
    * `freeze`: Freeze this object, preventing any further changes to it or any of its
      sub-objects.

    The following properties are always defined:

    * `params`: The current tuple of parameter values.
    * `nparams`: The number of required parameters.
    * `is_frozen`: True if this object (including all of its sub-objects) has been frozen.
    * `version`: An integer that starts at zero and increases whenever this object or one
      of its sub-objects changes.

    Programming Notes
    -----------------

    Note that many OOPS objects are not themselves Fittable, but may have a sub-object (or
    sub-sub-object, etc.) that is Fittable. Such objects are described as "mutable" and
    are addressed using the `mutable` API.

    Information about the Fittable state of all objects is maintained by a set of added
    attributes, which are all prefixed "_FITTABLE" or "_MUTABLE". These attributes are
    managed internally and should not be touched by the programmer.

    The programmer must define these as attributes or properties of a Fittable object:

    * params: The current tuple of parameter values.
    * nparams: The number of required parameters.

    This method must also be defined::
        _set_params(self, params)
    where `params` is a tuple, list, or array of one or more floating-point values that
    are used to update the object.

    If the Fittable object maintains cached information internally, it must also have this
    method::
        _refresh(self)
    which updates any internal attributes based on the currently defined parameters (along
    with the current values of any internal sub-objects that might also be Fittable or
    mutable). In addition, the method::
        _freeze(self)
    can be used to carry out any special actions that must take place if the object is
    frozen.
    """

    @property
    def is_frozen(self):
        """True if this object and all Fittable sub-objects have been frozen."""

        try:
            return self._FITTABLE_is_frozen
        except AttributeError:
            self._FITTABLE_is_frozen = False
            return False

    @property
    def version(self):
        """The version number of this object, incremented each time it is modified."""

        return Fittable._MUTABLE.version(self)

    def get_params(self, *, frozen=False, recursive=False):
        """The parameters defining the current state of this object.

        Parameters:
            obj (object): Object for which parameters are to be retrieved.
            frozen (bool, optional): True to include parameters associated with frozen
                objects.
            recursive (bool, optional): True to include the parameters of any mutable
                sub-objects recursively, in addition to this object's parameters.

        Returns:
            (tuple or dict): If recursive is False or if this object has no mutable
                sub-objects, this function returns the tuple of this object's current
                parameters. If recursive is True and this object has one or more mutable
                sub-objects, this function instead returns a dictionary keyed by the name
                of each mutable sub-object and containing the parameters of that
                sub-object (recursively). In this case, this object's parameters are
                returned keyed by an empty string "".
        """

        return Fittable._MUTABLE.get_params(self, frozen=frozen, recursive=True)

    def set_params(self, params):
        """Redefine this object using the given set of parameters.

        This function also refreshes the object.

        Parameters:
            params (tuple, list, np.ndarray, or dict): Parameter values to use. If a
                tuple, list or array is given, the values are applied this object. If a
                dictionary is given, each key in the dictionary must be the name of an
                attribute and the values are applied to that sub-object; in this case,
                parameters to be applied this object can be provided as a tuple, list, or
                array value keyed by an empty string "".

        Returns:
            (bool): True if this object has changed as a result of this function call.
        """

        return Fittable._MUTABLE.set_params(self, params)

    def refresh(self):
        """Update any internally cached information if this object has been modified.

        Use this function to ensure that an object is fully self-consistent, not
        containing any stale information.

        If the given object and all sub-object(s) are already up to date, the object is
        not changed and the function returns False.

        Parameters:
            obj (object): Object to be refreshed if necessary.

        Returns:
            (bool): True if this object was modified as a result of this call.
        """

        return Fittable._MUTABLE.refresh(self)

    def freeze(self):
        """Freeze this object and any Fittable sub-objects.

        A frozen object can no longer be modified.

        Returns:
            (bool): True if this object was frozen as a result of this call; False if it
                was already frozen or immutable.
        """

        return Fittable._MUTABLE.freeze(self)

    def _mark_as_frozen(self):
        """Mark this object as frozen."""

        self._FITTABLE_is_frozen = True

##########################################################################################
