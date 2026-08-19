##########################################################################################
# oops/fittable.py: Fittable interface
##########################################################################################
"""Support for Fittable (mutable) OOPS objects."""

from collections.abc import Iterable
from typing import Any


class Fittable(object):
    """The Fittable interface enables any class to be used within a fitting procedure.

    Most OOPS objects are static, but objects that subclass Fittable can be modified
    in-place via a call to the method `set_params`. This is primarily used when fitting
    unknown values such as pointing corrections, time shifts, plate scales, or orbital
    elements.

    The following methods are defined for all Fittable objects:

    * `set_params`: Updates the parameter values for this object and refreshes it.
    * `refresh`: Makes sure this object is internally consistent.
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
    where `params` is a scalar, tuple, list, or array of one or more floating-point values
    that are used to update the object. The function `set_params` of the public API uses
    this function

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
    def is_frozen(self) -> bool:
        """True if this object and all Fittable sub-objects have been frozen."""

        try:
            return self._FITTABLE_is_frozen
        except AttributeError:
            self._FITTABLE_is_frozen = False
            return False

    @property
    def version(self) -> int:
        """The version number of this object, incremented each time it is modified."""

        return Fittable._MUTABLE.version(self)

    def set_params(self, params: Any) -> None:
        """Redefine this object using the given set of parameters.

        This function also refreshes the object.

        Parameters:
            params: Parameter value or values to be applied to this object.

        Returns:
            True if this object has changed as a result of this function call.

        Raises:
            ValueError: If the number of parameters is incorrect or the object is frozen.
        """

        # Convert params to tuple if necessary
        if isinstance(params, Iterable):
            params = tuple(float(x) for x in params)
        else:
            params = (float(params),)

        # Check for valid parameters
        if not params:
            raise ValueError(f'missing parameters for {type(self).__name__}.set_params()')

        if params == self.params:
            return False
        if len(params) != self.nparams:
            plural = 's' if self.nparams > 1 else ''
            raise ValueError(f'{type(self).__name__} object requires {self.nparams} fit '
                             f'parameter{plural}')

        # Check for frozen object
        if self.is_frozen:
            raise ValueError(f'{type(self).__name__} object is frozen')

        # Update, increment, refresh
        self._set_params(params)
        Fittable._MUTABLE._increment(self)
        Fittable._MUTABLE.refresh(self)
        return True

    def refresh(self) -> bool:
        """Update any internally cached information if this object has been modified.

        Use this function to ensure that an object is fully self-consistent, not
        containing any stale information.

        If the given object and all sub-object(s) are already up to date, the object is
        not changed and the function returns False.

        Returns:
            True if this object was modified as a result of this call.
        """

        return Fittable._MUTABLE.refresh(self)

    def copy(self) -> 'Fittable':
        """An independent, unfrozen copy of this object.

        The copy begins with the same parameter values as this object, but the two can be
        fitted separately; changing one has no effect on the other. Any object to which
        this one applies, such as the reference frame of a Navigation or the FOV of a
        Platescale, is shared with the copy rather than duplicated. The copy is not
        registered under an ID.

        This implementation reconstructs the object from the constructor arguments
        returned by `__getstate__`, dropping the trailing ID of a registered Frame or
        Path. A subclass whose constructor cannot be called with those arguments
        positionally must override this method.

        Returns:
            (Fittable): A new, unfrozen object of the same class.
        """

        state = self.__getstate__()     # this also refreshes the object

        # Frame and Path subclasses return their registered ID as the last item; the copy
        # is left unregistered, so it is dropped
        if hasattr(self, 'stripped_id'):
            state = state[:-1]

        return type(self)(*state)

    def freeze(self) -> bool:
        """Freeze this object and any Fittable sub-objects.

        A frozen object can no longer be modified.

        Returns:
            True if this object was frozen as a result of this call; False if it was
            already frozen or immutable.
        """

        return Fittable._MUTABLE.freeze(self)

    def _mark_as_frozen(self) -> None:
        """Mark this object as frozen."""

        self._FITTABLE_is_frozen = True

##########################################################################################
