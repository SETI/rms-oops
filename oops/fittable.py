##########################################################################################
# oops/fittable.py: Fittable interface
##########################################################################################
"""Support for Fittable (mutable) OOPS objects."""

import numpy as np


class Fittable(object):
    """The Fittable interface enables any class to be used within a fitting procedure.

    Most OOPS objects are static, but objects that subclass Fittable can be modified
    in-place. This is primarily used when fitting unknown values such as pointing
    corrections, time shifts, or plate scales, or orbital elements.

    The following methods are defined:

    * `set_params`: Update the parameter values for this object and/or any of its
      sub-objects.
    * `get_params`: Retrieve the current parameter values.
    * `refresh`: Make sure this object is internally consistent. Always call this object
      before making use of an object if any of the sub-objects might have changed
      underneath it.
    * `freeze`: Freeze the parameter values, preventing any further changes to this object
      or any of its sub-objects.
    * `is_frozen`: True if this object is frozen.
    * `version`: An integer that starts at zero and increases whenever this or one of its
      sub-objects changes.

    When an object class subclasses Fittable, the method::
        _set_params(self, params)
    must be defined. It receives one or more floating-point parameters and uses their
    values to update the object. (Note that the programmer should call the method
    `set_params`, not `_set_params`, because the former handles a variety of additional
    "bookkeeping" tasks.)

    A Fittable object must have a property or attribute `_params`, which returns the
    tuple of the object's current parameters.

    If an object class maintains cached information internally, it should also have a
    method `_refresh`, which updates any internal attributes based on the new parameters.
    It may also need `_freeze`, which is called when the object is frozen.

    Note that it is possible for a Fittable object to have sub-objects that are also
    Fittable.

    Information about the fittable state of such an object is maintained by a set of added
    attributes. These attributes are maintained entirely by the Fittable interface and
    should not be touched by the programmer.

    * `_fittables`: An ordered list of the names of all the fittable sub-objects.
    * `_fittable_nparams`: The number of fittable parameters required by the object and
      its sub-objects.
    * `_fittable_params`: The current values of the fittable parameters for the object or
      its sub-objects.
    * `_fittable_version`: The version number of the parameters. This values begins at
      zero and is incremented each time the object or any of its sub-objects is modified.
    * `_fittable_is_frozen`: True if this object is now frozen. Once frozen, it can no
      longer be modified.
    * `_fittable_state`: A dictionary, keyed by attribute name, providing the version
      number of each fittable sub-object.
    """

    def set_params(self, params):
        """Redefine this object using a new set of parameters.

        This calls a defined `_set_params` method for any sub-objects and then for the
        object itself. It also refreshes as it goes.

        Parameters:
            obj (object): Object for which parameters are to be set.
            params (tuple, list, or np.ndarray): Parameter values to use.
        """

        Fittable_.set_params(self, params)

    def get_params(self, *, frozen=False, as_dict=False):
        """The parameters defining the current state of this object.

        Parameters:
            obj (object): Object for which parameters are to be retrieved.
            frozen (bool, optional): True to include parameters associated with frozen
                objects as well.
        """

        return Fittable_.get_params(self, frozen=frozen, as_dict=as_dict)

    def refresh(self):
        """Update any internally cached information if this object has been modified.

        Use this call to ensure that an object is fully self-consistent, not containing
        any stale information.

        If the given object and any Fittable sub-object(s) are already up to date, the
        object is not changed.

        Parameters:
            obj (object): Object to be refreshed if necessary.

        Returns:
            (bool): True if this object was modified as a result of this call.
        """

        return Fittable_.refresh(self)

    def freeze(self):
        """Freeze this object and any Fittable subobjects.

        A frozen object can no longer be modified.
        """

        Fittable_.freeze(self)

    def is_frozen(self):
        """True if the given object and all Fittable sub-objects are frozen."""

        return Fittable_.is_frozen(self)

    def version(self):
        """The Fittable version number of this object.

        The version number starts at zero and is incremented each time the object or one
        of its sub-objects is modified by a call to `set_params` or possibly `refresh`.
        """

        return Fittable_.version(self)

##########################################################################################
# Class Fittable_, holding static versions of all needed functions
##########################################################################################

class Fittable_:
    """Static functions that support Fittable operations on objects that are not
    necessarily subclasses of Fittable.

    Most OOPS objects are static. An object that subclasses Fittable can be modified
    in-place by making a call to a function named `_set_params`.

    If an object contains one or more Fittable sub-objects, it is effectively fittable
    even if it does not subclass the Fittable class. The following static methods can be
    applied to any object:

    * `set_params`: Update the parameter values for the given object and/or any of its
      sub-objects.
    * `get_params`: Retrieve the current parameter values.
    * `refresh`: Make sure the given object is internally consistent. Always call this
      function before making use of an object if it or any of the sub-objects might have
      changed via a recent call to `set_params`.
    * `freeze`: Freeze the parameter values, preventing any further changes to the given
      object or its sub-objects.
    * `is_frozen`: True if the given object is frozen.
    * `fittables`: A list of the names of any fittable sub-objects.
    * `is_fittable`: True if the given object is fittable, whether or not it is frozen.
    * `version`: An integer that starts at zero and increases whenever an object or one of
      its sub-objects changes.

    The programmer may wish to define these items for a class, even if the object class is
    not a subclass of Fittable:

    * `_refresh`: This optional method should update any internal attributes that depend
      on an updated sub-object. It ensures that this information cannot become "stale" if
      an underlying sub-object changes.
    * `_freeze`: This optional method should update any internal attributes that depend
      on the object being frozen.

    Information about the fittable state of such an object is maintained by a set of added
    attributes. These attributes are maintained entirely by the module and should not be
    touched by the programmer.

    * `_fittables`: An ordered list of the names of all the fittable sub-objects.
    * `_fittable_nparams`: The number of fittable parameters required by the object and
      its sub-objects.
    * `_fittable_params`: The current values of the fittable parameters for the object or
      its sub-objects.
    * `_fittable_version`: The version number of the parameters. This values begins at
      zero and is incremented each time the object or any of its sub-objects is modified.
    * `_fittable_is_frozen`: True if this object is now frozen. Once frozen, it can no
      longer be modified.
    * `_fittable_state`: A dictionary, keyed by attribute name, providing the version
      number of each fittable sub-object.
    """

    _FROZEN_IDS = set()     # for objects with __dict__ that can't have attributes set

    @staticmethod
    def set_params(obj, /, params):
        """Redefine the given object using a new set of parameters.

        This calls a defined `_set_params` method for any sub-objects and then for the
        object itself. It also refreshes as it goes.

        Parameters:
            obj (object): Object for which parameters are to be set.
            params (tuple, list, np.ndarray, or dict): Parameter values to use. Use a dict
                to apply parameters to subobjects, where the key is the name of the
                fittable attribute; in this case, use a blank key "" to set the parameters
                of the given object.
        """

        def clean_params(params):
            if isinstance(params, (dict, tuple)):
                return params
            if isinstance(params, (list, np.ndarray)):
                return tuple(params)
            return (params,)

        if not Fittable_.is_fittable(obj):
            raise ValueError(f'{type(obj).__name__} object is not Fittable')

        if Fittable_.is_frozen(obj):
            if clean_params(params) == obj.get_params():    # no error if unchanged
                return
            raise ValueError(f'{type(obj).__name__} object is frozen')

        # Make sure state is initialized
        _ = Fittable_._get_state(obj)

        # Handle a dictionary
        if isinstance(params, dict):
            for key, sub_params in params.items():
                if not key:
                    continue
                Fittable_.set_params(obj.__dict__[key], sub_params)

            if '' in params:
                params = params['']
            else:
                if hasattr(obj, '_refresh'):
                    obj._refresh()
                Fittable_._increment(obj)
                obj._fittable_state[''] = obj._fittable_version
                Fittable_.refresh(obj)
                return

        # Handle a tuple
        if not isinstance(obj, Fittable):
            raise ValueError(f'{type(obj).__name__} object is not Fittable')

        params = clean_params(params)
        if not params:
            raise ValueError(f'missing parameters for {type(obj).__name__}.set_params()')

        # Check or set the number of parameters
        if hasattr(obj, '_fittable_nparams'):
            if len(params) != obj._fittable_nparams:
                plural = 's' if obj._fittable_nparams > 1 else ''
                raise ValueError(f'{type(obj).__name__} object requires '
                                 f'{obj._fittable_nparams} fit parameter{plural}')
        else:
            obj._fittable_nparams = len(params)

        # Update the object with the new parameters and refresh
        obj._set_params(params)
        obj._fittable_params = params

        if hasattr(obj, '_refresh'):
            obj._refresh()
        Fittable_._increment(obj)
        obj._fittable_state[''] = obj._fittable_version
        Fittable_.refresh(obj)

    @staticmethod
    def get_params(obj, /, *, frozen=False, as_dict=False, _memo=None):
        """The parameters defining the current state of the given object.

        Parameters:
            obj (object): Object for which parameters are to be retrieved.
            frozen (bool, optional): True to include parameters associated with frozen
                objects.
            as_dict (bool, optional): True to return a dictionary that also includes the
                parameters of sub-objects keyed by attribute name; the parameters of the
                given object will be keyed by a blank string "". If False a tuple of the
                given object's parameters are returned.
            _memo (set, optional): Set to use internally for tracking sub-objects, which
                is needed to handle circular references.

        Returns:
            (tuple or dict): Parameters as a tuple or dictionary.
        """

        _memo = set() if _memo is None else _memo
        Fittable_.refresh(obj)

        result = ()
        if frozen or not Fittable_.is_frozen(obj):
            result = obj.__dict__.get('_fittable_params', ())
            if not result and hasattr(obj, '_params'):
                result = obj._params
        _memo.add(id(obj))

        if as_dict:
            result = {'': result} if result else {}
            for name in Fittable_.fittables(obj, frozen=frozen):
                subobj = obj.__dict__[name]
                subobj_id = id(subobj)
                if subobj_id in _memo:
                    continue

                value = Fittable_.get_params(subobj, frozen=frozen, as_dict=True,
                                             _memo=_memo)
                _memo.add(subobj_id)

                # Convert dict to tuple where appropriate
                result[name] = value[''] if list(value.keys()) == [''] else value

        return result

    @staticmethod
    def refresh(obj, /, *, _memo=None):
        """Update any internally cached information if the given object has been modified.

        Use this call to ensure that an object is fully self-consistent, not containing
        any stale information.

        If the given object and any Fittable sub-object(s) are already up to date, the
        object is not changed.

        Parameters:
            obj (object): Object to be refreshed if necessary.
            _memo (set, optional): Set to use internally for tracking sub-objects, which
                is needed to handle circular references.

        Returns:
            (bool): True if the given object was modified as a result of this call.
        """

        def refresh1(obj, _memo):
            """Apply a single iteration of `refresh`."""

            # Refresh any Fittable sub-objects first
            changed = False
            state = Fittable_._get_state(obj)
            for name, prev_version in state.items():
                if not name:                        # skip the object itself for now
                    continue

                subobj = obj.__dict__[name]

                # Make sure this object wasn't already refreshed
                subobj_id = id(subobj)
                if subobj_id in _memo:              # avoid a circular reference
                    state[name] = subobj.version()  # ...but update the state dict
                    continue
                _memo.add(subobj_id)

                # Refresh this sub-object recursively
                Fittable_.refresh(subobj, _memo=_memo)

                # Update the version of this sub-object
                new_version = Fittable_.version(subobj)
                if new_version != prev_version:
                    state[name] = new_version
                    subobj._fittable_params = Fittable_.get_params(subobj)
                    changed = True

            # If any sub-object has changed, this object needs a `_refresh`
            if changed:
                if hasattr(obj, '_refresh'):
                    obj._refresh()
                obj._fittable_params = Fittable_.get_params(obj)
                Fittable_._increment(obj)
                state[''] = obj._fittable_version

            return changed

        # Begin active code
        _memo = set() if _memo is None else _memo

        if not hasattr(obj, '__dict__'):
            return False

        if hasattr(obj, '_is_fittable') and not obj._is_fittable:
            return False

        if hasattr(obj, '_fittable_is_frozen') and obj._fittable_is_frozen:
            return False

        # Refresh once
        changed = refresh1(obj, _memo)

        # A circular reference can require multiple refreshes to get the correct state
        if changed:
            for i in range(9):
                _memo = set()
                if not refresh1(obj, _memo):
                    return True

            raise RuntimeError('Fittable_.refresh() did not complete after 10 iterations')

        return False

    @staticmethod
    def freeze(obj, /, _memo=None):
        """Freeze the given object and any Fittable subobjects.

        A frozen object can no longer be modified.

        Parameters:
            obj (object): Object to freeze.
            _memo (set, optional): Set to use internally for tracking sub-objects, which
                is needed to handle circular references.
        """

        _memo = set() if _memo is None else _memo

        # If not freezable, return
        if not hasattr(obj, '__dict__'):
            return

        # Don't revisit this object
        obj_id = id(obj)
        if obj_id in _memo:
            return
        _memo.add(obj_id)

        if hasattr(obj, '_fittable_is_frozen') and obj._fittable_is_frozen:
            return

        # If the frozen attribute can't be set, check the global list of frozen IDs
        if obj_id in Fittable_._FROZEN_IDS:
            return

        # Freeze the sub-objects
        values = list(obj.__dict__.values())
        for subobj in values:
            subobj_id = id(subobj)
            if subobj_id in _memo:
                continue
            Fittable_.freeze(subobj, _memo=_memo)
            _memo.add(subobj_id)

        # Refresh and freeze this
        if hasattr(obj, '_refresh'):
            obj._refresh()

        if hasattr(obj, '_freeze'):
            obj._freeze()

        # Set this object as frozen if possible
        try:
            obj._fittable_is_frozen = True
        except (AttributeError, TypeError):
            pass

    @staticmethod
    def is_frozen(obj, /, _memo=None):
        """True if the given object and all Fittable sub-objects are frozen.

        Parameters:
            obj (object): Object to test.
            _memo (set, optional): Set to use internally for tracking sub-objects, which
                is needed to handle circular references.

        Returns:
            (bool): True if the given object is frozen.
        """

        _memo = set() if _memo is None else _memo

        # If not a freezable object, return True
        if not hasattr(obj, '__dict__'):
            return True

        # Don't revisit this object
        obj_id = id(obj)
        _memo.add(obj_id)

        # If this object is already frozen, return True
        if hasattr(obj, '_fittable_is_frozen'):
            if obj._fittable_is_frozen:
                return True
        elif obj_id in Fittable_._FROZEN_IDS:
            return True

        # If any sub-object is not frozen, this is not frozen
        values = list(obj.__dict__.values())
        for subobj in values:
            subobj_id = id(subobj)
            if subobj_id in _memo:
                continue
            if not Fittable_.is_frozen(subobj, _memo=_memo):
                return False
            _memo.add(subobj_id)

        # If this object fittable, return False
        if isinstance(obj, Fittable):
            obj._fittable_is_frozen = False
            return False

        # Designate this object as frozen
        try:
            obj._fittable_is_frozen = True
        except (AttributeError, TypeError):
            Fittable_._FROZEN_IDS.add(obj_id)

        return True

    @staticmethod
    def fittables(obj, /, *, frozen=False, _memo=None):
        """Ordered list of Fittable subobjects of the given object.

        This is the list of Fittable sub-objects sorted alphabetically.

        Parameters:
            obj (object): Object for which to obtain the fittable attribute names.
            frozen (bool, optional): True to include names of attributes that have been
                frozen.
            _memo (set, optional): Set to use internally for tracking sub-objects, which
                is needed to handle circular references.

        Returns:
            (list): List of the Fittable sub-objects of the given objects.
        """

        _memo = set() if _memo is None else _memo

        if not hasattr(obj, '__dict__'):
            return []

        if hasattr(obj, '_fittables'):
            names = obj._fittables
        else:
            keys = list(obj.__dict__.keys())

            names = []
            for name in keys:
                subobj = obj.__dict__[name]
                subobj_id = id(subobj)
                if subobj_id in _memo:
                    continue
                if Fittable_.is_fittable(subobj, _memo=_memo):
                    names.append(name)
                _memo.add(subobj_id)

            names.sort()
            try:
                obj._fittables = names
            except (AttributeError, TypeError):
                pass

            _memo.add(id(obj))

        if not frozen:
            _memo = set()
            names = [n for n in names if not Fittable_.is_frozen(obj.__dict__[n],
                                                                 _memo=_memo)]

        _memo.add(id(obj))
        return names

    @staticmethod
    def is_fittable(obj, /, _memo=None):
        """True if the given object or any of its sub-objects is fittable, whether or not
        it is frozen.

        Parameters:
            obj (object): Object to test.
            _memo (set, optional): Set to use internally for tracking sub-objects, needed
                to handle circular references.

        Returns:
            (bool): True if the object is fittable, either by being a Fittable subclass or
                by having a Fittable subobject.
        """

        _memo = set() if _memo is None else _memo
        _memo.add(id(obj))

        if not hasattr(obj, '__dict__'):
            return False

        if hasattr(obj, '_is_fittable'):
            return obj._is_fittable

        if isinstance(obj, Fittable):
            obj._is_fittable = True
            return True

        fittables = Fittable_.fittables(obj, frozen=True, _memo=_memo)
        try:
            return obj.__dict__.setdefault('_is_fittable', bool(fittables))
        except (AttributeError, TypeError):
            return False

    @staticmethod
    def version(obj, /):
        """The Fittable version number of this object.

        The version number starts at zero and is incremented each time the object or one
        of its sub-objects is modified by a call to `set_params` or possibly `refresh`.
        """

        if hasattr(obj, '_fittable_version'):
            return obj._fittable_version

        try:
            obj._fittable_version = 0
        except (AttributeError, TypeError):
            pass

        return 0

    @staticmethod
    def _get_state(obj, /):
        """The state of the given object.

        This is a dictionary containing the version number of the object itself, as well
        as each Fittable sub-object, at the time of construction or the most recent call
        to `refresh`. Each dictionary entry is keyed by the attribute name; the version
        number of the object itself is keyed by "".
        """

        if Fittable_.is_frozen(obj):
            return {}

        if hasattr(obj, '_fittable_state'):
            return obj._fittable_state

        state = {}
        for name in Fittable_.fittables(obj):
            state[name] = 0
        state[''] = 0

        try:
            obj._fittable_state = state
        except (AttributeError, TypeError):
            return {}

        return state

    @staticmethod
    def _increment(obj, /):
        """Increment the Fittable version number of this object."""

        if not hasattr(obj, '__dict__'):
            return

        if hasattr(obj, '_fittable_version'):
            obj._fittable_version += 1

        obj._fittable_version = 1

##########################################################################################
