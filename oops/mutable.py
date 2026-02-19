##########################################################################################
# oops/mutable.py
##########################################################################################
"""Functions that support tracking in-place changes to OOPS objects.

OOPS objects are usually static. However, any object that subclasses Fittable can be
modified in-place by making a call to a function named `set_params`. The mutable API is
used to manage updates to objects that are either Fittable or might contain a Fittable
sub-object (recursively).

Any object that is a subclass of Fittable or might itself depend on a Fittable object is
considered "mutable".

The following functions are defined:

* `set_params`: Updates the parameter values for the given object and, optionally, its
  mutable sub-objects.
* `get_params`: Retrieves the current parameter values of the given object and,
  optionally, those of its mutable sub-objects.
* `refresh`: Makes sure the given object is internally consistent. Always call this
  function after an object or any of the sub-objects might have been modified.
* `freeze`: Freeze the given object, preventing any further changes to it or any of its
  sub-objects.
* `is_fittable`: True if the given object is a subclass of Fittable.
* `is_mutable`: True if the given object is mutable. An object is mutable if it is
  Fittable or if it depends on any mutable sub-objects (recursively).
* `is_frozen`: True if the given object is frozen or otherwise immutable.
* `mutable_names`: A list of the names of any mutable sub-objects.
* `unfrozen_names`: A list of the names of any mutable sub-objects that have not been
  frozen.
* `version`: An integer that starts at zero and increases whenever this object or one of
  its sub-objects changes.

It is important to call::

    mutable.refresh(obj)

after an object `obj` or any of its sub-objects might have been changed. This ensures that
any internal information maintained by the object has not become "stale". If an object is
currently up to date, this should be a very fast operation.

Programming Notes
-----------------

Information about the Fittable or mutable state of all objects is maintained by a set of
added attributes, which are all prefixed "_FITTABLE" or "_MUTABLE". These attributes are
managed internally and should not be touched by the programmer.

If an object could potentially depend one or more mutable sub-objects, then it may be
necessary to define this method::

    _refresh(self)

which must update any internal attributes that might become "stale" if they depend on a
sub-object that was modified. In addition, the method::

    _freeze(self)

can be used to carry out any special actions that must take place when the object is
frozen.
"""

import numpy as np
from collections import namedtuple

from polymath import Qube
from oops.fittable import Fittable

_Info = namedtuple('_Info', ['is_fittable', 'is_mutable', 'is_frozen', 'mutable_names',
                             'unfrozen_names', 'versions'])
_IMMUTABLE = _Info(False, False, True, [], [], {})

_IMMUTABLE_OBJECTS = set()  # for objects with __dict__ that can't have attributes set


def get_params(obj, /, *, frozen=False, recursive=False):
    """The parameters defining the current state of the given object.

    Parameters:
        obj (object): The object for which parameters are to be retrieved.
        frozen (bool, optional): True to include parameters associated with frozen
            objects.
        recursive (bool, optional): True to include the parameters of any mutable
            sub-objects recursively, in addition to `obj`'s parameters

    Returns:
        (tuple or dict): If recursive is False or if `obj` has no mutable sub-objects,
            this function returns the tuple of the current parameters of `obj` if it is
            Fittable (or an empty tuple if `obj` is not Fittable). If recursive is True
            and `obj` has one or more mutable sub-objects, this function instead returns a
            dictionary keyed by the name of each mutable sub-object, containing the
            parameters used by that sub-object (recursively). In this case, if `obj` is
            Fittable, its parameters are returned keyed by an empty string "".
    """

    def get_params_recursive(obj, memo):

        obj_id = id(obj)
        if obj_id in memo:
            return memo[obj_id]

        info = _get_info(obj)
        if info.is_frozen and not frozen:
            return ()

        names = info.mutable_names if frozen else info.unfrozen_names
        result = {}
        for name in names:
            params = get_params_recursive(obj.__dict__[name], memo)
            if params:
                result[name] = params

        if info.is_fittable:
            if result:
                result[''] = obj.params
            else:
                result = obj.params

        return result

    if recursive:
        return get_params_recursive(obj, memo={})
    elif isinstance(obj, Fittable) and (frozen or not obj.is_frozen):
        return obj.params
    else:
        return ()


def set_params(obj, /, params):
    """Redefine the given object using new parameters.

    This calls a defined `_set_params` method for any Fittable sub-objects and then for
    the object itself if it is Fittable. It also refreshes.

    Parameters:
        obj (object): Object for which parameters are to be set.
        params (tuple, list, np.ndarray, or dict): Parameter values to use. If a tuple,
            list or array is given, the values are applied to `obj`, which must be
            Fittable. If a dictionary is given, each key in the dictionary must be the
            name of an attribute in `obj` and the values are applied to that sub-object;
            in this case, parameters to be applied to `obj` itself (if it is Fittable) can
            be provided as a tuple, list, or array value keyed by an empty string "".

    Returns:
        (bool): True if `obj` is different as a result of this function call.
    """

    if isinstance(params, dict):
        changed = False
        for name, vals in params.items():
            if name:
                changed = changed or set_params(obj.__dict__[name], vals)
        if '' in params:
            changed = changed or set_params(obj, params[''])
        return changed

    if not isinstance(obj, Fittable):
        raise ValueError(f'{type(obj).__name__} object is not Fittable')

    # Convert params to tuple if necessary
    if isinstance(params, (list, np.ndarray)):
        params = tuple(params)
    elif not isinstance(params, tuple):
        params = (params,)

    # Check for valid parameters
    if not params:
        raise ValueError('missing parameters for {type(obj).__name__}.set_params()')

    if params == obj.params:
        return False
    if len(params) != obj.nparams:
        plural = 's' if obj._nparams > 1 else ''
        raise ValueError(f'{type(obj).__name__} object requires {obj.nparams} fit '
                         f'parameter{plural}')

    # Check for frozen object
    if obj.is_frozen:
        raise ValueError(f'{type(obj).__name__} object is frozen')

    obj._set_params(params)
    obj.refresh()
    _increment(obj)
    return True


def refresh(obj, /):
    """Update any internally cached information if the given object or any of its
    sub-objects has been modified.

    Use this call to ensure that an object is fully self-consistent, not containing
    any stale information.

    If the given object and all Fittable sub-object(s) are already up to date, the given
    object is not changed.

    Parameters:
        obj (object): Object to be refreshed if necessary.

    Returns:
        (bool): True if the given object was modified as a result of this call.
    """

    def refresh_recursive(obj, memo):
        nonlocal changed

        obj_id = id(obj)
        if obj_id in memo:
            return

        memo.add(obj_id)

        needs_refresh = not hasattr(obj, '_MUTABLE_info')   # True on first call
        info = _get_info(obj)
        if info.is_frozen:
            if needs_refresh and hasattr(obj, '_refresh'):
                obj._refresh()
            return

        # Refresh any sub-objects
        for name in info.unfrozen_names:
            subobj = obj.__dict__[name]
            refresh_recursive(subobj, memo)
            new_version = version(subobj)
            if info.versions[name] < new_version:
                needs_refresh = True
                info.versions[name] = new_version

        # Refresh this object
        if needs_refresh:
            if hasattr(obj, '_refresh'):
                obj._refresh()
            _increment(obj)
            changed = True

    # Begin active code
    changed = False
    refresh_recursive(obj, memo=set())

    # A circular reference can require multiple refreshes to get the correct state
    if changed:
        for i in range(9):
            changed = False
            refresh_recursive(obj, memo=set())
            if not changed:
                return True

        raise RuntimeError('refresh() did not complete after 10 iterations')

    return False


def freeze(obj, /):
    """Freeze the given object and all of its sub-objects.

    A frozen object can no longer be modified.

    Parameters:
        obj (object): The object to freeze.

    Returns:
        (bool): True if the given object was frozen as a result of this call; False if it
            was already frozen or immutable.
    """

    def freeze_recursive(obj, memo):
        nonlocal changed

        obj_id = id(obj)
        if obj_id in memo:
            return

        memo.add(obj_id)

        info = _get_info(obj)
        if info.is_frozen:
            return

        for name in info.unfrozen_names:
            freeze_recursive(obj.__dict__[name], memo)
            changed = True

        if info.is_fittable and not obj.is_frozen:
            obj._mark_as_frozen()
            changed = True

        if hasattr(obj, '_freeze'):
            obj._freeze()

        # Save the info if possible
        info = _Info(is_fittable, is_mutable, True, mutable_names, [], {})
        try:
            obj._MUTABLE_info = info
        except (AttributeError, TypeError):
            _IMMUTABLE_OBJECTS.add(obj_id)

    changed = False
    freeze_recursive(obj, memo=set())
    return changed


def is_fittable(obj, /):
    """True if the given object is Fittable.

    Parameters:
        obj (object): The object.

    Returns:
        (bool): True if `obj` is Fittable.
    """

    return _get_info(obj).is_fittable


def is_mutable(obj, /):
    """True if the given object is mutable.

    An object is mutable if it is Fittable or if it contains any Fittable sub-object
    (recursively).

    Parameters:
        obj (object): The object.

    Returns:
        (bool): True if `obj` is mutable.
    """

    return _get_info(obj).is_mutable


def is_frozen(obj, /):
    """True if the given object is frozen or immutable.

    Parameters:
        obj (object): The object.

    Returns:
        (bool): True if `obj` is frozen or immutable.
    """

    return _get_info(obj).is_frozen


def mutable_names(obj, /):
    """Names of the mutable sub-objects of the given object.

    Parameters:
        obj (object): The object.

    Returns:
        (list): List of the names of the mutable sub-objects of the given object.
    """

    return _get_info(obj).mutable_names


def unfrozen_names(obj, /):
    """Names of the mutable sub-objects of the given object that are not currently frozen.

    Parameters:
        obj (object): The object.

    Returns:
        (list): List of the names of the un-frozen sub-objects of the given object.
    """

    return _get_info(obj).unfrozen_names


def _get_info(obj, /, memo=None):
    """Returns the tuple (is_fittable, is_mutable, is_frozen, mutable_names,
    unfrozen_names, versions).

    Parameters:
        obj (object): The object to test.
        memo (dict): Used internally to prevent infinite recursion.

    Returns:
        (tuple): (`is_fittable`, `is_mutable`, `is_frozen`, `mutable_names`,
                  `unfrozen_names`, `versions`) where:

        * `is_fittable`: True either if `obj` is Fittable.
        * `is_mutable`: True either if the `obj` is Fittable or if it contains any
          Fittable sub-objects. This is a recursive test and does not depend on the frozen
          state of any object.
        * `is_frozen`: True if `obj` is frozen or immutable.
        * `mutable_names`: The list of names of all sub-objects that are mutable.
        * `unfrozen_names`: The list of names of all sub-objects that are not frozen.
        * `versions`: A dictionary that maps attribute names to version numbers.
    """

    if not hasattr(obj, '__dict__') or hasattr(obj, '_IS_IMMUTABLE'):
        return _IMMUTABLE

    # Treat all arrays and polymath objects as read-only
    if isinstance(obj, (Qube, np.ndarray)):
        return _IMMUTABLE

    if memo is None:
        memo = {}

    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]

    if obj_id in _IMMUTABLE_OBJECTS:
        memo[obj_id] = _IMMUTABLE
        return _IMMUTABLE

    memo[obj_id] = None

    # During a repeat call, just update the frozen info
    if hasattr(obj, '_MUTABLE_info'):
        info = obj._MUTABLE_info
        (is_fittable, is_mutable, is_frozen, mutable_names, _, versions) = info
        if is_frozen:
            memo[obj_id] = info
            return info

        unfrozen_names = []
        for name in info.unfrozen_names:
            subobj = obj.__dict__[name]
            if not _get_info(subobj, memo).is_frozen:
                unfrozen_names.append(name)

    else:
        # During the first call, initialize everything
        mutable_names = []
        unfrozen_names = []
        versions = {}
        for name, subobj in obj.__dict__.items():
            info = _get_info(subobj, memo)
            if info is not None and info.is_mutable:
                mutable_names.append(name)
                if not info.is_frozen:
                    unfrozen_names.append(name)
                    versions[name] = version(subobj)

        is_fittable = isinstance(obj, Fittable)
        is_mutable = bool(mutable_names) or is_fittable

    is_frozen = not (bool(unfrozen_names) or (is_fittable and not obj.is_frozen))
    info = _Info(is_fittable, is_mutable, is_frozen, mutable_names, unfrozen_names,
                 versions)
    try:
        obj._MUTABLE_info = info
    except (AttributeError, TypeError):
        _IMMUTABLE_OBJECTS.add(obj_id)

    memo[obj_id] = info
    return info


def version(obj, /):
    """The version number of the given object.

    Parameters:
        obj (object): The object.

    Returns:
        (int): The version number, which starts at zero and is incremented each time
            the object or one of its sub-objects is modified.
    """

    if hasattr(obj, '_MUTABLE_version'):
        return obj._MUTABLE_version

    if not hasattr(obj, '__dict__'):
        return 0

    try:
        obj._MUTABLE_version = 0
    except (AttributeError, TypeError):
        pass

    return 0


def _increment(obj, /):
    """Increment and return the version number of the given object.

    Parameters:
        obj (object): The object.

    Returns:
        (int): The new version number, or 0 if the given object cannot be mutable.
    """

    if not hasattr(obj, '__dict__'):
        return 0

    if hasattr(obj, '_MUTABLE_version'):
        obj._MUTABLE_version += 1
        return obj._MUTABLE_version

    try:
        obj._MUTABLE_version = 1
    except (AttributeError, TypeError):
        return 0

    return obj._MUTABLE_version

##########################################################################################
