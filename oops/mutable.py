##########################################################################################
# oops/mutable.py
##########################################################################################
"""Functions that support tracking in-place changes to OOPS objects.

OOPS objects are usually static. However, any object that subclasses Fittable can be
modified in-place by making a call to the function `set_params`. The mutable API is used
to manage updates to objects that are either Fittable or might contain a Fittable
sub-object (recursively).

Any object that is a subclass of Fittable or might itself depend on a Fittable object is
considered "mutable".

The following functions are defined:

* `needs_refresh`: True if the given object is not now internally consistent.
* `refresh`: Make sure the given object is internally consistent. Always call this
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

Programming Notes
-----------------

Information about the Fittable or mutable state of all objects is maintained by a set of
added attributes, which are all prefixed "_FITTABLE" or "_MUTABLE". These attributes are
managed internally and should not be touched by the programmer.

If an object could potentially depend one or more mutable sub-objects, then it may be
necessary to define this method::

    _refresh(self)

which updates any internal attributes that might become "stale" if they depend on a
sub-object that was modified. In addition, the method::

    _freeze(self)

can be used to carry out any special actions that must take place when the object is
frozen.
"""

import numpy as np
from collections import namedtuple
from collections.abc import Iterable
from typing import Any

from polymath import Qube
from oops.fittable import Fittable

_Info = namedtuple('_Info', ['is_fittable', 'is_mutable', 'is_frozen', 'mutable_names',
                             'unfrozen_names', 'versions'])
_IMMUTABLE = _Info(False, False, True, [], [], {})
# versions[name] = the version number that was used for the named subobject the last time
# this object was refreshed. If the named subobject has a later version number now, then
# a refresh is needed. versions[''] is the version of this object at last refresh.

_NEVER_REFRESHED = -1
# The value returned in place of versions[''] for an object that has not yet been
# refreshed. Version numbers start at zero, so this value is always smaller, ensuring that
# the first call to `refresh` always applies `_refresh`. This is what initializes the
# derived attributes of an object at the end of its constructor.

_IMMUTABLE_OBJECTS = set()  # for objects with __dict__ that can't have attributes set


def refresh(obj: Any, /) -> bool:
    """Update any internally cached information if the given object or any of its
    sub-objects has been modified.

    Use this call to ensure that an object is fully self-consistent, not containing any
    stale information.

    If the given object and all Fittable sub-object(s) are already up to date, the given
    object is not changed.

    Parameters:
        obj: Object to be refreshed if necessary.

    Returns:
        True if the given object was modified as a result of this call.
    """

    # memo[0] indicates whether the given object has changed; initially set to False
    return _refresh_internal(obj, memo={0: False}, info_memo={})


def _refresh_internal(obj: Any, /, memo: dict, info_memo: dict) -> bool:
    """Update any internally cached information if the given object or any of its
    sub-objects has been modified.

    This is the internal, recursive implementation of `refresh`.

    Parameters:
        obj: Object to be refreshed if necessary.
        memo: Tracks status; prevents infinite recursion.
        info_memo: `memo` input to `_get_info`.

    Returns:
        True if the given object was modified as a result of this call.
    """

    # Check memo; prevent infinite recursion
    changed = memo[0]
    obj_id = id(obj)
    if obj_id in memo:
        return changed

    # An object that cannot record its own state shares the read-only _IMMUTABLE info, so
    # it has no derived attributes to refresh and nowhere to save a version number
    info = _get_info(obj, info_memo)
    if info is _IMMUTABLE:
        memo[obj_id] = None
        return changed

    # Refresh sub-objects
    for name in info.unfrozen_names:
        subobj = obj.__dict__[name]
        if _needs_refresh_internal(subobj, info_memo):
            changed = _refresh_internal(subobj, memo, info_memo)
            info.versions[name] = version(subobj)
        elif info.versions[name] < version(subobj):
            changed = True

    # Check for an object that has never been refreshed or whose own version number has
    # been incremented since the last refresh, as happens when a Fittable is given new
    # parameter values
    if info.versions.get('', _NEVER_REFRESHED) < version(obj):
        changed = True

    # Refresh the given object
    if changed:
        _increment(obj)
        if hasattr(obj, '_refresh'):
            obj._refresh()

    # Now everything is up to date
    for name in info.unfrozen_names:
        info.versions[name] = version(obj.__dict__[name])
    info.versions[''] = version(obj)

    memo[obj_id] = None         # record that this object has been refreshed
    memo[0] |= changed
    return changed


def needs_refresh(obj: Any, /) -> bool:
    """True if any internally cached information of the given object or any of its
    sub-objects needs to be refreshed.

    If the given object and all Fittable sub-object(s) are already up to date, this
    function returns False.

    Parameters:
        obj (object): Object to test.

    Returns:
        (bool): True if the given object needs to be refreshed.
    """

    return _needs_refresh_internal(obj, info_memo={})


def _needs_refresh_internal(obj: Any, info_memo: dict) -> bool:
    """True if any internally cached information of the given object or any of its
    sub-objects needs to be refreshed.

    This is the internal, recursive implementation

    Parameters:
        obj: Object to test.
        info_memo: `memo` input to `_get_info`.

    Returns:
        True if the given object needs to be refreshed.
    """

    info = _get_info(obj, info_memo)

    # An object that cannot record its own state has nothing to refresh
    if info is _IMMUTABLE:
        return False

    # If this object has never been refreshed or is stale, return True
    if info.versions.get('', _NEVER_REFRESHED) < version(obj):
        return True

    # If any unfrozen subobject is stale, return True
    for name in info.unfrozen_names:
        subobj = obj.__dict__[name]
        if _needs_refresh_internal(subobj, info_memo):
            return True
        if info.versions[name] < version(subobj):
            return True

    return False


def freeze(obj: Any, /) -> bool:
    """Freeze the given object and all of its sub-objects.

    A frozen object can no longer be modified.

    Parameters:
        obj: The object to freeze.

    Returns:
        True if the given object was frozen as a result of this call; False if it is
        immutable or was already frozen.
    """

    info_memo = {}
    _refresh_internal(obj, memo={0: False}, info_memo=info_memo)
    return _freeze_internal(obj, memo={0: False}, info_memo=info_memo)


def _freeze_internal(obj: Any, /, memo: dict, info_memo: dict) -> bool:
    """Freeze the given object and all of its sub-objects.

    This is the internal, recursive implementation.

    Parameters:
        obj: The object to freeze.
        memo: Tracks status; prevents infinite recursion.
        info_memo: `memo` input to `_get_info`.

    Returns:
        True if the given object was frozen as a result of this call; False if it is
        immutable or was already frozen.
    """

    changed = memo[0]
    obj_id = id(obj)
    if obj_id in memo:
        return changed

    info = _get_info(obj, info_memo)
    if info.is_frozen:
        return changed

    for name in info.unfrozen_names:
        subobj = obj.__dict__[name]
        changed = _freeze_internal(subobj, memo, info_memo)

    if info.is_fittable:
        obj._mark_as_frozen()
        if hasattr(obj, '_freeze'):
            obj._freeze()
        changed = True

    # Save the info if possible
    if changed:
        info = _Info(info.is_fittable, info.is_mutable, True, info.mutable_names, [], {})
        try:
            obj._MUTABLE_info = info
        except (AttributeError, TypeError):
            _IMMUTABLE_OBJECTS.add(obj_id)

    memo[obj_id] = None         # record that this object has been frozen
    memo[0] = changed
    return changed

##########################################################################################
# Apply parameters for multiple sub-objects at once
##########################################################################################

def set_param_order(obj: Any, names: list[str]) -> None:
    """Define the order of the parameters for an object that might contain Fittable
    sub-objects.

    Parameters:
        obj: The object.
        names: The names of the sub-objects in the order that their parameters appear. Use
            an empty string to indicate the location of the parameters of `obj` if it is
            Fittable.

    Raises:
        ValueError: If a name in `names` is not a recognized sub-object or if parameter
            values have already been set.
        AttributeError: If this object or a sub-object is not mutable.
    """

    if hasattr(obj, '_MUTABLE_params'):
        raise ValueError('parameter order was already defined: '
                         f'{obj._MUTABLE_param_names}')

    nparams = 0
    params = []
    for name in names:
        if name:
            if name not in obj.__dict__:
                raise ValueError(f'no attribute {name}')
            temp_obj = obj.__dict__[name]
        else:
            if not isinstance(obj, Fittable):
                raise ValueError(f'object is not Fittable')
            temp_obj = obj

        nparams += temp_obj.nparams
        params += list(get_params(temp_obj))

    if nparams == 0:
        raise ValueError(f'no fittable parameters')

    obj._MUTABLE_param_names = list(names)
    obj._MUTABLE_nparams = nparams
    obj._MUTABLE_params = tuple(params)


def get_param_order(obj: Any) -> list[str]:
    """Get the order of the parameters for an object that might contain Fittable
    sub-objects.

    Parameters:
        obj: The object.
    """

    if hasattr(obj, '_MUTABLE_param_names'):
        return obj._MUTABLE_param_names

    return []


def get_nparams(obj: Any) -> int:
    """The number of parameters for a mutable or Fittable object.

    Parameters:
        obj: The object.
    """

    if hasattr(obj, '_MUTABLE_nparams'):
        return obj._MUTABLE_nparams

    if isinstance(obj, Fittable):
        return obj.nparams

    return 0


def set_params(obj: Any, params: Any) -> bool:
    """Set the parameters of an object, including for any mutable sub-objects.

    Parameters:
        obj: The object.
        params: Parameter values to apply.

    Returns:
        True if the given object has changed as a result of this function call.

    Raises:
        ValueError: If the number of parameters is incorrect or the object is frozen.
    """

    # Convert params to tuple if necessary
    if isinstance(params, Iterable):
        params = tuple(float(x) for x in params)
    else:
        params = (float(params),)

    # Check parameter count
    if len(params) != get_nparams(obj):
        raise ValueError('incorrect parameter count for mutable.set_params()')
    if len(params) == 0:
        return False

    # Enable this call to be used for a Fittable
    if not hasattr(obj, '_MUTABLE_param_names'):
        if isinstance(obj, Fittable):
            return obj.set_params(params)
        raise ValueError(f'unknown parameter order for {obj}')

    obj._MUTABLE_params = params

    # Set subobject parameters, skipping those for the given object
    k0 = 0
    blank_params = []
    changed = False
    for name in obj._MUTABLE_param_names:
        temp_obj = obj.__dict__[name] if name else obj
        if name:
            k1 = k0 + get_nparams(temp_obj)
            changed |= set_params(temp_obj, params[k0:k1])
        else:
            k1 = k0 + temp_obj.nparams
            blank_params = params[k0:k1]
        k0 = k1

    # If the given object is Fittable, set its parameters last
    if blank_params and blank_params != obj.params:
        obj._set_params(blank_params)
        changed = True

    # Refresh if needed
    if changed:
        _increment(obj)
        refresh(obj)

    return changed


def get_params(obj: Any) -> tuple[float, ...]:
    """Get the parameters of an object, including those for any mutable sub-objects.

    Parameters:
        obj: The object.
        params: Parameter values to apply.
    """

    if hasattr(obj, '_MUTABLE_param_names'):
        return obj._MUTABLE_params

    if isinstance(obj, Fittable):
        return obj.params

    return ()

##########################################################################################
# Support for attributes of the mutable class
##########################################################################################

def _get_info(obj: Any, /, memo: dict | None = None) -> _Info:
    """The tuple (is_fittable, is_mutable, is_frozen, mutable_names, unfrozen_names,
    versions).

    Parameters:
        obj: The object to test.
        memo: Used internally to prevent infinite recursion. memo[name] returns the info
            for the named subobject.

    Returns:
        (`is_fittable`, `is_mutable`, `is_frozen`, `mutable_names`, `unfrozen_names`,
        `versions`) where:

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


def is_fittable(obj: Any, /) -> bool:
    """True if the given object is Fittable.

    Parameters:
        obj: The object.

    Returns:
        True if `obj` is Fittable.
    """

    return _get_info(obj).is_fittable


def is_mutable(obj: Any, /) -> bool:
    """True if the given object is mutable.

    An object is mutable if it is Fittable or if it contains any Fittable sub-object
    (recursively).

    Parameters:
        obj: The object.

    Returns:
        True if `obj` is mutable.
    """

    return _get_info(obj).is_mutable


def is_frozen(obj: Any, /) -> bool:
    """True if the given object is frozen or immutable.

    Parameters:
        obj: The object.

    Returns:
        True if `obj` is frozen or immutable.
    """

    return _get_info(obj).is_frozen


def mutable_names(obj: Any, /) -> list[str]:
    """Names of the mutable sub-objects of the given object.

    Parameters:
        obj: The object.

    Returns:
        List of the names of the mutable sub-objects of the given object.
    """

    return _get_info(obj).mutable_names


def unfrozen_names(obj: Any, /) -> list[str]:
    """Names of the mutable sub-objects of the given object that are not currently frozen.

    Parameters:
        obj: The object.

    Returns:
        List of the names of the un-frozen sub-objects of the given object.
    """

    return _get_info(obj).unfrozen_names


def _versions(obj: Any, /) -> dict[str, int]:
    """Dictionary of the version number of each mutable sub-object at the time of last
    refresh.

    If the current version of any sub-object is greater than the version found in this
    dictionary, the object must be refreshed.

    If the given object is a Fittable, then the version of the internal parameters at last
    refresh is saved in the dictionary with a blank key.

    Parameters:
        obj: The object.

    Returns:
        Dictionary keyed by the name of each mutable sub-object, returning the version of
        that object at the time of last refresh.
    """

    return _get_info(obj).versions


def version(obj: Any, /) -> int:
    """The version number of the given object.

    Parameters:
        obj: The object.

    Returns:
        The version number, which starts at zero and is incremented each time the object
        or one of its sub-objects is modified.
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


def _increment(obj: Any, /) -> int:
    """Increment and return the version number of the given object.

    Parameters:
        obj: The object.

    Returns:
        The new version number, or 0 if the given object cannot be mutable.
    """

    if hasattr(obj, '_MUTABLE_version'):
        obj._MUTABLE_version += 1
        return obj._MUTABLE_version

    if not hasattr(obj, '__dict__'):
        return 0

    try:
        obj._MUTABLE_version = 1
    except (AttributeError, TypeError):
        return 0

    return obj._MUTABLE_version

##########################################################################################
