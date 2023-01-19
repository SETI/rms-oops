################################################################################
# polymath/extensions/shrinker.py: shrink and unshrink operations
################################################################################

import numpy as np
from polymath.qube   import Qube
from polymath.scalar import Scalar

def shrink(self, antimask):
    """A 1-D version of this object, containing only the samples in the antimask
    provided.

    The antimask array value of True indicates that an element should be
    included; False means that is should be discarded. A scalar value of True or
    False applies to the entire object.

    The purpose is to speed up calculations by first eliminating all the objects
    that are masked. Any calculation involving un-shrunken objects should
    produce the same result if the same objects are all shrunken by a common
    antimask first, the calculation is performed, and then the result is
    un-shrunken afterward.

    Shrunken objects are always converted to read-only.
    """

    #### For testing only...
    if Qube._DISABLE_SHRINKING:
        if not self._shape_ or Qube.is_one_true(antimask):
            return self
        return self.mask_where(np.logical_not(antimask))

    # A True antimask leaves an object unchanged
    if Qube.is_one_true(antimask):
        return self

    # If the antimask is a single False value, or if this object is already
    # entirely masked, return a single masked value
    if (Qube.is_one_true(self._mask_) or Qube.is_one_false(antimask) or
        not np.any(antimask & self.antimask)):
            obj = self.masked_single().as_readonly()
            if not Qube.DISABLE_CACHE:
                obj._cache_['unshrunk'] = self
            return obj

    # If this is a shapeless object, return it as is
    if not self._shape_:
        self._cache_['unshrunk'] = self
        return self

    # Beyond this point, the size of the last axis in the returned object
    # will have the same number of elements as the number of True elements
    # in the antimask.

    # Ensure that this object and the antimask have compatible dimensions.
    # If the antimask has extra dimensions, broadcast self to make it work
    self_rank = len(self._shape_)
    antimask_rank = antimask.ndim
    extras = self_rank - antimask_rank
    if extras < 0:
        self = self.broadcast_to(antimask.shape, recursive=False)
        self_rank = antimask_rank
        extras = 0

    # If self has extra dimensions, these will be retained and only the
    # rightmost axes will be flattened.
    before = self._shape_[:extras]      # shape of self retained
    after  = self._shape_[extras:]      # shape of self to be masked

    # Make the rightmost axes of self and the mask compatible
    new_after = tuple([max(after[k],antimask.shape[k])
                       for k in range(len(after))])
    new_shape = before + new_after
    if self._shape_ != new_shape:
        self = self.broadcast_to(new_shape, recursive=False)
    if antimask.shape != new_after:
        antimask = np.broadcast_to(antimask, new_after)

    # Construct the new mask
    if Qube.is_one_false(self._mask_):
        mask = np.zeros(antimask.shape, dtype=np.bool_)[antimask]
    else:
        mask = self._mask_[extras * (slice(None),) + (antimask,Ellipsis)]

    if np.all(mask):
        obj = self.masked_single().as_readonly()
        obj._cache_['unshrunk'] = self
        return obj

    if not np.any(mask):
        mask = False

    # Construct the new object
    obj = Qube.__new__(type(self))
    obj.__init__(self._values_[extras * (slice(None),)
                                + (antimask,Ellipsis)],
                 mask, example=self)
    obj.as_readonly()

    for (key, deriv) in self._derivs_.items():
        obj.insert_deriv(key, deriv.shrink(antimask))

    # Cache values to speed things up later
    obj._cache_['unshrunk'] = self
    return obj

#===========================================================================
def unshrink(self, antimask, shape=()):
    """Convert an object to its un-shrunken shape, based on a given
    antimask.

    If this object was previously shrunken, the antimask must match the one
    used to shrink it. Otherwise, the size of this object's last axis must
    match the number of True values in the antimask.

    Input:
        antimask    the antimask to apply.
        shape       in cases where the antimask is a literal False, this
                    defines the shape of the returned object. Normally, the
                    rightmost axes of the returned object match those of
                    the antimask.

    The returned object will be read-only.
    """

    #### For testing only...
    if Qube._DISABLE_SHRINKING:
        return self

    # Get the previous unshrunk version if available and delete from cache
    if Qube.DISABLE_CACHE:
        unshrunk = None
    else:
        unshrunk = self._cache_.get('unshrunk', None)
        if unshrunk is not None:
            del self._cache_['unshrunk']

            if Qube._IGNORE_UNSHRUNK_AS_CACHED:
                unshrunk = None

    # If the antimask is True, return this as is
    if Qube.is_one_true(antimask):
        return self

    # If the new object is entirely masked, return a shapeless masked object
    if not np.any(antimask) or np.all(self._mask_):
        return self.masked_single().broadcast_to(shape)

    # If this object is shapeless, return it as is
    if not self._shape_:
        return self

    # If we found a cached value, return it
    if unshrunk is not None:
        return unshrunk.mask_where(np.logical_not(antimask))

    # Create the new data array
    new_shape = self._shape_[:-1] + antimask.shape
    indx = (len(self._shape_)-1) * (slice(None),) + (antimask, Ellipsis)
    if isinstance(self._values_, np.ndarray):
        default = self._default_
        if isinstance(default, Qube):
            default = self._default_._values_

        new_values = np.empty(new_shape + self._item_,
                              self._values_.dtype)
        new_values[...] = default

        new_values[indx] = self._values_ # fill in non-default values

    # ...where single values can be handled by broadcasting...
    else:
        item = Scalar(self._values_)
        new_values = item.broadcast_to(new_shape)._values_

    # Create the new mask array
    new_mask = np.ones(new_shape, dtype=np.bool_)
    new_mask[indx] = self._mask_        # insert the shrunk mask values

    # Create the new object
    obj = Qube.__new__(type(self))
    obj.__init__(new_values, new_mask, example=self)
    obj = obj.as_readonly()

    # Unshrink the derivatives
    for (key, deriv) in self._derivs_.items():
        obj.insert_deriv(key, deriv.unshrink(antimask, shape))

    return obj

################################################################################
