################################################################################
# polymath/extensions/mask_ops.py: masking operations
################################################################################

import numpy as np
from ..qube import Qube

def mask_where(self, mask, replace=None, remask=True):
    """A copy of this object after a mask has been applied.

    If the mask is empty, this object is returned unchanged.

    Inputs:
        mask            the mask to apply as a boolean array.
        replace         a single replacement value or, an object of the same
                        shape and class as this object, containing replacement
                        values. These are inserted into returned object at every
                        masked location. Use None (the default) to leave values
                        unchanged.
        remask          True to include the new mask in the object's mask;
                        False to replace the values but leave them unmasked.
    """

    # Convert to boolean array if necessary
    if isinstance(mask, np.ndarray):
        mask = mask.astype(np.bool_)
    elif isinstance(mask, Qube):
        mask = mask.as_mask_where_nonzero_or_masked()
    else:
        mask = Qube.BOOLEAN_CLASS.as_boolean(mask)._values_

    # If the mask is empty, return the object as is
    if not np.any(mask):
        return self

    # Get the replacement value as this type
    if replace is not None:
        replace = self.as_this_type(replace, recursive=False)
        if replace._shape_ not in ((), self._shape_):
            raise ValueError('shape of replacement is incompatible with ' +
                             'shape of object being masked: %s, %s' %
                             (replace._shape_, self._shape_))

    # Shapeless case
    if np.isscalar(self._values_):
        if np.shape(mask):
            raise ValueError('object and mask have incompatible shapes: ' +
                             '%s, %s' % (self._shape_, np.shape(mask)))
        if replace is None:
            new_values = self._values_
        else:
            new_values = replace._values_

        if remask or replace._mask_:
            new_mask = True
        else:
            new_mask = self._mask_

        obj = self.clone(recursive=True)
        obj._set_values_(new_values, new_mask)
        return obj

    # Construct the new mask
    if remask:
        new_mask = self._mask_ | mask
    elif np.shape(self._mask_):
        new_mask = self._mask_.copy()
    else:
        new_mask = self._mask_

    # Construct the new array of values
    if replace is None:
        new_values = self._values_

    # If replacement is an array of values...
    elif replace._shape_:
        new_values = self._values_.copy()
        new_values[mask] = replace._values_[mask]

        # Update the mask if replacement values are masked
        if Qube.is_one_true(new_mask):
            pass
        elif Qube.is_one_false(replace._mask_):
            pass
        else:
            if Qube.is_one_false(new_mask):
                new_mask = np.zeros(self._shape_, dtype=np.bool_)

            if Qube.is_one_true(replace._mask_):
                new_mask[mask] = True
            else:
                new_mask[mask] = replace._mask_[mask]

    # If replacement is a single value...
    else:
        new_values = self._values_.copy()
        new_values[mask] = replace._values_

        # Update the mask if replacement values are masked
        if replace._mask_:
            if np.shape(new_mask):
                new_mask[mask] = True
            else:
                new_mask = True

    # Construct the new object and return
    obj = self.clone(recursive=True)
    obj._set_values_(new_values, new_mask)
    return obj

#===============================================================================
def mask_where_eq(self, match, replace=None, remask=True):
    """A copy of this object with items equal to a value masked.

        Instead of or in addition to masking the items, the values can be
        replaced. If no items need to be masked, this object is returned
        unchanged.

    Inputs:
        match           the item value to match.
        replace         a single replacement value or, an object of the same
                        shape and class as this object, containing replacement
                        values. These are inserted into returned object at every
                        masked location. Use None (the default) to leave values
                        unchanged.
        remask          True to include the new mask into the object's mask;
                        False to replace the values but leave them unmasked.
    """

    match = self.as_this_type(match, recursive=False)

    mask = (self._values_ == match._values_)
    for r in range(self._rank_):
        mask = np.all(mask, axis=-1)

    return self.mask_where(mask, replace, remask)

#===============================================================================
def mask_where_ne(self, match, replace=None, remask=True):
    """A copy of this object with items not equal to a value masked.

    Instead of or in addition to masking the items, the values can be replaced.
    If no items need to be masked, this object is returned unchanged.

    Inputs:
        match           the item value to match.
        replace         a single replacement value or, an object of the same
                        shape and class as this object, containing replacement
                        values. These are inserted into returned object at every
                        masked location. Use None (the default) to leave values
                        unchanged.
        remask          True to include the new mask into the object's mask;
                        False to replace the values but leave them unmasked.
    """

    match = self.as_this_type(match, recursive=False)

    mask = (self._values_ != match._values_)
    for r in range(self._rank_):
        mask = np.any(mask, axis=-1)

    return self.mask_where(mask, replace, remask)

#===============================================================================
def mask_where_le(self, limit, replace=None, remask=True):
    """A copy of this object with items <= a limit value masked.

    Instead of or in addition to masking the items, the values can be replaced.
    If no items need to be masked, this object is returned unchanged.

    Inputs:
        limit           the limiting value.
        replace         a single replacement value or, an object of the same
                        shape and class as this object, containing replacement
                        values. These are inserted into returned object at every
                        masked location. Use None (the default) to leave values
                        unchanged.
        remask          True to include the new mask into the object's mask;
                        False to replace the values but leave them unmasked.
    """

    if self._rank_:
        raise ValueError('mask_where_le requires item rank zero')

    if isinstance(limit, Qube):
        limit = limit._values_

    return self.mask_where(self._values_ <= limit, replace, remask)

#===============================================================================
def mask_where_ge(self, limit, replace=None, remask=True):
    """A copy of this object with items >= a limit value masked.

    Instead of or in addition to masking the items, the values can be replaced.
    If no items need to be masked, this object is returned unchanged.

    Inputs:
        limit           the limiting value.
        replace         a single replacement value or, an object of the same
                        shape and class as this object, containing replacement
                        values. These are inserted into returned object at every
                        masked location. Use None (the default) to leave values
                        unchanged.
        remask          True to include the new mask into the object's mask;
                        False to replace the values but leave them unmasked.
    """

    if self._rank_:
        raise ValueError('mask_where_ge requires item rank zero')

    if isinstance(limit, Qube):
        limit = limit._values_

    return self.mask_where(self._values_ >= limit, replace, remask)

#===============================================================================
def mask_where_lt(self, limit, replace=None, remask=True):
    """A copy with items less than a limit value masked.

    Instead of or in addition to masking the items, the values can be
    replaced. If no items need to be masked, this object is returned
    unchanged.

    Inputs:
        limit           the limiting value.
        replace         a single replacement value or, an object of the same
                        shape and class as this object, containing replacement
                        values. These are inserted into returned object at every
                        masked location. Use None (the default) to leave values
                        unchanged.
        remask          True to include the new mask into the object's mask;
                        False to replace the values but leave them unmasked.
    """

    if self._rank_:
        raise ValueError('mask_where_lt requires item rank zero')

    if isinstance(limit, Qube):
        limit = limit._values_

    return self.mask_where(self._values_ < limit, replace, remask)

#===============================================================================
def mask_where_gt(self, limit, replace=None, remask=True):
    """A copy with items greater than a limit value masked.

    Instead of or in addition to masking the items, the values can be replaced.
    If no items need to be masked, this object is returned unchanged.

    Inputs:
        limit           the limiting value.
        replace         a single replacement value or, an object of the same
                        shape and class as this object, containing replacement
                        values. These are inserted into returned object at every
                        masked location. Use None (the default) to leave values
                        unchanged.
        remask          True to include the new mask into the object's mask;
                        False to replace the values but leave them unmasked.
    """

    if self._rank_:
        raise ValueError('mask_where_gt requires item rank zero')

    if isinstance(limit, Qube):
        limit = limit._values_

    return self.mask_where(self._values_ > limit, replace, remask)

#===============================================================================
def mask_where_between(self, lower, upper, mask_endpoints=False,
                             replace=None, remask=True):
    """A copy with values between two limits masked.

    Instead of or in addition to masking the items, the values can be replaced.
    If no items need to be masked, this object is returned unchanged.

    Inputs:
        lower           the lower limit.
        upper           the upper limit.
        mask_endpoints  True to mask the endpoints, where values are equal to
                        the lower or upper limits; False to exclude the
                        endpoints. Use a tuple of two values to handle the
                        endpoints differently.
        replace         a single replacement value or, an object of the same
                        shape and class as this object, containing replacement
                        values. These are inserted into returned object at every
                        masked location. Use None (the default) to leave values
                        unchanged.
        remask          True to include the new mask into the object's mask;
                        False to replace the values but leave them unmasked.
    """

    if self._rank_:
        raise ValueError('mask_where_between requires item rank zero')

    if isinstance(lower, Qube):
        lower = lower._values_

    if isinstance(upper, Qube):
        upper = upper._values_

    # To minimize the number of array operations, identify the options first
    if not isinstance(mask_endpoints, (tuple, list)):
        mask_endpoints = (mask_endpoints, mask_endpoints)

    if mask_endpoints[0]:               # lower point included in the mask
        op0 = self._values_.__ge__
    else:                               # lower point excluded from the mask
        op0 = self._values_.__gt__

    if mask_endpoints[1]:               # upper point included in the mask
        op1 = self._values_.__le__
    else:                               # upper point excluded from the mask
        op1 = self._values_.__lt__

    mask = op0(lower) & op1(upper)

    return self.mask_where(mask, replace=replace, remask=remask)

#===============================================================================
def mask_where_outside(self, lower, upper, mask_endpoints=False, replace=None,
                             remask=True):
    """A copy with values outside two limits masked.

    Instead of or in addition to masking the items, the values can be replaced.
    If no items need to be masked, this object is returned unchanged.

    Inputs:
        lower           the lower limit.
        upper           the upper limit.
        mask_endpoints  True to mask the endpoints, where values are equal to
                        the lower or upper limits; False to exclude the
                        endpoints. Use a tuple of two values to handle the
                        endpoints differently.
        replace         a single replacement value or, an object of the same
                        shape and class as this object, containing replacement
                        values. These are inserted into returned object at every
                        masked location. Use None (the default) to leave values
                        unchanged.
        remask          True to include the new mask into the object's mask;
                        False to replace the values but leave them unmasked.
    """

    if self._rank_:
        raise ValueError('mask_where_outside requires item rank zero')

    if isinstance(lower, Qube):
        lower = lower._values_

    if isinstance(upper, Qube):
        upper = upper._values_

    # To minimize the number of array operations, identify the options first
    if not isinstance(mask_endpoints, (tuple, list)):
        mask_endpoints = (mask_endpoints, mask_endpoints)

    if mask_endpoints[0]:               # end points are included in the mask
        op0 = self._values_.__le__
    else:                               # end points are excluded from the mask
        op0 = self._values_.__lt__

    if mask_endpoints[1]:               # end points are included in the mask
        op1 = self._values_.__ge__
    else:                               # end points are excluded from the mask
        op1 = self._values_.__gt__

    mask = op0(lower) | op1(upper)

    return self.mask_where(mask, replace=replace, remask=remask)

#===============================================================================
def clip(self, lower, upper, remask=True, inclusive=True):
    """A copy with values clipped to fall within a pair of limits.

    Values below the lower limit become equal to the lower limit; values above
    the upper limit become equal to the upper limit.

    Instead of or in addition to masking the items, the values can be replaced.
    If no items need to be masked, this object is returned unchanged.

    Inputs:
        lower           the lower limit or an object of the same shape and type
                        as this, containing lower limits. None or masked values
                        to ignore.
        upper           the upper limit or an object of the same shape and type
                        as this, containing upper limits. None or masked values
                        to ignore.
        remask          True to include the new mask into the object's mask;
                        False to replace the values but leave them unmasked.
        inclusive       True to leave values that exactly match the upper
                        limit unmasked; False to mask them.
    """

    result = self

    if lower is not None:
        result = result.mask_where(result < lower, lower, remask)

    if upper is not None:
        if inclusive:
            result = result.mask_where(result > upper, upper, remask)
        else:
            result = result.mask_where(result >= upper, upper, remask)

    return result

################################################################################
