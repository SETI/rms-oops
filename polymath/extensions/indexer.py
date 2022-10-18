################################################################################
# polymath/extensions/indexer.py: indexing operations
################################################################################

import numpy as np
import numbers
from ..qube import Qube

def __getitem__(self, indx):

    # A single value can only be indexed with True or False
    if not self._shape_:
        if isinstance(indx, tuple) and len(indx) == 1:
            indx = indx[0]
        if Qube.is_one_true(indx):
            return self
        if Qube.is_one_false(indx):
            return self.as_all_masked()

        raise IndexError('too many indices')

    # Interpret and adapt the index
    (pre_index, post_mask, has_ellipsis,
     moved_to_front, array_shape, first_array_loc) = self._prep_index(indx)

    # Apply index to values
    if has_ellipsis and self._rank_:
        vals_index = pre_index + self._rank_ * (slice(None),)
    else:
        vals_index = pre_index
    result_values = self._values_[vals_index]

    # Make sure we have not indexed into the item
    result_vals_shape = np.shape(result_values)
    if len(result_vals_shape) < self._rank_:
        raise IndexError('too many indices')

    # Apply index to mask
    if self._rank_:
        result_shape = result_vals_shape[:-self._rank_]
    else:
        result_shape = result_vals_shape

    if not np.any(post_mask):                   # post-mask is False
        if np.shape(self._mask_):               # self-mask is array
            result_mask = self._mask_[pre_index]
        else:                                   # self-mask is True or False
            result_mask = post_mask or self._mask_
    elif np.all(post_mask):                     # post-mask is True
        result_mask = True
    else:                                       # post-mask is array
        if np.shape(self._mask_):               # self-mask is array
            result_mask = self._mask_[pre_index].copy()
            result_mask[post_mask] = True
        elif self._mask_:                       # self-mask is True
            result_mask = True
        else:                                   # self-mask is False
            if post_mask.shape == result_shape:
                result_mask = post_mask.copy()
            else:
                result_mask = np.zeros(result_shape, dtype=np.bool_)
                axes = len(result_shape) - post_mask.ndim
                new_shape = post_mask.shape + axes * (1,)
                mask = post_mask.reshape(new_shape)
                result_mask[...] = mask

    # Relocate the axes indexed by arrays if necessary
    if moved_to_front:
        before = np.arange(len(array_shape))
        after = before + first_array_loc
        result_values = np.moveaxis(result_values, tuple(before),
                                                   tuple(after))
        if np.shape(result_mask):
            result_mask = np.moveaxis(result_mask, tuple(before),
                                                   tuple(after))

    # Construct the object
    obj = Qube.__new__(type(self))
    obj.__init__(result_values, result_mask, example=self)
    obj._readonly_ = self._readonly_

    # Apply the same indexing to any derivatives
    for (key, deriv) in self._derivs_.items():
        obj.insert_deriv(key, deriv[indx])

    return obj

#===============================================================================
def __setitem__(self, indx, arg):

    self.require_writable()

    # Handle shapeless objects and a single index of True or False
    test_index = None
    if isinstance(indx, (bool, np.bool_)):
        test_index = bool(indx)
    elif (isinstance(indx, tuple)
          and len(indx) == 1
          and isinstance(indx[0], (bool, np.bool_))):
        test_index = bool(indx[0])

    if test_index is not None:
        if not test_index:                      # immediate return on False
            return

        if not self._shape_:                    # immediate copy on True
            arg = self.as_this_type(arg)
            self._values_ = arg._values_
            self._mask_   = arg._mask_
            self._cache_.clear()
            return

    # No other index is allowed for a shapeless object
    if not self._shape_:
        raise IndexError('too many indices')

    # Interpret the index
    (pre_index, post_mask, has_ellipsis,
     moved_to_front, array_shape, first_array_loc) = self._prep_index(indx)

    # If index is fully masked, we're done
    if np.all(post_mask):
        return

    # Convert the argument to this type
    arg = self.as_this_type(arg, recursive=True)

    # Derivatives must match
    for key in self._derivs_:
        if key not in arg._derivs_:
            raise ValueError('missing derivative d_d%s in replacement' %
                             key)

    # Create the values index
    if has_ellipsis and self._rank_:
        vals_index = pre_index + self._rank_ * (slice(None),)
    else:
        vals_index = pre_index

    # Convert this mask to an array if necessary
    if (np.isscalar(self._mask_)            # in this special case, the mask
        and np.isscalar(arg._mask_)         # is fine as is
        and self._mask_ == arg._mask_):
            pass

    elif np.isscalar(self._mask_):
        if self._mask_:
            self._mask_ = np.ones(self._shape_, dtype=np.bool_)
        else:
            self._mask_ = np.zeros(self._shape_, dtype=np.bool_)

    # Create a view of the arg with array-indexed axes moved to front
    if moved_to_front:
        rank = len(array_shape)
        arg_rank = len(arg._shape_)
        if first_array_loc > arg_rank:
            moved_to_front = False          # arg rank does not reach array loc
        if first_array_loc + rank > arg_rank:
            after = np.arange(first_array_loc, arg_rank)
            before = tuple(after - first_array_loc)
            after = tuple(after)
        else:
            before = np.arange(rank)
            after = tuple(before + first_array_loc)
            before = tuple(before)

    if moved_to_front:
        arg_values = np.moveaxis(arg._values_, after, before)
        if np.shape(arg._mask_):
            arg_mask = np.moveaxis(arg._mask_, after, before)
    else:
        arg_values = arg._values_
        arg_mask = arg._mask_

    # Set the new values and mask
    if not np.any(post_mask):               # post-mask is False

        self._values_[vals_index] = arg_values
        if np.shape(self._mask_):
            self._mask_[pre_index] = arg_mask

    else:                                   # post-mask is an array

        # antimask is False wherever the index is masked
        antimask = np.logical_not(post_mask)

        selection = self._values_[vals_index]

        if np.shape(arg_values):
            selection[antimask] = arg_values[antimask]
        else:
            selection[antimask] = arg_values

        self._values_[vals_index] = selection

        if np.shape(self._mask_):
            selection = self._mask_[pre_index]

            if np.shape(arg_mask):
                selection[antimask] = arg_mask[antimask]
            else:
                selection[antimask] = arg_mask

            self._mask_[pre_index] = selection

    self._cache_.clear()

    # Also update the derivatives (ignoring those not in self)
    for (key, self_deriv) in self._derivs_.items():
        self._derivs_[key][indx] = arg._derivs_[key]

    return

#===============================================================================
def _prep_index(self, indx):
  """Prepare the index for this object.

  Input:
      indx              index to prepare.

  Return: A tuple (mask_index, new_mask_index, has_ellipsis)
      pre_index         index to apply to array and mask first.
      post_mask         mask to apply after indexing.
      has_ellipsis      True if the index contains an ellipsis.
      moved_to_front    True for non-consecutive array indices after first,
                        which result in axis-reordering according to NumPy's
                        rules.
      array_shape       array shape resulting from all the array indices.
      first_array_loc   place to relocate the axes associated with an array, if
                        necessary.

  The index is represented by a single object or a tuple of objects.
  Out-of-bounds integer index values are replaced by masked values.
  """

  try:      # catch any error and convert it to an IndexError

    # Convert a non-tuple index to a tuple
    if not isinstance(indx, (tuple,list)):
        indx = (indx,)

    # Convert tuples to Qubes of NumPy arrays;
    # Convert Vectors to individual arrays;
    # Convert all other numeric and boolean items to Qube
    expanded = []
    for item in indx:
        if isinstance(item, (list,tuple)):
            ranks = np.array([len(np.shape(k)) for k in item])
            if np.any(ranks == 0):
                expanded += [Qube(np.array(item))]
            else:
                expanded += [Qube(np.array(x)) for x in item]

        elif isinstance(item, Qube) and item.is_int() and item._rank_ > 0:
            (index_list, mask_vals) = item.as_index_and_mask(purge=False,
                                                             masked=None)
            expanded += [Qube(index_list[0], mask_vals)]
            expanded += [Qube(k) for k in index_list[1:]]

        elif isinstance(item, (bool, np.bool_, numbers.Integral,
                               np.ndarray)):
            expanded += [Qube(item)]
        else:
            expanded += [item]

    # At this point, every item in the index list is a slice, Ellipsis, None, or
    # a Qube subclass. Ever item will consume exactly one axis of the object,
    # except for multidimensional boolean arrays, ellipses, and None.

    # Identify the axis of this object consumed by each index item.
    # Initially, treat an ellipsis as consuming zero axes.
    # One item in inlocs for every item in expanded.
    inlocs = []
    ellipsis_k = -1
    inloc = 0
    for k,item in enumerate(expanded):
        inlocs += [inloc]

        if type(item) == type(Ellipsis):
            if ellipsis_k >= 0:
                raise IndexError("an index can only have a single " +
                                 "ellipsis ('...')")
            ellipsis_k = k

        elif isinstance(item, Qube) and item._shape_ and item.is_bool():
            inloc += len(item._shape_)

        elif item is not None:
            inloc += 1

    # Each value in inlocs is the
    has_ellipsis = ellipsis_k >= 0
    if has_ellipsis:            # if ellipsis found
        correction = len(self._shape_) - inloc
        if correction < 0:
            raise IndexError('too many indices for array')
        for k in range(ellipsis_k + 1, len(inlocs)):
            inlocs[k] += correction

    # Process the components of the index tuple...
    pre_index = []              # Numpy index to apply to the object
    post_mask = False           # Mask to apply after indexing, to account for
                                # masked index values.

    # Keep track of additional info about array indices
    array_inlocs = []           # inloc of each array index
    array_lengths = []          # length of axis consumed
    array_shapes = []           # shape of object returned by each array index.

    for k,item in enumerate(expanded):
        inloc = inlocs[k]

        # None consumes to input axis
        if item is None:
            pre_index += [item]
            continue

        axis_length = self._shape_[inloc]

        # Handle Qube subclasses
        if isinstance(item, Qube):

          if item.is_float():
                raise IndexError('floating-point indexing is not permitted')

          # A Boolean index collapses one or more axes down to one, where the
          # new number of elements is equal to the number of elements True or
          # masked. After the index is applied, the entries corresponding to
          # masked index values will be masked. If no values are True or masked,
          # the axis collapses down to size zero.
          if item.is_bool():

            # Boolean array
            # Consumes one index for each array dimension; returns one axis with
            # length equal to the number of occurrences of True or masked;
            # masked items leave masked elements.
            if item._shape_:

                # Validate shape
                item_shape = item._shape_
                for k,item_length in enumerate(item_shape):
                  if self._shape_[inloc + k] != item_length:
                    raise IndexError((
                        'boolean index did not match indexed array along ' +
                        'dimension %d; dimension is %d but corresponding ' +
                        'boolean dimension is %d') % (inloc + k,
                            self._shape_[inloc + k], item_length))

                # Update index and mask
                index = item._values_ | item._mask_     # True or masked
                pre_index += [index]

                if np.shape(item._mask_):               # mask is an array
                    post_mask = post_mask | item._mask_[index]

                elif item._mask_:                       # mask is True
                    post_mask = True

                array_inlocs += [inloc]
                array_lengths += list(item_shape)
                array_shapes += [(np.sum(index),)]

            # One boolean item
            else:

                # One masked item
                if item._mask_:
                    pre_index += [slice(0,1)]           # unit-sized axis
                    post_mask = True

                # One True item
                elif item._values_:
                    pre_index += [slice(None)]

                # One False item
                else:
                    pre_index += [slice(0,0)]           # zero-sized axis

          # Scalar index behaves like a NumPy ndarray index, except masked index
          # values yield masked array values
          elif item._rank_ == 0:

            # Scalar array
            # Consumes one axis; returns the number of axes in this array;
            # masked items leave masked elements.
            if item._shape_:
                index_vals = item._values_
                mask_vals = item._mask_

                # Find indices out of bounds
                out_of_bounds_mask = ((index_vals >= axis_length) |
                                      (index_vals < -axis_length))
                any_out_of_bounds = np.any(out_of_bounds_mask)
                if any_out_of_bounds:
                    mask_vals = mask_vals | out_of_bounds_mask
                    any_masked = True
                else:
                    any_masked = np.any(mask_vals)

                # Find an unused index value, if any
                index_vals = index_vals % axis_length
                if np.shape(mask_vals):
                    antimask = np.logical_not(mask_vals)
                    unused_set = (set(range(axis_length)) -
                                  set(index_vals[antimask]))
                elif mask_vals:
                    unused_set = ()
                else:
                    unused_set = (set(range(axis_length))
                                  - set(index_vals.ravel()))

                if unused_set:
                    unused_index_value = unused_set.pop()
                else:
                    unused_index_value = -1             # -1 = no unused element

                # Apply mask to index; update masked values
                if any_masked:
                    index_vals = index_vals.copy()
                    index_vals[mask_vals] = unused_index_value

                pre_index += [index_vals.astype(np.intp)]

                if np.shape(mask_vals):                 # mask is also an array
                    post_mask = post_mask | mask_vals
                elif mask_vals:                         # index is fully masked
                    post_mask = True

                array_inlocs += [inloc]
                array_lengths += [axis_length]
                array_shapes += [item._shape_]

            # One scalar item
            else:

                # Compare to allowed range
                index_val = item._values_
                mask_val = item._mask_

                if not mask_val:
                    if index_val < 0:
                        index_val += axis_length

                    if index_val < 0 or index_val >= axis_length:
                        mask_val = True

                # One masked item
                # Remove this axis and mark everything masked
                if mask_val:
                    pre_index += [0]                    # use 0 on a masked axis
                    post_mask = True

                # One unmasked item
                else:
                    pre_index += [index_val % axis_length]

        # Handle any other index element the NumPy way, with no masking
        elif isinstance(item, (slice, type(Ellipsis))):
            pre_index += [item]

        else:
            raise IndexError('invalid index type: ' + str(type(item)))

    # Get the shape of the array indices
    array_shape = Qube.broadcasted_shape(*array_shapes)

    # According to NumPy indexing rules, if there are non-consecutive array
    # array indices, the array indices are moved to the front of the axis
    # order in the result!
    if array_inlocs:
        first_array_loc = array_inlocs[0]
        diffs = np.diff(array_inlocs)
        moved_to_front = np.any(diffs > 1) and first_array_loc > 0
    else:
        first_array_loc = 0
        moved_to_front = False

    # Simplify the post_mask if possible
    if not all(array_shape):        # mask doesn't matter if size is zero
        post_mask = False
    elif np.all(post_mask):
        post_mask = True

    return (tuple(pre_index), post_mask, has_ellipsis,
            moved_to_front, array_shape, first_array_loc)

  except Exception as e:
    raise IndexError(e)

################################################################################
