################################################################################
# polymath/__init__.py
################################################################################

from polymath.qube       import Qube
from polymath.boolean    import Boolean
from polymath.scalar     import Scalar
from polymath.vector     import Vector
from polymath.vector3    import Vector3
from polymath.pair       import Pair
from polymath.matrix     import Matrix
from polymath.matrix3    import Matrix3
from polymath.quaternion import Quaternion

from polymath.polynomial import Polynomial

from polymath.units      import Units

################################################################################
# Extensions
################################################################################

from polymath.extensions import indexer
Qube.__getitem__        = indexer.__getitem__
Qube.__setitem__        = indexer.__setitem__
Qube._prep_index        = indexer._prep_index
Qube._prep_scalar_index = indexer._prep_scalar_index

from polymath.extensions import item_ops
Qube.extract_numer      = item_ops.extract_numer
Qube.slice_numer        = item_ops.slice_numer
Qube.transpose_numer    = item_ops.transpose_numer
Qube.reshape_numer      = item_ops.reshape_numer
Qube.flatten_numer      = item_ops.flatten_numer
Qube.transpose_denom    = item_ops.transpose_denom
Qube.reshape_denom      = item_ops.reshape_denom
Qube.flatten_denom      = item_ops.flatten_denom
Qube.join_items         = item_ops.join_items
Qube.split_items        = item_ops.split_items
Qube.swap_items         = item_ops.swap_items
Qube.chain              = item_ops.chain

from polymath.extensions import iterator
Qube.__iter__           = iterator.__iter__
Qube.ndenumerate        = iterator.ndenumerate

from polymath.extensions import mask_ops
Qube.mask_where         = mask_ops.mask_where
Qube.mask_where_eq      = mask_ops.mask_where_eq
Qube.mask_where_ne      = mask_ops.mask_where_ne
Qube.mask_where_le      = mask_ops.mask_where_le
Qube.mask_where_ge      = mask_ops.mask_where_ge
Qube.mask_where_lt      = mask_ops.mask_where_lt
Qube.mask_where_gt      = mask_ops.mask_where_gt
Qube.mask_where_between = mask_ops.mask_where_between
Qube.mask_where_outside = mask_ops.mask_where_outside
Qube.clip               = mask_ops.clip
Qube.is_below           = mask_ops.is_below
Qube.is_above           = mask_ops.is_above
Qube.is_outside         = mask_ops.is_outside
Qube.is_inside          = mask_ops.is_inside

from polymath.extensions import math_ops
Qube._mean_or_sum       = math_ops._mean_or_sum
Qube._check_axis        = math_ops._check_axis
Qube._zero_sized_result = math_ops._zero_sized_result
Qube.dot                = math_ops.dot
Qube.norm               = math_ops.norm
Qube.norm_sq            = math_ops.norm_sq
Qube.cross              = math_ops.cross
Qube.outer              = math_ops.outer
Qube.as_diagonal        = math_ops.as_diagonal
Qube.rms                = math_ops.rms

from polymath.extensions import pickler
Qube.__getstate__       = pickler.__getstate__
Qube.__setstate__       = pickler.__setstate__
Qube._encode_floats     = pickler._encode_floats
Qube._decode_floats     = pickler._decode_floats
Qube._encode_ints       = pickler._encode_ints
Qube._decode_ints       = pickler._decode_ints
Qube._encode_bools      = pickler._encode_bools
Qube._decode_bools      = pickler._decode_bools
Qube.set_pickle_digits         = pickler.set_pickle_digits
Qube.set_default_pickle_digits = pickler.set_default_pickle_digits
Qube._check_pickle_digits      = pickler._check_pickle_digits
Qube._pickle_debug             = pickler._pickle_debug

from polymath.extensions import shaper
Qube.reshape            = shaper.reshape
Qube.flatten            = shaper.flatten
Qube.swap_axes          = shaper.swap_axes
Qube.roll_axis          = shaper.roll_axis
Qube.move_axis          = shaper.move_axis
Qube.stack              = shaper.stack

from polymath.extensions import shrinker
Qube.shrink             = shrinker.shrink
Qube.unshrink           = shrinker.unshrink

from polymath.extensions import tvl
Qube.tvl_and            = tvl.tvl_and
Qube.tvl_or             = tvl.tvl_or
Qube.tvl_any            = tvl.tvl_any
Qube.tvl_all            = tvl.tvl_all
Qube.tvl_eq             = tvl.tvl_eq
Qube.tvl_ne             = tvl.tvl_ne
Qube.tvl_lt             = tvl.tvl_lt
Qube.tvl_gt             = tvl.tvl_gt
Qube.tvl_le             = tvl.tvl_le
Qube.tvl_ge             = tvl.tvl_ge
Qube._tvl_op            = tvl._tvl_op

################################################################################
