################################################################################
# polymath/__init__.py
################################################################################

from .qube       import Qube
from .boolean    import Boolean
from .scalar     import Scalar
from .vector     import Vector
from .vector3    import Vector3
from .pair       import Pair
from .matrix     import Matrix
from .matrix3    import Matrix3
from .quaternion import Quaternion

from .polynomial import Polynomial

from .units      import Units

################################################################################
# Extensions
################################################################################

from polymath.extensions import indexer
Qube.__getitem__        = indexer.__getitem__
Qube.__setitem__        = indexer.__setitem__
Qube._prep_index        = indexer._prep_index

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

from polymath.extensions import math_ops
Qube._mean              = math_ops._mean
Qube._sum               = math_ops._sum
Qube.dot                = math_ops.dot
Qube.norm               = math_ops.norm
Qube.norm_sq            = math_ops.norm_sq
Qube.cross              = math_ops.cross
Qube.outer              = math_ops.outer
Qube.as_diagonal        = math_ops.as_diagonal
Qube.rms                = math_ops.rms

from polymath.extensions import pickler
Qube.set_pickle_digits  = pickler.set_pickle_digits
Qube.__getstate__       = pickler.__getstate__
Qube.__setstate__       = pickler.__setstate__
Qube._interpret_digits  = pickler._interpret_digits
Qube._encode_one_float  = pickler._encode_one_float
Qube._encode_float_array= pickler._encode_float_array
Qube._decode_one_float  = pickler._decode_one_float
Qube._decode_float_array= pickler._decode_float_array

from polymath.extensions import shaper
Qube.reshape            = shaper.reshape
Qube.flatten            = shaper.flatten
Qube.swap_axes          = shaper.swap_axes
Qube.roll_axis          = shaper.roll_axis
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
