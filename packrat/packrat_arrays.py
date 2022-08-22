import numpy as np
import base64
from xml.sax.saxutils import escape, unescape

from packrat_entities import ENTITIES, UNENTITIES

NATIVE_BYTEORDER = '<' if np.dtype('<f8').isnative else '>'
FLOAT64 = NATIVE_BYTEORDER + 'f8'
FLOAT32 = NATIVE_BYTEORDER + 'f4'

INT8  = '|i1'
INT16 = NATIVE_BYTEORDER + 'i2'
INT32 = NATIVE_BYTEORDER + 'i4'
INT64 = NATIVE_BYTEORDER + 'i8'

UINT8  = '|u1'
UINT16 = NATIVE_BYTEORDER + 'u2'
UINT32 = NATIVE_BYTEORDER + 'u4'
UINT64 = NATIVE_BYTEORDER + 'u8'

NATIVE_BYTEORDER_ENC = '[' if np.dtype('<f8').isnative else ']'
FLOAT64_ENC = NATIVE_BYTEORDER_ENC + 'f8'
FLOAT32_ENC = NATIVE_BYTEORDER_ENC + 'f4'

INT8_ENC  = '|i1'
INT16_ENC = NATIVE_BYTEORDER_ENC + 'i2'
INT32_ENC = NATIVE_BYTEORDER_ENC + 'i4'
INT64_ENC = NATIVE_BYTEORDER_ENC + 'i8'

UINT8_ENC  = '|u1'
UINT16_ENC = NATIVE_BYTEORDER_ENC + 'u2'
UINT32_ENC = NATIVE_BYTEORDER_ENC + 'u4'
UINT64_ENC = NATIVE_BYTEORDER_ENC + 'u8'

################################################################################
# Top-level functions
################################################################################

#===============================================================================
# encode_array
#===============================================================================
def encode_array(values, b64_savings=1000, npz_savings=10000, **params):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Encode an array; generate the attribute list.

    Input:
        value       the NumPy ndarray to encode.
        b64_savings number of bytes to be saved before base64 encoding will be
                    considered. Base64 encoding is a relatively compact way to
                    store a binary value inside XML.
        npz_savings number of bytes to be saved before npz encoding will be
                    considered. This encoding stores the data in a separate file
                    using np.save_compressed().

        params      a dictionary of additional input parameters to define the
                    precision used for arrays of floats:
            single      True to convert to single precision.
            absolute    the upper limit to the absolute roundoff error; None to
                        ignore.
            relative    the upper limit to the relative (fractional) roundoff
                        error; None to ignore.
            worstcase   True to enforce the stronger of the absolute and
                        relative errors if both are specified; False to enforce
                        only the weaker of the two error limits.

    Return:         a tuple (xml_value, npz_value, attr_list).
        xml_value   the string to include in the XML; blank for none. Quotes and
                    apostrophes have been escaped.
        npz_value   the array to store an the associated .npz file; None to
                    skip.
        attr_list   a list of attributes (name,value), where the value is always
                    a string. Quotes and apostrophes have been escaped.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    dtype = values.dtype
    kind = dtype.kind

    if kind == 'f':
        return encode_float_array(values, b64_savings, npz_savings, **params)
    elif kind == 'b':
        return encode_bool_array(values, b64_savings, npz_savings)
    elif kind in 'ui':
        return encode_int_array(values, b64_savings, npz_savings)
    elif kind in 'Sc':
        return encode_string_array(values, b64_savings, npz_savings)
    else:
        raise ValueError('unsupported encoding for array dtype "%s"' % dtype)
#===============================================================================



#===============================================================================
# decode_array
#===============================================================================
def decode_array(value, attr):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Decode an array.

    Inputs:
        value       the is the XML value string if one is present in the file;
                    However, if attr['encoding'] = 'npz', it is the array as
                    read from the associated .npz file.
        attr        a dictionary of attributes for this XML node.

    Return:         the decoded array.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    dtype = decode_dtype(attr['dtype'])
    kind = dtype.kind

    if kind == 'f':
        return decode_float_array(value, attr)
    elif kind == 'b':
        return decode_bool_array(value, attr)
    elif kind in 'ui':
        return decode_int_array(value, attr)
    elif kind in 'Sc':
        return decode_string_array(value, attr)
    else:
        raise ValueError('unsupported decoding for array dtype "%s"' % dtype)
#===============================================================================



################################################################################
# Attribute list
################################################################################

#===============================================================================
# first_attributes
#===============================================================================
def first_attributes(values):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Start a list of attributes for a NumPy array.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #--------------------------------
    # Start with shape and dtype
    #--------------------------------
    dtype = values.dtype
    kind = dtype.kind

    attr_list = [('shape', str(values.shape).replace(' ','')),
                 ('dtype', encode_dtype(values.dtype))]

    #----------------------------------------------------
    # Include first, second and third for validation
    #----------------------------------------------------
    raveled = values.ravel()

    NAMES = ['first', 'second', 'third']
    for k in range(len(NAMES)):
        if k >= len(raveled): break

        if kind in 'Sc':
            attr_list.append((NAMES[k], escape(raveled[k], ENTITIES)))
        else:
            attr_list.append((NAMES[k], repr(raveled[k])))

    #----------------------------------------------------------------
    # For numbers, include the min and max for further validation
    #----------------------------------------------------------------
    if kind not in 'Scb' and raveled.size:
        attr_list.append(('min', repr(np.min(raveled))))
        attr_list.append(('max', repr(np.max(raveled))))

    return attr_list
#===============================================================================



################################################################################
# Routines to maniupulate dtypes
################################################################################

#===============================================================================
# convert_dtype
#===============================================================================
def convert_dtype(dtype, dtype2):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Return the name of an another dtype, retaining the byteorder.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    dtype = np.dtype(dtype)
    dtype2 = np.dtype(dtype2)

    #------------------------------------------------------------
    # Always express dtype using byteorder + kind + itemsize
    #------------------------------------------------------------
    byteorder = dtype.byteorder
    if byteorder == '=':
        byteorder = NATIVE_BYTEORDER

    #----------------------------------------------------------
    # Convert dtype name if necessary, preserving byteorder
    #----------------------------------------------------------
    kind = dtype2.kind
    itemsize = dtype2.itemsize

    if itemsize == 1:
        byteorder = '|'

    if itemsize > 1 and byteorder == '|':
        byteorder = NATIVE_BYTEORDER

    return byteorder + kind + str(itemsize)
#===============================================================================



#===============================================================================
# matches_dtype
#===============================================================================
def matches_dtype(dtype, dtype2):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Returns true if the dtypes are equivalent exept for byte order.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    dtype = np.dtype(dtype)
    dtype2 = np.dtype(dtype2)

    return dtype.kind == dtype2.kind and dtype.itemsize == dtype2.itemsize
#===============================================================================



################################################################################
# Routines to make dtype strings more XML-friendly
################################################################################

#===============================================================================
# encode_dtype
#===============================================================================
def encode_dtype(dtype):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Encode dtype for use in the XML file.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    dtype = np.dtype(dtype)

    #-----------------------------------------------------------
    # Always express dtype using byteorder + kind + itemsize
    #-----------------------------------------------------------
    byteorder = dtype.byteorder
    if byteorder == '=':
        byteorder = NATIVE_BYTEORDER

    dtype = byteorder + dtype.kind + str(dtype.itemsize)

    #---------------------------------------------------------------------------
    # Within the XML, use [] to improve readability, because <> must be escaped
    #---------------------------------------------------------------------------
    return dtype.replace('<','[').replace('>',']')
#===============================================================================



#===============================================================================
# decode_dtype
#===============================================================================
def decode_dtype(dtype_str):
    return np.dtype(dtype_str.replace('[','<').replace(']','>'))
#===============================================================================



################################################################################
# Integer array encoding/decoding
################################################################################

#===============================================================================
# encode_int_array
#===============================================================================
def encode_int_array(values, b64_savings, npz_savings):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Encode an integer array; generate the attribute list.

    Return a tuple(text_value, npz_value, attribute_list).
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #----------------------------------------
    # Generate the default attribute list
    #----------------------------------------
    attr_list = first_attributes(values)

    values = values.ravel()

    #-----------------------------
    # Handle zero-size object
    #-----------------------------
    if values.size == 0:
        attr_list.append(('encoding', 'text'))
        return ('', None, attr_list)

    #-------------------------------------
    # Check sizes of encoding options
    #-------------------------------------
    byte_size = values.itemsize * values.size
    xml_bytes = values.size * (len(str(np.median(values))) + 1)

    #----------------------------------------------------
    # Find optimal binary encoding and optional offset
    #----------------------------------------------------
    (compressed, offset) = shrink_int_array(values)
    npz_bytes = 0.7 * compressed.size * compressed.itemsize
            # 0.7 assumes zip compression
    b64_bytes = 1.3 * compressed.size * compressed.itemsize

    #------------------------------------------------------------------
    # Hold the data for the npz file if the savings is large enough
    #------------------------------------------------------------------
    if min(xml_bytes, b64_bytes) > npz_bytes + npz_savings:
        if offset:
            attr_list.append(('offset', str(offset)))
        attr_list += [('storage_dtype', encode_dtype(compressed.dtype)),
                      ('encoding', 'npz')]

        return ('', compressed, attr_list)

    #---------------------------------------
    # For something short, return text
    #---------------------------------------
    if xml_bytes < b64_bytes + b64_savings:
        attr_list.append(('encoding', 'text'))
        string = ','.join([repr(v) for v in values])
        return (string, None, attr_list)

    #------------------------------
    # Otherwise, base64 encode
    #------------------------------
    if offset:
        attr_list.append(('offset', offset))

    attr_list += [('storage_dtype', encode_dtype(compressed.dtype)),
                  ('encoding', 'base64')]

    return (base64.b64encode(compressed.tobytes()), None, attr_list)
#===============================================================================



#===============================================================================
# decode_int_array
#===============================================================================
def decode_int_array(value, attr):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Decode an integer array.

    Input parameter 'value' is either the string value of this node or else the
    array extracted from the npz file.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    encoding = attr['encoding']
    shape  = eval(attr['shape'])
    offset = eval(attr.get('offset', '0'))
    dtype  = decode_dtype(attr['dtype'])

    if np.prod(shape) == 0:
        return np.arange(1).astype(dtype)[:0].reshape(shape)

    if encoding == 'text':
        decoded = np.array(eval(value), dtype=dtype)

    elif encoding == 'base64':
        decoded = base64.b64decode(value)
        decoded = np.fromstring(decoded,
                                dtype=decode_dtype(attr['storage_dtype']))
        decoded = decoded.astype(dtype)
        if offset: decoded += offset

    else:
        decoded = value.astype(dtype)
        if offset: decoded += offset

    return decoded.reshape(shape)
#===============================================================================



#===============================================================================
# shrink_int_array
#===============================================================================
def shrink_int_array(values):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Encode an integer array into the smallest space possible

    Return a tuple(compressed_array, offset).
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if values.size == 0: return (values, 0)

    values = values.ravel()
    dtype = values.dtype

    amin = np.min(values)
    amax = np.max(values)
    diff = amax - amin

    if amin >= -128 and amax < 128:
        offset = 0
        compressed = values.astype(convert_dtype(dtype, INT8))

    elif amin >= 0 and amax < 256:
        offset = 0
        compressed = values.astype(convert_dtype(dtype, UINT8))

    elif diff < 256:
        offset = amin
        compressed = (values - amin).astype(convert_dtype(dtype, UINT8))

    elif amin >= -2**15 and amax < 2**15 :
        offset = 0
        compressed = values.astype(convert_dtype(dtype, INT16))

    elif amin >= 0 and amax < 2**16 :
        offset = 0
        compressed = values.astype(UINT16)

    elif diff < 2**16:
        offset = amin
        compressed = (values - amin).astype(convert_dtype(dtype, UINT16))

    elif amin >= -2**31 and amax < 2**31:
        offset = 0
        compressed = values.astype(INT32)

    elif amin >= 0 and amax < 2**32:
        offset = 0
        compressed = values.astype(UINT32)

    elif diff < 2**32:
        offset = amin
        compressed = (values - amin).astype(convert_dtype(dtype, UINT32))

    else:
        offset = 0
        compressed = values

    return (compressed, offset)
#===============================================================================



################################################################################
# Boolean array encoding/decoding
################################################################################

#===============================================================================
# encode_bool_array
#===============================================================================
def encode_bool_array(values, b64_savings, npz_savings):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Encode an boolean array; generate the attribute list.

    Return a tuple(text_value, npz_value, attribute_list).
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #-----------------------------------------
    # Generate the default attribute list
    #-----------------------------------------
    attr_list = first_attributes(values)

    values = values.ravel()

    #----------------------------
    # Handle zero-size object
    #----------------------------
    if values.size == 0:
        attr_list.append(('encoding', 'text'))
        return ('', None, attr_list)

    #-------------------------------------
    # Check sizes of encoding options
    #-------------------------------------
    byte_size = values.itemsize * values.size
    text_bytes = values.size

    #---------------------------------
    # Find optimal binary encoding
    #---------------------------------

    #- - - - - - - - - - - - - - - - - - - - - - - -
    # Count the number of True and False values
    #- - - - - - - - - - - - - - - - - - - - - - - -
    total_items = len(values)
    true_items = np.count_nonzero(values)
    false_items = total_items - true_items

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Method 'all_equal' works only if all mask values are the same
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if true_items == 0 or false_items == 0:
        attr_list += [('method', 'all_equal'), ('encoding', 'text')]

        if true_items == 0:
            return ('False', None, attr_list)
        else:
            return ('True', None, attr_list)

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # method 'list_false' and 'list_true' explicitly list the locations of
    # False or True values
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if total_items < 256:
        index_bytes = 1
        index_dtype = UINT8
    elif total_items < 2**16:
        index_bytes = 2
        index_dtype = UINT16
    else:
        index_bytes = 4
        index_dtype = UINT32

    if true_items < false_items:
        list_value  = True
        list_method = 'list_true'
        list_items  = true_items
    else:
        list_value  = False
        list_method = 'list_false'
        list_items  = false_items

    list_bytes = list_items * index_bytes

    #- - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Method 'packbits' combines eight values per byte
    #- - - - - - - - - - - - - - - - - - - - - - - - - - -
    packbits_bytes = (len(values) + 7) // 8

    #- - - - - - - - - - - - - - - - - - - 
    # Select the best npz/base64 option
    #- - - - - - - - - - - - - - - - - - - 
    if packbits_bytes <= list_bytes:
        compressed_method = 'packbits'
        compressed_bytes = packbits_bytes
    else:
        compressed_method = list_method
        compressed_bytes = list_bytes

    npz_bytes = 0.8 * compressed_bytes
    b64_bytes = 1.3 * compressed_bytes

    #- - - - - - - - - - - - - - - - 
    # Select the best text option
    #- - - - - - - - - - - - - - - - 
    list_text_bytes = (len(str(total_items)) + 1) * list_items
    if list_text_bytes < text_bytes:
        text_bytes = list_text_bytes
        text_method = list_method
    else:
        text_method = 'TF'

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Hold the data for the npz file if the savings is large enough
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    if min(text_bytes, b64_bytes) > npz_bytes + npz_savings:
        attr_list += [('method', compressed_method)]

        if compressed_method == 'packbits':
            compressed = np.packbits(values)
        else:
            compressed = np.where(values == list_value)[0].astype(index_dtype)
            attr_list += [('index_dtype', encode_dtype(index_dtype))]

        attr_list += [('encoding', 'npz')]
        return ('', compressed, attr_list)  # XML value, npz value, attributes

    #- - - - - - - - - - - - - - - - - - - 
    # For something short, return text
    #- - - - - - - - - - - - - - - - - - - 
    if text_bytes < b64_bytes + b64_savings:
        if text_method == 'TF':
            string = ''.join(['FT'[v] for v in values])
        else:
            indices = np.where(values == list_value)[0]
            string = ','.join([str(i) for i in indices])

        attr_list += [('method', text_method), ('encoding', 'text')]
        return (string, None, attr_list)

    #- - - - - - - - - - - - - - - 
    # Otherwise, base64 encode
    #- - - - - - - - - - - - - - - 
    attr_list += [('method', compressed_method)]

    if compressed_method == 'packbits':
        compressed = np.packbits(values)
    else:
        compressed = np.where(values == list_value)[0].astype(index_dtype)
        attr_list += [('index_dtype', encode_dtype(index_dtype))]

    attr_list += [('encoding', 'base64')]
    return (base64.b64encode(compressed.tobytes()), None, attr_list)
#===============================================================================



#===============================================================================
# decode_bool_array
#===============================================================================
def decode_bool_array(value, attr):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Decode a boolean array.

    Input parameter 'value' is either the string value of this node or else the
    array extracted from the npz file.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    shape = eval(attr['shape'])
    size = np.prod(shape)

    if size == 0:
        return np.array([False])[:0].reshape(shape)

    encoding = attr['encoding']
    method = attr['method']

    #---------------------------
    # Hande all_equal case
    #---------------------------
    if attr['method'] == 'all_equal':
        if value == 'True':
            return np.ones(shape, dtype='bool')
        else:
            return np.zeros(shape, dtype='bool')

    #-----------------------------
    # Handle TF text encoding
    #-----------------------------
    if method == 'TF':
        return np.array([(tf == 'T') for tf in value]).reshape(shape)

    #--------------------------
    # Convert text to array
    #--------------------------
    if encoding == 'text':
        value = np.array(eval(value))

    #-----------------------------
    # Convert base64 to array
    #-----------------------------
    elif encoding == 'base64':
        decoded = base64.b64decode(value)
        dtype = decode_dtype(attr.get('index_dtype', UINT8))
        value = np.fromstring(decoded, dtype=dtype)

    #--------------------
    # Handle packbits
    #--------------------
    if method == 'packbits':
        decoded = np.unpackbits(value).astype('bool')[:size]

    #----------------------
    # Handle list_false
    #----------------------
    elif method == 'list_false':
        decoded = np.ones(shape, dtype='bool').ravel()
        decoded[(value,)] = False

    #--------------------
    # Handle list_true
    #--------------------
    else:
        decoded = np.zeros(shape, dtype='bool').ravel()
        decoded[(value,)] = True

    return decoded.reshape(shape)
#===============================================================================



################################################################################
# Float array encoding/decoding
################################################################################

#===============================================================================
# encode_float_array
#===============================================================================
def encode_float_array(values, b64_savings, npz_savings, single=False,
                       absolute=None, relative=None, worstcase=True, **ignore):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Encode a float array; generate the attribute list.

    Return a tuple(text_value, npz_value, attribute_list).

    Input parameters:
        single      True to convert to single precision.
        absolute    the upper limit to the absolute roundoff error; None to
                    ignore.
        relative    the upper limit to the relative (fractional) roundoff error;
                    None to ignore.
        worstcase   True to enforce the stronger of the absolute and relative
                    errors if both are specified; False to enforce only the
                    weaker of the two error limits.

    Return a tuple(text_value, npz_value, attribute_list).
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #------------------------------------------
    # Generate the default attribute list
    #------------------------------------------
    attr_list = first_attributes(values)

    values = values.ravel()
    dtype = values.dtype

    #-----------------------------
    # Handle zero-size object
    #-----------------------------
    if values.size == 0:
        attr_list.append(('encoding', 'text'))
        return ('', None, attr_list)

    #-----------------------------
    # Study the dynamic range
    #-----------------------------
    amin = np.min(values)
    amax = np.max(values)

    abs_values = np.abs(values)
    amin_abs = np.min(abs_values)
    amax_abs = np.max(abs_values)

    storage_dtype = values.dtype
    storage_bytes = values.dtype.itemsize

    #---------------------------------------------
    # Convert to single precision if warranted
    #---------------------------------------------
    if single:
        storage_dtype = convert_dtype(dtype, FLOAT32)
        storage_bytes = 4

    #-----------------------------------------
    # Investigate integer encoding options
    #-----------------------------------------
    absolute_dominates = False
    if relative is None:
        if absolute is None:
            test_absolute = 0.
        else:
            test_absolute = absolute
            absolute_dominates = True
    else:
        if absolute is None:
            test_absolute = amin_abs * relative
        elif worstcase:
            test_absolute = min(absolute, amin_abs * relative)
            absolute_dominates = (absolute < amin_abs * relative)
        else:
            test_absolute = max(absolute, amin_abs * relative)
            absolute_dominates = (absolute > amin_abs * relative)

    if test_absolute:
        discrete_values = (amax - amin) / test_absolute
        if discrete_values < 256:
            storage_dtype = convert_dtype(dtype, UINT8)
            storage_bytes = 1
        elif discrete_values < 2**16:
            storage_dtype = convert_dtype(dtype, UINT16)
            storage_bytes = 2
        elif discrete_values < 2**32 and storage_bytes > 4:
            storage_dtype = convert_dtype(dtype, UINT32)
            storage_bytes = 4

    #--------------------------------
    # Investigate float32 option
    #--------------------------------
    if storage_bytes >= 4 and relative and not absolute_dominates:
        if relative >= 2.**(-24):
            storage_dtype = convert_dtype(dtype, FLOAT32)
            storage_bytes = 4

    #-------------------------------------
    # Check sizes of encoding options
    #-------------------------------------
    npz_bytes = 0.8 * values.size * storage_bytes
    b64_bytes = 1.3 * values.size * storage_bytes

    if matches_dtype(storage_dtype, FLOAT64):
        text_bytes = values.size * 24
    else:
        text_bytes = values.size * 16

    #---------------------------
    # Update the attributes
    #---------------------------
    encoding_list = [('storage_dtype', encode_dtype(storage_dtype))]
    if absolute is not None:
        encoding_list += [('absolute_precision', str(absolute))]
    if relative is not None:
        encoding_list += [('relative_precision', str(relative))]

    #------------------------------------------------------------------
    # Hold the data for the npz file if the savings is large enough
    #------------------------------------------------------------------
    if min(text_bytes, b64_bytes) > npz_bytes + npz_savings:
        attr_list += encoding_list + [('encoding', 'npz')]

        if matches_dtype(storage_dtype, FLOAT64):
            storage_values = values
        elif matches_dtype(storage_dtype, FLOAT32):
            storage_values = values.astype(storage_dtype)
        else:
            floats = (values - amin) / (amax - amin)
            storage_values = (floats *
                              (2.**(8*storage_bytes) - 1)).astype(storage_dtype)

        return ('', storage_values, attr_list)

    #--------------------------------------
    # For something short, return text
    #--------------------------------------
    if text_bytes < b64_bytes + b64_savings:
        if not matches_dtype(storage_dtype, FLOAT64):
            storage_dtype = convert_dtype(storage_dtype, FLOAT32)
            attr_list.append(('storage_dtype', encode_dtype(storage_dtype)))

        attr_list.append(('encoding', 'text'))
        string = ','.join([repr(v) for v in values.astype(storage_dtype)])

        return (string, None, attr_list)

    #------------------------------
    # Otherwise, base64 encode
    #------------------------------
    attr_list += encoding_list + [('encoding', 'base64')]

    if matches_dtype(storage_dtype, FLOAT64):
        storage_values = values
    elif matches_dtype(storage_dtype, FLOAT32):
        storage_values = values.astype(storage_dtype)
    else:
        floats = (values - amin) / (amax - amin)
        storage_values = (floats *
                          (2.**(8*storage_bytes) - 1)).astype(storage_dtype)

    return (base64.b64encode(storage_values.tobytes()), None, attr_list)
#===============================================================================



#===============================================================================
# decode_float_array
#===============================================================================
def decode_float_array(value, attr):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Decode a float array.

    Input parameter 'value' is either the string value of this node or else the
    array extracted from the npz file.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    encoding = attr['encoding']
    shape    = eval(attr['shape'])
    dtype    = decode_dtype(attr['dtype'])

    if np.prod(shape) == 0:
        return np.arange(1.).astype(dtype)[:0].reshape(shape)

    storage_dtype = decode_dtype(attr.get('storage_dtype', str(dtype)))
    amin = eval(attr['min'])
    amax = eval(attr['max'])

    if encoding == 'text':
        decoded = np.array(eval(value), dtype=dtype)

    elif encoding == 'base64':
        decoded = base64.b64decode(value)
        decoded = np.fromstring(decoded, dtype=storage_dtype)
        decoded = decoded.astype(dtype)

    else:
        decoded = value.astype(dtype)

    if storage_dtype.kind == 'u':
        steps = (2.**(storage_dtype.itemsize * 8) - 1)
        decoded = amin + (amax - amin) * decoded/steps

    return decoded.reshape(shape)
#===============================================================================



################################################################################
# Character and string array encoding/decoding
################################################################################

#===============================================================================
# encode_string_array
#===============================================================================
def encode_string_array(values, b64_savings, npz_savings):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Encode a string and character array; generate the attribute list.

    Return a tuple(text_value, npz_value, attribute_list).
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #-----------------------------------------
    # Generate the default attribute list
    #-----------------------------------------
    attr_list = first_attributes(values)

    values = values.ravel()

    #----------------------------
    # Handle zero-size object
    #----------------------------
    if values.size == 0:
        attr_list.append(('encoding', 'text'))
        return ('', None, attr_list)

    #--------------------
    # Estimate sizes
    #--------------------
    kind = values.dtype.kind
    if values.itemsize == 1:
        raw_bytes = values.size
        xml_bytes = raw_bytes
        b64_bytes = 1.3 * raw_bytes
        npz_bytes = 0.7 * raw_bytes
    else:
        raw_bytes = values.size * values.itemsize
        xml_bytes = values.size * (values.itemsize + 3)
        b64_bytes = 1.3 * raw_bytes
        npz_bytes = 0.7 * raw_bytes

    #------------------------------------------------------------------
    # Hold the data for the npz file if the savings is large enough
    #------------------------------------------------------------------
    if min(xml_bytes, b64_bytes) > npz_bytes + npz_savings:
        attr_list.append(('encoding', 'npz'))
        return ('', values, attr_list)

    #-------------------------------------
    # For something short, return text
    #-------------------------------------
    if xml_bytes < b64_bytes + b64_savings:
        attr_list.append(('encoding', 'text'))

        if values.itemsize == 1:
            content = escape(''.join([c for c in values]))
        else:
            content = ','.join(['"' + escape(s,ENTITIES) + '"' for s in values])

        return (content, None, attr_list)

    #------------------------------------------------------------------------
    # Otherwise, base64 encode
    #------------------------------------------------------------------------
    attr_list += encoding_list + [('encoding', 'base64')]
    return (base64.b64encode(values.tobytes()), None, attr_list)
#===============================================================================



#===============================================================================
# decode_string_array
#===============================================================================
def decode_string_array(value, attr):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Decode a string or character array.

    Input parameter 'value' is either the string value of this node or else the
    array extracted from the npz file.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    encoding = attr['encoding']
    shape  = eval(attr['shape'])
    dtype  = decode_dtype(attr['dtype'])
    kind = dtype.kind

    if np.prod(shape) == 0:
        return np.array([], dtype=dtype)[:0].reshape(shape)

    if encoding == 'text':
        if dtype.itemsize == 1:
            values = list(unescape(value, UNENTITIES))
            decoded = np.array(values, dtype=dtype)
        else:
            values = value.split('","')
            values[0] = values[0][1:]
            values[-1] = values[-1][:-1]
            values = [unescape(v, UNENTITIES) for v in values]
            decoded = np.array(values, dtype=dtype)

    elif encoding == 'base64':
        decoded = base64.b64decode(value)
        decoded = np.fromstring(decoded, dtype=dtype)

    else:
        decoded = value

    return decoded.reshape(shape)
#===============================================================================



################################################################################
# Unit tests
################################################################################

import unittest

#*******************************************************************************
# Test_Packrat_arrays
#*******************************************************************************
class Test_Packrat_arrays(unittest.TestCase):

  #=============================================================================
  # runTest
  #=============================================================================
  def runTest(self):

    #-------------------------
    # Integer array tests
    #-------------------------

    #===========================================================================
    # test_int_array
    #===========================================================================
    def test_int_array(ints, b64_savings, npz_savings):
        ints = np.array(ints)
        (string, values, attr) = encode_int_array(ints, b64_savings,
                                                        npz_savings)
        attr = dict(attr)

        ravel = ints.ravel()
        if ravel.size > 0:
            self.assertEqual(eval(attr['first']),  ravel[0])

        if ravel.size > 1:
            self.assertEqual(eval(attr['second']), ravel[1])

        if ravel.size > 2:
            self.assertEqual(eval(attr['third']),  ravel[2])

        if ravel.size > 0:
            self.assertEqual(eval(attr['min']), ints.min())
            self.assertEqual(eval(attr['max']), ints.max())

        self.assertEqual(eval(attr['shape']), ints.shape)

        if string:
            test = decode_int_array(string, attr)
        else:
            test = decode_int_array(values, attr)

        self.assertEqual(ints.shape, test.shape)
        self.assertEqual(np.dtype(decode_dtype(attr['dtype'])), test.dtype)
        self.assertTrue(np.all(test == ints))

        return attr
    #===========================================================================



    attr = test_int_array(np.arange(0) - 5, 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')

    attr = test_int_array(np.array(1), 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')

    attr = test_int_array(np.arange(10) - 5, 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')

    attr = test_int_array(np.arange(10).reshape(2,5), 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')

    attr = test_int_array(np.arange(10) - 5, 0, 1000)
    self.assertEqual(attr['encoding'], 'base64')
    self.assertEqual(attr['storage_dtype'], INT8_ENC)

    attr = test_int_array(np.arange(10).reshape(2,5), 0, 1000)
    self.assertEqual(attr['encoding'], 'base64')
    self.assertEqual(attr['storage_dtype'], INT8_ENC)

    attr = test_int_array(np.arange(10) - 5, 1000, 0)
    self.assertEqual(attr['encoding'], 'npz')
    self.assertEqual(attr['storage_dtype'], INT8_ENC)

    attr = test_int_array(np.arange(10).reshape(2,1,5), 1000, 0)
    self.assertEqual(attr['encoding'], 'npz')
    self.assertEqual(attr['storage_dtype'], INT8_ENC)

    attr = test_int_array(np.arange(10000) - 5, 1000, 10000)
    self.assertEqual(attr['encoding'], 'npz')
    self.assertEqual(attr['storage_dtype'], INT16_ENC)

    attr = test_int_array(np.arange(10000) - 50000, 1000, 10000)
    self.assertEqual(attr['encoding'], 'npz')
    self.assertEqual(attr['storage_dtype'], UINT16_ENC)
    self.assertEqual(eval(attr['offset']), -50000)

    attr = test_int_array(np.arange(10000) * 50000, 1000, 10000)
    self.assertEqual(attr['encoding'], 'npz')
    self.assertEqual(attr['storage_dtype'], INT32_ENC)

    attr = test_int_array(np.arange(65535) * 65535, 1000, 10000)
    self.assertEqual(attr['encoding'], 'npz')
    self.assertEqual(attr['storage_dtype'], UINT32_ENC)

    attr = test_int_array(np.arange(65535) * 65535 - 10, 1000, 10000)
    self.assertEqual(attr['encoding'], 'npz')
    self.assertEqual(attr['storage_dtype'], UINT32_ENC)
    self.assertEqual(eval(attr['offset']), -10)

    attr = test_int_array(np.arange(65535) * 65535 + 2**40, 1000, 10000)
    self.assertEqual(attr['encoding'], 'npz')
    self.assertEqual(attr['storage_dtype'], UINT32_ENC)
    self.assertEqual(eval(attr['offset']), 2**40)

    attr = test_int_array(np.arange(1), 1000, 1000)
    attr = test_int_array(np.arange(2), 1000, 1000)

    #------------------------
    # Boolean array tests
    #------------------------

    #===========================================================================
    # test_bool_array
    #===========================================================================
    def test_bool_array(bools, b64_savings, npz_savings):
        bools = np.array(bools).astype('bool')
        (string, values, attr) = encode_bool_array(bools, b64_savings,
                                                          npz_savings)
        attr = dict(attr)

        ravel = bools.ravel()
        if ravel.size > 0:
            self.assertEqual(eval(attr['first']), ravel[0])

        if bools.size > 1:
            self.assertEqual(eval(attr['second']), ravel[1])

        if bools.size > 2:
            self.assertEqual(eval(attr['third']), ravel[2])

        self.assertEqual(eval(attr['shape']), bools.shape)

        if string:
            test = decode_bool_array(string, attr)
        else:
            test = decode_bool_array(values, attr)

        self.assertEqual(bools.shape, test.shape)
        self.assertEqual(np.dtype(decode_dtype(attr['dtype'])), test.dtype)
        self.assertTrue(np.all(test == bools))

        return attr
    #===========================================================================



    attr = test_bool_array([], 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')

    attr = test_bool_array(np.array(True), 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')

    attr = test_bool_array([0,0,0,0,0], 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')
    self.assertEqual(attr['method'], 'all_equal')

    attr = test_bool_array(np.array([0,0,0,0,0,0]).reshape(2,3), 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')
    self.assertEqual(attr['method'], 'all_equal')

    attr = test_bool_array([1,1,1,1,1], 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')
    self.assertEqual(attr['method'], 'all_equal')

    attr = test_bool_array(np.array([1,1,1,1,1,1]).reshape(3,1,2), 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')
    self.assertEqual(attr['method'], 'all_equal')

    attr = test_bool_array([0,0], 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')
    self.assertEqual(attr['method'], 'all_equal')

    attr = test_bool_array([[0,0]], 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')
    self.assertEqual(attr['method'], 'all_equal')

    attr = test_bool_array([1,1], 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')
    self.assertEqual(attr['method'], 'all_equal')

    attr = test_bool_array([[1,1]], 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')
    self.assertEqual(attr['method'], 'all_equal')

    attr = test_bool_array([0], 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')
    self.assertEqual(attr['method'], 'all_equal')

    attr = test_bool_array([1], 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')
    self.assertEqual(attr['method'], 'all_equal')

    attr = test_bool_array([[0] + 99*[1]], 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')
    self.assertEqual(attr['method'], 'list_false')

    attr = test_bool_array([10*[0] + 65525*[1]], 0, 1000000)
    self.assertEqual(attr['encoding'], 'base64')
    self.assertEqual(attr['method'], 'list_false')
    self.assertEqual(attr['index_dtype'][1:], 'u2')
    self.assertNotEqual(attr['index_dtype'][0], '|')

    attr = test_bool_array([10*[0] + 999990*[1]], 0, 1000000)
    self.assertEqual(attr['encoding'], 'base64')
    self.assertEqual(attr['method'], 'list_false')
    self.assertEqual(attr['index_dtype'][1:], 'u4')
    self.assertNotEqual(attr['index_dtype'][0], '|')

    attr = test_bool_array([[1] + 99*[0]], 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')
    self.assertEqual(attr['method'], 'list_true')

    attr = test_bool_array([10*[1] + 65525*[0]], 0, 1000000)
    self.assertEqual(attr['encoding'], 'base64')
    self.assertEqual(attr['method'], 'list_true')
    self.assertEqual(attr['index_dtype'][1:], 'u2')
    self.assertNotEqual(attr['index_dtype'][0], '|')

    attr = test_bool_array([10*[1] + 999990*[0]], 0, 1000000)
    self.assertEqual(attr['encoding'], 'base64')
    self.assertEqual(attr['method'], 'list_true')
    self.assertEqual(attr['index_dtype'][1:], 'u4')
    self.assertNotEqual(attr['index_dtype'][0], '|')

    attr = test_bool_array([[50*[1,0]]], 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')
    self.assertEqual(attr['method'], 'TF')

    attr = test_bool_array([[0,0,0,0,0]], 0, 0)
    self.assertEqual(attr['encoding'], 'text')
    self.assertEqual(attr['method'], 'all_equal')

    attr = test_bool_array([[[1,1,1,1,1]]], 0, 0)
    self.assertEqual(attr['encoding'], 'text')
    self.assertEqual(attr['method'], 'all_equal')

    attr = test_bool_array([[0] + 99*[1]], 0, 1000)
    self.assertEqual(attr['encoding'], 'base64')
    self.assertEqual(attr['method'], 'list_false')

    attr = test_bool_array([[[1] + 99*[0]]], 1000, 0)
    self.assertEqual(attr['encoding'], 'npz')
    self.assertEqual(attr['method'], 'list_true')

    attr = test_bool_array(10000*[1,1,0,1], 1000, 1000)
    self.assertEqual(attr['encoding'], 'npz')
    self.assertEqual(attr['method'], 'packbits')

    attr = test_bool_array(10000*[1,1,0,1], 0, 1000000)
    self.assertEqual(attr['encoding'], 'base64')
    self.assertEqual(attr['method'], 'packbits')

    attr = test_bool_array(np.array(1000*[1,1,0,1]).reshape(4,10,100), 0, 10000)
    self.assertEqual(attr['encoding'], 'base64')
    self.assertEqual(attr['method'], 'packbits')

    attr = test_bool_array(np.array(1000*[1,1,0,1]).reshape(4,10,100), 10000, 0)
    self.assertEqual(attr['encoding'], 'npz')
    self.assertEqual(attr['method'], 'packbits')

    #-----------------------
    # Float array tests
    #-----------------------

    #===========================================================================
    # test_float_array
    #===========================================================================
    def test_float_array(floats, b64_savings, npz_savings, single=False,
                                 absolute=None, relative=None, worstcase=True):
        floats = np.array(floats)
        (string, values,
         attr) = encode_float_array(floats, b64_savings, npz_savings,
                                    single, absolute, relative, worstcase)
        attr = dict(attr)

        ravel = floats.ravel()
        if ravel.size > 0:
            self.assertEqual(eval(attr['first']),  ravel[0])

        if ravel.size > 1:
            self.assertEqual(eval(attr['second']), ravel[1])

        if ravel.size > 2:
            self.assertEqual(eval(attr['third']),  ravel[2])

        if ravel.size > 0:
            self.assertEqual(eval(attr['min']),   floats.min())
            self.assertEqual(eval(attr['max']),   floats.max())

        self.assertEqual(eval(attr['shape']), floats.shape)

        if string:
            test = decode_float_array(string, attr)
        else:
            test = decode_float_array(values, attr)

        self.assertEqual(floats.shape, test.shape)
        self.assertEqual(np.dtype(decode_dtype(attr['dtype'])), test.dtype)

        if single:
            test = test.astype('float32')
            floats = floats.astype('float32')

        diffs = np.abs(test - floats)
        abs_floats = np.abs(floats)
        if absolute is None:
            if relative is None and not single:
                self.assertTrue(np.all(test == floats))
            elif relative is None and single:
                self.assertTrue(np.all(diffs <= 8.e-8 * abs_floats))
            else:
                self.assertTrue(np.all(diffs <= relative * abs_floats))
        else:
            if relative is None:
                k = np.argmax(diffs.ravel())
                self.assertTrue(np.all(diffs <= absolute))
            elif worstcase:
                self.assertTrue(np.all(diffs <= relative * abs_floats))
                self.assertTrue(np.all(diffs <= absolute))
            else:
                self.assertTrue(np.all(diffs <= np.maximum(relative*abs_floats,
                                                           absolute)))
        return attr
    #===========================================================================



    randoms = np.random.randn(1000)
    cleaned = randoms[np.abs(randoms) > 1.e-3] # exclude values too close to 0

    attr = test_float_array(randoms[:0], 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')

    attr = test_float_array(np.array(1.), 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')

    attr = test_float_array(randoms[:1], 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')

    attr = test_float_array([[randoms[:2]]], 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')

    attr = test_float_array(randoms[:10], 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')

    attr = test_float_array([randoms[:100]], 0, 1000)
    self.assertEqual(attr['encoding'], 'base64')
    self.assertEqual(attr['storage_dtype'], FLOAT64_ENC)

    attr = test_float_array(randoms[:100], 1000, 0)
    self.assertEqual(attr['encoding'], 'npz')
    self.assertEqual(attr['storage_dtype'], FLOAT64_ENC)

    attr = test_float_array([[randoms[:300]]], 1000, 1000)
    self.assertEqual(attr['encoding'], 'npz')
    self.assertEqual(attr['storage_dtype'], FLOAT64_ENC)

    attr = test_float_array(randoms[:20], 1000, 1000, single=True)
    self.assertEqual(attr['encoding'], 'text')
    self.assertEqual(attr['storage_dtype'], FLOAT32_ENC)

    attr = test_float_array(randoms[:20].reshape(5,4), 1000, 1000, single=True)
    self.assertEqual(attr['encoding'], 'text')
    self.assertEqual(attr['storage_dtype'], FLOAT32_ENC)

    #- - - - - - - - - - - - - - - - - - -
    # In text mode, we never use ints
    #- - - - - - - - - - - - - - - - - - -
    attr = test_float_array(randoms[:20], 1000, 1000, absolute=0.05)
    self.assertEqual(attr['storage_dtype'], FLOAT32_ENC)

    attr = test_float_array([randoms[:20]], 1000, 1000, relative=0.05)
    self.assertEqual(attr['storage_dtype'], FLOAT32_ENC)

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Test int usage for npz encoding, absolute or relative
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    attr = test_float_array([randoms[:100].clip(-2,2)], 1000, 0, absolute=0.1)
    self.assertEqual(attr['storage_dtype'], UINT8_ENC)

    attr = test_float_array(randoms[:100].clip(-2,2), 1000, 0, absolute=0.001)
    self.assertEqual(attr['storage_dtype'], UINT16_ENC)

    attr = test_float_array([randoms[:100].clip(-2,2)], 1000, 0, absolute=0.00001)
    self.assertEqual(attr['storage_dtype'], UINT32_ENC)

    attr = test_float_array([cleaned[:100].clip(-2,2)], 1000, 0, relative=0.1)
    self.assertEqual(attr['storage_dtype'], UINT16_ENC)

    attr = test_float_array(cleaned[:100].clip(-2,2), 1000, 0, relative=0.0001)
    self.assertEqual(attr['storage_dtype'], FLOAT32_ENC)

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Test int usage for npz encoding, absolute and relative
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    attr = test_float_array([randoms[:100].clip(-2,2)], 1000, 0, absolute=0.1, relative=1.e-8, worstcase=False)
    self.assertEqual(attr['storage_dtype'], UINT8_ENC)

    attr = test_float_array(randoms[:100].clip(-2,2), 1000, 0, absolute=0.001, relative=1.e-8, worstcase=False)
    self.assertEqual(attr['storage_dtype'], UINT16_ENC)

    attr = test_float_array([randoms[:100].clip(-2,2)], 1000, 0, absolute=0.00001, relative=1.e-8, worstcase=False)
    self.assertEqual(attr['storage_dtype'], UINT32_ENC)

    attr = test_float_array([cleaned[:100].clip(-2,2)], 1000, 0, relative=0.1, absolute=1, worstcase=True)
    self.assertEqual(attr['storage_dtype'], UINT16_ENC)

    attr = test_float_array(cleaned[:100].clip(-2,2), 1000, 0, relative=0.0001, absolute=1, worstcase=True)
    self.assertEqual(attr['storage_dtype'], FLOAT32_ENC)

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Test int usage for base64 encoding, absolute or relative
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    attr = test_float_array([randoms[:100].clip(-2,2)], 0, 1000, absolute=0.1)
    self.assertEqual(attr['storage_dtype'], UINT8_ENC)

    attr = test_float_array(randoms[:100].clip(-2,2), 0, 1000, absolute=0.001)
    self.assertEqual(attr['storage_dtype'], UINT16_ENC)

    attr = test_float_array([randoms[:100].clip(-2,2)], 0, 1000, absolute=0.00001)
    self.assertEqual(attr['storage_dtype'], UINT32_ENC)

    attr = test_float_array([cleaned[:100].clip(-2,2)], 0, 1000, relative=0.1)
    self.assertEqual(attr['storage_dtype'], UINT16_ENC)

    attr = test_float_array(cleaned[:100].clip(-2,2), 0, 1000, relative=0.0001)
    self.assertEqual(attr['storage_dtype'], FLOAT32_ENC)

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Test int usage for base64 encoding, absolute and relative
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    attr = test_float_array([randoms[:100].clip(-2,2)], 0, 1000, absolute=0.1, relative=1.e-8, worstcase=False)
    self.assertEqual(attr['storage_dtype'], UINT8_ENC)

    attr = test_float_array(randoms[:100].clip(-2,2), 0, 1000, absolute=0.001, relative=1.e-8, worstcase=False)
    self.assertEqual(attr['storage_dtype'], UINT16_ENC)

    attr = test_float_array([randoms[:100].clip(-2,2)], 0, 1000, absolute=0.00001, relative=1.e-8, worstcase=False)
    self.assertEqual(attr['storage_dtype'], UINT32_ENC)

    attr = test_float_array([cleaned[:100].clip(-2,2)], 0, 1000, relative=0.1, absolute=1, worstcase=True)
    self.assertEqual(attr['storage_dtype'], UINT16_ENC)

    attr = test_float_array(cleaned[:100].clip(-2,2), 0, 1000, relative=0.0001, absolute=1, worstcase=True)
    self.assertEqual(attr['storage_dtype'], FLOAT32_ENC)

    #------------------------------------------
    # Test byte ordering in base64 encoding
    #------------------------------------------
    before = randoms.astype('<f8')
    (string, values, attr) = encode_float_array(before, 0, 1.e9)
    after = decode_array(string, dict(attr))
    self.assertTrue(np.all(before == after))

    before = randoms.astype('>f8')
    (string, values, attr) = encode_float_array(before, 0, 1.e9)
    after = decode_array(string, dict(attr))
    self.assertEqual(after.dtype, np.dtype('>f8'))
    self.assertTrue(np.all(before == after))

    #-----------------------
    # Character tests
    #-----------------------

    #===========================================================================
    # test_char_array
    #===========================================================================
    def test_char_array(chars, b64_savings, npz_savings):
        if type(chars) == str:
            chars = list(chars)

        chars = np.array(chars).astype('c')
        (string, values, attr) = encode_string_array(chars, b64_savings,
                                                            npz_savings)
        attr = dict(attr)

        ravel = chars.ravel()
        if ravel.size > 0:
            self.assertEqual(attr['first'], unescape(ravel[0], ENTITIES))

        if ravel.size > 1:
            self.assertEqual(attr['second'], unescape(ravel[1], ENTITIES))

        if ravel.size > 2:
            self.assertEqual(attr['third'], unescape(ravel[2], ENTITIES))

        self.assertEqual(eval(attr['shape']), chars.shape)

        if string:
            test = decode_string_array(string, attr)
        else:
            test = decode_string_array(values, attr)

        self.assertEqual(chars.shape, test.shape)
        self.assertEqual(np.dtype(decode_dtype(attr['dtype'])), test.dtype)
        self.assertTrue(np.all(test == chars))

        return attr
    #===========================================================================


    attr = test_char_array([], 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')

    attr = test_char_array(np.array('a'), 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')

    attr = test_char_array(10*'abcdefghijklmnopqrstuvwxyz', 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')

    attr = test_char_array(100*'abcdefghijklmnopqrstuvwxyz', 1000, 0)
    self.assertEqual(attr['encoding'], 'npz')

    attr = test_char_array(100*'abcdefghijklmnopqrstuvwxyz', 0, 1000)
    self.assertEqual(attr['encoding'], 'text')  # no benefit to base64

    attr = test_char_array('a\'b\"\c', 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')

    #------------------
    # String tests
    #------------------

    #===========================================================================
    # test_string_array
    #===========================================================================
    def test_string_array(strs, b64_savings, npz_savings):
        strs = np.array(strs)
        (string, values, attr) = encode_string_array(strs, b64_savings,
                                                           npz_savings)
        attr = dict(attr)

        ravel = strs.ravel()
        if ravel.size > 0:
            self.assertEqual(attr['first'], unescape(ravel[0], ENTITIES))

        if ravel.size > 1:
            self.assertEqual(attr['second'], unescape(ravel[1], ENTITIES))

        if ravel.size > 2:
            self.assertEqual(attr['third'], unescape(ravel[2], ENTITIES))

        self.assertEqual(eval(attr['shape']), strs.shape)

        if string:
            test = decode_string_array(string, attr)
        else:
            test = decode_string_array(values, attr)

        self.assertEqual(strs.shape, test.shape)
        self.assertEqual(np.dtype(decode_dtype(attr['dtype'])), test.dtype)
        self.assertTrue(np.all(test == strs))

        return attr
    #===========================================================================


    attr = test_string_array([], 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')

    attr = test_string_array(np.array('abc'), 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')

    attr = test_string_array(10*['abcdefghijklmnopqrstuvwxyz'], 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')

    attr = test_string_array(100*['abcdefghijklm','nopqrstuvwxyz'], 1000, 0)
    self.assertEqual(attr['encoding'], 'npz')

    attr = test_string_array(100*['a', 'bcdefghijklm','nopqrstuvwxyz'], 0, 1000000)
    self.assertEqual(attr['encoding'], 'text')  # no benefit to base64

    attr = test_string_array(3*['abc\'\'def\"\"ghi'], 1000, 1000)
    self.assertEqual(attr['encoding'], 'text')
  #=============================================================================

#*******************************************************************************


################################################################################
# Perform unit testing if executed from the command line
################################################################################

if __name__ == '__main__':
    unittest.main()

################################################################################
