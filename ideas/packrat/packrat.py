import os
import sys
import numpy as np
import base64
import pickle
import getpass
import time
import numbers
import decimal
from xml.sax.saxutils import escape, unescape
import xml.etree.ElementTree as ET

from packrat_arrays import encode_array, decode_array, \
                           encode_dtype, decode_dtype

from packrat_entities import ENTITIES, UNENTITIES

#===============================================================================
def clean_attr(attr_name):
    """Function to strip away '_Type__attr_', leaving 'attr'."""

    if attr_name[0] != '_' or attr_name[-1] != '_':
        return attr_name

    try:
        j = attr_name.index('__')
        return attr_name[j+2:-1]
    except ValueError:
        return attr_name

#===============================================================================
#===============================================================================
class Packrat(object):
    """A class that supports the reading and writing of objects and their
    attributes.

    Attributes and their values are saved in a readable, reasonably compact XML
    file. When necessary, large objects are stored in a NumPy "savez" file of
    the same name but extension '.npz', with a pointer in the XML file.

    Like standard files, Packrat files can be opened for read ("r") or write
    ("w"). When open for write, each call to
        write(value, key=name)
    appends a new value and its optional name to the end of the file. When
    opened for read, each call to
        read()
    returns the value of the next object in the file. In addition,
        read(key=name)
    returns the value associated with the given name.

    The procedure handles the reading and writing of objects of all types.
    Support is specifically included for all the  standard Python types: int,
    float, bool, Decimal, str, list, tuple, dict and set. All NumPy array dtypes
    except 'U' (Unicode) and 'o' (object) are also supported.

    Packrat can write and read objectsof most other classes as well. However,
    users can customize the way Packrat handles them by defining class constant
    PACKRAT_ARGS, object method PACKRAT__args__, and/or class method
    PACKRAT__init__. Details are provided below.

    Packrat's procedure for writing objects:

    1. Packrat looks for an object method
        PACKRAT__args__(self)
    If this function exists, it must return None, a list of attributes names, or
    another object. If None, then Packrat proceeds with the next step. If a list
    of attribute names, then these are the attributes of the object that Packrat
    will save in the file. If another object, then Packrat will proceed to Step
    2 using this other object.

    2. Packrat looks for a class attribute PACKRAT_ARGS. If this exists, it must
    contain a list of attribute names. The values of these attributes are saved
    in the file. This approach is more common than defining a PACKRAT__args__
    method, but theat option allows customization of what is saved on an
    object-by-object basis; this can be useful in some circumstances.

    3. If neither are defined, Packrat saves all the attributes of the object in
    the file.

    Note: In the list of attribute names, one can be prefixed with "**" and any
    number can be prefixed with "+". These attributes are handled
    specially when the object is read and reconstructed, as discussed below.

    Any attribute name can have a the definition of a Python dictionary as a
    suffix, starting with "{". This suffix is converted to a dictionary as used
    to provide additional parameters passed to function encode_array, which
    define how to pack the attribute. For example, "time{'single':True}" will
    save the time attribute in single precision.

    Packrat's procedure for reading and reconstructing objects:

    1. If the class has a function PACKRAT__init__, then Packrat calls the
    function as follows:
        PACKRAT__init__(cls, **args)
    where cls is the object class and args is a dictionary of all the attribute
    (attribute, value) pairs that were saved for this object. This function
    should return the new object or else None on failure.

    2. If PACKRAT__init__ does not exist or returns None, the standard
    constructor for the class is called as follows:
        cls.__init__(self, arg1, arg2, ..., **argdict)
    where arg1, arg2, ... are all the un-prefixed attributes save in the file,
    in order. If an attribute was listed by PACKRAT__args__ or PACKRAT_ARGS
    using a '**' prefix, then this is included as 'argdict' in the constructor;
    note that it must be a dictionary.

    3. Attribute names that PACKRAT__args__ or PACKRAT_ARGS prefixed with "+"
    are not passed to the constructor. Instead, Packrats sets these as
    (attribute, value) pairs for the object after it has been constructed.

    4. As noted above, if Packrat does not find explicit support for an object,
    all of its attributes are save. In this case, an object is constructed via
        cls.__new__()
    and then all of its attributes are defined. This is not a recommended way to
    construct an object, but it often works.
    """

    UNLINKABLE_TYPENAMES = {'int', 'bool', 'float', 'str', 'None', 'Decimal'}
    MUTABLE_TYPENAMES = {'list', 'dict', 'set', 'np.ndarray'}

    VERSION = '1.0'

    #===========================================================================
    def __init__(self, filename, access='w', indent=2, savings=(1.e3,1.e4),
                       crlf=None, compress=False):
        """Create a Packrat object for write or append.

        It opens a new file of the given name. Use the close() method when
        finished. An associated .npz file is opened only if it is needed.

        Input:
            filename    the name of the file to open. It should end in ".xml"
            access      'w' to write a new file (replacing one if it already
                        exists); 'r' to read an existing file.
            indent      the number of characters to indent entries in the XML
                        hierarchy.
            savings     a tuple containing two values:
                savings[0]  the approximate minimum number of bytes that need to
                            be saved before data values will be written into the
                            XML file using base64 encoding.
                savings[1]  the approximate minimum number of bytes that need to
                            be saved before data values will be written into an
                            associated .npz file. None to prevent the use of an
                            .npz file.
                        Either value can be None to disable that option.
            crlf        True to use Windows <cr><lf> line terminators; False to
                        use Unix/MacOS <lf> line terminators. Use None (the
                        default) to use the line terminator native to this OS.
                        Note that, when opening for append, whatever line
                        terminator is already in the file will continue to be
                        used.
            compress    True to use zip compression on the npz file objects;
                        False to leave them uncompressed.
        """

        if not filename.lower().endswith('.xml'):
            raise ValueError('filename does not end in ".xml": ' + filename)

        if access not in ('w', 'r'):
            raise ValueError('access is not "w" or "r"')

        self.filename = filename
        self.file = None
        self.access = access
        self._version = Packrat.VERSION

        # Attributes used for write...
        self.write_index = 0

        try:
            if len(savings) == 1:
                savings = (savings[0], savings[0])
        except TypeError:
            savings = (savings, savings)

        self.base64_savings = savings[0] or np.inf
        self.npz_savings = savings[1] or np.inf

        self.npz_filename = os.path.splitext(filename)[0] + '.npz'
        self.npz_list = []
        self.npz_no = 0

        self.indent = indent
        self.compress = compress

        if crlf is None:
            self.linesep = os.linesep
        elif crlf:
            self.linesep = '\r\n'
        else:
            self.linesep = '\n'

        self.xml_id_by_python_id = {}   # Dictionary of XML IDs keyed by id()
        self.python_id_by_xml_id = []   # Ordered list of Python IDs, indexed by
                                        # XML ID
        self.object_by_python_id = {}   # Dictionary of objects in XML file
                                        # keyed by Python id()

        # Attributes used for read...
        self.objects = []
        self.object_names = []
        self.object_no = 0              # To emulate sequential read operations

        self.object_by_xml_id = {}

        # Handle the existing npz file if necessary
        if os.path.exists(self.npz_filename):
            if access == 'r':
                npz_dict = np.load(self.npz_filename)
                count = len(npz_dict.keys())
                for i in range(count):
                    key = 'arr_' + str(i)
                    self.npz_list.append(npz_dict[key])

            else:
                os.remove(self.npz_filename)

        # When opening for read, load the whole file initially
        if access == 'r':
            self.file = None
            tree = ET.parse(filename)
            root = tree.getroot()

            # On read, use the file's Packrat version number
            self._version = root.attrib['version']

            # Load the tree recursively
            for node in root:
                try:
                    name = unescape(node.attrib['name'], UNENTITIES)
                except KeyError:
                    name = None

                self.objects.append(self._read_node(node))
                self.object_names.append(name)

            # Check the npz count
            if len(self.npz_list) != self.npz_no:
                raise IOError('Packrat file "%s" does not match its .npz file' %
                              filename)

            return

        # When opening for write, initialize the file
        self.file = open(filename, 'wb')

        self.file.write('<?xml version="1.0" encoding="ASCII"?>')
        self.file.write(self.linesep)
        self.file.write(self.linesep)
        self.file.write('<packrat version="%s">' % Packrat.VERSION)
        self.file.write(self.linesep)
        self.begun = ['packrat']

    #===========================================================================
    @staticmethod
    def open(filename, access='r', indent=2, savings=(1.e3,1.e4), crlf=None,
                       compress=False):
        """Open a Packrat object for read or write.

        This is an alternative to calling the constructor directly.

        It opens a new file of the given name. Use the close() method when
        finished. An associated .npz file is opened only if it is needed.

        Input:
            filename    the name of the file to open. It should end in ".xml"
            access      'w' to write a new file (replacing one if it already
                        exists); 'r' to read an existing file.
            indent      the number of characters to indent entries in the XML
                        hierarchy.
            savings     a tuple containing two values:
                savings[0]  the approximate minimum number of bytes that need to
                            be saved before data values will be written into the
                            XML file using base64 encoding.
                savings[1]  the approximate minimum number of bytes that need to
                            be saved before data values will be written into an
                            associated .npz file.
                        Either value can be None to disable that option.
            crlf        True to use Windows <cr><lf> line terminators; False to
                        use Unix/MacOS <lf> line terminators. Use None (the
                        default) to use the line terminator native to this OS.
                        Note that, when opening for append, whatever line
                        terminator is already in the file will continue to be
                        used.
            compress    True to use zip compression on the npz file objects;
                        False to leave them uncompressed.
        """

        return Packrat(filename, access, indent, savings, crlf, compress)

    #===========================================================================
    def close(self):
        """Close this Packrat file."""

        # Close a file open for write or append
        if self.file is not None:

            # Terminate anything already begun
            levels = len(self.begun) - 1
            for (k, typename) in enumerate(self.begun[::-1]):
                self.file.write(self.indent * (levels - k) * ' ')
                self.file.write('</' + typename + '>' + self.linesep)

            self.file.close()
            self.file = None

            if self.npz_no > 0:
              if self.compress:
                np.savez_compressed(self.npz_filename, *tuple(self.npz_list))
              else:
                np.savez(self.npz_filename, *tuple(self.npz_list))

        # Close a file open for read
        else:
            self.tuples = ()
            self.tuple_no = 0

            self.npz_list = []
            self.npz_no = 0

    ############################################################################
    # Write methods
    ############################################################################

    def write(self, obj, name=None, **params):
        """Write an object into a Packrat file, top-level version.

        The object is appended to the end of file.

        Input:
            obj         the object to write.
            name        optional name of the object.
            params      additional parameters, e.g., for defining the
                        compression to use on float arrays.
        """

        self._write(obj, name, self.write_index, level=0, **params)
        self.write_index += 1

    #===========================================================================
    def _write(self, obj, name=None, index=None, level=0, **params):
        """Write an object into a Packrat file, recursive version

        The object is appended to the end of file.

        Input:
            obj         the object to write.
            name        optional name of the object.
            index       optional integer index of the object.
            level       the indent level of this element relative to the
                        previous indentation.
            params      additional parameters, e.g., for defining the
                        compression to use on float arrays.
        """

        # Determine the type name
        if isinstance(obj, decimal.Decimal):
            typename = 'Decimal'
        elif isinstance(obj, (bool, np.bool_)):
            typename = 'bool'
        elif isinstance(obj, numbers.Integral):
            typename = 'int'
        elif isinstance(obj, numbers.Real):
            typename = 'float'
        elif isinstance(obj, str):
            typename = 'str'
        elif obj is None:
            typename = 'None'
        elif isinstance(obj, np.ndarray):
            typename = 'array'
        elif isinstance(obj, frozenset):
            typename = 'frozenset'
        elif isinstance(obj, set):
            typename = 'set'
        elif isinstance(obj, dict):
            typename = 'dict'
        elif isinstance(obj, list):
            typename = 'list'
        elif isinstance(obj, tuple):
            typename = 'tuple'
        else:
            typename = 'object'

        # Get the parameter name and dictionary
        if name is not None:
            name = name.lstrip('*').lstrip('+')
            brace = name.find('{')
            if brace >= 0:
                new_params = eval(name[brace:])
                params = params.copy()
                for (key,value) in new_params.iteritems():
                    params[key] = value
                name = name[:brace]

        # Determine the Python id; None for elementary objects and if len == 0
        python_id = None
        if typename not in self.UNLINKABLE_TYPENAMES:
            try:
                if len(obj) > 0:
                    python_id = id(obj)
            except (TypeError, AttributeError):
                python_id = id(obj)

        # Be careful when tracking mutable objects; test link before using
        object_to_cache = obj
        if typename in ('set', 'list', 'dict', 'array'):
            if python_id in self.object_by_python_id:
                test = self.object_by_python_id[python_id]

                if typename == 'array':
                    changed = not np.all(test == obj)
                else:
                    changed = (test != obj)

                if changed:
                    del self.xml_id_by_python_id[python_id]

            # For mutable objects, save a copy
            if typename == 'list':
                object_to_cache = list(obj)
            else:
                object_to_cache = obj.copy()

        # Look for object in cache; define XML id and link status; update cache
        if python_id:
            try:
                xml_id = self.xml_id_by_python_id[python_id]
                xml_link = True
            except KeyError:
                xml_id = len(self.python_id_by_xml_id)
                xml_link = False

                self.python_id_by_xml_id.append(python_id)
                self.xml_id_by_python_id[python_id] = xml_id

            self.object_by_python_id[python_id] = object_to_cache
        else:
            xml_id = None
            xml_link = False

        # Initialize the XML attributes
        xml_attr = []
        if xml_link:
            xml_attr += [('link', str(xml_id))]
        elif xml_id is not None:
            xml_attr += [('id', str(xml_id))]

        # Write a standard Python class without caching

        # float, int, bool, decimal, str, None
        if typename in ('int', 'float', 'bool'):
            self._start_node(typename, level, name, index, xml_attr)
            self.file.write(repr(obj))
            self._end_node(typename, level, indent=False)
            return

        if typename == 'None':
            self._start_node(typename, level, name, index, xml_attr,
                             slash='/')
            return

        if typename == 'str':
            self._start_node(typename, level, name, index, xml_attr)
            self.file.write(escape(obj, ENTITIES))
            self._end_node(typename, level, indent=False)
            return

        if typename == 'Decimal':
            self._start_node(typename, level, name, index, xml_attr)
            self.file.write(str(obj))
            self._end_node(typename, level, indent=False)
            return

        # Write a Python class with caching

        # tuple, list, set
        if typename in ('tuple', 'list', 'set', 'frozenset'):
            lenval = len(obj)
            xml_attr += [('len', str(lenval))]
            slash = self._start_node(typename, level, name, index, xml_attr,
                                     slash=(lenval==0), terminate=True)
            if slash:
                return

            if typename in ('set', 'frozenset'):
                obj = list(obj)
                obj.sort()

            for indx in range(len(obj)):
                self._write(obj[indx], None, indx, level=level+1, **params)

            self._end_node(typename, level)
            return

        # dict
        if typename == 'dict':
            lenval = len(obj)
            xml_attr += [('len', str(lenval))]
            slash = self._start_node(typename, level, name, index, xml_attr,
                                     slash=(lenval==0), terminate=True)
            if slash:
                return

            keys = list(obj.keys())
            keys.sort()

            compact_mode = all([isinstance(k,str) for k in keys])
            if compact_mode:
                compact_mode = all([(k == escape(k,ENTITIES)) for k in keys])

            if compact_mode:
                for indx in range(len(keys)):
                    key = keys[indx]
                    self._write(obj[key], key, indx, level=level+1, **params)
            else:
                for indx in range(len(keys)):
                    key = keys[indx]
                    self._start_node('dict_pair', level+1, None, indx,
                                     terminate=True)
                    self._write(key,      'key', level=level+2)
                    self._write(obj[key], 'value', level=level+2, **params)
                    self._end_node('dict_pair', level+1)

            self._end_node(typename, level)
            return

        # Write a NumPy ndarray

        if typename == 'array':
            if xml_link:
                xml_attr += [('shape', str(obj.shape).replace(' ','')),
                             ('dtype', encode_dtype(obj.dtype))]

                self._start_node(typename, level, name, index, xml_attr)
                return

            (xml_text, npz_value,
             attr_list) = encode_array(obj, self.base64_savings,
                                            self.npz_savings, **params)
            xml_attr += attr_list

            slash = self._start_node(typename, level, name, index, xml_attr,
                                     slash=(npz_value is not None))

            # Hold the data for the npz file if the savings is large enough
            if npz_value is not None:
                self.npz_list.append(npz_value)
                self.npz_no += 1
                return

            if slash:
                return

            self.file.write(xml_text)
            self._end_node(typename, level, indent=False)
            return

        # Otherwise write an object

        # Use the special attribute list if available
        obj_attr = None
        if hasattr(obj, 'PACKRAT__args__'):
            result = obj.PACKRAT__args__()
            if result is None or type(result) == list:
                obj_attr = result
            else:
                obj = result    # object might be replaced by this func

        # Write the XML header for this object
        xml_attr += [('module', type(obj).__module__),
                     ('class', type(obj).__name__)]
        slash = self._start_node(typename, level, name, index, xml_attr,
                                 terminate=True)
        if slash:
            return

        # Use the class attribute list if available
        if obj_attr is None:
            if hasattr(obj, 'PACKRAT_ARGS'):
                obj_attr = obj.PACKRAT_ARGS

        # Otherwise, use all the attributes
        if obj_attr is None:
            obj_attr = obj.__dict__.keys()
            obj_attr.sort()

        # Generate (name, value) pairs
        # Also clean up the internal names of attributes
        stripped = [k.lstrip('*').lstrip('+') for k in obj_attr]
        pairs = [(clean_attr(k), obj.__dict__[k]) for k in stripped]

        # Write the subnodes
        for indx in range(len(pairs)):
            (attr_name, value) = pairs[indx]
            self._write(value, attr_name, indx, level=level+1, **params)

        # End this object
        self._end_node(typename, level)

    #===========================================================================
    def _start_node(self, typename, level, name=None, index=None, attr=[],
                          slash='', terminate=False):

        # Determine if a trailing slash is needed
        if slash:
            slash = '/'

        if not slash:
            for (attr_name, _) in attr:
                if attr_name == 'link':
                    slash = '/'
                    break

        # Determine
        if slash:
            terminate = True

        # Indent
        self.file.write(self.indent * (len(self.begun) + level) * ' ')

        # Write node type
        self.file.write('<%s' % typename)

        # Include name and/or index for list items, object attributes, etc.
        if name:
            self.file.write(' name="%s"' % name)
        if index is not None:
            self.file.write(' index="%d"' % index)

        # Write the additional attributes
        for (attr_name, value) in attr:
            self.file.write(' %s="%s"' % (attr_name, escape(value, ENTITIES)))

        # Write trailing slash if needed
        if slash:
            self.file.write('/>')
        else:
            self.file.write('>')

        # Terminate if necessary and report termination state
        if terminate:
            self.file.write(self.linesep)

        return slash

    #===========================================================================
    def _end_node(self, typename, level, indent=True):

        # Indent
        if indent:
            self.file.write(self.indent * (len(self.begun) + level) * ' ')

        # Terminate node
        self.file.write('</%s>' % typename + self.linesep)

    ############################################################################
    # Read methods
    ############################################################################

    def read(self, key=None, index=None):
        """Read the next item from the file.

        The read process is actually emulated, because all of the objects are
        read by the constructor when access is 'r'
        """

        if key is not None:
            self.object_no = self.object_names.index(key)

        if index is not None:
            self.object_no = index

        if self.object_no >= len(self.objects):
            return None

        obj = self.objects[self.object_no]
        self.object_no += 1
        return obj

    #===========================================================================
    def read_as_list(self, key=None, index=None):
        """Read the next item from the file.

        The read process is actually emulated, because all of the objects are
        read by the constructor when access is 'r'
        """

        return self.objects

    #===========================================================================
    def read_as_dict(self, key=None, index=None):
        """Read the next item from the file.

        The read process is actually emulated, because all of the objects are
        read by the constructor when access is 'r'
        """

        result = {k:self.objects[k] for k in range(len(self.objects))}

        for (k,key) in enumerate(self.object_names):
            if key is not None:
                result[key] = self.objects[k]

        return result

    #===========================================================================
    def _read_node(self, node):
        """Interprets one node of the XML tree, recursively."""

        node_type = node.tag

        # Return the value if this node is a link
        key = None
        try:
            key = int(node.attrib['link'])
        except KeyError:
            pass

        if key is not None:
            return self.object_by_xml_id[key]

        if node_type == 'int':
            obj = int(node.text)

        elif node_type == 'float':
            obj = float(node.text)

        elif node_type == 'bool':
            obj = eval(node.text)

        elif node_type == 'Decimal':
            obj = decimal.Decimal(node.text)

        elif node_type == 'None':
            obj = None

        elif node_type == 'str':
            obj = unescape(node.text, UNENTITIES)

        elif node_type in ('list', 'tuple', 'set'):
            obj = []
            for subnode in node:
                obj.append(self._read_node(subnode))

            if node_type == 'tuple':
                obj = tuple(obj)

            if node_type == 'set':
                obj = set(obj)

        elif node_type == 'dict':
            obj = {}
            for subnode in node:
                if subnode.tag == 'dict_pair':
                    key = self._read_node(subnode[0])
                    value = self._read_node(subnode[1])
                else:
                    key = subnode.attrib['name']
                    value = self._read_node(subnode)
                obj[key] = value

        elif node_type == 'array':
            encoding = node.attrib['encoding']
            if encoding == 'npz':
                obj = decode_array(self.npz_list[self.npz_no], node.attrib)
                self.npz_no += 1
            else:
                obj = decode_array(node.text, node.attrib)

        elif node_type == 'object':
            module = node.attrib['module']
            classname = node.attrib['class']
            try:
                cls = sys.modules[module].__dict__[classname]
            except KeyError:
                cls = None

            obj = None

            # Create a dictionary of elements
            object_dict = {}
            for subnode in node:
                key = subnode.attrib['name']
                value = self._read_node(subnode)
                object_dict[key] = value

            # For an unrecognized class, just return the attribute dictionary
            if cls is None:
                obj = object_dict

            # If the class has a PACKRAT__init__ function, call it with the full
            # dictionary
            if obj is None and hasattr(cls,'PACKRAT__init__'):
                obj = cls.PACKRAT__init__(cls, **object_dict)

            # If the class has a PACKRAT_ARGS list, call the constructor
            if obj is None and hasattr(cls, 'PACKRAT_ARGS'):
                arglist = []
                argdict = {}
                extras = []
                for key in cls.PACKRAT_ARGS:
                    if key.startswith('**'):
                        argdict = object_dict[clean_attr(key[2:])]
                    elif key.startswith('+'):
                        extras.append((key[1:],
                                       object_dict[clean_attr(key[1:])]))
                    else:
                        arglist.append(object_dict[clean_attr(key)])

                obj = object.__new__(cls)
                obj.__init__(*arglist, **argdict)

                for (attr_name, value) in extras:
                    obj.__dict__[attr_name] = value

            # Otherwise, create a new object and install the attributes.
            # This approach is not recommended but it will often work.
            if obj is None:
                obj = object.__new__(cls)
                obj.__dict__ = object_dict

        else:
            raise TypeError('unrecognized Packrat element type: ' + node_type)

        # Save in dictionary
        try:
            key = int(node.attrib['id'])
            self.object_by_xml_id[key] = obj
        except KeyError:
            pass

        return obj

################################################################################
# Unit tests
################################################################################

import unittest

class Foo(object):
    def __init__(self, a, b):
        self.ints = a
        self.floats = b
        self.sum = a + b

class Bar(object):
    PACKRAT_ARGS = ['ints', 'floats']
    def __init__(self, a, b):
        self.ints = a
        self.floats = b
        self.sum = a + b

class Test_Packrat(unittest.TestCase):

  def runTest(self):

    random = np.random.randn(20).reshape(2,2,5)
    random *= 10**(30. * np.random.randn(20).reshape(2,2,5))

    more_randoms = np.random.randn(200).reshape(2,5,4,5)
    more_randoms *= 10**(30. * np.random.randn(200).reshape(2,5,4,5))

    more_randoms_lsb = more_randoms.copy().astype('<f8')
    more_randoms_msb = more_randoms.copy().astype('>f8')
    self.assertTrue(np.all(more_randoms_lsb == more_randoms_msb))

    random_ints = np.random.randint(2**63, size=200).reshape(2,5,4,5)
    random_uints_lsb = random_ints.astype('<u8')
    random_uints_msb = random_ints.astype('>u8')
    random_ints_lsb = random_ints.astype('<i8')
    random_ints_msb = random_ints.astype('>i8')

    prefix = 'packrat_unittests_'
    if os.path.exists('unittest_results'):
        prefix = os.path.join('unittest_results', prefix)

    for (suffix,crlf) in [('crlf', True), ('lf', False), ('native', None)]:
        filename = prefix + suffix + '.xml'

        ####################
        f = Packrat.open(filename, access='w', crlf=crlf)
        f.write('\\', 'zero')
        f.write('\\1', 'one')
        f.write(2, 'two')
        f.write(3., 'three')
        f.write('4', 'four')
        f.write((5,5.,'five'), 'five', )
        f.write([6,6.,'six'], 'six')
        f.write(set([7,'7']), 'seven')
        f.write({8:'eight', 'eight':8.}, 'eight')
        f.write(True, 'nine')
        f.write(False, 'ten')
        f.write({'>11':'<=13', '>=11':12, '"elev"':"'11'"}, 'eleven')
        f.write('eleven<"12"<<thirteen>', 'twelve')
        f.write(None, 'thirteen')

        bools = np.array([True, False, False, True]).reshape(2,1,2)
        ints = np.arange(20)
        floats = np.arange(20.)

        strings = np.array(['1', '22', '333', '4444']).reshape(2,2)
        uints = np.arange(40,60).astype('uint')
        chars = np.array([str(i) for i in (range(10) + list('<>"'))])

        f.write(bools, 'bools')
        f.write(ints, 'ints')
        f.write(floats, 'floats')

        f.write(random, 'random')

        f.write(strings, 'strings')
        f.write(uints, 'uints')
        f.write(chars, 'chars')

        #################### uses an npz file
        more_ints = np.arange(10000).reshape(4,100,25)
        f.write(more_ints, 'more_ints')

        #################### uses base64
        more_floats = np.arange(200.)
        f.write(more_floats, 'more_floats')
        f.write(more_randoms, 'more_randoms')
        f.write(more_randoms_msb, 'more_randoms_msb')
        f.write(more_randoms_lsb, 'more_randoms_lsb')

        f.write(random_ints, 'random_ints')
        f.write(random_uints_lsb, 'random_uints_lsb')
        f.write(random_uints_msb, 'random_uints_msb')
        f.write(random_ints_lsb, 'random_ints_lsb')
        f.write(random_ints_msb, 'random_ints_msb')

        #################### uses a new class foo, without PACKRAT_ARGS
        f.write(Foo(np.arange(10), np.arange(10.)), 'foo')

        #################### uses a new class bar, with PACKRAT_ARGS
        sample_bar = Bar(np.arange(10), np.arange(10.))
        f.write(sample_bar, 'bar')

        f.close()

        ####################
        f = Packrat.open(filename, access='r')
        rec = f.read()
        self.assertEqual(rec, '\\')
        self.assertEqual(type(rec), str)

        rec = f.read()
        self.assertEqual(rec, '\\1')
        self.assertEqual(type(rec), str)

        rec = f.read()
        self.assertEqual(rec, 2)
        self.assertEqual(type(rec), int)

        rec = f.read()
        self.assertEqual(rec, 3.)
        self.assertEqual(type(rec), float)

        rec = f.read()
        self.assertEqual(rec, '4')
        self.assertEqual(type(rec), str)

        rec = f.read()
        self.assertEqual(rec, (5,5.,'five'))
        self.assertEqual(type(rec), tuple)
        self.assertEqual(type(rec[0]), int)
        self.assertEqual(type(rec[1]), float)
        self.assertEqual(type(rec[2]), str)

        rec = f.read()
        self.assertEqual(rec, [6,6.,'six'])
        self.assertEqual(type(rec), list)
        self.assertEqual(type(rec[0]), int)
        self.assertEqual(type(rec[1]), float)
        self.assertEqual(type(rec[2]), str)

        rec = f.read()
        self.assertEqual(rec, set([7,'7']))
        self.assertEqual(type(rec), set)

        rec = f.read()
        self.assertEqual(rec, {8:'eight', 'eight':8.})
        self.assertEqual(rec[8], 'eight')
        self.assertEqual(rec['eight'], 8.)

        rec = f.read()
        self.assertEqual(rec, True)
        self.assertEqual(type(rec), bool)

        rec = f.read()
        self.assertEqual(rec, False)
        self.assertEqual(type(rec), bool)

        rec = f.read()
        self.assertEqual(rec, {'>11':'<=13', '>=11':12, '"elev"':"'11'"})
        self.assertEqual(rec['>11'], '<=13')
        self.assertEqual(rec['>=11'], 12)
        self.assertEqual(rec['"elev"'], "'11'")

        rec = f.read()
        self.assertEqual(rec, 'eleven<"12"<<thirteen>')

        rec = f.read()
        self.assertEqual(rec, None)

        f.close()

        # Now spot-check using indices and keys
        f = Packrat.open(filename, access='r')
        rec = f.read(key='two')
        self.assertEqual(rec, 2)

        rec = f.read(key='thirteen')
        self.assertEqual(rec, None)

        rec = f.read(index=3)
        self.assertEqual(rec, 3.)
        self.assertEqual(type(rec), float)

        rec = f.read(key='nine')
        self.assertEqual(rec, True)
        self.assertEqual(type(rec), bool)

        rec = f.read(index=13)
        self.assertEqual(rec, None)

        rec = f.read(key='four')
        self.assertEqual(rec, '4')
        self.assertEqual(type(rec), str)

        rec = f.read()
        self.assertEqual(rec, (5,5.,'five'))

        rec = f.read()
        self.assertEqual(rec, [6,6.,'six'])
        self.assertEqual(type(rec[2]), str)

        f.close()

        # read_as_dict
        f = Packrat.open(filename, access='r')
        recs = f.read_as_dict()
        f.close()

        self.assertTrue(np.all(recs['bools'] == bools))
        self.assertTrue(np.all(recs['ints'] == ints))
        self.assertTrue(np.all(recs['floats'] == floats))
        self.assertTrue(np.all(recs['random'] == random))
        self.assertTrue(np.all(recs['strings'] == strings))
        self.assertTrue(np.all(recs['uints'] == uints))
        self.assertTrue(np.all(recs['chars'] == chars))

        self.assertTrue(np.all(recs['more_ints'] == more_ints))

        self.assertTrue(np.all(recs['more_floats'] == more_floats))
        self.assertTrue(np.all(recs['more_randoms'] == more_randoms))

        self.assertTrue(np.all(recs['more_randoms_lsb'] == more_randoms))
        self.assertTrue(np.all(recs['more_randoms_msb'] == more_randoms))
        self.assertTrue(recs['more_randoms_lsb'].dtype, np.dtype('<f8'))
        self.assertTrue(recs['more_randoms_msb'].dtype, np.dtype('>f8'))

        self.assertTrue(np.all(recs['random_ints'] == random_ints))
        self.assertTrue(np.all(recs['random_uints_lsb'] == random_ints))
        self.assertTrue(np.all(recs['random_uints_msb'] == random_ints))
        self.assertTrue(np.all(recs['random_ints_lsb'] == random_ints))
        self.assertTrue(np.all(recs['random_ints_msb'] == random_ints))

        self.assertTrue(recs['random_uints_lsb'].dtype, np.dtype('<u8'))
        self.assertTrue(recs['random_uints_msb'].dtype, np.dtype('>u8'))
        self.assertTrue(recs['random_ints_lsb'].dtype, np.dtype('<i8'))
        self.assertTrue(recs['random_ints_msb'].dtype, np.dtype('>i8'))

        self.assertEqual(type(recs['foo']), Foo)
        self.assertTrue(np.all(recs['foo'].ints == np.arange(10)))
        self.assertTrue(np.all(recs['foo'].floats == np.arange(10.)))
        self.assertTrue(np.all(recs['foo'].sum == 2.*np.arange(10)))

        self.assertEqual(type(recs['bar']), Bar)
        self.assertTrue(np.all(recs['bar'].ints   == sample_bar.ints))
        self.assertTrue(np.all(recs['bar'].floats == sample_bar.floats))
        self.assertTrue(np.all(recs['bar'].sum    == sample_bar.sum))

        # read_as_list
        f = Packrat.open(filename, access='r')
        recs = f.read_as_list()
        self.assertEqual(len(recs), f.object_names.index('bar')+1)
        f.close()

        f = Packrat.open(filename, access='r')

        for (k,value) in enumerate(recs):
            value2 = f.read()
            if isinstance(value, (list, np.ndarray, Foo, Bar)):
                self.assertEqual(type(value), type(value2))
            else:
                self.assertEqual(value, value2)

        f.close()

    # Test all line terminators

    f = open(prefix + 'native.xml', 'rb')
    native_lines = f.readlines()
    f.close()
    native_rec = 0

    f = open(prefix + 'crlf.xml', 'rb')
    for line in f:
        self.assertTrue(line.endswith('\r\n'))
        if len(os.linesep) > 1:
            self.assertTrue(line == native_lines[native_rec])
            native_rec += 1

    f.close()

    f = open(prefix + 'lf.xml', 'rb')
    for line in f:
        if len(line) >= 2:
            self.assertTrue(line[-2] != '\r')

        if len(os.linesep) == 1:
            self.assertTrue(line == native_lines[native_rec])
            native_rec += 1

    f.close()

################################################################################
# Perform unit testing if executed from the command line
################################################################################

if __name__ == '__main__':
    unittest.main()

################################################################################
