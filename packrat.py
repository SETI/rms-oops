import os
import sys
import numpy as np
import base64
import pickle
import getpass
import time
from xml.sax.saxutils import escape, unescape
import xml.etree.ElementTree as ET

class Packrat(object):
    """A class that supports the reading and writing of objects and their
    attributes.

    Attributes and their values are saved in a readable, reasonably compact XML
    file. When necessary, large objects are stored in a NumPy "savez" file of
    the same name but extension '.npz', with a pointer in the XML file.

    The procedure handles the reading and writing of all the standard Python
    types: int, float, bool, str, list, tuple, dict and set.

    The procedure also handles NumPy arrays of all dtypes except 'U' (Unicode)
    and 'o' (object).

    A special class "Session" is designed for grouping together information that
    has been added to a Packrat file all it once. A Session records the user
    ID and time, and exposes its contents as attributes and by indexing. A call
    to start_session() is all that is required.

    For other object classes, it looks for a class attribute 'PACKRAT_ARGS',
    containing a list of the names of attributes. When writing, the values of
    these attributes are written in the order listed. When reading, it calls
    the constructor using the argument values in the order they appear in this
    list.

    For classes that do not have a PACKRAT_ARGS attribute, _all_ of the
    attributes are written to the XML file in alphabetical order. Reading
    returns a tuple (module_name, class_name, attribute_dictionary).

    Like standard files, Packrat files can be open for read ("r"), write ("w")
    or append ("a"). When open for write or append, each call to
        write(name, value)
    appends a new name/value to the end of the file. When open for read, each
    call to
        read()
    returns a tuple containing the name and value of the next object in the 
    file.
    """

    class Session(object):
        """Subclass for logically grouped information.

        Values are saved by name as session attributes. Sessions can also be
        indexed to return (name,value) tuples in order.
        """

        def __init__(self, node, pack):
            self._mylist = []
            self._mydict = {}

            for subnode in node:
                (key, value) = pack._read_node(subnode)
                self._mylist.append((key, value))
                self._mydict[key] = value
                self.__dict__[key] = value

            self.user = node.attrib['user']
            self.time = node.attrib['time']
            self.note = node.attrib['note']

        def __getitem__(self, i):
            if type(i) == int:
                return self._mylist[i]
            else:
                return self._mydict[i]

        def __str__(self):
            return 'Session(user="%s", time="%s")' % (self.user, self.time)

        def __repr__(self): return str(self)

    VERSION = '1.0'

    ENTITIES = {'"': '&quot;', "'": '&apos;'}
    UNENTITIES = {'&quot;':'"', '&apos;': "'"}

    def __init__(self, filename, access='w', indent=2, savings=1.e5,
                       crlf=None):
        """Create a Packrat object for write or append.

        It opens a new file of the given name. Use the close() method when
        finished. An associated .npz file is opened only if it is needed.

        Input:
            filename    the name of the file to open. It should end in ".xml"
            access      'w' to write a new file (replacing one if it already
                        exists); 'a' to append new objects to an existing file;
                        'r' to read an existing file.
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
        """

        if not filename.lower().endswith('.xml'):
            raise ValueError('filename does not end in ".xml": ' + filename)

        if access not in ('w', 'a', 'r'):
            raise ValueError('access is not "w", "a" or "r"')

        self.filename = filename
        self.file = None
        self.access = access
        self._version = Packrat.VERSION
        self.tuples = []
        self.tuple_no = 0       # To emulate sequential read operations
        self.indent = indent

        if crlf is None:
            self.linesep = os.linesep
        elif crlf:
            self.linesep = '\r\n'
        else:
            self.linesep = '\n'

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

        # Handle the existing npz file if necessary
        if os.path.exists(self.npz_filename):
            if access in ('a', 'r'):
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
            self.tuples = []
            for node in root:
                self.tuples.append(self._read_node(node))

            # Check the npz count
            if len(self.npz_list) != self.npz_no:
                raise IOError('Packrat file %s does not match its .npz file' %
                              filename)

            return

        # When opening for write, initialize the file
        if self.access == 'w':
            self.file = open(filename, 'wb')

            self.file.write('<?xml version="1.0" encoding="ASCII"?>')
            self.file.write(self.linesep)
            self.file.write(self.linesep)
            self.file.write('<packrat version="%s">' % Packrat.VERSION)
            self.file.write(self.linesep)
            self.begun = ['packrat']

        # When opening for append, position before the last line
        else:
            self.file = open(filename, 'r+b')

            # Figure out line termination
            self.file.seek(-2, 2)
            char = self.file.read(1)
            if char == '\r':     # Windows CRLF line terminators
                self.linesep = '\r\n'
            else:
                self.linesep = '\n'

            # Jump to just before the last line of the file
            self.file.seek(0, 0)
            for line in self.file:
                pass

            self.file.seek(1 - len(line) - len(self.linesep), 2)
            self.begun = ['packrat']

    @staticmethod
    def open(filename, access='r', indent=2, savings=(1000,1000), crlf=None):
        """Create and open a Packrat object for write or append.

        This is an alternative to calling the constructor directly.

        It opens a new file of the given name. Use the close() method when
        finished. An associated .npz file is opened only if it is needed.

        Input:
            filename    the name of the file to open. It should end in ".xml"
            access      'w' to write a new file (replacing one if it already
                        exists); 'a' to append new objects to an existing file;
                        'r' to read an existing file.
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
        """

        return Packrat(filename, access, indent, savings, crlf)

    def close(self):
        """Close this Packrat file."""

        # Close a file open for write or append
        if self.file is not None:

            # Terminate anything already begun
            levels = len(self.begun) - 1
            for (k,element) in enumerate(self.begun[::-1]):
                self.file.write(self.indent * (levels - k) * ' ')
                self.file.write('</' + element + '>' + self.linesep)

            self.file.close()
            self.file = None

            if self.npz_no > 0:
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

    def write(self, element, value, level=0, attributes=[]):
        """Write an object into a Packrat file.

        The object and its value is appended to this file.

        Input:
            element     name of the XML element.
            value       the value of the element.
            level       the indent level of this element relative to the
                        previous indentation.
            skip        True to skip a line before and after.
            attributes  a list of additional attributes to include as (name,
                        value) tuples.
        """

        close_element = True

        # Write a NumPy ndarray

        if type(value) == np.ndarray:
            self._write_element(element, level,
                                [('type', 'array'),
                                 ('shape', str(value.shape)),
                                 ('dtype', value.dtype.str)] + attributes, '')

            raw_bytes = value.size * value.itemsize
            kind = value.dtype.kind
            flattened = value.ravel()

            if kind in ('S','c'): first = flattened[0]
            else:                 first = repr(flattened[0])

            # Estimate the size when formatted
            if kind == 'f':
                xml_bytes = value.size * 24
            elif kind in 'iu':
                median = np.median(flattened)
                xml_bytes = value.size * (len(str(median)) + 3)
            elif kind == 'b':
                xml_bytes = value.size
            elif kind == 'c' or value.dtype.str == '|S1':
                xml_bytes = value.size
            elif kind in 'S':
                xml_bytes = value.size * (value.itemsize + 3)
            else:
                raise ValueError('arrays of kind %s are not supported' % kind)

            b64_bytes = int(1.3 * raw_bytes)

            # Hold the data for the npz file if the savings is large enough
            if min(xml_bytes, b64_bytes) > raw_bytes + self.npz_savings:

                self.npz_list.append(value)
                self.npz_no += 1

                self.file.write(' first="%s"' % first)
                self.file.write(' encoding="npz"/>' + self.linesep)
                close_element = False

            # Otherwise, write data as base64 if the savings is large enough
            elif xml_bytes > b64_bytes + self.base64_savings:
                if kind == 'S': first = escape(flattened[0], Packrat.ENTITIES)
                else:           first = repr(flattened[0])

                self.file.write(' first="%s"' % first)
                self.file.write(' encoding="base64">' + self.linesep)

                string = base64.b64encode(value.tostring())
                self.file.write(string) # escape() not needed
                self.file.write(self.linesep)
                self.file.write(self.indent * (len(self.begun) + level) * ' ')

            # Handle integers and floats
            elif kind in 'iuf':
                self.file.write(' encoding="text">')
                for v in flattened[:-1]:
                    self.file.write(repr(v))
                    self.file.write(', ')
                self.file.write(repr(flattened[-1]))

            # Handle booleans
            elif kind == 'b':
                self.file.write(' encoding="text">')
                for v in flattened:
                    self.file.write('FT'[v])

            # Handle characters or 1-bytes strings
            elif kind == 'c' or value.dtype.str == '|S1':
                self.file.write(' encoding="text">')
                for v in flattened:
                    self.file.write(escape(v, Packrat.ENTITIES))

            # Handle longer strings
            elif kind == 'S':
                self.file.write(' encoding="text">')

                for v in flattened[:-1]:
                    self.file.write('"')
                    self.file.write(escape(v, Packrat.ENTITIES))
                    self.file.write('", ')

                self.file.write('"')
                self.file.write(escape(flattened[-1], Packrat.ENTITIES))
                self.file.write('"')

            if close_element:
                self.file.write('</' + element + '>' + self.linesep)

        # Write a standard Python class

        # int, float, bool
        elif type(value) in (int, float, bool):
            self._write_element(element, level,
                                [('type', type(value).__name__)] + attributes)
            self.file.write(repr(value))
            self.file.write('</' + element + '>' + self.linesep)

        # str
        elif type(value) == str:
            self._write_element(element, level,
                                [('type', type(value).__name__)] + attributes)
            self.file.write('"')
            self.file.write(escape(value, Packrat.ENTITIES))
            self.file.write('"')
            self.file.write('</' + element + '>' + self.linesep)

        # None
        elif value == None:
            self._write_element(element, level, [('type', 'None')] + attributes,
                                terminate='/>')
            self.file.write(self.linesep)

        # tuple, list
        elif type(value) in (tuple,list):
            self._write_element(element, level,
                                [('type', type(value).__name__)] + attributes)
            self.file.write(self.linesep)

            for (i,item) in enumerate(value):
                self.write('item', item, level=level+1,
                           attributes=[('index',str(i))])

            self.file.write(self.indent * (len(self.begun) + level) * ' ')
            self.file.write('</' + element + '>' + self.linesep)

        # set
        elif type(value) == set:
            self._write_element(element, level,
                                [('type', type(value).__name__)] + attributes)
            self.file.write(self.linesep)

            for item in value:
                self.write('item', item, level=level+1)

            self.file.write(self.indent * (len(self.begun) + level) * ' ')
            self.file.write('</' + element + '>' + self.linesep)


        # dict
        elif type(value) == dict:
            self._write_element(element, level,
                                [('type', type(value).__name__)] + attributes)
            self.file.write(self.linesep)

            keys = value.keys()
            keys.sort()
            for key in keys:
                self._write_dict_pair(key, value[key], level=level+1)

            self.file.write(self.indent * (len(self.begun) + level) * ' ')
            self.file.write('</' + element + '>' + self.linesep)

        # Otherwise write an object

        else:
            self._write_element(element, level,
                                [('type', 'object'),
                                 ('module', type(value).__module__),
                                 ('class', type(value).__name__)] + attributes)
            self.file.write(self.linesep)

            # Use the special attribute list if available
            if hasattr(type(value), 'PACKRAT_ARGS'):
                attr_list = type(value).PACKRAT_ARGS
            else:
                attr_list = value.__dict__.keys()
                attr_list.sort()

            for key in attr_list:
                self.write(key, value.__dict__[key], level=level+1)

            self.file.write(self.indent * (len(self.begun) + level) * ' ')
            self.file.write('</' + element + '>' + self.linesep)

    def start(self, element, attributes=[]):
        """Write the beginning of an object into a Packrat file."""

        self.file.write(self.linesep)
        self.file.write(self.indent * len(self.begun) * ' ')
        self.file.write('<' + element)

        for tuple in attributes:
            self.file.write(' ' + tuple[0] + '="' +
                            escape(tuple[1], Packrat.ENTITIES) + '"')

        self.file.write('>' + self.linesep)

        self.begun.append(element)

    def start_session(self, note=''):
        """Write the beginning of a new session into a Packrat file.

        A session is just an optional way to group information in a file. A
        session saves the user name, date and Packrat version ID into the file
        as attributes. The next call to finish() ends the session.
        """

        self.start('session', [('type', 'session'),
                               ('user', getpass.getuser()),
                               ('time', time.strftime('%Y-%m-%dT%H:%M:%S')),
                               ('note', note),
                               ('version', Packrat.VERSION)])

    def finish(self):
        """End of the most recently begun object in a Packrat file."""

        self.file.write(self.indent * (len(self.begun) - 1) * ' ')
        self.file.write('</' + self.begun[-1] + '>' + self.linesep)

        del self.begun[-1]

    def _write_element(self, element, level, attributes, terminate='>'):
        """Internal method to write the beginning of one element."""

        self.file.write(self.indent * (len(self.begun) + level) * ' ')
        self.file.write('<' + element)

        for tuple in attributes:
            self.file.write(' ' + tuple[0] + '="' +
                            escape(tuple[1], Packrat.ENTITIES) + '"')

        self.file.write(terminate)

    def _write_dict_pair(self, key, value, level):
        """Internal write method for key/value pairs from dictionaries."""

        self.file.write(self.indent * (len(self.begun) + level) * ' ')
        self.file.write('<dict_pair>' + self.linesep)

        self.write('key', key, level=level+1)
        self.write('value', value, level=level+1)

        self.file.write(self.indent * (len(self.begun) + level) * ' ')
        self.file.write('</dict_pair>' + self.linesep)

    ############################################################################
    # Read methods
    ############################################################################

    def read(self):
        """Read the next item from the file.

        The read process is actually emulated, because all of the objects are
        read by the constructor when access is 'r'
        """

        if self.tuple_no >= len(self.tuples):
            return ()

        else:
            result = self.tuples[self.tuple_no]
            self.tuple_no += 1
            return result

    def read_list(self):
        """Return the complete contents as (element, value) pairs."""

        self.tuple_no = len(self.tuples)
        return self.tuples

    def read_dict(self):
        """Return the complete contents as a dictionary."""

        self.tuple_no = len(self.tuples)

        result = {}
        for (key, value) in self.tuples:
            result[key] = value

        return result

    def _read_node(self, node):
        """Interprets one node of the XML tree recursively."""

        node_type = node.attrib['type']

        if node_type == 'int':
            return (node.tag, int(node.text))

        if node_type == 'float':
            return (node.tag, float(node.text))

        if node_type == 'bool':
            return (node.tag, eval(node.text))

        if node_type == 'None':
            return (node.tag, None)

        if node_type == 'str':
            assert node.text[0] == '"'
            assert node.text[-1] == '"'
            return (node.tag, unescape(node.text[1:-1], Packrat.UNENTITIES))

        if node_type in ('list', 'tuple', 'set'):
            result = []
            for subnode in node:
                result.append(self._read_node(subnode)[1])

            if node_type == 'tuple':
                return (node.tag, tuple(result))

            if node_type == 'set':
                return (node.tag, set(result))

            return (node.tag, result)

        if node_type == 'dict':
            result = {}
            for subnode in node:
                key = self._read_node(subnode[0])[1]
                value = self._read_node(subnode[1])[1]
                result[key] = value

            return (node.tag, result)

        if node_type == 'array':
            dtype = node.attrib['dtype']
            kind = np.dtype(dtype).kind
            shape = eval(node.attrib['shape'])
            source = None

            if 'first' in node.attrib:
                first = node.attrib['first']
                if kind in ('S','c'):
                    first = unescape(first, Packrat.UNENTITIES)
                else:
                    first = eval(first)
            else:
                first = None

            if node.attrib['encoding'] == 'npz':
                result = self.npz_list[self.npz_no]
                self.npz_no += 1
                flattened = result.ravel()
                source = 'npz file'

            elif node.attrib['encoding'] == 'base64':
                decoded = base64.b64decode(unescape(node.text,
                                                    Packrat.UNENTITIES))
                flattened = np.fromstring(decoded, dtype=dtype)
                result = flattened.reshape(shape)
                source = 'base64 string'

            elif kind == 'b':
                result = []
                for v in node.text:
                    result.append(v == 'T')
                result = np.array(result).reshape(shape)

            elif kind == 'c' or dtype == '|S1':
                result = []
                for v in unescape(node.text, Packrat.UNENTITIES):
                    result.append(v)
                result = np.array(result, dtype=dtype).reshape(shape)

            elif kind == 'S':
                result = eval('[' + node.text + ']')
                for (i,value) in enumerate(result):
                    if '&' in value:
                        result[i] = unescape(value, Packrat.UNENTITIES)
                result = np.array(result, dtype=dtype).reshape(shape)

            else:
                result = np.fromstring(node.text, sep=',', dtype=dtype)
                result = result.reshape(shape)

            if (first is not None and
                source is not None and
                flattened[0] != first):
                    raise IOError('error decoding %s: %s != %s' %
                                  (source, repr(first), repr(flattened[0])))

            return (node.tag, result)

        if node_type == 'object':
            module = node.attrib['module']
            classname = node.attrib['class']
            try:
                cls = sys.modules[module].__dict__[classname]
            except KeyError:
                cls = None

            # Create a dictionary of elements
            object_dict = {}
            for subnode in node:
                (key, value) = self._read_node(subnode)
                object_dict[key] = value

            # For an unrecognized class, just return the attribute dictionary
            if cls is None:
                return (node.tag, object_dict)

            # If the class has a PACKRAT_ARGS list, call the constructor
            if hasattr(cls, 'PACKRAT_ARGS'):
                args = []
                for key in cls.PACKRAT_ARGS:
                    args.append(object_dict[key])

                obj = object.__new__(cls)
                obj.__init__(*args)
                return (node.tag, obj)

            # Otherwise, create a new object and install the attributes.
            # This approach is not generally recommended but it will often work.
            obj = object.__new__(cls)
            obj.__dict__ = object_dict

            return (node.tag, obj)

        if node_type == 'session':
            return Packrat.Session(node, self)

        raise TypeError('unrecognized Packrat element type: ' + node_type)

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

class test_packrat(unittest.TestCase):

  def runTest(self):

    random = np.random.randn(20).reshape(2,2,5)
    random *= 10**(30. * np.random.randn(20).reshape(2,2,5))

    more_randoms = np.random.randn(200).reshape(2,5,4,5)
    more_randoms *= 10**(30. * np.random.randn(200).reshape(2,5,4,5))

    for (suffix,crlf) in [('crlf', True), ('lf', False), ('native', None)]:
        filename = 'packrat_unittests_' + suffix + '.xml'

        ####################
        f = Packrat.open(filename, access='w', crlf=crlf)
        f.write('two', 2)
        f.close()

        ####################
        f = Packrat.open(filename, access='r')
        rec = f.read()
        self.assertEqual(rec[0], 'two')
        self.assertEqual(rec[1], 2)
        self.assertEqual(type(rec[1]), int)

        rec = f.read()
        self.assertEqual(rec, ())
        f.close()

        ####################
        f = Packrat.open(filename, access='a')
        f.write('three', 3.)
        f.write('four', '4')
        f.write('five', (5,5.,'five'))
        f.write('six', [6,6.,'six'])
        f.write('seven', set([7,'7']))
        f.write('eight', {8:'eight', 'eight':8.})
        f.write('nine', True)
        f.write('ten', False)
        f.close()

        ####################
        f = Packrat.open(filename, access='r')
        rec = f.read()
        self.assertEqual(rec[0], 'two')
        self.assertEqual(rec[1], 2)
        self.assertEqual(type(rec[1]), int)

        rec = f.read()
        self.assertEqual(rec[0], 'three')
        self.assertEqual(rec[1], 3.)
        self.assertEqual(type(rec[1]), float)

        rec = f.read()
        self.assertEqual(rec[0], 'four')
        self.assertEqual(rec[1], '4')
        self.assertEqual(type(rec[1]), str)

        rec = f.read()
        self.assertEqual(rec[0], 'five')
        self.assertEqual(rec[1], (5,5.,'five'))
        self.assertEqual(type(rec[1]), tuple)
        self.assertEqual(type(rec[1][0]), int)
        self.assertEqual(type(rec[1][1]), float)
        self.assertEqual(type(rec[1][2]), str)

        rec = f.read()
        self.assertEqual(rec[0], 'six')
        self.assertEqual(rec[1], [6,6.,'six'])
        self.assertEqual(type(rec[1]), list)
        self.assertEqual(type(rec[1][0]), int)
        self.assertEqual(type(rec[1][1]), float)
        self.assertEqual(type(rec[1][2]), str)

        rec = f.read()
        self.assertEqual(rec[0], 'seven')
        self.assertEqual(rec[1], set([7,'7']))
        self.assertEqual(type(rec[1]), set)

        rec = f.read()
        self.assertEqual(rec[0], 'eight')
        self.assertEqual(rec[1], {8:'eight', 'eight':8.})
        self.assertEqual(rec[1][8], 'eight')
        self.assertEqual(rec[1]['eight'], 8.)

        rec = f.read()
        self.assertEqual(rec[0], 'nine')
        self.assertEqual(rec[1], True)
        self.assertEqual(type(rec[1]), bool)

        rec = f.read()
        self.assertEqual(rec[0], 'ten')
        self.assertEqual(rec[1], False)
        self.assertEqual(type(rec[1]), bool)

        ####################
        f = Packrat.open(filename, access='a')
        f.write('eleven', {'>11':'<=13', '>=11':12, '"elev"':"'11'"})
        f.write('twelve', 'eleven<"12"<<thirteen>')
        f.write('thirteen', None)
        f.close()

        ####################
        f = Packrat.open(filename, access='r')
        recs = f.read_list()

        self.assertEqual(recs[-4][0], 'ten')

        self.assertEqual(recs[-3][0], 'eleven')
        self.assertEqual(recs[-3][1], {'>11':'<=13',
                                       '>=11':12,
                                       '"elev"':"'11'"})
        self.assertEqual(recs[-3][1]['>11'], '<=13')
        self.assertEqual(recs[-3][1]['>=11'], 12)
        self.assertEqual(recs[-3][1]['"elev"'], "'11'")

        self.assertEqual(recs[-2][0], 'twelve')
        self.assertEqual(recs[-2][1], 'eleven<"12"<<thirteen>')

        self.assertEqual(recs[-1][0], 'thirteen')
        self.assertEqual(recs[-1][1], None)

        ####################
        f = Packrat.open(filename, access='a')

        bools = np.array([True, False, False, True]).reshape(2,1,2)
        ints = np.arange(20)
        floats = np.arange(20.)

        strings = np.array(['1', '22', '333', '4444']).reshape(2,2)
        uints = np.arange(40,60).astype('uint')
        chars = np.array([str(i) for i in (range(10) + list('<>"'))])

        f.write('bools', bools)
        f.write('ints', ints)
        f.write('floats', floats)

        f.write('random', random)

        f.write('strings', strings)
        f.write('uints', uints)
        f.write('chars', chars)
        f.close()

        ####################
        f = Packrat.open(filename, access='r')
        recs = f.read_dict()

        self.assertTrue(np.all(recs['bools'] == bools))
        self.assertTrue(np.all(recs['ints'] == ints))
        self.assertTrue(np.all(recs['floats'] == floats))
        self.assertTrue(np.all(recs['random'] == random))
        self.assertTrue(np.all(recs['strings'] == strings))
        self.assertTrue(np.all(recs['uints'] == uints))
        self.assertTrue(np.all(recs['chars'] == chars))
        f.close()

        #################### uses a pickle file
        f = Packrat.open(filename, access='a')

        more_ints = np.arange(10000).reshape(4,100,25)
        f.write('more_ints', more_ints)
        f.close()

        ####################
        f = Packrat.open(filename, access='r')
        recs = f.read_dict()

        self.assertTrue(np.all(recs['more_ints'] == more_ints))

        #################### uses base64
        f = Packrat.open(filename, access='a', savings=(1,1.e99))

        more_floats = np.arange(200.)
        f.write('more_floats', more_floats)

        f.write('more_floats', more_floats)
        f.write('more_randoms', more_randoms)
        f.write('more_randoms_msb', more_randoms.astype('>f8'))
        f.write('more_randoms_lsb', more_randoms.astype('<f8'))
        f.close()

        ####################
        f = Packrat.open(filename, access='r')
        recs = f.read_dict()

        self.assertTrue(np.all(recs['more_floats'] == more_floats))
        self.assertTrue(np.all(recs['more_randoms'] == more_randoms))
        self.assertTrue(np.all(recs['more_randoms_lsb'] == more_randoms))
        self.assertTrue(np.all(recs['more_randoms_msb'] == more_randoms))

        #################### uses a new class foo, without PACKRAT_ARGS
        f = Packrat.open(filename, access='a')
        f.write('foo', Foo(np.arange(10), np.arange(10.)))
        f.close()

        ####################
        f = Packrat.open(filename, access='r')
        recs = f.read_dict()

        self.assertEqual(type(recs['foo']), Foo)
        self.assertTrue(np.all(recs['foo'].ints == np.arange(10)))
        self.assertTrue(np.all(recs['foo'].floats == np.arange(10.)))
        self.assertTrue(np.all(recs['foo'].sum == 2.*np.arange(10)))

        #################### uses a new class bar, with PACKRAT_ARGS
        sample_bar = Bar(np.arange(10), np.arange(10.))
        f = Packrat.open(filename, access='a')
        f.write('bar', sample_bar)
        f.close()

        ####################
        f = Packrat.open(filename, access='r')
        recs = f.read_dict()

        self.assertEqual(type(recs['bar']), Bar)
        self.assertTrue(np.all(recs['bar'].ints   == sample_bar.ints))
        self.assertTrue(np.all(recs['bar'].floats == sample_bar.floats))
        self.assertTrue(np.all(recs['bar'].sum    == sample_bar.sum))

    # Test all line terminators

    f = open('packrat_unittests_native.xml', 'rb')
    native_lines = f.readlines()
    f.close()
    native_rec = 0

    f = open('packrat_unittests_crlf.xml', 'rb')
    for line in f:
        self.assertTrue(line.endswith('\r\n'))
        if len(os.linesep) > 1:
            self.assertTrue(line == native_lines[native_rec])
            native_rec += 1

    f.close()

    filename = 'packrat_unittests_lf.xml'
    f = open('packrat_unittests_lf.xml', 'rb')
    for line in f:
        if len(line) >= 2:
            self.assertTrue(line[-2] != '\r')

        if len(os.linesep) == 1:
            self.assertTrue(line == native_lines[native_rec])
            native_rec += 1

    f.close()

    ############################################################################
    # Test start(), finish(), sessions

    filename = 'packrat_unittests2.xml'

    f = Packrat.open(filename, 'w')
    f.start_session(note='test')
    f.write('one23', [1,2,3])
    f.close()

    ####################
    f = Packrat.open(filename, access='r')
    recs = f.read_list()
    f.close()

    self.assertEqual(len(recs), 1)
    self.assertEqual(type(recs[0]), Packrat.Session)
    self.assertEqual(recs[0][0][0], 'one23')
    self.assertEqual(recs[0][0][1], [1,2,3])
    self.assertEqual(recs[0].one23, [1,2,3])
    self.assertEqual(recs[0].note, 'test')
    self.assertTrue(hasattr(recs[0], 'user'))
    self.assertTrue(hasattr(recs[0], 'time'))

    ####################
    time.sleep(1)

    f = Packrat.open(filename, 'a')
    f.start_session(note='another test')
    f.write('four56',  [4,5,6])
    f.write('seven89', [7,8,9])
    f.close()

    ####################
    f = Packrat.open(filename, access='r')
    recs = f.read_list()
    f.close()

    self.assertEqual(len(recs), 2)
    self.assertEqual(type(recs[1]), Packrat.Session)
    self.assertEqual(len(recs[1]._mylist), 2)
    self.assertEqual(len(recs[1]._mydict), 2)

    self.assertEqual(recs[1][0][0], 'four56')
    self.assertEqual(recs[1][0][1], [4,5,6])
    self.assertEqual(recs[1].four56, [4,5,6])
    self.assertEqual(recs[1]['four56'], [4,5,6])

    self.assertEqual(recs[1][1][0], 'seven89')
    self.assertEqual(recs[1][1][1], [7,8,9])
    self.assertEqual(recs[1].seven89, [7,8,9])
    self.assertEqual(recs[1]['seven89'], [7,8,9])

    self.assertEqual(recs[1].note, 'another test')

    self.assertEqual(recs[0].user, recs[1].user)
    self.assertTrue(recs[0].time < recs[1].time)

    #################### use start/finish to create a fake tuple
    f = Packrat.open(filename, 'a')
    f.start('fake_tuple', attributes=[('type', 'tuple')])
    f.write('item', 1)
    f.write('item', 2)
    f.finish()
    f.close()

    ####################
    f = Packrat.open(filename, access='r')
    recs = f.read_list()
    f.close()

    self.assertEqual(recs[2][0], 'fake_tuple')
    self.assertEqual(recs[2][1], (1,2))

################################################################################
# Perform unit testing if executed from the command line
################################################################################

if __name__ == '__main__':
    unittest.main()

################################################################################
