################################################################################
# oops/hosts/pds3.py
#
# Utility functions for finding and loading PDS3 labels in host modules.
#
################################################################################
import os
import pdsparser
import re
from pathlib import Path

#===============================================================================
def clean_label(lines, cleaner=None):
    """
    Correct known errors in PDS3 labels.

    Inputs:
        lines:          List of strings comprising the label.
        cleaner:        Function that takes a label as input and
                        return a cleaned label.
    """
    if cleaner is not None:
        lines = cleaner(lines)

    return lines

#===============================================================================
def find_label(filespec):
    """Find the PDS3 label corresponding to the given filespec."""
    # Construct candidate label filenames
    spec = Path(filespec)
    filespec = str(filespec)
    labelspecs = [
        spec.with_suffix('.lbl'),
        spec.with_suffix('.LBL') ]
    if filespec.upper().endswith('.LBL'):
        labelspecs.append(filespec)

    # Check candidates; return first one that exists
    for labelspec in labelspecs:
        if os.path.isfile(labelspec):
            return labelspec

    # If no successful candidate, assume attached label
    return spec

#===============================================================================
def get_label(filespec, cleaner=None):
    """
    Find the PDS3 label, clean it, load it, and parse into dictionary.

    Inputs:
        filespec:       File specification for the label.
        cleaner:        Function that takes a label as input and
                        return a cleaned label.
        no_standard:    If True, the standard cleaners are not used.
    """
    assert os.path.isfile(filespec), f'Not found: {filespec}'

    # Find the label
    labelspec = find_label(filespec)

    # Load and clean the PDS label
    lines = clean_label(pdsparser.PdsLabel.load_file(labelspec), cleaner=cleaner)

    # Parse the label into a dictionary
    return pdsparser.PdsLabel.from_string(lines).as_dict()

#===============================================================================
def pointer_int(value):
    """Return the integer part of a PDS3 pointer in units of records.

    Works whether or not as_dict() has been applied to the PdsLabel object.

    ("v1793917030_1.qub", 47) -> 47
    (47, 'RECORDS') -> 47
    (47, 'BYTES') -> ValueError raised
    47 -> 47
    """

    if isinstance(value, pdsparser.PdsLocalPointer):    # without as_dict()
        return value.value

    if isinstance(value, (list, tuple)):                # with as_dict()
        if value[1] == 'BYTES':
            raise ValueError('pointer has units of BYTES: ' + str(value))
        return [v for v in value if isinstance(v, int)][0]

    if isinstance(value, int):
        return value

    raise ValueError('not a valid pointer value: ' + str(value))

#=========================================================================================
def fast_dict(label, units=False, strings='concatenate'):
    """Return a hierarchical dictionary containing the content of a PDS3 label.

    This is _MUCH_ faster than pdsparser.PdsLabel.from_file (about 100x). However it does
    almost no syntax checking and may not work on some labels that are otherwise PDS3
    compliant.

    Note that, when a keyword or OBJECT/GROUP name is repeated, a numeric suffix "_2",
    "_3", etc. is appended to make it unique. This ensures that information in the label
    is not lost.

    Input:
        label       a PDS3 label as a file path (string or pathlib object), string, or
                    list of strings.
        units       False to ignore units inside "<>" characters. True to include them, in
                    which case a value with units is returned as a list with the unit
                    string as the second value.
        strings     how to handle extended strings within labels:
            'indent'        preserve indents and newlines;
            'newlines'      preserve newlines but not indents;
            'concatenate'   concatenate into one string with single space separators.
    """

    def clean_lines(lines):
        """Return a cleaned list of strings.

        Remove blank lines and comments; strip final END; merge extended lists and strings
        into a single line.
        """

        if strings not in {'concatenate', 'newlines', 'indent'}:
            raise ValueError(f'invalid strings option: "{strings}"')

        sep = ' ' if strings == 'concatenate' else '\n'

        cleaned = []
        prefix = ''
        quoted = False
        for line in lines:
            line = line.partition('/*')[0]      # strip trailing comments

            if strings == 'indent':
                text = line.rstrip()
                line = text.lstrip()
            else:
                line = line.strip()
                text = line

            # If we are currently inside a quoted string...
            if quoted:
                quote_count = len([c for c in line if c == '"'])
                if quote_count % 2:     # if odd
                    line = prefix + sep + text
                    prefix = ''
                    quoted = False
                else:
                    prefix += sep + text
                    line = ''
                    continue

            # Check for an unbalanced quote
            else:
                quote_count = len([c for c in line if c == '"'])
                if quote_count % 2:     # if odd
                    prefix = line
                    quoted = True
                    continue

            if not line:
                continue
            if line.startswith('/*'):
                continue
            if line == 'END':
                break

            # Check for a continuation line
            # This will fail if a comma is not the last char in an incomplete list
            if line[-1] in (',', '='):
                prefix = prefix + line
                continue

            # Otherwise, this line is complete
            cleaned.append(prefix + line)
            prefix = ''

        if prefix:
            cleaned.append(prefix)  # this will trigger an error in evaluate()

        return cleaned

    def evaluate(value):
        """Evaluate one PDS3-compliant value. Lists and sets are supported."""

        value = value.partition('/*')[0]
        value = value.strip()

        # Convert a set to a list
        is_set = value[0] == '{'
        if is_set:
            if value[-1] != '}':
                raise ValueError(f'unbalanced braces at "{value}"')
            value = '(' + value[1:-1] + ')'

        # Handle trailing units outside a list
        if value.endswith('>'):
            parts = value[:-1].partition('<')
            if units:
                value = '(' + parts[0] + ', "' + parts[-1] + '")'
            else:
                value = value.partition('<')[0].rstrip()

        # Evaluate. We take advantage of the fact that most PDS3-compliant values are also
        # Python-compliant.
        try:
            result = eval(value)
        except Exception:
            result = None

        # If evaluation failed...
        if result is None:

            # If the value is inside parens, attempt each component separately
            if value[0] == '(':
                if value[-1] != ')':
                    raise ValueError(f'unbalanced parentheses at "{value}"')

                parts = re.split(''',(?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', value[1:-1])
                    # from https://stackoverflow.com/questions/2785755/...
                    # ...how-to-split-but-ignore-separators-in-quoted-strings-in-python
                return [evaluate(p) for p in parts]

            # Otherwise, this is probably an unquoted string
            result = value

        # Otherwise, it's probably an un-quoted string

        # Convert a tuple to a list
        if isinstance(result, tuple):
            result = list(result)

        return result

    def unique_key(name, dict_):
        """This name if it is not in the dict_; otherwise with a numeric suffix appended
        to make it unique.
        """

        if name not in dict_:
            return name

        k = 2
        while (key := name +  '_' + str(k)) in dict_:
            k += 1

        return key

    def to_dict(lines):
        """The dictionary from a "cleaned" list of records."""

        state = [('', '', {})]              # list of (OBJECT or GROUP, name, dict)
        for k, line in enumerate(lines):

            # Get the name and value
            (name, equal, value) = line.partition('=')
            if not equal:
                raise ValueError(f'missing "=" at "{line}"')

            name = name.strip()
            value = evaluate(value)

            # If this begins an object, append it to the state list
            if name in ('OBJECT', 'GROUP'):
                state.append((name, value, {}))

            # If this ends an object, add this sub-dictionary to the dictionary
            elif name.startswith('END_'):
                if not state:
                    raise ValueError(f'unmatched {name} = {value}')

                (obj_type, obj_name, obj_dict) = state.pop()
                if obj_type != name[4:] or (value and value != obj_name):
                    raise ValueError(f'unbalanced {obj_type} = {obj_name}')
                        # tolerate END_OBJECT without name

                dict_ = state[-1][-1]
                key = unique_key(obj_name, dict_)
                dict_[key] = obj_dict

            # Otherwise, just append one new value to the current dictionary
            else:
                dict_ = state[-1][-1]
                key = unique_key(name, dict_)
                dict_[key] = value

        # Make sure all objects and groups were terminated
        (obj_type, obj_name, obj_dict) = state.pop()
        if len(state) > 1:
            raise ValueError(f'unbalanced {obj_type} = {obj_name}')

        return obj_dict

    #### Active code starts here

    if strings not in {'concatenate', 'newlines', 'indent'}:
        raise ValueError(f'invalid strings option: "{strings}"')

    # Convert to a list of strings
    if '\n' in label:                       # split a long string containing newlines
        lines = label.split('\n')
    elif isinstance(label, (str, Path)):    # read the content of a file path
        with open(label, 'r', encoding='latin8') as f:
            lines = f.readlines()
    elif isinstance(label, list):           # leave a list alone
        lines = label
    else:
        raise ValueError('not a recognized label')

    cleaned = clean_lines(lines)
    return to_dict(cleaned)

# now = datetime.now()
# for i in range(1000):
#     x = fast_read(label)
# datetime.now() - now
#
# now = datetime.now()
# for i in range(10):
#     x = pdsparser.PdsLabel.from_string(label)
# datetime.now() - now

################################################################################
# UNIT TESTS
################################################################################
import unittest
import os.path

from oops.hosts                   import pds3
from oops.unittester_support import TESTDATA_PARENT_DIRECTORY

#===========================================================================
def clean_VIMS(lines):
    for i in range(len(lines)):
        lines[i] = lines[i].replace('(N/A',  '("N/A"')
        lines[i] = lines[i].replace('N/A)',  '"N/A")')

        if lines[i].strip().endswith('*/') and '/*' not in lines[i]:
            lines[i] = '\n'

    return lines

#===========================================================================
def clean_UVIS(lines):
    for i in range(len(lines)):
        if 'CORE_UNIT' in lines[i]:
            if '"COUNTS/BIN"' not in lines[i]:
                lines[i] = lines[i].replace('COUNTS/BIN', '"COUNTS/BIN"')
        if lines[i].startswith('ODC_ID'):
            lines[i] = lines[i].replace(',', '')

    return lines

#===============================================================================
class Test_PDS3(unittest.TestCase):

    #===========================================================================
    def runTest(self):

        # Test extension replacement
        filespec = 'cassini/ISS/W1575634136_1.IMG'
        label_dict = pds3.get_label(os.path.join(TESTDATA_PARENT_DIRECTORY, filespec))
        self.assertTrue(label_dict['PDS_VERSION_ID'] == 'PDS3')

        # Test .LBL file input
        filespec = 'cassini/ISS/W1575634136_1.LBL'
        label_dict = pds3.get_label(os.path.join(TESTDATA_PARENT_DIRECTORY, filespec))
        self.assertTrue(label_dict['PDS_VERSION_ID'] == 'PDS3')

        # Test non-existent file
        filespec = 'nofile.crap'
        self.assertRaises(AssertionError, pds3.get_label, os.path.join(TESTDATA_PARENT_DIRECTORY, filespec))

        # Test Cassini VIMS file
        filespec = 'cassini/VIMS/v1793917030_1.lbl'
        label_dict = pds3.get_label(os.path.join(TESTDATA_PARENT_DIRECTORY, filespec),
                                    cleaner=clean_VIMS)
        self.assertTrue(label_dict['PDS_VERSION_ID'] == 'PDS3')

        # Test Cassini UVIS file
        filespec = 'cassini/UVIS/HSP2014_197_21_29.LBL'
        label_dict = pds3.get_label(os.path.join(TESTDATA_PARENT_DIRECTORY, filespec),
                                                 cleaner=clean_UVIS)
        self.assertTrue(label_dict['PDS_VERSION_ID'] == 'PDS3')

##############################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
