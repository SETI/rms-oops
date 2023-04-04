################################################################################
# hosts/pds3.py
#
# Utility functions for finding and loading PDS3 labels in host modules.
#
################################################################################
import os
import pdsparser
from pathlib import Path


#===========================================================================
def clean_comments(line):
    """
    Standard cleaner to handle commenting errors that could affect any host.
    """

    # Close-comment w/o open-comment
    if line.strip().endswith('*/') and '/*' not in line:
        line = '\n'

    return line

#===========================================================================
def call_cleaners(cleaners, lines):
    """
    Call the given cleaner functions.

    Inputs:
        cleaners:       List of cleaner functions.
        lines:          List of lines comprising the label.
    """

    for i in range(len(lines)):
        for cleaner in cleaners:
            lines[i] = cleaner(lines[i])

    return lines

#===========================================================================
def clean_label(lines, cleaners=None, no_standard=False):
    """
    Correct known errors in PDS3 labels.

    Inputs:
        lines:          List of lines comprising the label.
        cleaners:       List of functions that take a line as input and
                        return a cleaned line.  By default, these functions
                        are appended to a list of standard cleaner functions.
        no_standard:    If True, the standard cleaners are not used.
    """

    # Call standard cleaners
    standard_cleaners = [clean_comments]
    if not no_standard:
        lines = call_cleaners(standard_cleaners, lines)

    # Call custom cleaners
    if cleaners is not None:
        lines = call_cleaners(cleaners, lines)

    return lines

#===========================================================================
def find_label(filespec):
    """Find the PDS3 label corresponding to the given filespec."""

    # Construct candidate label filenames
    spec = Path(filespec)
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
    return filespec

#===========================================================================
def get_label(filespec, cleaners=None, no_standard=False):
    """
    Find the PDS3 label, clean it, load it, and parse into dictionary.

    Inputs:
        filespec:       File specification for the label.
        cleaners:       List of functions that take a label line as input and
                        return a cleaned line.  By default, these functions
                        are appended to a list of standard cleaner functions.
        no_standard:    If True, the standard cleaners are not used.
    """

    assert os.path.isfile(filespec), f'Not found: {filespec}'

    # Find the label
    labelspec = find_label(filespec)

    # Load and clean the PDS label
    lines = clean_label(pdsparser.PdsLabel.load_file(labelspec),
                        cleaners=cleaners, no_standard=no_standard)

    # Parse the label into a dictionary
    return pdsparser.PdsLabel.from_string(lines).as_dict()


################################################################################
# UNIT TESTS
################################################################################
import unittest
import os.path

from hosts                   import pds3
from oops.unittester_support import TESTDATA_PARENT_DIRECTORY


#===========================================================================
def clean_VIMS(line):
    line = line.replace('(N/A',  '("N/A"')
    line = line.replace('N/A)',  '"N/A")')
    return line

#===========================================================================
def clean_UVIS(line):
    if 'CORE_UNIT' in line:
        if '"COUNTS/BIN"' not in line:
            lines = line.replace('COUNTS/BIN', '"COUNTS/BIN"')
    if line.startswith('ODC_ID'):
        line = line.replace(',', '')
    return line


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
                                    cleaners=[clean_VIMS])
        self.assertTrue(label_dict['PDS_VERSION_ID'] == 'PDS3')

        # Test Cassini UVIS file
        filespec = 'cassini/UVIS/HSP2014_197_21_29.LBL'
        label_dict = pds3.get_label(os.path.join(TESTDATA_PARENT_DIRECTORY, filespec),
                                                 cleaners=[clean_UVIS])
        self.assertTrue(label_dict['PDS_VERSION_ID'] == 'PDS3')

##############################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
