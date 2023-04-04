################################################################################
# hosts/pds3.py
#
# Utility functions for finding and loading PDS3 labels in host modules.
#
################################################################################
import os
import pdsparser
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


################################################################################
# UNIT TESTS
################################################################################
import unittest
import os.path

from hosts                   import pds3
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
