################################################################################
# oops/inst/pds3.py
#
# Utility functions for finding and loading PDS3 labels in host modules.
#
################################################################################
import os
import pdsparser
from pathlib import Path

#===============================================================================
class PDS3(object):
    """An instance-free class for PDS3 label management methods."""

    #===========================================================================
    @staticmethod
    def clean_label(lines):
        """Correct common errors in PDS3 labels."""

        for i in range(len(lines)):

            # Relevant to Cassini VIMS
            line = lines[i]
            lines[i] = line.replace('(N/A',  '("N/A"')
            lines[i] = line.replace('N/A)',  '"N/A")')

            if line.strip().endswith('*/') and '/*' not in line:
                lines[i] = '\n'

            # Relevant to Cassini UVIS
            if 'CORE_UNIT' in line:
                if '"COUNTS/BIN"' not in line:
                    lines[i] = line.replace('COUNTS/BIN', '"COUNTS/BIN"')
            if line.startswith('ODC_ID'):
                lines[i] = line.replace(',', '')

        return lines

    #===========================================================================
    @staticmethod
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
    @staticmethod
    def get_label(filespec):
        """Find the PDS3 label, clean it, load it, and parse into dictionary."""

        assert os.path.isfile(filespec), f'Not found: {filespec}'

        # Find the label
        labelspec = PDS3.find_label(filespec)

        # Load and clean the PDS label
        lines = PDS3.clean_label(pdsparser.PdsLabel.load_file(labelspec))

        # Parse the label into a dictionary
        return pdsparser.PdsLabel.from_string(lines).as_dict()


################################################################################
# UNIT TESTS
################################################################################
import unittest
import os.path

from hosts.pds3              import PDS3
from oops.unittester_support import TESTDATA_PARENT_DIRECTORY


#*******************************************************************************
class Test_PDS3(unittest.TestCase):

    #===========================================================================
    def runTest(self):

        # Test extension replacement
        filespec = 'cassini/ISS/W1575634136_1.IMG'
        label_dict = PDS3.get_label(os.path.join(TESTDATA_PARENT_DIRECTORY, filespec))
        self.assertTrue(label_dict['PDS_VERSION_ID'] == 'PDS3')

        # Test .LBL file input
        filespec = 'cassini/ISS/W1575634136_1.LBL'
        label_dict = PDS3.get_label(os.path.join(TESTDATA_PARENT_DIRECTORY, filespec))
        self.assertTrue(label_dict['PDS_VERSION_ID'] == 'PDS3')

        # Test non-existent file
        filespec = 'nofile.crap'
        self.assertRaises(AssertionError, PDS3.get_label, os.path.join(TESTDATA_PARENT_DIRECTORY, filespec))


##############################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
