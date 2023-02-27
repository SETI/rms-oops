################################################################################
# oops/inst/pds3.py
#
# Utility functions for finding and loading PDS3 labels in host modules.
#
################################################################################
import os
import pdsparser


#===============================================================================
class PDS3(object):
    """An instance-free class to for PDS3 label management methods."""

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
        """Find the PDS3 label corresponding to the given file spec."""

        # Construct candidate label filenames
        labelspecs = [filespec]
        if not filespec.upper().endswith('.LBL'):
            labelspecs.append(filespec.replace('.img', '.lbl'))
            labelspecs.append(filespec.replace('.IMG', '.lbl'))

        # Check candidates; return first one that exists
        for labelspec in labelspecs:
            if os.path.isfile(labelspec) :
                return labelspec 
                
        # If no successful candidate, assume attached label
        return filespec       

    #===========================================================================
    @staticmethod
    def get_label(filespec):
        """FInd the PDS3 label, clean it, load it, and parse into dictionary."""

        # Find the label
        labelspec = find_label(filespec)

        # Load and clean the PDS label
        lines = clean_label(pdsparser.PdsLabel.load_file(labelspec))

        # Parse the label into a dictionary
        return pdsparser.PdsLabel.from_string(lines).as_dict()


