################################################################################
# oops/format_/unittester.py
################################################################################

import unittest

from oops.format_.format     import Test_Format
from oops.format_.hms        import Test_HMS
from oops.format_.pythonfmt  import Test_PythonFmt

########################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
