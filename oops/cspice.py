################################################################################
# oops/cspice.py: cspice configuration based on version used
#
# Note that this overrides the default behavior of
#   import cspice
# inside the OOPS modules. This version makes it possible to test different
# versions of the cspice module within OOPS.
#
# The default version is cspice2. However, this can be overridden using the
# environment variable OOPS_CSPICE=1 or OOPS_CSPICE=0.
#
# Within OOPS, oops.cspice.VERSION will tell you the version loaded.
################################################################################

import os
import imp

try:
    # Always import cspice1
    import cspice1

    # This is tricky. The loaded version of cspice is saved as
    # cspice1.SELECTED_MODULE. By saving it inside cspice1, it persists between
    # the loading of other modules. This ensures that once a cspice version gets
    # loaded, it is assured that the same version will be used by all the OOPS
    # modules.
    module = cspice1.SELECTED_MODULE
    version = cspice1.SELECTED_VERSION

    # If we get this far, then some version of cspice has already been imported.
    # Here we make all of the functions of that module visible as modules of
    # this module.
    for (key, value) in module.__dict__.iteritems():
        globals()[key] = value

# If cspice has not yet been imported, cspice1.SELECTED_MODULE will not exist,
# so we land here
except AttributeError:

    # The default version of cspice is cspice2, but this can be modified at
    # runtime through the environment variable OOPS_CSPICE. For example,
    #   OOPS_CSPICE=0 ipython --pylab
    # will start an interactive session, where OOPS will use the original
    # version of cspice.
    version = int(os.getenv('OOPS_CSPICE', 2))

    if version == 1:
        # use_errors() is for backward compatibililty with the original cspice.
        cspice1.use_errors()
        cspice1.VERSION = 1

        # Define the cspice1 functions globally.
        # This is equivalent to "from cspice1 import *" but that is disallowed.
        for (key, value) in cspice1.__dict__.iteritems():
            globals()[key] = value

        # Save info about the selected module in cspice1 to avoid reload
        cspice1.SELECTED_MODULE = cspice1
        cspice1.SELECTED_VERSION = version

    elif version == 2:

        import cspice2
        cspice2.set_options('ERRORS', 'ALIASES')
        cspice2.use_noaliases() # Alias functions must be selected explicitly
        cspice2.VERSION = version

        # Define the cspice2 functions globally.
        # This is equivalent to "from cspice2 import *" but that is disallowed.
        for (key, value) in cspice2.__dict__.iteritems():
            globals()[key] = value

        cspice1.SELECTED_MODULE = cspice2
        cspice1.SELECTED_VERSION = version

    else:

        # Deal with issue that the original module is also named cspice
        # Here we define cspice0 to be the original module
        info = imp.find_module('cspice1')
        cspice0 = imp.load_module('cspice0', info[0], info[1][:-1], info[2])
        cspice0.VERSION = 0

        # Define the cspice0 functions globally.
        for (key, value) in cspice0.__dict__.iteritems():
            globals()[key] = value

        # Save info about the selected module in cspice1 to avoid reload
        cspice1.SELECTED_MODULE = cspice0
        cspice1.SELECTED_VERSION = version

    # Report the selection
    print 'Using cspice module version %d' % version

################################################################################

