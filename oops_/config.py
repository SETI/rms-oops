################################################################################
# oops_/config.py: General configuration parameters
#
# 2/29/12 Created (MRS)
################################################################################
# QuickPath and QuickFrame default parameters
#
# Disable the use of QuickPaths/Frames on an individual basis by calling the
# function with quick=False. The default set of parameters will be used whenever
# quick=True. If a function is called with quick as a dictionary, then any
# values in the dictionary override these defaults and the merged dictionary of
# parameters is used.
################################################################################

QUICK_DICTIONARY = {
    "use_quickpaths": True,
    "path_time_step": 0.01,     # time step in seconds.
    "path_time_extension": 0.,  # secs by which to extend interval at each end.
    "path_self_check": None,    # fractional precision for self-testing.
    "path_extra_steps": 4,      # number of extra time steps at each end.

    "use_quickframes": True,
    "frame_time_step": 0.05,    # time interval in seconds.
    "frame_time_extension": 0., # secs by which to extend interval at each end.
    "frame_self_check": None,   # fractional precision for self-testing.
    "frame_extra_steps": 4      # number of extra time steps at each end.
}

QUICK = True                    # Defines the default input argument as
                                # quick=True or quick=False.

################################################################################
# Photon solver parameters
################################################################################

# For Path.solve_photons()

class PATH_PHOTONS(object):
    max_iterations = 4          # Maximum number of iterations.
    dlt_precision = 1.e-8       # Iterations stops when every change in light
                                # travel time from one iteration to the next
                                # drops below this threshold.
    dlt_limit = 10.             # The allowed range of variations in light
                                # travel time before they are truncated. This
                                # should be related to the physical scale of
                                # the system being studied.

# For Surface.solve_photons()

class SURFACE_PHOTONS(object):
    max_iterations = 4          # Maximum number of iterations.
    dlt_precision = 1.e-8       # See PATH_PHOTONS for more info.
    dlt_limit = 10              # See PATH_PHOTONS for more info.

################################################################################
# Logging and Monitoring
################################################################################

class LOGGING(object):
    quickpath_creation = False  # Log the creation of QuickPaths.
    quickframe_creation = False # Log the creation of QuickFrames.
    path_iterations = False     # Log iterations of Path._solve_photons().
    surface_iterations = False  # Log iterations of Surface._solve_photons()

    @staticmethod
    def off():
        
        LOGGING.quickpath_creation = False
        LOGGING.quickframe_creation = False
        LOGGING.path_iterations = False
        LOGGING.surface_iterations = False

    @staticmethod
    def on():
        LOGGING.quickpath_creation = True
        LOGGING.quickframe_creation = True
        LOGGING.path_iterations = True
        LOGGING.surface_iterations = True

################################################################################

