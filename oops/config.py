################################################################################
# oops/config.py: General configuration parameters
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

class QUICK(object):
  flag = True                   # Defines the default behavior as quick=True or
                                # quick=False.

  dictionary = {
    "use_quickpaths": True,
    "path_time_step": 0.05,     # time step in seconds.
    "path_time_extension": 0.,  # secs by which to extend interval at each end.
    "path_self_check": None,    # fractional precision for self-testing.
    "path_extra_steps": 4,      # number of extra time steps at each end.
    "quickpath_cache": 4,       # maximum number of non-overlapping quickpaths
                                # to cache for any given path.

    "use_quickframes": True,
    "frame_time_step": 0.5,     # time interval in seconds.
    "frame_time_extension": 0., # secs by which to extend interval at each end.
    "frame_self_check": None,   # fractional precision for self-testing.
    "frame_extra_steps": 4,     # number of extra time steps at each end.
    "quickframe_cache": 4       # maximum number of non-overlapping quickframes
                                # to cache for any given frame.
}

################################################################################
# Photon solver parameters
################################################################################

# For Path._solve_photons()

class PATH_PHOTONS(object):
    max_iterations = 4          # Maximum number of iterations.
    dlt_precision = 1.e-6       # Iterations stops when every change in light
                                # travel time from one iteration to the next
                                # drops below this threshold.
    dlt_limit = 10.             # The allowed range of variations in light
                                # travel time before they are truncated. This
                                # should be related to the physical scale of
                                # the system being studied.

# For Surface._solve_photons()

class SURFACE_PHOTONS(object):
    max_iterations = 4          # Maximum number of iterations.
    dlt_precision = 1.e-6       # See PATH_PHOTONS for more info.
    dlt_limit = 10.             # See PATH_PHOTONS for more info.

################################################################################
# Event precision
################################################################################

class EVENT_CONFIG(object):
    collapse_threshold = 3.     # When an event returned by a calculation spans
                                # a range of times smaller than this threshold,
                                # the time field is converted to a single value.
                                # This approximation can speed up some
                                # calculations substantially.

################################################################################
# Logging and Monitoring
################################################################################

class LOGGING(object):
    prefix = ""                     # Prefix in front of a log message
    quickpath_creation = False      # Log the creation of QuickPaths.
    quickframe_creation = False     # Log the creation of QuickFrames.
    path_iterations = False         # Log iterations of Path._solve_photons().
    surface_iterations = False      # Log iterations of Surface._solve_photons()
    event_time_collapse = False     # Report event time collapse

    @staticmethod
    def all(flag):
        LOGGING.quickpath_creation = flag
        LOGGING.quickframe_creation = flag
        LOGGING.path_iterations = flag
        LOGGING.surface_iterations = flag
        LOGGING.event_time_collapse = flag

    @staticmethod
    def off(): LOGGING.all(False)

    @staticmethod
    def on(prefix=""):
        LOGGING.all(True)
        LOGGING.prefix = prefix

################################################################################

