##########################################################################################
# oops/config.py: General configuration parameters
##########################################################################################

import datetime
import logging
import sys
import traceback
import warnings

##########################################################################################
# QuickPath and QuickFrame default parameters
##########################################################################################

class QUICK(object):
    """Default parameters for :class:`~oops.path.QuickPath` and
    :class:`~oops.frame.QuickFrame` interpolation.

    These are the default parameters. Any of these parameters can be overridden on a
    one-time basis by calling a function with `quick` as an input dictionary containing
    temporary replacement values. Use ``quick = False`` to temporarily disable the use of
    QuickPaths and QuickFrames.
    """

    flag = True                 # Defines the default behavior as quick=True or
                                # quick=False.

    dictionary = {
        'use_quickpaths': True,
        'path_time_step': 0.05,             # time step in seconds.
        'path_time_extension': 5.,          # secs by which to extend interval at each
                                            # end.
        'path_self_check': None,            # fractional precision for self-testing.
        'path_extra_steps': 4,              # number of extra time steps at each end.
        'quickpath_cache_size': 40,         # maximum number of non-overlapping
                                            # quickpaths to cache for any given path.
        'quickpath_linear_interpolation_threshold': 3.,
                                            # if a time span is less than this amount,
                                            # perform linear interpolation instead of
                                            # using InterpolatedUnivariateSpline; this
                                            # improves performance

        'use_quickframes': True,
        'frame_time_step': 0.05,            # time interval in seconds.
        'frame_time_extension': 5.,         # secs by which to extend interval at each
                                            # end.
        'frame_self_check': None,           # fractional precision for self-testing.
        'frame_extra_steps': 4,             # number of extra time steps at each end.
        'quickframe_cache_size': 40,        # maximum number of non-overlapping
                                            # quickframes to cache for any given frame.
        'quickframe_linear_interpolation_threshold': 1.,
                                            # if a time span is less than this amount,
                                            # perform linear interpolation instead of
                                            # using InterpolatedUnivariateSpline; this
                                            # improves performance
        'quickframe_numerical_omega': False,
                                            # True to derive the omega rotation vectors
                                            # via numerical derivatives rather than via
                                            # interpolation of the vector components.
        'ignore_quickframe_omega': False,
                                            # True to treat the omega rotation vectors
                                            # as zero within a QuickFrame.
    }

##########################################################################################
# Photon solver parameters
##########################################################################################

# For Path._solve_photon()

class PATH_PHOTONS(object):
    """Convergence parameters for the Path photon solver."""

    max_iterations = 4          # Maximum number of iterations.
    dlt_precision = 3.e-7       # Iterations stops when every change in light
                                # travel time from one iteration to the next
                                # drops below this threshold. This is roughly
                                # the accuracy of a double-precision time in
                                # seconds for dates within a few decades of the
                                # year 2000.
    dlt_limit = 10.             # The allowed range of variations in light
                                # travel time before they are truncated. This
                                # should be related to the physical scale of
                                # the system being studied.
    km_precision = 1.e-4        # Default precision goal in geometry solutions
                                # is ten cm.
    rel_precision = 1.e-10      # Between sqrt(machine precision) and machine
                                # precision.

# For Surface._solve_photon_by_los()

class SURFACE_PHOTONS(object):
    """Convergence parameters for the Surface photon solver."""

    max_iterations = 6          # Maximum number of iterations.
    dlt_precision = 3.e-7       # See PATH_PHOTONS for more info.
    dlt_limit = 10.             # See PATH_PHOTONS for more info.
    collapse_threshold = 3.     # When a surface intercept consists of a range
                                # of times smaller than this threshold, the
                                # times are converted to a single value.
                                # This approximation can speed up some
                                # calculations substantially.
    km_precision = 1.e-4        # Default precision goal in geometry solutions
                                # is ten cm.
    rel_precision = 1.e-10      # Between sqrt(machine precision) and machine
                                # precision.

##########################################################################################
# Event precision
##########################################################################################

class EVENT_CONFIG(object):
    """Precision parameters for Event times."""

    collapse_threshold = 3.     # When an event returned by a calculation spans
                                # a range of times smaller than this threshold,
                                # the time field is converted to a single value.
                                # This approximation can speed up some
                                # calculations substantially.

##########################################################################################
# Logging and Monitoring
##########################################################################################

LOG_DATEFMT = '%Y-%m-%d %H:%M:%S.%f'
LOG_FORMAT  = '%(mytime)s | %(name)s | %(mylevelname)-5s | %(message)s'
LOG_FORMATTER = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)

LOGGING_STACK = []

class LOGGING(object):
    """Switches controlling what the library logs."""

    prefix = '   '                  # Prefix in front of a log message
    quickpath_creation = False      # Log the creation of QuickPaths
    quickframe_creation = False     # Log the creation of QuickFrames
    fov_iterations = False          # Log iterations of FOV solvers
    path_iterations = False         # Log iterations of Path photon solvers
    surface_iterations = False      # Log iterations of Surface photon solvers
    observation_iterations = False  # Log iterations of Observation solvers
    event_time_collapse = False     # Report event time collapse
    surface_time_collapse = False   # Report surface time collapse

    stdout = True                   # Write logging info to stdout.
    stderr = False                  # Write logging info to stderr.
    file_path = None                # Additional/alternative output file.
    _file = None                    # File object for the above.
    logger = None                   # logger or PdsLogger object.
    level = logging.DEBUG           # Minimum logging level.
    handlers = set()                # Set of handlers for logger.
    log_formatting = True           # False while the handlers carry the formatter of
                                    # a literal message instead of LOG_FORMATTER.
    warnings = 0                    # Warning count.
    errors = 0                      # Error count.
    lines = 0                       # Number of lines logged.
    python_warnings = False         # Use Python warnings on LOGGING.warn()

    LEVELS = {                      # Static dictionary of logging levels
        'debug'   : logging.DEBUG,
        'info'    : logging.INFO,
        'warn'    : logging.WARNING,
        'warning' : logging.WARNING,
        'error'   : logging.ERROR,
        'fatal'   : logging.FATAL,
        'critical': logging.CRITICAL,
    }

    # Create a dictionary mapping level numbers to names
    LEVEL_NAMES = {}
    for name in LEVELS.keys():
        level_val = LEVELS[name]
        LEVEL_NAMES[level_val] = name.upper()
        for k in range(level_val+1, level_val+10):
            LEVEL_NAMES[k] = name.upper() + '[%d]' % k

    for k in range(1, 10):
        LEVEL_NAMES[k] = str(k)

    @staticmethod
    def reset():
        """Reset the error, warning, and line counts to zero."""

        LOGGING.warnings = 0
        LOGGING.errors = 0
        LOGGING.lines = 0

    @staticmethod
    def all(flag, category='', reset=False):
        """Turn one or more categories of messages on or off.

        Parameters:
            flag (bool): True to turn the selected categories on; False to turn them off.
            category (str, optional): "convergence" to select the FOV, path, surface, and
                observation iteration messages, or "diagnostics" to select the QuickPath
                and QuickFrame creation and time-collapse messages. Default is "", meaning
                every category.
            reset (bool, optional): True to zero the warning, error, and line counts;
                default False.
        """

        if not category or 'convergence' in category:
            LOGGING.fov_iterations = flag
            LOGGING.path_iterations = flag
            LOGGING.surface_iterations = flag
            LOGGING.observation_iterations = flag

        if not category or 'diagnostics' in category:
            LOGGING.quickpath_creation = flag
            LOGGING.quickframe_creation = flag
            LOGGING.event_time_collapse = flag
            LOGGING.surface_time_collapse = flag

        if reset:
            LOGGING.reset()

        # Never allow stdout=False if other logging methods are off
        if flag and not any([LOGGING._file, LOGGING.logger, LOGGING.stderr]):
            LOGGING.stdout = True

    @staticmethod
    def off(category='', reset=True):
        """Turn one or more categories of messages off.

        Parameters:
            category (str, optional): The category to turn off, as for `all`; default "",
                meaning every category.
            reset (bool, optional): True to zero the warning, error, and line counts;
                default True.
        """

        LOGGING.all(False, category=category, reset=reset)

    @staticmethod
    def on(prefix='   ', category='', reset=False):
        """Turn one or more categories of messages on.

        Parameters:
            prefix (str, optional): The string to write in front of each log message;
                default is three spaces.
            category (str, optional): The category to turn on, as for `all`; default "",
                meaning every category.
            reset (bool, optional): True to zero the warning, error, and line counts;
                default False.
        """

        LOGGING.all(True, category=category, reset=reset)
        LOGGING.prefix = prefix

    @staticmethod
    def set_stdout(flag, reset=False):
        """Enable or disable log messages to stdout.

        Parameters:
            flag (bool): True to write log messages to stdout; False to stop. Stdout
                remains enabled regardless if no file, logger, or stderr destination is
                active, because that would leave the messages nowhere to go.
            reset (bool, optional): True to zero the warning, error, and line counts;
                default False.
        """

        LOGGING.stdout = bool(flag)

        # Never allow stdout=False if other logging methods are off
        if not any([LOGGING._file, LOGGING.logger, LOGGING.stderr]):
            LOGGING.stdout = True

        if reset:
            LOGGING.reset()

    @staticmethod
    def set_stderr(flag, reset=False):
        """Enable or disable log messages to stderr.

        Parameters:
            flag (bool): True to write log messages to stderr; False to stop.
            reset (bool, optional): True to zero the warning, error, and line counts;
                default False.
        """

        LOGGING.stderr = bool(flag)

        # Never allow stdout=False if other logging methods are off
        if not any([LOGGING._file, LOGGING.logger, LOGGING.stderr]):
            LOGGING.stdout = True

        if reset:
            LOGGING.reset()

    @staticmethod
    def set_file(file_path='', reset=False):
        """Send log messages to a file; use a blank file path to disable.

        Any file already open for logging is closed first.

        Parameters:
            file_path (str, optional): Path of the file to write; default "", which
                disables file logging.
            reset (bool, optional): True to zero the warning, error, and line counts;
                default False.

        Raises:
            OSError: If the named file cannot be opened for writing.
        """

        if LOGGING._file:
            LOGGING._file.close()
            LOGGING._file = None        # otherwise the next message writes to a
                                        # closed file

        LOGGING.file_path = file_path
        if LOGGING.file_path:
            try:
                LOGGING._file = open(LOGGING.file_path, 'w+')
            except OSError:
                LOGGING._file = None
                LOGGING.file_path = ''
                if not any([LOGGING.logger, LOGGING.stderr]):
                    LOGGING.stdout = True
                raise

        else:
            # Never allow stdout=False if other logging methods are off
            if not any([LOGGING.logger, LOGGING.stderr]):
                LOGGING.stdout = True

        if reset:
            LOGGING.reset()

    @staticmethod
    def set_logger(logger=None, level='DEBUG', reset=False):
        """Send log messages to a logger; None to disable Python logging.

        Parameters:
            logger (Logger, optional): The logger to receive log messages; None, the
                default, disables Python logging and restores the default level.
            level (str or int, optional): Minimum level for the logger, given as a name
                such as "DEBUG" or as an integer; default "DEBUG". Ignored if `logger` is
                None.
            reset (bool, optional): True to zero the warning, error, and line counts;
                default False.
        """

        LOGGING.logger = logger

        # Never allow stdout=False if other logging methods are off
        if not any([LOGGING._file, LOGGING.logger, LOGGING.stderr]):
            LOGGING.stdout = True

        if reset:
            LOGGING.reset()

        if logger is None:      # reset defaults
            LOGGING.level = logging.DEBUG
            LOGGING.handlers = set()
            LOGGING.log_formatting = True
            return

        # Apply the formatter to each handler
        LOGGING.handlers = set()
        LOGGING._check_logger_formatters()

        # Interpret and set the minimum logging level
        if isinstance(level, str):
            level = LOGGING.LEVELS[level.lower()]
        LOGGING.logger.setLevel(level)
        LOGGING.level = level

    @staticmethod
    def set_logger_level(level):
        """Set the logging level of the logger.

        Parameters:
            level (str or int): The minimum level, given as a name such as "DEBUG" or as
                an integer.
        """

        if isinstance(level, str):
            level = LOGGING.LEVELS[level.lower()]
        LOGGING.logger.setLevel(level)
        LOGGING.level = level

    @staticmethod
    def _check_logger_formatters():
        """Make sure all handlers have their formatter set properly.

        A literal message installs a formatter of its own on every handler and clears
        `log_formatting`. This restores LOG_FORMATTER, and the flag with it, so the
        handlers are only re-formatted once rather than on every message afterward.
        """

        if LOGGING.log_formatting:
            # Every handler seen before still carries the formatter, so only one added
            # since the last message needs it
            handlers = set(LOGGING.logger.handlers) - LOGGING.handlers
        else:
            # A literal message replaced the formatter on every handler
            handlers = set(LOGGING.logger.handlers)
            LOGGING.log_formatting = True

        for handler in handlers:
            handler.setFormatter(LOG_FORMATTER)

        LOGGING.handlers = set(LOGGING.logger.handlers)

    @staticmethod
    def print(*args, level=logging.INFO, literal=False, force=False):
        """Print a log message for a given log level.

        The message is constructed by converting each argument to a string, and
        then concatenating them with spaces in between.

        Parameters:
            *args: The values to log; each is converted to a string.
            level (int or str, optional): Logging level, one of "DEBUG"=10, "INFO"=20,
                "WARN" or "WARNING"=30, "ERROR"=40, or "FATAL"=50; default "INFO". A
                message reaches a defined logger only if the level is at or above that
                logger's threshold, but it is sent to other streams regardless.
            literal (bool, optional): True to log the message as it is, without a time
                tag, level, or other information, though any specified prefix is still
                included. Default is False.
            force (bool, optional): True to log the message even if its level falls below
                that of the logger; default False.
        """

        # Interpret level
        if isinstance(level, str):
            level = LOGGING.LEVELS[level.lower()]

        # Update the prefix based on the level
        prefix = LOGGING.prefix
        if not literal:
            if level >= logging.ERROR:
                prefix += 'ERROR:'
                LOGGING.errors += 1
            elif level >= logging.WARNING:
                prefix += 'WARNING:'
                LOGGING.warnings += 1

        # The Python warnings filter mechanism can be used, so a warning might
        # be suppressed or converted to an error. If suppressed, it will still
        # be sent to a logger.
        user_was_warned = False
        if (not literal and LOGGING.python_warnings
                and logging.WARNING <= level < logging.ERROR):
            warnings.warn(' '.join([str(x) for x in args]))
            user_was_warned = True

        if prefix:
            prefix_ = prefix + ' '
        else:
            prefix_ = ''

        # Write to stdout
        if LOGGING.stdout and level >= LOGGING.level and not user_was_warned:
            if prefix:
                print(prefix, *args)
            else:
                print(*args)

        # Prep message for other destinations
        if LOGGING.stderr or LOGGING._file or LOGGING.logger:
            message = ' '.join([str(x) for x in args])

        # Write to stderr
        if LOGGING.stderr and not user_was_warned:
            sys.stderr.write(prefix_)
            sys.stderr.write(message)
            sys.stderr.write('\n')

        # Write to a file
        if LOGGING._file and not user_was_warned:
            LOGGING._file.write(prefix_)
            LOGGING._file.write(message)
            LOGGING._file.write('\n')

        # Write to a logger
        if LOGGING.logger:

            # Turn log formatting on or off as needed
            if literal:
                LOGGING.log_formatting = False
                formatter = logging.Formatter(prefix_ + '%(message)s')
                for handler in LOGGING.logger.handlers:
                    handler.setFormatter(formatter)
                mytime = ''     # not used but must be defined

            else:
                LOGGING._check_logger_formatters()
                now = datetime.datetime.now()
                mytime = now.strftime('%Y-%m-%d %H:%M:%S.%f')

            extras = {'mytime': mytime,
                      'mylevelname': LOGGING.LEVEL_NAMES[level]}

            if force and level < LOGGING.level:
                LOGGING.logger.log(LOGGING.level, message, extra=extras)
            else:
                LOGGING.logger.log(level, message, extra=extras)

        LOGGING.lines += (force or level >= LOGGING.level)

    @staticmethod
    def debug(*args, force=False):
        """Same as print(*args, level='DEBUG')."""
        LOGGING.print(*args, level=logging.DEBUG, literal=False, force=force)

    @staticmethod
    def info(*args, force=False):
        """Same as print(*args, level='INFO')."""
        LOGGING.print(*args, level=logging.INFO, literal=False, force=force)

    @staticmethod
    def warn(*args, force=False):
        """Same as print(*args, level='WARN')."""
        LOGGING.print(*args, level=logging.WARN, literal=False, force=force)

    @staticmethod
    def error(*args, force=False):
        """Same as print(*args, level='ERROR')."""
        LOGGING.print(*args, level=logging.ERROR, literal=False, force=force)

    @staticmethod
    def fatal(*args, force=False):
        """Same as print(*args, level='FATAL')."""
        LOGGING.print(*args, level=logging.FATAL, literal=False, force=force)

    @staticmethod
    def convergence(*args, force=False):
        """Print a convergence message."""
        LOGGING.print(*args, level=logging.DEBUG, literal=True, force=force)

    @staticmethod
    def diagnostic(*args, force=False):
        """Print a diagnostic message."""
        LOGGING.print(*args, level=logging.DEBUG, literal=True, force=force)

    @staticmethod
    def diagnostics(*args, force=False):
        """Print a diagnostic message."""
        LOGGING.print(*args, level=logging.DEBUG, literal=True, force=force)

    @staticmethod
    def performance(*args, force=False):
        """Print a performance message."""
        LOGGING.print(*args, level=logging.DEBUG, literal=True, force=force)

    @staticmethod
    def exception(exception, message=''):
        """Log an exception with its traceback at FATAL level.

        If no logger is defined, the exception is raised instead of logged.
        """

        if not LOGGING.logger:
            raise exception

        LOGGING._check_logger_formatters()
        now = datetime.datetime.now()
        mytime = now.strftime('%Y-%m-%d %H:%M:%S.%f')

        (etype, value, tb) = sys.exc_info()
        if etype is None:
            error_msg = type(exception).__name__ + ' ' + str(exception)
            value = str(exception)
            tb_msg = ''
        else:
            error_msg = etype.__name__ + ' ' + str(value)
            tb_msg = '\n'.join(traceback.format_tb(tb)) + '\n'

        if tb_msg:
            message = (message or error_msg) + '\n' \
                      + tb_msg + '  ' + error_msg + '\n'
        elif message:
            message += '\n  ' + error_msg
        else:
            message = error_msg

        LOGGING.logger.fatal(message, extra={'mytime': mytime,
                                             'mylevelname': 'FATAL'})
        LOGGING.errors += 1

    @staticmethod
    def literal(*args, level=logging.DEBUG, force=True):
        """Print a literal message to the log."""
        LOGGING.print(*args, level=level, literal=True, force=force)

    @staticmethod
    def push():
        """Push the current LOGGING settings onto a stack."""

        global LOGGING_STACK

        state = {}
        for key, value in LOGGING.__dict__.items():
            if not key.startswith('_'):
                state[key] = value

        # Limit the stack to ten
        LOGGING_STACK.append(state)
        if len(LOGGING_STACK) > 10:
            LOGGING_STACK = LOGGING_STACK[1:]

    @staticmethod
    def pop():
        """Pop the previous LOGGING settings from a stack."""

        global LOGGING_STACK

        state = LOGGING_STACK.pop()
        old_level = LOGGING.level

        if not LOGGING_STACK:
            LOGGING_STACK.append(state)

        for key, value in state.items():
            setattr(LOGGING, key, value)

        if LOGGING.logger and old_level != LOGGING.level:
            LOGGING.set_logger_level(LOGGING.level)

LOGGING.push()      # At initialization, put the default settings onto the stack

##########################################################################################
# Serialization parameters
##########################################################################################

class PICKLE_CONFIG(object):
    """Switches controlling how much internal detail is pickled."""

    quickpath_details = True    # Save the internals of a QuickPath
    quickframe_details = True   # Save the internals of a QuickFrame
    backplane_events = True     # Save the events dictionary inside backplanes

##########################################################################################
# Support for the old FOV.area_factor(), for backward compatibility
##########################################################################################

class AREA_FACTOR(object):
    """Switch selecting the definition used by :meth:`~oops.FOV.area_factor`."""

    old = False     # True to use old area factors, which have a small error due
                    # to the fact that off-axis lines of sight in an FOV are
                    # not quite unit length.

##########################################################################################

