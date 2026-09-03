##########################################################################################
# tests/test_config.py
##########################################################################################

import logging
from collections.abc import Iterator

import pytest

from oops.config import LOG_FORMATTER, LOGGING


# Every mutable class attribute of LOGGING, saved and restored around each test.
LOGGING_ATTRS = ('logger', 'level', 'handlers', 'log_formatting', 'stdout', 'stderr',
                 'lines', 'errors', 'warnings', 'prefix', 'file_path', '_file',
                 'fov_iterations', 'path_iterations', 'surface_iterations',
                 'observation_iterations', 'quickpath_creation', 'quickframe_creation',
                 'path_time_collapse', 'frame_time_collapse', 'surface_time_collapse',
                 'event_time_collapse')


@pytest.fixture(autouse=True)
def restore_logging() -> Iterator[None]:
    """Save and restore the LOGGING class attributes that these tests modify.

    LOGGING is a module-level namespace of mutable class attributes, so a test that
    installs a logger has to put the previous state back or it leaks into every test
    that runs afterward.
    """

    saved = {name: getattr(LOGGING, name) for name in LOGGING_ATTRS
             if hasattr(LOGGING, name)}

    yield

    for name, value in saved.items():
        setattr(LOGGING, name, value)


@pytest.fixture
def logger() -> logging.Logger:
    """A quiet logger carrying one handler, installed as the LOGGING destination."""

    log = logging.Logger('test_config')
    log.propagate = False
    log.addHandler(logging.NullHandler())

    LOGGING.set_logger(log)
    LOGGING.set_stdout(False)

    return log


def test_literal_message_replaces_the_formatter(logger: logging.Logger) -> None:
    """A literal message installs a formatter of its own on every handler."""

    LOGGING.print('literal', literal=True)

    assert logger.handlers[0].formatter is not LOG_FORMATTER


def test_literal_message_clears_the_formatting_flag(logger: logging.Logger) -> None:
    """A literal message records that the handlers no longer carry LOG_FORMATTER."""

    LOGGING.print('literal', literal=True)

    assert LOGGING.log_formatting is False


def test_formatted_message_restores_the_formatter(logger: logging.Logger) -> None:
    """The next formatted message puts LOG_FORMATTER back on the handler."""

    LOGGING.print('literal', literal=True)
    LOGGING.print('formatted')

    assert logger.handlers[0].formatter is LOG_FORMATTER


def test_formatted_message_restores_the_formatting_flag(logger: logging.Logger) -> None:
    """The flag returns to True once the handlers carry LOG_FORMATTER again."""

    LOGGING.print('literal', literal=True)
    LOGGING.print('formatted')

    assert LOGGING.log_formatting is True


def test_handler_added_later_gets_the_formatter(logger: logging.Logger) -> None:
    """A handler added after the logger was installed is formatted on the next message."""

    LOGGING.print('formatted')
    added = logging.NullHandler()
    logger.addHandler(added)
    LOGGING.print('formatted again')

    assert added.formatter is LOG_FORMATTER


def test_handlers_accumulate(logger: logging.Logger) -> None:
    """Every handler that has been given the formatter is remembered, not just the
    newest one."""

    LOGGING.print('formatted')
    added = logging.NullHandler()
    logger.addHandler(added)
    LOGGING.print('formatted again')

    assert LOGGING.handlers == set(logger.handlers)


def test_formatter_is_not_reapplied_to_a_known_handler(logger: logging.Logger) -> None:
    """A handler already carrying the formatter is left alone by a later message."""

    LOGGING.print('formatted')
    logger.handlers[0].setFormatter(None)
    LOGGING.print('formatted again')

    assert logger.handlers[0].formatter is None


def test_disabling_the_logger_restores_the_formatting_flag(logger: logging.Logger) -> None:
    """set_logger(None) resets the flag along with the rest of the defaults."""

    LOGGING.print('literal', literal=True)
    LOGGING.set_logger(None)

    assert LOGGING.log_formatting is True


def test_reset_zeroes_the_counts(logger: logging.Logger) -> None:
    """reset() returns the error, warning, and line counts to zero."""

    LOGGING.error('an error')
    LOGGING.warn('a warning')
    LOGGING.reset()

    assert LOGGING.errors == 0
    assert LOGGING.warnings == 0
    assert LOGGING.lines == 0


def test_counts_accumulate(logger: logging.Logger) -> None:
    """Each message increments the line count, and errors and warnings their own."""

    LOGGING.reset()
    LOGGING.error('an error')
    LOGGING.warn('a warning')
    LOGGING.info('a note')

    assert LOGGING.errors == 1
    assert LOGGING.warnings == 1
    assert LOGGING.lines == 3


def test_on_enables_every_category() -> None:
    """on() with no category turns every category of message on."""

    LOGGING.off()
    LOGGING.on()

    assert LOGGING.fov_iterations
    assert LOGGING.path_iterations
    assert LOGGING.surface_iterations
    assert LOGGING.observation_iterations
    assert LOGGING.quickpath_creation
    assert LOGGING.quickframe_creation


def test_off_disables_every_category() -> None:
    """off() with no category turns every category of message off."""

    LOGGING.on()
    LOGGING.off()

    assert not LOGGING.fov_iterations
    assert not LOGGING.path_iterations
    assert not LOGGING.surface_iterations
    assert not LOGGING.observation_iterations
    assert not LOGGING.quickpath_creation
    assert not LOGGING.quickframe_creation


def test_convergence_category_selects_the_iteration_messages() -> None:
    """"convergence" selects the FOV, path, surface, and observation messages."""

    LOGGING.off()
    LOGGING.on(category='convergence')

    assert LOGGING.fov_iterations
    assert LOGGING.path_iterations
    assert LOGGING.surface_iterations
    assert LOGGING.observation_iterations

    # ...and leaves the diagnostics alone
    assert not LOGGING.quickpath_creation
    assert not LOGGING.quickframe_creation


def test_diagnostics_category_selects_the_quick_messages() -> None:
    """"diagnostics" selects the QuickPath and QuickFrame messages."""

    LOGGING.off()
    LOGGING.on(category='diagnostics')

    assert LOGGING.quickpath_creation
    assert LOGGING.quickframe_creation
    assert not LOGGING.fov_iterations
    assert not LOGGING.path_iterations


def test_all_turns_categories_on_and_off() -> None:
    """all() takes the flag directly rather than through on() or off()."""

    LOGGING.all(True, category='convergence')
    assert LOGGING.fov_iterations

    LOGGING.all(False, category='convergence')
    assert not LOGGING.fov_iterations


def test_on_records_the_prefix() -> None:
    """The prefix is written in front of each log message."""

    LOGGING.on(prefix='>>> ')

    assert LOGGING.prefix == '>>> '


def test_off_resets_the_counts_by_default(logger: logging.Logger) -> None:
    """off() zeroes the counts unless told otherwise."""

    LOGGING.error('an error')
    LOGGING.off()
    assert LOGGING.errors == 0

    LOGGING.error('another error')
    LOGGING.off(reset=False)
    assert LOGGING.errors == 1


def test_on_leaves_the_counts_alone_by_default(logger: logging.Logger) -> None:
    """on() preserves the counts unless told otherwise."""

    LOGGING.reset()
    LOGGING.error('an error')
    LOGGING.on()
    assert LOGGING.errors == 1

    LOGGING.on(reset=True)
    assert LOGGING.errors == 0


def test_set_stdout(logger: logging.Logger) -> None:
    """Log messages to stdout can be switched on and off."""

    LOGGING.set_stdout(True)
    assert LOGGING.stdout

    LOGGING.set_stdout(False)
    assert not LOGGING.stdout


def test_stdout_stays_on_with_no_other_destination() -> None:
    """Stdout remains enabled when disabling it would leave messages nowhere to go."""

    LOGGING.set_logger(None)
    LOGGING.set_stderr(False)
    LOGGING.set_file('')
    LOGGING.set_stdout(False)

    assert LOGGING.stdout


def test_set_stderr(logger: logging.Logger) -> None:
    """Log messages to stderr can be switched on and off."""

    LOGGING.set_stderr(True)
    assert LOGGING.stderr

    LOGGING.set_stderr(False)
    assert not LOGGING.stderr


def test_stderr_receives_messages(logger: logging.Logger, capsys) -> None:
    """A message reaches stderr once stderr is enabled."""

    LOGGING.set_stdout(False)
    LOGGING.set_stderr(True)
    LOGGING.info('to stderr')

    assert 'to stderr' in capsys.readouterr().err


def test_stdout_receives_messages(logger: logging.Logger, capsys) -> None:
    """A message reaches stdout once stdout is enabled."""

    LOGGING.set_stdout(True)
    LOGGING.info('to stdout')

    assert 'to stdout' in capsys.readouterr().out


def test_set_file_writes_the_messages(tmp_path, logger: logging.Logger) -> None:
    """Messages are appended to the named file until file logging is disabled."""

    log_file = tmp_path / 'oops.log'
    LOGGING.set_stdout(False)
    LOGGING.set_file(str(log_file))
    try:
        LOGGING.info('into the file')
    finally:
        LOGGING.set_file('')

    assert 'into the file' in log_file.read_text()


def test_set_file_records_the_path(tmp_path, logger: logging.Logger) -> None:
    """The active log file path is remembered while the file is open."""

    log_file = tmp_path / 'oops.log'
    LOGGING.set_file(str(log_file))
    try:
        assert LOGGING.file_path == str(log_file)
    finally:
        LOGGING.set_file('')

    assert not LOGGING.file_path


def test_blank_file_path_stops_the_file_logging(tmp_path,
                                                logger: logging.Logger) -> None:
    """A blank file path closes the log file, so later messages do not reach it."""

    log_file = tmp_path / 'oops.log'
    LOGGING.set_stdout(False)
    LOGGING.set_file(str(log_file))
    LOGGING.info('first')
    LOGGING.set_file('')
    LOGGING.info('second')

    assert 'first' in log_file.read_text()
    assert 'second' not in log_file.read_text()


def test_set_file_rejects_an_unwritable_path(logger: logging.Logger) -> None:
    """A file that cannot be opened for writing raises OSError."""

    with pytest.raises(OSError):
        LOGGING.set_file('/no/such/directory/oops.log')


def test_set_logger_level_accepts_a_name(logger: logging.Logger) -> None:
    """A level given as a name is converted to its integer value."""

    LOGGING.set_logger_level('WARNING')

    assert LOGGING.level == logging.WARNING


def test_set_logger_level_accepts_an_integer(logger: logging.Logger) -> None:
    """A level given as an integer is used as it is."""

    LOGGING.set_logger_level(35)

    assert LOGGING.level == 35


def test_set_logger_sets_the_level(logger: logging.Logger) -> None:
    """The level given to set_logger becomes the logger's threshold."""

    LOGGING.set_logger(logger, level='ERROR')

    assert LOGGING.level == logging.ERROR


def test_set_logger_none_restores_the_default_level(logger: logging.Logger) -> None:
    """Disabling Python logging also restores the default level."""

    LOGGING.set_logger(logger, level='ERROR')
    LOGGING.set_logger(None)

    assert LOGGING.logger is None
    assert LOGGING.level == logging.DEBUG


@pytest.mark.parametrize('method', ['debug', 'info', 'warn', 'error', 'fatal'])
def test_level_shortcuts_log_the_message(method: str, logger: logging.Logger,
                                         capsys) -> None:
    """debug(), info(), warn(), error(), and fatal() each write their message."""

    LOGGING.set_stdout(True)
    getattr(LOGGING, method)('a message')

    assert 'a message' in capsys.readouterr().out


@pytest.mark.parametrize('method, tag', [('warn', 'WARNING:'),
                                         ('error', 'ERROR:'),
                                         ('fatal', 'ERROR:')])
def test_severe_levels_are_tagged(method: str, tag: str, logger: logging.Logger,
                                  capsys) -> None:
    """A message at WARNING or above is tagged with its severity."""

    LOGGING.set_stdout(True)
    getattr(LOGGING, method)('a message')

    assert tag in capsys.readouterr().out


@pytest.mark.parametrize('method', ['debug', 'info'])
def test_ordinary_levels_are_untagged(method: str, logger: logging.Logger,
                                      capsys) -> None:
    """A message below WARNING carries no severity tag."""

    LOGGING.set_stdout(True)
    getattr(LOGGING, method)('a message')
    out = capsys.readouterr().out

    assert 'WARNING' not in out
    assert 'ERROR' not in out


@pytest.mark.parametrize('method', ['convergence', 'diagnostic', 'diagnostics',
                                    'performance'])
def test_category_shortcuts_log_a_message(method: str, logger: logging.Logger,
                                          capsys) -> None:
    """The category shortcuts write their message like any other."""

    LOGGING.set_stdout(True)
    getattr(LOGGING, method)('a categorized message')

    assert 'a categorized message' in capsys.readouterr().out


def test_print_joins_its_arguments_with_spaces(logger: logging.Logger, capsys) -> None:
    """Each argument is converted to a string and joined by single spaces."""

    LOGGING.set_stdout(True)
    LOGGING.print('one', 2, 3.5)

    assert 'one 2 3.5' in capsys.readouterr().out


def test_print_accepts_a_level_name(logger: logging.Logger, capsys) -> None:
    """A level given by name selects the same level as its integer value."""

    LOGGING.set_stdout(True)
    LOGGING.print('named level', level='WARNING')

    assert 'WARNING' in capsys.readouterr().out


def test_literal_message_omits_the_time_tag(logger: logging.Logger, capsys) -> None:
    """A literal message is logged as it is, without a time tag or level."""

    LOGGING.set_stdout(True)
    LOGGING.literal('bare text')
    out = capsys.readouterr().out

    assert 'bare text' in out
    assert 'DEBUG' not in out


def test_prefix_is_included_in_a_literal_message(logger: logging.Logger,
                                                 capsys) -> None:
    """A specified prefix is still written in front of a literal message."""

    LOGGING.on(prefix='>>> ')
    LOGGING.set_stdout(True)
    LOGGING.literal('bare text')
    out = capsys.readouterr().out

    assert 'bare text' in out
    assert out.startswith('>>> ')


def test_exception_is_raised_when_no_logger_is_defined() -> None:
    """With no logger defined, an exception is raised rather than logged."""

    LOGGING.set_logger(None)

    with pytest.raises(ValueError, match='sample failure'):
        LOGGING.exception(ValueError('sample failure'))


def test_exception_is_logged_when_a_logger_is_defined(logger: logging.Logger) -> None:
    """With a logger defined, the exception is logged rather than raised."""

    records: list[logging.LogRecord] = []

    class _Recorder(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger.addHandler(_Recorder())
    LOGGING.set_stdout(False)
    LOGGING.exception(ValueError('sample failure'), 'while testing')

    assert any('sample failure' in record.getMessage() for record in records)


def test_exception_is_logged_at_fatal_level(logger: logging.Logger) -> None:
    """The exception is logged at FATAL level."""

    records: list[logging.LogRecord] = []

    class _Recorder(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger.addHandler(_Recorder())
    LOGGING.set_stdout(False)
    LOGGING.exception(ValueError('sample failure'))

    assert records[0].levelno == logging.FATAL


def test_push_and_pop_restore_the_settings(logger: logging.Logger) -> None:
    """push() saves the settings that pop() puts back."""

    LOGGING.on(prefix='   ')
    LOGGING.push()
    LOGGING.on(prefix='### ')
    assert LOGGING.prefix == '### '

    LOGGING.pop()
    assert LOGGING.prefix == '   '


def test_push_and_pop_restore_the_categories(logger: logging.Logger) -> None:
    """The category flags are saved and restored along with everything else."""

    LOGGING.off()
    LOGGING.push()
    LOGGING.on()
    assert LOGGING.fov_iterations

    LOGGING.pop()
    assert not LOGGING.fov_iterations


def test_push_and_pop_nest(logger: logging.Logger) -> None:
    """The saved settings form a stack, so pushes and pops nest."""

    LOGGING.on(prefix='a')
    LOGGING.push()
    LOGGING.on(prefix='b')
    LOGGING.push()
    LOGGING.on(prefix='c')

    LOGGING.pop()
    assert LOGGING.prefix == 'b'
    LOGGING.pop()
    assert LOGGING.prefix == 'a'


##########################################################################################
