##########################################################################################
# tests/test_config.py
##########################################################################################

import logging
from collections.abc import Iterator

import pytest

from oops.config import LOG_FORMATTER, LOGGING


@pytest.fixture(autouse=True)
def restore_logging() -> Iterator[None]:
    """Save and restore the LOGGING class attributes that these tests modify.

    LOGGING is a module-level namespace of mutable class attributes, so a test that
    installs a logger has to put the previous state back or it leaks into every test
    that runs afterward.
    """

    saved = (LOGGING.logger, LOGGING.level, LOGGING.handlers, LOGGING.log_formatting,
             LOGGING.stdout, LOGGING.lines)

    yield

    (LOGGING.logger, LOGGING.level, LOGGING.handlers, LOGGING.log_formatting,
     LOGGING.stdout, LOGGING.lines) = saved


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

##########################################################################################
