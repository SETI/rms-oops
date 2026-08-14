---
description: Generic best practices for logging with PdsLogger (rms-pdslogger), including sectioning, deferred formatting, and FCPath interaction.
---

# Logging Best Practices

These practices apply to any project that logs through `pdslogger.PdsLogger`
(the `rms-pdslogger` package). For the project-specific logger wiring (which
named loggers exist and how they are configured), see
`logging_nav`.

Use `pdslogger.PdsLogger` exclusively. **Never** use the standard Python
`logging` module to emit log messages, and never call bare `print()` from
library code. The only legitimate use of `import logging` is for type
annotations in low-level plumbing code (e.g. `logging.Handler`,
`logging.FileHandler`) where `pdslogger` factory functions return standard
handler objects. Do not add `import logging` to new files for any other reason.

## 1. Structuring Output with `logger.open()`

Use `with logger.open(header)` to group related log lines under a named section
header. This creates a visual block in both the console and log files, and is
the expected way to delimit a logical unit of work.

```python
# GOOD - wrap a logical unit of work in a named section
with logger.open(f'CREATE MODEL FOR: {name}'):
    render()

# GOOD - optional level= to control visibility
with logger.open('EXPENSIVE STEP', level=log_level):
    ...

# GOOD - attach handlers (e.g. stdout + file) only for this context
with logger.open(str(item_id), handler=local_handlers):
    ...
```

- Sections may be nested; indentation in the output reflects the nesting.
- Open the outermost section for a unit of work (e.g. one input item) with
  `handler=...` so per-unit handlers are active only within that window,
  rather than attached permanently to the logger.
- f-strings ARE acceptable in the `logger.open()` header argument, because the
  header is always rendered.

## 2. Logging Calls

Use `%`-style format strings (never f-strings) in logging calls so that the
interpolation is deferred until the message is actually emitted at the active
level:

```python
# GOOD - arguments interpolated only if the level is enabled
logger.info('No data visible in observation')
logger.info('Writing metadata to %s', metadata_file)
logger.warning('No reference found -- cannot continue')
logger.debug('Failed; keys: %s', sorted(metadata.keys()))
logger.exception('Error reading "%s": %s', path, message)

# BAD - f-string is always evaluated, even when the level is suppressed
logger.info(f'Writing metadata to {metadata_file}')
```

- Available levels: `debug`, `info`, `warning`, `error`, `exception`, `fatal`.
- Use `exception` (not `error`) inside an `except` block so the traceback is
  captured automatically.

## 3. Interaction with FCPath

Log file paths are typically `FCPath` objects (from `rms-filecache`). Follow the
`filecache` rule: never call `mkdir` on the log directory --
`PdsLogger` creates the directory through `FileCache` internally when it opens
the file handler.

```python
# BAD - do not mkdir the log directory yourself
log_dir.mkdir(parents=True, exist_ok=True)
```

Pass the `FCPath` (local or remote) straight to the handler factory; transparent
local-or-remote handling is provided by `filecache`.

## 4. Quick Reference

| Task | Pattern |
|------|---------|
| Emit a message | `logger.info('msg')` / `logger.warning('msg')` |
| Deferred formatting | `logger.info('value=%s', value)` |
| Named section | `with logger.open('SECTION HEADER'):` |
| Named section + level | `with logger.open('HEADER', level='DEBUG'):` |
| Per-unit handlers | `with logger.open(item_id, handler=handlers):` |
| Exception in except block | `logger.exception('msg: %s', detail)` |
| Standard-library logging | Only for type annotations in low-level plumbing |
