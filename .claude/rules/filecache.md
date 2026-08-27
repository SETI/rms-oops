---
description: Best practices for using rms-filecache (FCPath) for transparent local/remote file access.
---

# rms-filecache (FCPath) Best Practices

This project uses `rms-filecache` to read and write files transparently across local
filesystems and remote storage (e.g. `gs://`, `s3://`, `https://`). All path handling
MUST go through `FCPath`. Casting an `FCPath` to a plain `Path` discards the remote
source information and breaks remote I/O.

## 1. Core Rules

- ALWAYS represent file paths as `FCPath`. NEVER store, return, or downcast an
  `FCPath` to `pathlib.Path` (or `str`) just to "simplify". `FCPath` already supports
  the full `pathlib.Path` API surface (e.g. `parent`, `suffix`, `stem`, `name`,
  `with_suffix`, `/` joining, `exists`, `iterdir`, `glob`, `is_dir`, `is_file`).
- ALWAYS normalize at the boundary: convert `str` and `Path` inputs to `FCPath` as
  the first step in any function that handles paths. `FCPath(x)` is safe and cheap
  even when `x` is already an `FCPath`, so a single `FCPath(x)` call is the
  idiomatic way to handle a `str | Path | FCPath` parameter.
- NEVER hand an `FCPath` to `os.path.*`, `shutil.*`, `open()` (the builtin), or any
  other API that does not understand remote URLs. Use `FCPath` methods instead
  (e.g. `fcpath.open(...)`, `fcpath.iterdir()`, `fcpath.glob(...)`). See section 4
  for the preferred pattern for existence checks.
- When an external API requires a plain string URL/path, use `fcpath.as_posix()`
  rather than `str(fcpath)` so the result is a stable POSIX-style URL.

### Type hints at API boundaries

- **Public API** (functions called from outside the module): accept
  `str | Path | FCPath`. Internally convert with `FCPath(...)` on the first line.
- **Internal helpers**: accept and return `FCPath` only.

```python
from pathlib import Path
from filecache import FCPath

def load_table(path: str | Path | FCPath) -> Table:
    fcpath = FCPath(path)            # cheap if already an FCPath
    return _load_table_impl(fcpath)

def _load_table_impl(fcpath: FCPath) -> Table:
    ...
```

## 2. Modules That Already Accept FCPath

The following project / RMS modules accept `FCPath` directly. Pass the `FCPath`
straight through; do NOT convert to `Path` or `str` first:

- `oops`
- `rms-vicar`
- `rms-pdslogger`
- `rms-starcat`
- `rms-textkernel`
- `rms-cloud-tasks`
- `rms-julian`
- `rms-pdsparser`
- `rms-pdstable`

## 3. Modules That Do NOT Accept FCPath

For libraries that require a real local filesystem path (e.g. `astropy.io.fits`,
`numpy.load`/`numpy.save`, `PIL.Image`, `matplotlib.pyplot.savefig`, `csv` over a
file object opened by a third-party API, etc.), follow the patterns below.

### 3a. Reading a file with normal Python I/O

ALWAYS use `fcpath.open('r')` (or `'rb'`) inside a `with` block. This does store a
local copy but follows Python best practices and automatically closes the file on
context exit.

```python
fcpath = FCPath(path)
with fcpath.open('r') as f:
    text = f.read()
```

### 3b. Reading a file that requires a real local path

When the consumer cannot accept a file-like object (e.g. `astropy.io.fits.open`,
`np.load`), call `fcpath.retrieve()` to download the file to the local cache and
get back a `Path`. `retrieve()` is safe and a no-op for already-local files.

`retrieve()` returns `Path | Exception | list[Path | Exception]` because it can
operate on a list of paths and can return failures inline. For a single-file call
with `exception_on_fail=True` (the default), the result is always a `Path`, so it
is correct to assert that with `cast(Path, ...)` for mypy.

```python
from typing import cast

local_path = cast(Path, fcpath.retrieve())
with fits.open(local_path) as hdul:
    data = hdul[0].data
```

### 3c. Writing a file with normal Python I/O

ALWAYS use `fcpath.open('w')` (or `'wb'`) inside a `with` block. The data is
uploaded to the remote location automatically when the context exits.

```python
with fcpath.open('w') as f:
    f.write(text)
```

### 3d. Writing a file that requires a real local path

When the producer cannot accept a file-like object (e.g. `hdul.writeto`,
`np.save`/`np.savez`, `plt.savefig`), use `fcpath.get_local_path()` to obtain the
local cache path, write to it, and then call `fcpath.upload()` to push it to the
remote.

`get_local_path()` returns `Path | list[Path]` for the same reason `retrieve()`
returns a union. For a single path, `cast(Path, ...)` is the correct assertion.

**`get_local_path()` creates all parent directories automatically.** NEVER call
`.mkdir(parents=True, exist_ok=True)` on the returned path or any of its parents.

```python
from typing import cast

local_path = cast(Path, fcpath.get_local_path())
hdul.writeto(local_path, overwrite=True)
fcpath.upload()
```

## 4. Existence Checks: Prefer Try/Except Over `exists()`

For remote backends, `FCPath.exists()` triggers a network round-trip - and when the
file does exist, the very next step is almost always `retrieve()` or `open()`,
which performs a *second* round-trip. This is wasteful.

ALWAYS prefer the EAFP pattern: just try to retrieve / open the file and catch
`FileNotFoundError`. `retrieve()` raises `FileNotFoundError` when the file is
missing (and `open()` does the same for the underlying transport).

```python
# BAD - two round-trips when the file exists, and a TOCTOU race besides
if fcpath.exists():
    local_path = cast(Path, fcpath.retrieve())
    process(local_path)
else:
    handle_missing()

# GOOD - one round-trip; existence is determined as a side effect of retrieval
try:
    local_path = cast(Path, fcpath.retrieve())
except FileNotFoundError:
    handle_missing()
else:
    process(local_path)
```

The same applies to `fcpath.open(...)` - just open it inside `try` / `except
FileNotFoundError` rather than guarding with `exists()`.

`exists()`, `iterdir()`, and `glob()` are still legitimate when the *answer itself*
is what you need (e.g. listing a directory, surfacing a "not found" message to the
user without trying to read the file). Just don't use them as a pre-flight check
before a read.

## 5. Never Call `mkdir` Through FCPath

`FCPath.get_local_path()` (and `retrieve()`) already create the necessary parent
directories in the local cache. Calling `mkdir` is at best redundant and at worst
breaks the abstraction for remote backends (where directories are not a real
concept).

```python
# BAD - redundant and confuses the abstraction
local_path = cast(Path, fcpath.get_local_path())
local_path.parent.mkdir(parents=True, exist_ok=True)   # remove this
hdul.writeto(local_path, overwrite=True)
fcpath.upload()

# GOOD
local_path = cast(Path, fcpath.get_local_path())
hdul.writeto(local_path, overwrite=True)
fcpath.upload()
```

This applies in particular to **PdsLogger log directories**. `PdsLogger` uses
`FileCache` internally, so the log directory is an `FCPath` whose backing local
directory is materialized on first write. NEVER `mkdir` a logs directory.

```python
# BAD
log_dir = FCPath(config.log_dir)
log_dir.mkdir(parents=True, exist_ok=True)             # remove this
logger = PdsLogger('mytool', logfile=log_dir / 'run.log')

# GOOD
log_dir = FCPath(config.log_dir)
logger = PdsLogger('mytool', logfile=log_dir / 'run.log')
```

## 6. Quick Reference

| Need | Use |
|------|-----|
| Accept `str`, `Path`, or `FCPath` | `fcpath = FCPath(path)` |
| Read text/bytes | `with fcpath.open('r'/'rb') as f: ...` |
| Read via library that needs a real file | `local = cast(Path, fcpath.retrieve())` |
| Write text/bytes | `with fcpath.open('w'/'wb') as f: ...` |
| Write via library that needs a real file | `local = cast(Path, fcpath.get_local_path())`; write; `fcpath.upload()` |
| "Does this file exist *and* I want to read it?" | `try: ...retrieve()... except FileNotFoundError:` (do NOT pre-check with `exists()`) |
| List / glob a directory | `fcpath.iterdir()`, `fcpath.glob(...)` |
| Pass to an external CLI / URL API | `fcpath.as_posix()` |
| Make parent directories | Do nothing - `get_local_path()` handles it |
