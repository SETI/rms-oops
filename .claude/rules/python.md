---
description: Python coding standards for writing correct, readable, maintainable, and well-tested library code.
---

# Python Best Practices

Apply these rules to ALL new and modified Python code. This project is a Python library published on PyPI and documented on ReadTheDocs. **Minimum Python version: 3.11.**

## 1. Naming and Style

- **Maximum line length**: 90 characters. Enforce via Ruff; use editor rulers at 80 and 90 as visual guides.
- **Functions and local variables**: Use `lowercase_with_underscores`.
- **Class names**: Use `TitleCase`.
- **Module-level constants (global variables)**: Use `ALL_CAPS_WITH_UNDERSCORES`.
- **Private names**: Prepend a single underscore for names that are not part of the public API: private attributes (e.g. `_cache`), module-private global variables, and non-public helper functions (e.g. `_parse_header`). Public API names have no leading underscore.
- **Built-in names**: Do NOT use variable or function names that shadow Python built-ins (e.g. `float`, `filter`, `id`, `list`, `type`). If you must use such a name, append a single underscore (e.g. `filter_`, `type_`).
- **Explicit checks over exceptions**: Prefer explicit membership or presence checks over catching exceptions for control flow. Example: use `if "a" in b: x = b["a"]` (or a clear `get` with a sentinel) rather than `try: x = b["a"]` / `except KeyError: ...` for normal flow. Use exceptions for genuinely exceptional conditions.

## 2. General Coding

- Always match the coding style of pre-existing Python files.
- Always make the minimal changes necessary. Never modify code outside the scope of the current task.
- Do not include backwards-compatibility code unless explicitly requested.
- Always write simple, clear code; avoid unnecessary complexity.
- Define magic constants as module-level constants, in a config module, or via environment variables.
- Catch exceptions at the smallest granularity possible. Do not wrap large blocks in a single `try`/`except`.
- **Libraries:** Let exceptions propagate unless you are adding context, converting to a library-specific exception, or the exception represents a recoverable internal state. When re-raising, use `raise ... from` to preserve the full traceback for debugging.
- **Applications:** Do not allow uncaught exceptions to reach the top level; use a top-level handler (e.g. in the main loop or HTTP framework) so that failures are logged and the process stays predictable. In both cases, always provide full exception information for debugging (e.g. traceback, `raise ... from` when re-raising).
- Include meaningful, structured logging (use the `logging` module) that can be disabled or redirected. Never use bare `print()` for diagnostic output in library code.
- Avoid mutable global variables. If unavoidable, document purpose and limit scope. Prefer module-level constants (ALL_CAPS) or dependency injection.
- Always prefer comprehensions (list, dict, set, generator) over manual loops when the result is a new collection and the expression remains readable.
- Apply DRY. Avoid duplicating code. Place reusable logic in a utility module. Search existing utilities before writing new functions. Parameterize utility functions to increase generality.
- Place imports at the top of the file in three alphabetically-sorted groups separated by a blank line: (1) standard library, (2) third-party, (3) local project. When adding new code or tests, add new imports to the appropriate group at the top; do not place them adjacent to the new code. Inline imports are permitted only to avoid heavy optional dependencies (e.g., GUI libraries).
- Limit new functions to at most 5 positional parameters. Additional parameters should be keyword-only (after `*`). Choose a logical grouping of 0-5 positional parameters before enforcing keyword-only parameters.
- Use the Receive-an-Object, Return-an-Object (RORO) pattern when a function takes or returns more than a few related values: accept a dataclass or TypedDict and return one, rather than long positional tuples.
- Never use `getattr` just as a defensive measure if it is guaranteed that the object has the attribute. Reference the attribute directly unless there is a specific reason to know the attribute may not be present. NEVER use getattr to reference the result of an `argparse` namespace when the argument name is a constant string.

## 3. Public API Design

- Clearly separate public API from internal implementation. Prefix internal functions, classes, and modules with `_`.
- Use `__all__` in `__init__.py` to explicitly declare the public API surface.
- Design for stability: think carefully before adding to the public API, because removing it later is a breaking change.
- Include a `py.typed` marker file so downstream users get type-checking support.

## 4. Comments

- ALWAYS write self-documenting code: meaningful names, simple structure, limited nesting.
- NEVER include comments that merely restate the code, reference user requests, or describe modification history.
- ALWAYS include comments that explain the **rationale** behind non-obvious or complex logic.
- ALWAYS preserve existing comments that are still accurate and relevant. Remove or update stale comments.

## 5. Lint and Type Checking

### Types

- NEVER use type annotations in the src directory tree. Types of input parameters and returns should be indicated in the docstrings.
- Annotate all test function/method parameters and return values, including `-> None` for functions (and `__init__`) that return nothing.
- Use modern generic syntax (`list[str]`, `dict[str, int]`, `X | None`) for Python 3.11+.

### Mypy

- NEVER run `mypy` on the src directory tree.
- Run `mypy` on the tests after changes. Fix all errors before delivering.
- In exceptional, unfixable cases use a minimal line-level ignore: `# type: ignore[error-code]  # <brief justification>`.

### Ruff / Linting

- Include `mypy` and `ruff` in the project's dev dependencies (e.g. in `pyproject.toml`).
- Run `ruff check` on the full codebase after changes
- Follow PEP 8 for formatting and naming conventions.
- Use the project's explicit Ruff rule set in `pyproject.toml` (see **Ruff rule categories** below). Do not disable categories that enforce project conventions (e.g. **A** for no builtin shadowing, **N** for naming).

## 6. Docstrings

- Include a docstring for every module, class, function, and method.
- Follow **PEP 257** using **Google style**. Use `Parameters:` (not `Args:`).
- Include `Returns:`, `Raises:`, and any important behavioral notes.
- NEVER mention backwards compatibility, a user request, change history, or an issue/ticket number in a docstring. Docstrings are usage documentation for the published API, not a place to explain the code's provenance; describe only observable behavior. (Issue references are allowed in inline `#` code comments per Section 4, and in commit messages and PR descriptions.)
- Docstrings MUST be detailed enough to write a black-box test from the docstring alone.
- Wrap docstring text to **90** characters.
- ALWAYS update docstrings when the associated code changes.

## 7. Testing

- All testing standards live in `python_testing` (pytest, fixtures,
  parametrization, coverage targets, TDD, and test hygiene). Tests are
  first-class code and MUST follow the naming, typing, docstring, and DRY rules
  in this file as well.

## 8. Ruff Rule Categories (Default Set)

The template enables these Ruff lint categories in `pyproject.toml`. Use them as the default for new repos; add or ignore specific codes as needed.

| Code | Source | Purpose |
|------|--------|---------|
| **E**, **W** | pycodestyle | Style and formatting (indent, whitespace, line length). |
| **F** | Pyflakes | Unused imports, undefined names, syntax issues. |
| **I** | isort | Import sorting and grouping. |
| **UP** | pyupgrade | Prefer modern Python (e.g. 3.11+ syntax). |
| **B** | flake8-bugbear | Common bugs (mutable defaults, assert, loop vars). |
| **SIM** | flake8-simplify | Simpler alternatives (e.g. `in` instead of `not x == y`). |
| **C4** | flake8-comprehensions | Prefer comprehensions over loops where clear. |
| **A** | flake8-builtins | No shadowing of builtins (`id`, `filter`, `type`, etc.). |
| **N** | pep8-naming | Class = TitleCase, functions/variables = lowercase_with_underscores. |
| **PT** | flake8-pytest-style | Pytest best practices (fixtures, parametrize, raises). |
| **RUF** | Ruff | Ruff-specific (e.g. unused noqa, deprecated). |

Optional categories to consider adding later: **D** (pydocstyle) or **DOC** (pydoclint) for docstring linting; **PTH** (pathlib); **RET** (return simplification); **PERF** (perflint). Enable only if the team agrees to fix or ignore the resulting diagnostics.
