---
description: Standards for Python library documentation using Sphinx, ReadTheDocs, and docstrings.
---

# Documentation Standards

## 1. Documentation System

- Use **Sphinx** for all project documentation, hosted on **ReadTheDocs**.
- After any code or doc change, run `sphinx-build` on the full documentation tree and fix all warnings and errors before delivering.

## 2. Documentation Standard

- Always use one space between the period at the end of a sentence and the next sentence.
- Always use American spelling instead of British spelling (e.g. color instead of colour).

## 3. Required Documentation

| Document | Contents | Keep up-to-date? |
|----------|----------|-------------------|
| **Module index** | Every module that exists or is planned (placeholders for future modules). | Yes |
| **Architecture overview** | Class hierarchy, public API surface, and interface contracts. | Yes |
| **Install guide** | `pip install` instructions, supported Python versions, optional dependencies. | Yes |
| **Usage examples** | Common workflows with code snippets and expected output. | Yes |
| **README** | Project summary, PyPI/ReadTheDocs badges, quickstart, and links to full docs. | Yes |

## 4. Docstrings

- EVERY class, method, function, and module MUST have a descriptive docstring.
- Follow **PEP 257** using **Google style** with `Parameters:` (not `Args:`).
- Include `Returns:` and `Raises:` only if there are return values or exceptions raised.
- Include behavioral notes sufficient to write a black-box test but do not reference the internal details of the code.
- Wrap docstring text to **90** characters.

## 5. Cross-Reference Completeness

- EVERY mention of a class, method, function, module, attribute, or data
  constant in narrative prose MUST use the appropriate Sphinx cross-reference
  role:
    - `:class:`~nav.path.module.Class``
    - `:meth:`~nav.path.module.Class.method``
    - `:func:`~nav.path.module.func``
    - `:mod:`nav.path.module``
    - `:attr:`~nav.path.module.Class.attr``
    - `:data:`~nav.path.module.NAME``
- Bare CamelCase or `module.symbol` text in narrative prose is a violation,
  even when wrapped in inline literals (`` `` ``). Inline literals are for
  YAML/JSON keys, file paths, CLI tokens, and shell snippets — not for API
  symbols.
- Cross-references are NOT required (and should be omitted) inside
  `.. code-block::` directives, `::` literal blocks, Mermaid / other diagram
  blocks, YAML examples, or section titles.
- When a class, method, function, module, attribute, or data constant is
  added, removed, or renamed, every cross-reference to it across the docs
  tree MUST be updated in the same change. A rename without ref updates is a
  documentation regression.
- Validate by building with `sphinx-build -W -b html` (warnings as errors)
  AND `sphinx-build -n -b html` (nitpicky mode); both MUST succeed with zero
  warnings before delivering.

## 6. Change Discipline

- Any code change MUST update the relevant docstrings and the README if affected.
- NEVER leave stale or contradictory documentation. If a feature is removed, remove its docs.
