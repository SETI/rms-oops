---
name: git-workflow
description: Commit message format, branch naming, and the pull request workflow for this project. Use when the user asks to commit, name a branch, or prepare changes for review.
---

# Git Workflow

## 1. Commit Messages

Write the subject as a plain capitalized sentence in the imperative mood, with no type
prefix:

```
Add caching to profile lookup

[Optional body — wrap at 72 characters. Explain *what* and *why*, not *how*.
A bulleted list is fine for a change that touches several areas.]

[Optional footer — e.g., Closes #123, Co-authored-by: Name <email>]
```

### Rules

- Subject line MUST be imperative mood ("Add X", not "Added X" or "Adds X").
- Subject line MUST be capitalized and MUST NOT end with a period.
- Do NOT prefix the subject with a Conventional Commits type such as `feat:` or `fix:`.
  This project does not use them.
- Keep the subject under 72 characters, including the `(#N)` that a squash merge appends.
- Separate subject from body with a blank line.
- Body lines MUST NOT exceed 72 characters.
- Reference related issues in the footer.
- Each commit MUST represent one logical change. Do NOT mix unrelated changes.

When a pull request is squash-merged, GitHub appends its number to the subject, producing
history entries such as `Increase test coverage, improve docstrings, minor bug fixes (#13)`.
Do not add the `(#N)` yourself.

## 2. Branching Strategy

- **`main`** — Always releasable. Protected; requires PR review and passing CI. Releases
  are created by tagging commits on `main`.
- **Work branches** — Named `<initials>_<YYMMDD>_<topic>`, for example `rf_251204_mixins`
  or `rf_250712`. A short descriptive name such as `mark-reorg` is also acceptable for
  one-off work.

Branch from `main`. Do NOT use `feature/` or `bugfix/` prefixes, and do NOT create
separate release, hotfix, or develop branches. All work merges back to `main` via pull
request.

## 3. Pull Requests and Merging

- ALWAYS create a PR for merging into `main`; direct pushes are prohibited.
- PRs MUST pass all CI checks (lint, type-check, tests) before merge.
- Prefer **squash merge** to keep `main` history linear and readable.
- Delete the source branch after merge.

## 4. Tagging and Releases

- Tag releases on `main` with semantic versioning: `v<MAJOR>.<MINOR>.<PATCH>`.
- Let `setuptools_scm` derive the package version from tags automatically.
- Creating a GitHub Release from the tag triggers the PyPI publish workflow.
