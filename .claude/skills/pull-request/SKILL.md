---
name: pull-request
description: Standards for pull request structure, covering purpose, implementation details, testing evidence, and the review checklist. Use when the user asks to write a pull request description, open a PR, or review one.
---

# Pull Request Standards

## Scope of review

Treat the PR as a **single unit of change**. The diff to review is the set of all commits on the current branch back to its **immediate root** (the merge-base with the target branch). Consider the net result of those commits together; do **not** comment on differences that exist only between commits within the PR (e.g. "you fixed X in a later commit" or "commit 2 undid part of commit 1"). Review the final state of the branch against the base. Do not explicitly word wrap lines.

## Principles

1. **Descriptive title** — Summarize the change in an imperative sentence (e.g., "Add caching to profile lookup").
2. **Purpose first** — Explain *why* the change is needed before *how* it was done.
3. **Scope** — One logical change per PR. Split unrelated changes into separate PRs.
4. **Testing evidence** — Document automated and manual testing performed.
5. **Impact assessment** — Note potential effects on the public API, performance, or dependent packages (Potential Impacts section).
6. **Linked issues** — Reference related GitHub issues using `Closes #NNN` syntax.

## Template

The PR template is in `.github/pull_request_template.md` and is applied automatically when a new PR is opened. Fill out every section:

- **Purpose** — Why the change is needed; link issue with `Closes #NNN`.
- **Changes / Implementation Details** — What changed and how it was implemented; technical approaches chosen and non-obvious design decisions.
- **Type of Change** — Check all that apply (bug fix, feature, breaking, refactor, docs, tests, CI/build).
- **Testing** — Check boxes for unit tests, integration tests, E2E tests run; describe new tests added and manual verification performed.
- **Potential Impacts** — Public API, backward compatibility, performance, downstream; write "None" if straightforward.
- **Checklist** — Style, mypy, docs, no debug code, no secrets/credentials, no warnings/errors, performance impact assessed, breaking changes flagged.
- **Notes** — Optional; delete only if not needed (tricky areas, follow-up work).

## Guidance

- **Library-specific** — Call out public API changes, deprecations, and migration notes in Potential Impacts.
- **Required reviewers** — Tag maintainers for changes to core modules.
- **Brevity vs. completeness** — Short enough that authors fill everything out; detailed enough for a reviewer with no other context.
