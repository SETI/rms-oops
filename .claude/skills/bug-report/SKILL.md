---
name: bug-report
description: Standards for writing clear, reproducible bug reports, covering required components, severity levels, evidence, and environment details. Use when the user asks to write, file, or review a bug report or issue.
---

# Bug Report Standards

## 1. Core Components

Every bug report MUST include:

- **Clear title** — Describes the symptom and its location (e.g., "`Profile.from_file` raises KeyError on valid FITS header").
- **Reproduction steps** — Numbered, minimal steps (ideally a short script) anyone can follow.
- **Expected vs. actual behavior** — Side-by-side comparison.
- **Environment** — Python version, OS, package version, relevant dependency versions.
- **Severity** — Assessed per the scale below.
- **Evidence** — Tracebacks, log output, or screenshots of incorrect results.

## 2. Severity Scale

| Level | Criteria |
|-------|----------|
| **Critical** | Crash, data corruption, silent wrong results, or security vulnerability. |
| **High** | Major feature broken or blocking for many users. |
| **Medium** | Non-critical feature broken or produces degraded results. |
| **Low** | Minor issue, documentation error, or cosmetic problem. |
| **Trivial** | Very minor issue with negligible user impact. |

## 3. Report Template

```markdown
# Bug Report: [Concise title]

## Description
[1-2 sentences: what is broken and its impact.]

## Environment
- **Python version**: [e.g., 3.12.4]
- **OS**: [e.g., Ubuntu 24.04, macOS 14.5, Windows 11]
- **Package version**: [e.g., rms-polymath 0.3.1]
- **Key dependency versions**: [e.g., numpy 2.2.1, scipy 1.14.0]
- **Installation method**: [e.g., pip install rms-polymath, editable install]

## Severity
[Level] — [Brief justification]

## Steps to Reproduce
1. Install: `pip install rms-polymath==0.3.1`
2. Run:
   (Example)
3. Observe the error.

## Expected Behavior
[What should happen.]

## Actual Behavior
[What actually happens, including the full traceback.]

## Traceback / Logs
[Paste full traceback or relevant log output here]

## Additional Notes
[Workarounds, frequency, related issues.]

## Possible Fix
[Optional: suspected root cause or fix direction.]
```

## 4. Writing Guidelines

1. Be objective and factual — no blame or subjective language.
2. One issue per report.
3. Include exact version numbers and full tracebacks.
4. Keep reproduction steps as short as possible while remaining unambiguous.
5. Verify the bug is reproducible before submitting.

## 5. Adaptation

Adjust the template for the project's GitHub Issues and add project-specific fields (e.g., affected data set, mission, instrument).
