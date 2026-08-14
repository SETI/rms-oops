---
description: Guidelines for writing user-facing how-to documentation with steps, prerequisites, and troubleshooting.
paths:
  - "docs/**/*.rst"
  - "docs/**/*.md"
---

# How-To Documentation

## 1. Audience and Tone

- Write for **Python users** who are familiar with `pip` and the command line but may not know the library's internals.
- Use clear, direct language; define domain-specific terms on first use.
- Focus on what the user needs to do and what they should observe.

## 2. Best Practices

1. **Action-oriented title** — e.g., "How To Process a Cassini Image", not "Image Processing Overview".
2. **Brief introduction** — 1-3 sentences explaining purpose and value.
3. **Prerequisites** — Python version, package installation, required data or environment variables.
4. **Numbered steps** — One action per step in logical order. Include code snippets for API usage or CLI commands.
5. **Expected results** — State what the user should see after each significant step AND in a summary section at the end. Keep both consistent.
6. **Troubleshooting** — Common failures (import errors, missing data, version mismatches) and their fixes.
7. **Related features** — Mention next steps or related guides.

## 3. Document Structure

```markdown
# How To [Action]

[1-3 sentence introduction explaining purpose and value.]

## Prerequisites

- Python >= 3.11
- `pip install rms-<package>`
- [Any required data, environment variables, or configuration]

## Steps

1. Import the module:
   ```python
   from package import SomeClass
   ```
2. [Action]. You should see [result].
3. [Action].

## Expected Results

[Summary of the successful end state — expected output, files created, etc.]

## Troubleshooting

- **[Problem]**: [Solution].

## Additional Information

[Tips, performance notes, or links to related guides.]
```

## 4. Converting Technical Content

When turning docstrings, test scripts, or internal notes into How-To guides:

1. Identify the user-facing feature or workflow.
2. Determine the target audience (library user, CLI user, contributor).
3. Extract user actions from technical steps.
4. Translate internal terminology to user-friendly language.
5. Add code examples, expected output, and troubleshooting.

## 5. Diagrams and Figures

- **When to use**: Multi-step workflows, data pipelines, or architecture that is clearer as a visual.
- **Placement**: Inline, immediately after the relevant step or section.
- **Format**: Prefer Mermaid diagrams (e.g., rendered by Sphinx via `sphinxcontrib-mermaid`) for process flows. Use PNG/SVG for screenshots or data visualizations.
- **Naming**: Descriptive filenames (e.g., `backplane-pipeline.svg`). Include alt text for accessibility.
