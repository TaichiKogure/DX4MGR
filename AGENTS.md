# Repository Guidelines

## Scope
These instructions apply to the entire repository unless a deeper `AGENTS.md` overrides them.

## Project Layout
- `2026/` contains the current planning, data analysis, and R&D simulation work. Treat this as the active area by default.
- `archive/` contains historical versions and backups. Do not modify archived content unless the user explicitly asks for a versioned or migration task.
- `others/` contains helper scripts, ad hoc utilities, and generated comparison artifacts.
- `reports/` contains generated analysis outputs and summaries. Prefer adding new generated artifacts under this tree instead of scattering them elsewhere.

## Working Norms
- Before changing code, confirm which version or plan directory is the intended target when the request is ambiguous.
- Prefer small, surgical changes that preserve the existing folder naming and documentation style.
- When editing documentation in Japanese, keep Japanese headings and prose unless the user asks for translation.
- Preserve historical traceability: when replacing a workflow, update the nearby README or docs that describe how to run it.

## Files And Outputs
- Do not commit large generated outputs, caches, or local IDE artifacts outside the existing project files.
- Put new one-off scripts in `others/` unless a subproject already has a clearer home.
- Put new reports, CSV outputs, charts, or experiment summaries in `reports/` or the nearest existing report folder for that subproject.

## Validation
- For code or script changes, run the narrowest relevant validation first, such as the specific Python entrypoint or report-generation command affected by the change.
- If a task touches simulation behavior, mention any required input files or scenario data needed to reproduce results.
- If validation cannot be run locally, state that clearly in the handoff.

## Safety
- Avoid destructive cleanup in `archive/` and avoid deleting report artifacts unless the user explicitly requests it.
- If a request could apply to both active work and archived versions, ask before changing multiple branches of the repository.
