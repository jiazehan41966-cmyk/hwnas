# Repository Review

Current review entry point: `docs/FIRST_PRINCIPLES_AUDIT_20260703.md`.

The earlier external review was produced from the pre-merge `main` snapshot
(`40aaa8a`), which contained only two commits and did not include the local
Phase0/HLS work. Pull request #1 later merged the `codex/code-work` history into
`main`. Statements that the current repository has no HLS builder or no Phase0
documentation are therefore obsolete.

The merge does not make every historical claim final. The current audit keeps
the following boundaries:

- Phase0 search and retrain scores are legacy fold-0 validation evidence, not
  untouched-test generalization estimates.
- COM5 proves fixed-harness latency/output behavior, not NKSID board accuracy or
  power.
- The existing `denoise` and `edge` HLS templates are not semantically identical
  to their train-time PyTorch blocks.
- Measured board power and energy remain unavailable.
- Generated local result trees remain excluded from Git; compact reproducible
  evidence is published under `artifacts/first_principles_audit_20260703/`.

Use `docs/PROJECT_MEMORY.md` for the complete evidence and archive index.
