# First-Principles Audit Evidence

This directory is the Git-tracked, compact handoff for the 2026-07-03
first-principles audit. Large and regenerated outputs remain under
`results/first_principles_audit_20260703/`, which is intentionally ignored by
Git.

Reproduce the dataset protocol audit:

```powershell
python scripts/audit_nksid_protocol.py `
  --data-dir data/NKSID `
  --fold 0 `
  --neighbor-radius 1 `
  --hash-files `
  --output-dir results/first_principles_audit_20260703
```

Reproduce the PyTorch/HLS operator-semantic audit:

```powershell
python scripts/audit_operator_semantic_parity.py `
  --output-dir results/first_principles_audit_20260703
```

The compact values, commands, generated-file sizes, and SHA256 hashes are in
`evidence_summary.json`. See `docs/FIRST_PRINCIPLES_AUDIT_20260703.md` for
interpretation and claim boundaries.
