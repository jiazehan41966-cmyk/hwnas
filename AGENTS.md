# Sonar HW-NAS layered mainline

- This source-only worktree migrates E0/E1 / elastic_neck72_v1 detection. Read README.md and docs/MIGRATION.md before changing protocol or importing legacy results.
- Preserve unrelated work and immutable experiment artifacts. Plans and audits remain read-only; implementation authorization does not authorize training, HLS, board activity, remote jobs or cloud spending.
- Space is the immutable identity layer. Models may depend on space; models must never import training, quantization, hardware or search. Search consumes evaluator interfaces and never builds models. scripts/ is the composition root.
- Genotype JSON and full SHA256 identify candidates; precision plans have separate q_hash. Seed, budget, checkpoint, calibration, hardware mapping, device, clock, toolchain and bitstream identities must remain explicit.
- Keep software accuracy, PTQ integer correctness, operator costs, full HLS, route, board runtime and measured power separate. Missing evidence is NEEDS_MEASUREMENT; never substitute zero or cached costs from a different context.
- Keep test sealed during model/precision/mapping selection. Preserve the registered frozen training, initialization and validation protocol unless the user explicitly requests a new experimental variant.
- Prefer project-local Python environments; outputs belong under results/, outputs/ or logs/. Never put credentials in source, artifacts, commands or logs.
- Verify meaningful changed behavior and static import boundaries. Do not execute research campaigns as a test. Do not push branches/tags without authorization.
