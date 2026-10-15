# Lightweight deployment handoff

This isolated package checks existing `elastic_neck72_v1` integer exports and
binds them to a selected full-budget checkpoint. It does not modify the old
B1/C/D deployment allowlist, generate a bitstream, start HLS, or program a board.

## API

`inspect_ptq_directory(path)` checks the saved software export. It verifies the
manifest, source code/contracts, all 64 train calibration file hashes, canonical
architecture and topology, parameter bytes and integer dtypes, graph hashes,
all-node parity coverage, and the four explicit INT8-to-INT16 head boundaries.
It returns logical tensor byte sizes, **not** a hardware frame ABI or DDR plan.
It reads prior parity evidence; it does not re-execute inference.

`prepare_handoff(request, base_dir=Path('.'))` validates a selected candidate and
returns separate stage statuses. Missing dependencies produce a blocking report.
Malformed top-level identity raises `ValueError`. Paths resolve against
`base_dir`; the CLI uses the request file's parent directory.

Request schema (shown as Python, placeholders are not valid evidence):

```python
{
    "schema_version": 1,
    "space_version": "elastic_neck72_v1",
    "architecture": {"pan": False, "td_kernels": [3, 3, 3], "pan_kernels": []},
    "architecture_id": "E0_TD_P2k3_P3k3_P4k3",
    "architecture_hash": "<canonical architecture SHA256>",
    "num_classes": 2,
    "profile": "W8A8",  # also HEAD_W8A16 or HEAD_W16A16
    "mapping_policy": "reference_low",  # registered six policies only
    "seed": 42, "budget": 80,
    "evidence_kind": "trained_checkpoint",
    "calibration_sha256": "<selected train64 manifest identity>",
    "artifacts": {
        "checkpoint": {"path": "<best_80.pt>", "sha256": "<file SHA256>"},
        "training_receipt": {"path": "<train task JSON>", "sha256": "<file SHA256>"},
        "confirmation": {"path": "<confirmation JSON>", "sha256": "<file SHA256>"},
        "ptq_directory": {"path": "<profile export directory>",
                          "sha256": "<artifact_manifest.json file SHA256>"}
    }
}
```

Checkpoint and receipt follow the existing `joint_v2_elastic.train_candidate`
format. The PTQ profile must contain its existing `budget_checkpoint_binding`;
architecture, classes, seed, budget, freeze, checkpoint and calibration hashes,
validation split and sealed-test status must match. Thus diagnostic exports
without this binding cannot be substituted for final trained exports.

The confirmation envelope binds `status=PASS`,
`evidence_kind=three_seed_quantized_confirmation`, `architecture_hash`,
`profile`, `selected_checkpoint_sha256`, `seeds=[42,43,44]`,
`accuracy_split=valid`, and `test_evaluated=false`.
**This header alone only checks identity. Confirmation remains
`PENDING_INDEPENDENT_GATE_REPLAY` until the formal driver supplies real
per-seed receipts and the existing admission gate is independently replayed.**
No actual such final candidate or confirmation is supplied in this revision.

Hardware parameter package/loader, HLS parity, routed implementation, board
parity, full-valid board accuracy, latency, and power stay `NOT_VERIFIED`.
Arbitrary `hardware_evidence` or `board_parity=PASS` flags cannot change this.
Accordingly this revision always returns `NOT_READY_FOR_DEPLOYMENT`,
`formal_eligible=false` and `launch_hardware=false`.

## Commands

```powershell
.venv_cuda/Scripts/python.exe -m tools.sonar_light_deploy.handoff --request <request.json> --output <new-report.json>
.venv_cuda/Scripts/python.exe -m tools.sonar_light_deploy.handoff --inspect-ptq <saved-directory> --output <new-inspection.json>
.venv_cuda/Scripts/python.exe -m pytest -q tests/test_sonar_light_deploy.py
```

The command never overwrites an existing report. The six archived E0/E1
software packages used by regression tests contain diagnostic weights; passing
these tests proves neither detection accuracy nor hardware feasibility.
