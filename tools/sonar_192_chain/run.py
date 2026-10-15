"""Explicit, bounded 192 v2 interface commands. There is no search/launch command."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .contract import ROOT, model_contract, make_identity, digest, write_json_new


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare(args):
    import torch
    from .model import load_checkpoint
    from .quantization import (calibration_from_training_manifest, prepare_ptq, verify_parity,
                               save_artifacts, numeric_contract)
    from .handoff import prepare_parameter_package
    torch.set_num_threads(args.threads)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new output directory; existing evidence is immutable")
    purpose = "diagnostic" if args.diagnostic else "formal"
    model, checkpoint = load_checkpoint(args.checkpoint, architecture=args.architecture, seed=args.seed,
        purpose=purpose, execution_manifest=args.execution_manifest, training_result=args.training_result)
    calibration = calibration_from_training_manifest(args.dataset_root, args.execution_manifest,
        num_images=args.calibration_images, formal=not args.diagnostic)
    if args.mapping:
        mapping = _read(args.mapping)
    elif args.diagnostic:
        mapping = dict(implementation_id="sonar192_software_reference_v1", PE=1, SIMD=1,
            buffer_profile="logical_CHW_host_reference", status="PROPOSED",
            numeric_contract_sha256=numeric_contract(args.profile)[1],
            interface_contract_sha256=digest(dict(abi="sonar192_integer_transport_v1",
                                                  scope="software_transport_not_hardware_implementation")))
    else:
        raise ValueError("Formal preparation requires an explicit frozen concrete --mapping")
    cal = dict(calibration.manifest, sha256=calibration.manifest["calibration_sha256"], source_verified=True)
    identity = make_identity(args.architecture, checkpoint=checkpoint, profile=args.profile,
                             mapping=mapping, calibration=cal, purpose=purpose)
    write_json_new(args.output / "candidate_identity.json", identity)
    ptq = prepare_ptq(model, calibration, identity=identity, profile=args.profile)
    images = next(calibration.batches())
    witness = verify_parity(ptq, images)
    save_artifacts(args.output / "quantization", ptq, witness=witness)
    report = dict(status="SOFTWARE_PARITY_FAIL", identity_sha256=identity["identity_sha256"],
                  software_parity=witness, formal_candidate_admitted=False, hardware_launched=False,
                  search_launched=False, training_launched=False, test_evaluated=False)
    if witness["mismatches"] == 0:
        confirmation = None
        if args.confirmation:
            from .confirmation import reference
            confirmation = reference(args.confirmation)
        if purpose == "formal" and confirmation is None:
            report.update(status="BLOCKED_AWAITING_THREE_SEED_CONFIRMATION",
                          quantization_export_complete=True,
                          next_step="confirm real three-seed prediction receipts, then package this same export")
        else:
            package = prepare_parameter_package(identity, args.output / "quantization/artifact_manifest.json",
                output=args.output / "parameter_package", confirmation=confirmation)
            report.update(status="SOFTWARE_HANDOFF_READY_AWAITING_BOARD_EVIDENCE", package=package)
    write_json_new(args.output / "single_candidate_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("contract", help="Inspect a frozen model contract without dataset/model execution")
    p.add_argument("--architecture", default="9ee366")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("prepare", help="One checkpoint -> PTQ -> integer transport; never launch hardware")
    p.add_argument("--architecture", required=True)
    p.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--execution-manifest", type=Path, required=True)
    p.add_argument("--training-result", type=Path)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--mapping", type=Path)
    p.add_argument("--profile", default="W8A8", choices=("W8A8", "HEAD_W8A16", "HEAD_W16A16"))
    p.add_argument("--calibration-images", type=int, default=64)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--confirmation", type=Path)
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("package", help="Verify an existing PTQ export and write a transport package")
    p.add_argument("--identity", type=Path, required=True)
    p.add_argument("--export-manifest", type=Path, required=True)
    p.add_argument("--confirmation", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("verify-board", help="Verify a returned board artifact; never contact a board")
    p.add_argument("--identity", type=Path, required=True)
    p.add_argument("--loader-manifest", type=Path, required=True)
    p.add_argument("--board-receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("confirm", help="Recompute paired three-seed quantization admission from raw predictions")
    p.add_argument("--identity", type=Path, required=True)
    p.add_argument("--receipts", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "contract":
        result = model_contract(args.architecture)
        write_json_new(args.output, result)
    elif args.command == "prepare":
        result = prepare(args)
    elif args.command == "package":
        from .handoff import prepare_parameter_package
        from .confirmation import reference
        result = prepare_parameter_package(_read(args.identity), args.export_manifest, output=args.output,
            confirmation=reference(args.confirmation) if args.confirmation else None)
    elif args.command == "verify-board":
        from .handoff import validate_board_return
        result = validate_board_return(_read(args.identity), args.loader_manifest, args.board_receipt)
        write_json_new(args.output, result)
    else:
        from .confirmation import confirm_candidates
        result = confirm_candidates(_read(args.identity), _read(args.receipts), _read(args.protocol))
        write_json_new(args.output, result)
    print(json.dumps({k: result[k] for k in ("status", "space_version", "identity_sha256", "formal_eligible") if k in result}, indent=2))
    return 2 if "FAIL" in result.get("status", "") or "BLOCKED" in result.get("status", "") else 0


if __name__ == "__main__":
    raise SystemExit(main())
