"""Apply the predeclared screening gate after all twelve runs complete."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(os.environ.get("SONAR_BACKGROUND_SNAPSHOT", Path(__file__).resolve().parents[2]))
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from scripts.detection_controlled_study import summarize_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default=str(ROOT / "results/sonar_background_pilot_20261010"))
    args = parser.parse_args()
    campaign = Path(args.campaign)
    data = campaign / "remote_results" if (campaign / "remote_results").is_dir() else campaign
    plan = json.loads((campaign / "PLAN.json").read_text(encoding="utf-8"))
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    details = {}
    for variant in plan["arms"]:
        for seed in plan["seeds"]:
            run = data / "runs" / f"{variant}_seed{seed}"
            status = json.loads((run / "status.json").read_text(encoding="utf-8"))
            if status["status"] != "COMPLETE" or status["epochs"] != plan["epochs_per_arm_seed"]:
                raise RuntimeError("Incomplete bounded campaign")
            rows = [json.loads(line) for line in (run / "valid_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            if len(rows) != plan["splits"]["valid"] or status["test_accessed"]:
                raise RuntimeError("Evaluation split contract violated")
            fixed = {}
            for threshold in (0.1, 0.3, 0.5):
                summary = summarize_errors(rows, ["cylider", "manta"], threshold)
                fixed[str(threshold)] = {name: {
                    "background": counts.get("background", 0), "precision": counts["precision"],
                    "recall": counts["recall"], "true_positive": counts["true_positive"],
                    "targets": counts["targets"]} for name, counts in summary["per_class"].items()}
            details[(variant, seed)] = {"status": status, "fixed_thresholds": fixed}
    verdicts = {}
    for variant in plan["arms"]:
        rows = comparison["variants"][variant]["runs"]
        deltas = {}
        for scope in ("all", "manta"):
            values = []
            for seed in plan["seeds"]:
                actual = details[(variant, seed)]["status"]["recall_at_fppi"][scope]["0.5"]
                base = details[("baseline", seed)]["status"]["recall_at_fppi"][scope]["0.5"]
                values.append(None if actual is None or base is None else actual["recall"] - base["recall"])
            deltas[scope] = None if any(value is None for value in values) else float(np.mean(values))
        summary = comparison["variants"][variant]
        gates = {"map_delta_gt_0_005": summary["mean_map_delta"] > 0.005,
                 "at_least_two_positive_seeds": summary["positive_map_seeds"] >= 2,
                 "f1_not_lower": summary["mean_f1_delta"] >= 0,
                 "manta_recall_protected": deltas["manta"] is not None and deltas["manta"] >= -0.02,
                 "overall_fppi_recall_not_lower": deltas["all"] is not None and deltas["all"] >= 0}
        verdicts[variant] = {"screening_gate": gates, "passed": all(gates.values()) if variant != "baseline" else None,
            "recall_delta_at_fppi_0_5": deltas,
            "mean_ap50": float(np.mean([details[(variant, seed)]["status"]["best_validation"]["map50"] for seed in plan["seeds"]])),
            "mean_ap75": float(np.mean([details[(variant, seed)]["status"]["best_validation"]["map75"] for seed in plan["seeds"]])),
            "mean_coco101_ap": float(np.mean([details[(variant, seed)]["status"]["validation_coco101"]["coco_ap"] for seed in plan["seeds"]])),
            "fixed_thresholds": {str(seed): details[(variant, seed)]["fixed_thresholds"] for seed in plan["seeds"]},
            "paired": rows}
    result = {"test_accessed": False, "predeclared_gate": plan["promotion_gate"],
              "variants": verdicts, "limitation": "Screening only; not a significance claim; validation-selected epochs and thresholds"}
    (campaign / "acceptance.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({variant: value["passed"] for variant, value in verdicts.items()}))


if __name__ == "__main__":
    main()
