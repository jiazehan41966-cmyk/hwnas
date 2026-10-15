"""One bounded campaign controller; wait for free cards, then run frozen code."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--max-wait-hours", type=float, default=None)
    args = parser.parse_args()
    campaign = Path(args.campaign).resolve()
    expected = json.loads((campaign / "cpu640_data_bindings.json").read_text())
    assert json.loads((campaign / "independent_doc_check.json").read_text())["status"] == "PASS"
    started = time.monotonic()
    state = {"status": "WAITING_FOR_IDLE", "budget": {"jobs": 12, "epochs_each": 30},
             "gpu_policy": "GPU0/1 each <500MiB for three consecutive 10-second samples",
             "other_jobs_modified": False, "test_accessed": False, "maximum_wait_hours": args.max_wait_hours}
    def save():
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        temporary = campaign / "scheduling_status.tmp"
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(campaign / "scheduling_status.json")
    try:
        consecutive = 0
        while args.max_wait_hours is None or time.monotonic() - started < args.max_wait_hours * 3600:
            values = []
            for gpu in (0, 1):
                values.append(int(subprocess.check_output(["nvidia-smi", "-i", str(gpu),
                    "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip()))
            consecutive = consecutive + 1 if all(value < 500 for value in values) else 0
            state.update(memory_used_mib=values, consecutive_idle_samples=consecutive)
            save()
            if consecutive >= 3:
                break
            time.sleep(10)
        else:
            state["status"] = "WAIT_EXPIRED_NO_TRAINING"
            save()
            return
        command = [sys.executable, "-m", "tools.sonar_background_pilot.campaign"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        state["status"] = "CUDA_PREFLIGHT_RUNNING"
        save()
        with (campaign / "cuda_preflight.log").open("x") as handle:
            subprocess.run(command + ["preflight", "--campaign", str(campaign), "--data-dir", args.data_dir,
                           "--device", "cuda"], cwd=campaign / "snapshot", env=env,
                           stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=240)
        preflight = json.loads((campaign / "preflight_cuda.json").read_text())
        if preflight["status"] != "PASS" or preflight["data_bindings"] != expected:
            raise RuntimeError("CUDA preflight or local/remote dataset parity failed")
        for variant in ("rank", "fg_bg"):
            if preflight["variants"][variant]["auxiliary"]["raw"] <= 0:
                raise RuntimeError(f"Auxiliary inactive on real data: {variant}")
        state["status"] = "TRAINING_QUEUE_RUNNING"
        save()
        with (campaign / "queue.log").open("x") as handle:
            subprocess.run(command + ["queue", "--campaign", str(campaign), "--data-dir", args.data_dir,
                "--gpus", "0", "1", "--parallel-per-gpu", "2", "--epochs", "30"],
                cwd=campaign / "snapshot", stdout=handle, stderr=subprocess.STDOUT, check=True)
        queue = json.loads((campaign / "queue_status.json").read_text())
        if queue["status"] == "COMPLETE":
            report_env = dict(env, SONAR_BACKGROUND_SNAPSHOT=str(campaign / "snapshot"))
            with (campaign / "acceptance.log").open("x") as handle:
                subprocess.run([sys.executable, str(campaign / "analysis.py"), "--campaign", str(campaign)],
                    cwd=campaign / "snapshot", env=report_env, stdout=handle, stderr=subprocess.STDOUT,
                    check=True, timeout=300)
        state["status"] = queue["status"]
        save()
    except Exception as error:
        state.update(status="FAILED", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()
