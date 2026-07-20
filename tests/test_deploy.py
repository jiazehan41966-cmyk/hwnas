import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

from hwnas_fpga.deploy import (
    HLSProjectConfig,
    attach_hls_report,
    create_hls_project_stub,
    export_checkpoint_quantized_weights,
    export_checkpoint_to_onnx,
)
from hwnas_fpga.models import build_model
from hwnas_fpga.search_space import SearchSpace, SearchSpaceConfig


class DeployTests(unittest.TestCase):
    def test_export_checkpoint_and_prepare_hls_stub(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "run"
            checkpoints_dir = run_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True)

            config = {
                "dataset": {
                    "image_size": 32,
                    "input_channels": 1,
                }
            }
            with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle)

            space = SearchSpace(SearchSpaceConfig(image_size=32))
            architecture = space.baseline_architecture()
            model = build_model(architecture, num_classes=8)
            checkpoint_path = checkpoints_dir / "best_model.pt"
            torch.save(
                {
                    "candidate": {
                        "arch_id": "arch_0",
                        "encoding": architecture.to_dict(),
                        "metrics": {},
                    },
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )

            def fake_export(_model, _dummy_input, target, **_kwargs):
                Path(target).write_bytes(b"fake-onnx")

            with mock.patch("torch.onnx.export", side_effect=fake_export):
                onnx_path, metadata = export_checkpoint_to_onnx(
                    checkpoint_path,
                    export_path=root / "exported" / "model.onnx",
                    device="cpu",
                    image_size_override=32,
                    input_channels_override=1,
                )

            self.assertTrue(onnx_path.exists())
            self.assertEqual(metadata["image_size"], 32)

            hls_dir = root / "hls_project"
            manifest = create_hls_project_stub(
                onnx_path,
                hls_dir,
                config=HLSProjectConfig(board="zynq7020", clock_mhz=200),
            )
            self.assertTrue((hls_dir / "hls_project.json").exists())
            self.assertTrue((hls_dir / "run_hls.tcl").exists())
            self.assertEqual(manifest["config"]["board"], "zynq7020")

            report_path = root / "csynth.rpt"
            report_path.write_text(
                "\n".join(
                    [
                        "Latency (cycles): 12345",
                        "BRAM_18K | 12",
                        "DSP48E   | 34",
                        "LUT      | 4321",
                    ]
                ),
                encoding="utf-8",
            )
            metrics = attach_hls_report(hls_dir, report_path)
            self.assertEqual(metrics["cycles"], 12345)
            self.assertEqual(metrics["dsp"], 34)
            self.assertTrue((hls_dir / "hls_report.json").exists())

    def test_export_quantized_weight_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoints_dir = root / "checkpoints"
            checkpoints_dir.mkdir(parents=True)

            space = SearchSpace(SearchSpaceConfig(image_size=32))
            architecture = space.baseline_architecture()
            model = build_model(architecture, num_classes=8)
            checkpoint_path = checkpoints_dir / "best_model.pt"
            torch.save(
                {
                    "candidate": {
                        "arch_id": "arch_0",
                        "encoding": architecture.to_dict(),
                        "metrics": {},
                    },
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )

            package_path, summary = export_checkpoint_quantized_weights(
                checkpoint_path,
                output_path=root / "quantized" / "weights_int8.pt",
                device="cpu",
            )

            self.assertTrue(package_path.exists())
            self.assertGreater(summary["num_quantized_tensors"], 0)
            self.assertFalse(summary["parity_ready"])
            self.assertTrue((package_path.with_suffix(".json")).exists())

            payload = torch.load(package_path, map_location="cpu")
            self.assertEqual(payload["format"], "hwnas_fpga_int8_weights_v1")
            self.assertIn("head.fc.weight", payload["weights"])
            self.assertEqual(payload["weights"]["head.fc.weight"].dtype, torch.int8)


if __name__ == "__main__":
    unittest.main()
