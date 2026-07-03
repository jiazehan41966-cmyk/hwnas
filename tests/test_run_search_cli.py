import argparse
import json
import tempfile
import unittest
from pathlib import Path
import sys
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_search


class RunSearchCliTests(unittest.TestCase):
    def _write_pool_file(self, tmpdir: str) -> Path:
        pool_path = Path(tmpdir) / "selected_backbone_pool.json"
        pool_path.write_text(
            json.dumps(
                {
                    "recommended_profiles": {
                        "mobile_anchor": {"source_role": "search_anchor"},
                        "accuracy_biased": {"source_role": "accuracy_anchor"},
                        "lightweight_sonar": {"source_role": "lightweight_anchor"},
                    }
                }
            ),
            encoding="utf-8",
        )
        return pool_path

    def test_parse_args_supports_backbone_pool_and_anchor_role(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "run_search.py",
                "--config",
                "configs/search/nksid_fpga_search_av7k325.yaml",
                "--backbone-pool",
                "selected_backbone_pool.json",
                "--anchor-role",
                "accuracy_anchor",
            ],
        ):
            args = run_search.parse_args()
        self.assertEqual(args.backbone_pool, "selected_backbone_pool.json")
        self.assertEqual(args.anchor_role, "accuracy_anchor")

    def test_resolve_family_profile_uses_backbone_pool_when_yaml_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pool_path = self._write_pool_file(tmpdir)
            args = argparse.Namespace(
                family_profile=None,
                backbone_pool=str(pool_path),
                anchor_role="search_anchor",
            )
            config = {"search_space": {}}
            family_profile, source = run_search._resolve_family_profile(args, config)
        self.assertEqual(family_profile, "mobile_anchor")
        self.assertEqual(source, "pool")
        self.assertEqual(config["search_space"]["family_profile"], "mobile_anchor")

    def test_resolve_family_profile_prefers_cli_over_pool_and_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pool_path = self._write_pool_file(tmpdir)
            args = argparse.Namespace(
                family_profile="accuracy_biased",
                backbone_pool=str(pool_path),
                anchor_role="search_anchor",
            )
            config = {"search_space": {"family_profile": "lightweight_sonar"}}
            family_profile, source = run_search._resolve_family_profile(args, config)
        self.assertEqual(family_profile, "accuracy_biased")
        self.assertEqual(source, "cli")
        self.assertEqual(config["search_space"]["family_profile"], "accuracy_biased")

    def test_resolve_family_profile_keeps_yaml_when_cli_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pool_path = self._write_pool_file(tmpdir)
            args = argparse.Namespace(
                family_profile=None,
                backbone_pool=str(pool_path),
                anchor_role="search_anchor",
            )
            config = {"search_space": {"family_profile": "lightweight_sonar"}}
            family_profile, source = run_search._resolve_family_profile(args, config)
        self.assertEqual(family_profile, "lightweight_sonar")
        self.assertEqual(source, "config")
        self.assertEqual(config["search_space"]["family_profile"], "lightweight_sonar")


if __name__ == "__main__":
    unittest.main()
