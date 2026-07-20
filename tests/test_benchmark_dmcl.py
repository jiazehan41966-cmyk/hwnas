from pathlib import Path

import torch

from hwnas_fpga.benchmarks.dmcl import (
    DmclAuthorRecipeConfig,
    load_dmcl_author_components,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMIT = "eea8dc07ce007988150ac208cd09e00daedba2ca"
ARCHIVE_SHA = "4bd5158c491821bb1de3138856344949ca3ce1747f033601809d30774e7d5a61"


def test_dmcl_author_loss_loads_from_verified_isolated_archive():
    components = load_dmcl_author_components(
        REPO_ROOT / "reference/_local/Sonar-OLTR",
        pinned_commit=COMMIT,
        archive_sha256=ARCHIVE_SHA,
    )
    assert components.commit == COMMIT
    assert components.archive_sha256 == ARCHIVE_SHA
    assert len(components.source_hashes) == 4
    loss = components.module.DynamicMarginLoss(3)
    targets = torch.tensor([0, 0, 1, 1])
    loss.update_class_counts(targets)
    loss.update_margins()
    value = loss(torch.randn(4, 8), targets)
    assert torch.isfinite(value)


def test_dmcl_recipe_matches_author_defaults_and_frozen_bridge():
    recipe = DmclAuthorRecipeConfig()
    recipe.validate()
    assert recipe.lr == 0.01
    assert recipe.lambda_contrast == 0.5
    assert recipe.lambda_uncertainty == 0.1
    assert recipe.milestones == (35, 75)
    assert recipe.to_dict()["author_component"] == "DynamicMarginLoss"
