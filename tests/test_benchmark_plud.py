from pathlib import Path

import torch

from hwnas_fpga.benchmarks.plud import (
    PludAuthorRecipeConfig,
    load_plud_author_components,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMIT = "eea8dc07ce007988150ac208cd09e00daedba2ca"
ARCHIVE_SHA = "4bd5158c491821bb1de3138856344949ca3ce1747f033601809d30774e7d5a61"


def test_plud_author_push_loss_loads_from_verified_isolated_archive():
    components = load_plud_author_components(
        REPO_ROOT / "reference/_local/Sonar-OLTR",
        pinned_commit=COMMIT,
        archive_sha256=ARCHIVE_SHA,
    )
    logits = torch.tensor([[-2.0, 0.0, 2.0]])
    value = components.module.push_logit_loss(logits, gamma=0.5)
    expected = torch.sigmoid(0.5 * logits).mean()
    assert torch.equal(value, expected)
    assert components.commit == COMMIT
    assert len(components.source_hashes) == 4


def test_plud_recipe_matches_author_defaults_and_records_bridge():
    recipe = PludAuthorRecipeConfig()
    recipe.validate()
    assert recipe.lr == 0.01
    assert recipe.classifier_lr_multiplier == 10.0
    assert recipe.momentum == 0.0
    assert recipe.alpha == 1.5
    assert recipe.gamma == 0.5
    assert recipe.milestones == (35, 75)
    assert recipe.to_dict()["feature_bridge"] == "input to final Linear classifier"
