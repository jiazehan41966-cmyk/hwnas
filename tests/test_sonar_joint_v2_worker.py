"""Worker binding regressions using synthetic fixtures, never campaign metrics.

The 64 tiny files are byte fixtures, not sonar images.  The model, evaluator,
calibration tensors, and PTQ constructor are mocked: no training or real PTQ
executes.  Actual serialization/hashes exercise budget/checkpoint receipts.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.sonar_joint_v2 import worker as w
from tools.sonar_joint_v2 import run as runner


FIXTURE_AID = "synthetic_fixture_architecture"
FIXTURE_ARCHITECTURE = {"output_combination": "P3_P4_P5", "kernels": {"stage1": 3}}
FIXTURE_PROFILES = ("W8A8", "HEAD_W8A16", "HEAD_W16A16")


@pytest.fixture
def contract(tmp_path, monkeypatch):
    """An isolated train/valid contract with 65 ordered byte fixtures."""
    data = tmp_path / "synthetic_data"
    rows, samples = [], []
    for index in range(65):
        image = data / "train" / "images" / f"{index:03d}.png"
        annotation = data / "train" / "labels" / f"{index:03d}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        annotation.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"synthetic fixture image {index}".encode())
        annotation.write_bytes(f"synthetic fixture annotation {index}".encode())
        samples.append(image)
        rows.append(dict(
            image=image.relative_to(data).as_posix(), image_sha256=w.sha(image),
            annotation=annotation.relative_to(data).as_posix(), annotation_sha256=w.sha(annotation),
        ))
    freeze = dict(
        protocol=dict(
            budgets=[40, 60, 80], confirmation_seeds=[42, 43, 44],
            quant_profiles=list(FIXTURE_PROFILES),
            calibration=dict(split="train", images=64, augmentation=False),
            test_policy="sealed", evaluation_split="valid",
        ),
        dataset=dict(train=rows, valid=[]), files={}, test_evaluated=False,
    )
    baseline = dict(dataset=dict(data_dir="synthetic_data"))
    entries = deepcopy(rows[:64])
    manifest = dict(
        split="train", calibration_seed=42, number_of_calibration_images=64,
        batch_size=4, order="frozen_loader_sorted_first_64", train_image_count=65,
        valid_used_for_selection=False, test_evaluated=False, augmentation=False, shuffle=False,
        entries=entries, split_hash=w._digest([row["image"] for row in rows]),
        dataset_hash=w._digest(entries, sort_keys=True),
    )
    loader = SimpleNamespace(dataset=SimpleNamespace(root=data, samples=samples))
    loader_splits = []

    def fixture_loader(_baseline, _data, split, _seed, _workers, **_kwargs):
        loader_splits.append(split)
        assert split in {"train", "valid"}, "worker attempted to read sealed test data"
        return loader

    monkeypatch.setattr(w, "ROOT", tmp_path)
    monkeypatch.setattr(w, "architectures", lambda: {FIXTURE_AID: deepcopy(FIXTURE_ARCHITECTURE)})
    monkeypatch.setattr(w.frozen, "make_loader", fixture_loader)
    out = tmp_path / "synthetic_campaign"
    w.save(out / "freeze.json", freeze)
    return SimpleNamespace(
        out=out, data=data, freeze=freeze, baseline=baseline, manifest=manifest,
        loader=loader, loader_splits=loader_splits,
    )


def fixture_profile(name):
    """A minimal profile record for precision/bypass validation only."""
    head_weight, head_activation = w.PROFILES[name]

    def layer(path, weight, activation):
        return dict(name=path, weight_bits=weight, input_bits=activation, output_bits=activation,
                    fp32_weight=False, fp32_activation=False)

    return dict(
        layers=[layer("backbone.fixture_conv", 8, 8), layer("head.fixture_conv", head_weight, head_activation)],
        fp32_bypass=dict(software_diagnostic_only=False, neck_adds_quantized=True),
    )


def write_fixture_checkpoint(contract, requested_budget, **updates):
    """Real .pt serialization of a tiny fixture, with no trained weights."""
    state = dict(
        freeze_sha256=w.sha(contract.out / "freeze.json"),
        architecture_id=FIXTURE_AID, architecture=deepcopy(FIXTURE_ARCHITECTURE),
        seed=42, budget=requested_budget, epoch=requested_budget, test_evaluated=False,
        model_state_dict={"synthetic_fixture_token": w.torch.tensor(float(requested_budget))},
    )
    state.update(updates)
    checkpoint = contract.out / "training" / FIXTURE_AID / "seed_42" / f"best_{requested_budget}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    w.torch.save(state, checkpoint)
    return checkpoint


def install_fixture_quantization(contract, monkeypatch):
    """Mock all numeric/model work while retaining real binding and file I/O."""
    calls = dict(ptq=[], evaluation=[], calibration=0)

    class FixtureModel:
        def __init__(self):
            self.synthetic_fixture_token = None

        def to(self, _device):
            return self

        def eval(self):
            return self

        def load_state_dict(self, state, *, strict):
            assert strict is True
            assert set(state) == {"synthetic_fixture_token"}
            self.synthetic_fixture_token = float(state["synthetic_fixture_token"])

        def loss_config(self, _baseline):
            return {}

    def build_fixture_calibration(folder):
        calls["calibration"] += 1
        w.save(folder / "calibration_manifest.json", contract.manifest)
        return deepcopy(contract.manifest)

    def fixture_ptq(model, batches, **kwargs):
        assert len(batches) == 16
        assert kwargs["quantize_neck_adds"] is True
        bits = (kwargs["weight_bits_by_prefix"]["head"], kwargs["activation_bits_by_prefix"]["head"])
        name = next(name for name, expected in w.PROFILES.items() if expected == bits)
        calls["ptq"].append((model.synthetic_fixture_token, name))
        # The fixture evaluator reads the loaded checkpoint token, proving that
        # each budget's result originates from that budget's state dictionary.
        return deepcopy(model), deepcopy(model), {}, fixture_profile(name)

    def synthetic_fixture_metrics(model, _loader, _criterion, _baseline, _device):
        calls["evaluation"].append(model.synthetic_fixture_token)
        value = model.synthetic_fixture_token / 100
        return {key: value for key in ("mAP50_95", "AP50", "AP75", "Precision", "Recall")}, None

    monkeypatch.setattr(w, "build_calibration", build_fixture_calibration)
    monkeypatch.setattr(w, "calibration_tensor_batches", lambda _manifest, _device: iter(
        [w.torch.zeros(4, 1, 2, 2)] * 16
    ))
    monkeypatch.setattr(w, "build_sonar_candidate", lambda *_args: FixtureModel())
    monkeypatch.setattr(w, "DenseAnchorDetectionLoss", lambda _config: w.torch.nn.Identity())
    monkeypatch.setattr(w, "prepare_h2_ptq", fixture_ptq)
    monkeypatch.setattr(w.frozen, "evaluate", synthetic_fixture_metrics)
    return calls


def test_calibration_matches_frozen_train_order_and_content(contract):
    w.validate_calibration(contract.manifest, contract.freeze, contract.baseline)
    assert contract.loader_splits == ["train"]


@pytest.mark.parametrize("field,value", [
    ("split", "valid"), ("split", "test"), ("calibration_seed", 43),
    ("number_of_calibration_images", 63), ("batch_size", 8),
    ("order", "shuffled"), ("train_image_count", 64),
    ("split_hash", "fixture-wrong-full-order"), ("dataset_hash", "fixture-wrong-selected-hash"),
    ("valid_used_for_selection", True), ("test_evaluated", True),
    ("augmentation", True), ("shuffle", True),
])
def test_calibration_rejects_selection_or_manifest_drift(contract, field, value):
    manifest = deepcopy(contract.manifest)
    manifest[field] = value
    with pytest.raises(ValueError, match="Calibration"):
        w.validate_calibration(manifest, contract.freeze, contract.baseline)
    assert contract.loader_splits == []


@pytest.mark.parametrize("change", ["order", "image_identity", "image_hash", "annotation_identity", "annotation_hash"])
def test_calibration_rejects_changed_frozen_entry_binding(contract, change):
    manifest = deepcopy(contract.manifest)
    if change == "order":
        manifest["entries"].reverse()
    else:
        field = {"image_identity": "image", "image_hash": "image_sha256",
                 "annotation_identity": "annotation", "annotation_hash": "annotation_sha256"}[change]
        manifest["entries"][0][field] = "synthetic-fixture-drift"
    with pytest.raises(ValueError, match="identity/hash/order"):
        w.validate_calibration(manifest, contract.freeze, contract.baseline)


def test_calibration_rejects_live_loader_order_drift(contract):
    contract.loader.dataset.samples.reverse()
    with pytest.raises(ValueError, match="Live train-loader identity/order"):
        w.validate_calibration(contract.manifest, contract.freeze, contract.baseline)


@pytest.mark.parametrize("field", ["image", "annotation"])
def test_calibration_rejects_live_file_content_drift(contract, field):
    (contract.data / contract.manifest["entries"][0][field]).write_bytes(b"synthetic changed content")
    with pytest.raises(ValueError, match="file content changed"):
        w.validate_calibration(contract.manifest, contract.freeze, contract.baseline)


@pytest.mark.parametrize("seed,budget", [(42, 40), (42, 60), (42, 80), (43, 80), (44, 80)])
def test_task_accepts_only_supported_stage_or_confirmation_domain(contract, seed, budget):
    w._check_task(FIXTURE_AID, seed, budget, contract.freeze)


@pytest.mark.parametrize("aid,seed,budget", [
    (FIXTURE_AID, 43, 40), (FIXTURE_AID, 44, 60), (FIXTURE_AID, 41, 80),
    (FIXTURE_AID, 42, 20), ("unknown_fixture_architecture", 42, 40),
])
def test_task_rejects_early_confirmation_unknown_candidate_or_budget(contract, aid, seed, budget):
    with pytest.raises(ValueError, match="Task outside|Confirmation seeds"):
        w._check_task(aid, seed, budget, contract.freeze)


def test_task_requires_all_profiles_including_uniform_control(contract):
    freeze = deepcopy(contract.freeze)
    freeze["protocol"]["quant_profiles"] = ["HEAD_W8A16", "HEAD_W16A16"]
    with pytest.raises(ValueError, match="including W8A8"):
        w._check_task(FIXTURE_AID, 42, 40, freeze)


@pytest.mark.parametrize("name", FIXTURE_PROFILES)
def test_profile_accepts_body_int8_and_claimed_head_precision(name):
    w._check_profile(fixture_profile(name), name)


@pytest.mark.parametrize("layer,field,value", [
    (0, "weight_bits", 16), (0, "input_bits", 16), (0, "output_bits", 16),
    (1, "weight_bits", 8), (1, "input_bits", 8), (1, "output_bits", 8),
    (1, "fp32_weight", True), (1, "fp32_activation", True),
])
def test_profile_rejects_precision_drift_and_float_bypass(layer, field, value):
    profile = fixture_profile("HEAD_W16A16")
    profile["layers"][layer][field] = value
    with pytest.raises(ValueError, match="precision|bypasses"):
        w._check_profile(profile, "HEAD_W16A16")


@pytest.mark.parametrize("field,value", [("software_diagnostic_only", True), ("neck_adds_quantized", False)])
def test_profile_rejects_neck_or_diagnostic_bypass(field, value):
    profile = fixture_profile("W8A8")
    profile["fp32_bypass"][field] = value
    with pytest.raises(ValueError, match="FPN integer quantization"):
        w._check_profile(profile, "W8A8")


@pytest.mark.parametrize("field,value,match", [
    ("freeze_sha256", "synthetic-wrong-freeze", "freeze_sha256"),
    ("architecture_id", "synthetic-wrong-architecture", "architecture_id"),
    ("seed", 43, "seed"), ("budget", 80, "budget"),
    ("architecture", {}, "architecture configuration"),
    ("epoch", 80, "epoch outside"), ("test_evaluated", True, "test split"),
])
def test_quantization_rejects_checkpoint_identity_drift_before_evaluation(contract, field, value, match):
    write_fixture_checkpoint(contract, 40, **{field: value})
    with pytest.raises(ValueError, match=match):
        w.quant_candidate(contract.out, FIXTURE_AID, 42, 40, "cpu", contract.freeze,
                          w.sha(contract.out / "freeze.json"), contract.baseline)
    assert contract.loader_splits == []
    assert not (contract.out / "quantization").exists()


def test_quantization_does_not_substitute_later_budget_checkpoint(contract):
    write_fixture_checkpoint(contract, 80)
    with pytest.raises(FileNotFoundError, match="best_40"):
        w.quant_candidate(contract.out, FIXTURE_AID, 42, 40, "cpu", contract.freeze,
                          w.sha(contract.out / "freeze.json"), contract.baseline)
    assert not (contract.out / "quantization").exists()


def test_mocked_budget_quantization_keeps_results_profiles_and_receipts_isolated(contract, monkeypatch):
    calls = install_fixture_quantization(contract, monkeypatch)
    checkpoints = {budget: write_fixture_checkpoint(contract, budget) for budget in (40, 60, 80)}
    monkeypatch.setattr(w, "assert_binding", lambda _out: (
        contract.freeze, w.sha(contract.out / "freeze.json"), contract.baseline
    ))

    def fixture_train(_out, aid, seed, budget, _device, _freeze, _freeze_sha, _baseline):
        assert aid == FIXTURE_AID and seed == 42
        checkpoint = checkpoints[budget]
        return dict(architecture_id=aid, seed=seed, budget=budget, mAP50_95=budget / 100,
                    checkpoint=checkpoint.relative_to(contract.out).as_posix(),
                    checkpoint_sha256=w.sha(checkpoint), test_evaluated=False), [checkpoint]

    monkeypatch.setattr(w, "train_candidate", fixture_train)
    retained_hashes = {}
    for budget in (40, 60, 80):
        for mode in ("train", "quant"):
            monkeypatch.setattr(sys, "argv", [
                "synthetic-worker-fixture", mode, "--output", str(contract.out),
                "--architecture", FIXTURE_AID, "--seed", "42", "--budget", str(budget), "--device", "cpu",
            ])
            w.main()
            result = runner.verify_receipt(contract.out, mode, FIXTURE_AID, 42, budget)
            assert result["budget"] == budget
            assert result["checkpoint"] == checkpoints[budget].relative_to(contract.out).as_posix()
            if mode == "quant":
                assert result["fp32"]["mAP50_95"] == pytest.approx(budget / 100)
                assert set(result["profiles"]) == set(FIXTURE_PROFILES)
                assert result["split"] == "valid" and result["test_evaluated"] is False
                folder = contract.out / "quantization" / FIXTURE_AID / "seed_42" / f"budget_{budget}"
                for name in FIXTURE_PROFILES:
                    profile = w.read(folder / f"{name}_profile.json")
                    assert profile["binding"]["budget"] == budget
                    assert profile["binding"]["checkpoint_sha256"] == result["checkpoint_sha256"]
                    assert profile["precision_profile"] == name
        for path, previous_hash in retained_hashes.items():
            assert w.sha(path) == previous_hash, "a later stage overwrote a prior budget artifact"
        for path in (contract.out / "tasks").glob("*.json"):
            retained_hashes[path] = w.sha(path)
        for path in (contract.out / "quantization").rglob("*.json"):
            retained_hashes[path] = w.sha(path)
    assert len(list((contract.out / "tasks").glob("quant_*.json"))) == 3
    assert calls["calibration"] == 1
    assert calls["ptq"] == [(float(budget), name) for budget in (40, 60, 80) for name in FIXTURE_PROFILES]
    assert calls["evaluation"] == [float(budget) for budget in (40, 60, 80) for _ in range(4)]
    assert set(contract.loader_splits) == {"train", "valid"}


def test_runner_receipt_rejects_checkpoint_changed_after_mocked_quantization(contract, monkeypatch):
    install_fixture_quantization(contract, monkeypatch)
    checkpoint = write_fixture_checkpoint(contract, 40)
    freeze_sha = w.sha(contract.out / "freeze.json")
    result, artifacts = w.quant_candidate(contract.out, FIXTURE_AID, 42, 40, "cpu",
                                          contract.freeze, freeze_sha, contract.baseline)
    w.save(contract.out / "tasks" / f"quant_{FIXTURE_AID}_42_40.json", dict(
        freeze_sha256=freeze_sha, result=result,
        outputs={path.relative_to(contract.out).as_posix(): w.sha(path) for path in artifacts},
    ))
    checkpoint.write_bytes(b"synthetic later-checkpoint replacement")
    with pytest.raises(ValueError, match="Task artifact drift"):
        runner.verify_receipt(contract.out, "quant", FIXTURE_AID, 42, 40)
