import inspect
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import architecture_transfer_common as transfer
import finalize_phase3c_transfer as finalizer


def test_formal_training_never_constructs_test_loader():
    source = inspect.getsource(transfer.formal_run)
    assert 'make_loader(config, "test"' not in source
    assert '"test_evaluated": False' in source
    finalizer_source = inspect.getsource(finalizer.evaluate_all_frozen_checkpoints)
    assert finalizer_source.index("_verify_frozen_manifest()") < finalizer_source.index('make_loader(config, "test"')
    assert finalizer_source.index('"status": "STARTED"') < finalizer_source.index('make_loader(config, "test"')


def test_smoke_never_constructs_test_loader():
    assert 'make_loader(config, "test"' not in inspect.getsource(transfer.smoke)
