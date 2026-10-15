"""Static package contracts: discovery does not import ML or execute a pipeline."""
import ast
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("space", "models", "train", "quant", "hw", "search", "scripts")
SOURCE_FILES = sorted(path for name in PACKAGES for path in (ROOT / name).rglob("*.py"))
SOURCE_FILES += [ROOT / "experiment.py"]
STDLIB = sys.stdlib_module_names | {"__future__"}
FORBIDDEN = {"hwnas_fpga", "tools"}


def _qualified_from(path, node):
    """Resolve relative imports without importing either side of the boundary."""
    relative = path.relative_to(ROOT)
    package_parts = list(relative.parts[:-1])
    if node.level:
        if node.level > len(package_parts):
            return "<outside_package>"
        base = package_parts[:len(package_parts) - node.level + 1]
        return ".".join(base + (node.module.split(".") if node.module else []))
    return node.module or ""


def _imports(path, tree):
    # ast.walk includes imports inside functions, conditionals and TYPE_CHECKING.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                yield node.lineno, item.name
        elif isinstance(node, ast.ImportFrom):
            yield node.lineno, _qualified_from(path, node)
        elif isinstance(node, ast.Call):
            function = node.func
            is_dynamic = ((isinstance(function, ast.Name) and function.id in {"__import__", "import_module"})
                          or (isinstance(function, ast.Attribute) and function.attr == "import_module"))
            if is_dynamic:
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    yield node.lineno, node.args[0].value
                else:
                    yield node.lineno, "<computed_dynamic_import>"


def _boundary_violations(path, source):
    tree = ast.parse(source, filename=str(path))
    package = path.relative_to(ROOT).parts[0]
    errors = []
    for line, imported in _imports(path, tree):
        root = imported.split(".", 1)[0]
        message = None
        if root in FORBIDDEN or root.startswith("<"):
            message = "legacy or unresolvable runtime import"
        elif package == "space" and root not in STDLIB | {"space"}:
            message = "space must remain independent of ML and orchestration runtimes"
        elif package == "models" and root not in STDLIB | {"models", "space", "torch"}:
            message = "models may import only space, torch, standard library and its own package"
        elif package == "quant" and root in {"train", "hw", "search"}:
            message = "quant cannot depend on training, hardware or search"
        elif package == "hw" and root in {"models", "train", "search"}:
            message = "hardware evidence cannot depend on models, training or search"
        elif package == "search" and root == "models":
            message = "search cannot construct or import models"
        if message:
            errors.append(f"{path.name}:{line}: {message}: {imported}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and package == "search":
            called = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else None
            if called == "build_model":
                errors.append(f"{path.name}:{node.lineno}: model construction belongs outside search")
        if isinstance(node, ast.Attribute) and node.attr == "path" and isinstance(node.value, ast.Name) and node.value.id == "sys":
            errors.append(f"{path.name}:{node.lineno}: package must not bootstrap legacy code through sys.path")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value.replace("\\", "/").lower()
            if ("e:/1/hwnas/hwnas" in text or "src/hwnas_fpga" in text
                    or text.startswith("hwnas_fpga.") or text.startswith("tools.")):
                errors.append(f"{path.name}:{node.lineno}: legacy runtime path embedded in package")
    return errors


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda path: path.relative_to(ROOT).as_posix())
def test_package_dependency_boundaries(path):
    violations = _boundary_violations(path, path.read_text(encoding="utf-8-sig"))
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("package,source", [
    ("space", "import torch"),
    ("models", "def delayed():\n    from train import evaluate"),
    ("quant", "from hw import evaluate"),
    ("hw", "from search import pareto_records"),
    ("search", "model_factory.build_model(candidate)"),
    ("search", "from models import build_model as create"),
    ("quant", "__import__('tools.fpga_h4.numeric')"),
    ("hw", "import importlib\nimportlib.import_module('hwnas_fpga.hardware')"),
    ("space", "import importlib\nimportlib.import_module(module_name)"),
    ("train", "import sys\nsys.path.insert(0, 'legacy')"),
    ("models", "ROOT = r'E:\\1\\hwnas\\hwnas\\src'"),
])
def test_boundary_auditor_detects_hidden_upward_and_legacy_dependencies(package, source):
    assert _boundary_violations(ROOT / package / "example.py", source)


def test_relative_same_package_imports_are_allowed():
    assert not _boundary_violations(ROOT / "models" / "neck.py", "from .blocks import MBConvBlock")
    assert not _boundary_violations(ROOT / "hw" / "backends" / "report.py", "from .. import HardwareResult")
