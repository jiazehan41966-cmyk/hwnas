#!/usr/bin/env python3
"""HW-NAS 完整流程测试脚本"""

import sys
sys.path.insert(0, 'src')

import torch

print("=" * 50)
print("HW-NAS FPGA Sonar 完整流程测试")
print("=" * 50)

# Step 1: 数据加载
print("\n[Step 1] 数据加载")
from hwnas_fpga.data import NKSIDDataset
ds = NKSIDDataset('data/NKSID', use_kfold=False)
print(f"  ✅ NKSID: {len(ds)} 张图片, 8类")

# Step 2: 搜索空间
print("\n[Step 2] 搜索空间定义")
from hwnas_fpga.search_space import SearchSpace, SearchSpaceConfig
from hwnas_fpga.interfaces import SearchConstraints
constraints = SearchConstraints(max_latency_ms=50, max_dsp=200)
ss = SearchSpace(SearchSpaceConfig(num_classes=8, hardware_constraints=constraints))
print(f"  ✅ {ss.config.stage_count} stages")

# Step 3: 硬件成本模型
print("\n[Step 3] 硬件成本模型")
from hwnas_fpga.hardware import FPGACostEstimator
from hwnas_fpga.interfaces import HardwareSpec
hw = HardwareSpec('Zynq7020', 200, max_dsp=220, max_bram=140, max_lut=53200)
est = FPGACostEstimator(hw, constraints)
print(f"  ✅ 目标: Zynq7020")

# Step 4: 架构搜索
print("\n[Step 4] 架构搜索")
pruned = ss.pre_prune(est)
arch = pruned.sample(seed=42)
cost = est.estimate(arch, pruned)
print(f"  ✅ 剪枝后采样: latency={cost.latency_ms:.1f}ms, DSP={cost.peak_dsp}")

# Step 5: 模型构建
print("\n[Step 5] 模型构建")
from hwnas_fpga.models import build_model
model = build_model(arch, num_classes=8)
params = sum(p.numel() for p in model.parameters())
y = model(torch.randn(1, 1, ss.config.image_size, ss.config.image_size))
print(f"  ✅ 参数量: {params:,}, 输出: {y.shape}")

# Step 6: 部署量化准备
print("\n[Step 6] 部署量化准备")
from hwnas_fpga.deploy import build_quantized_weight_package

package, summary = build_quantized_weight_package(
    model,
    architecture=arch.to_dict(),
    class_names=[
        "big_propeller",
        "cylinder",
        "fishing_net",
        "floats",
        "iron_pipeline",
        "small_propeller",
        "soft_pipeline",
        "tire",
    ],
)
print(
    "  ✅ INT8包: "
    f"{summary['num_quantized_tensors']} 个张量, "
    f"压缩比 {summary['compression_ratio']:.2f}x"
)
print(f"  ✅ 包格式: {package['format']}")

print("\n" + "=" * 50)
print("✅ 结论: HW-NAS 6步流程全部正常!")
print("=" * 50)
