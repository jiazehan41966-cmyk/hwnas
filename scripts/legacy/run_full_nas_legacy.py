#!/usr/bin/env python3
"""
HW-NAS 完整端到端流程
====================

输入:
  - NKSID声呐数据集
  - Zynq7020硬件约束

输出:
  1. 最优网络架构 (layer type, kernel size, channel, depth)
  2. 训练后的模型权重 (.pth)
  3. 部署版本 (ONNX)
"""

import sys
sys.path.insert(0, 'src')

import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# HW-NAS 模块
from hwnas_fpga.data import NKSIDDataset, create_nksid_dataloaders, NKSID_CLASSES
from hwnas_fpga.search_space import SearchSpace, SearchSpaceConfig, ArchitectureSpec
from hwnas_fpga.interfaces import SearchConstraints, HardwareSpec
from hwnas_fpga.hardware import FPGACostEstimator
from hwnas_fpga.search import RandomSearcher
from hwnas_fpga.models import build_model
from hwnas_fpga.training import Trainer
from hwnas_fpga.deploy.export import export_to_onnx, get_model_info


def print_header(title: str):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_architecture(arch: ArchitectureSpec, title: str = "网络架构"):
    """打印架构详情"""
    print(f"\n📐 {title}")
    print("-" * 50)
    print(f"{'Stage':<8} {'Operator':<20} {'Kernel':<8} {'Channels':<10} {'Depth':<6}")
    print("-" * 50)
    
    for i, stage in enumerate(arch.stages):
        for j, block in enumerate(stage.blocks):
            stage_name = f"Stage{i+1}" if j == 0 else ""
            print(f"{stage_name:<8} {block.op_type:<20} {block.kernel_size}x{block.kernel_size:<6} {block.out_channels:<10} {j+1}")
    
    print("-" * 50)
    print(f"Stem channels: {arch.stem_channels}")
    print(f"Head channels: {arch.head_channels}")


def search_best_architecture(
    search_space: SearchSpace,
    cost_estimator: FPGACostEstimator,
    constraints: SearchConstraints,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    num_candidates: int = 20,
    proxy_epochs: int = 5,
    device: str = "cpu",
) -> tuple:
    """搜索最优架构"""
    
    print_header("Step 1: 架构搜索")
    
    # 硬件驱动剪枝
    print("\n🔧 应用硬件驱动剪枝...")
    pruned_space = search_space.pre_prune(cost_estimator)
    print(f"   剪枝后通道选择: {pruned_space.config.channel_choices}")
    print(f"   剪枝后算子选择: {pruned_space.config.op_choices}")
    
    # 创建搜索器
    searcher = RandomSearcher(pruned_space, cost_estimator, constraints, seed=42)
    
    best_candidate = None
    best_accuracy = 0.0
    candidates_log = []
    
    print(f"\n🔍 搜索 {num_candidates} 个候选架构...")
    print(f"{'#':<4} {'Latency':<12} {'DSP':<8} {'Feasible':<10} {'Accuracy':<10}")
    print("-" * 50)
    
    for i in range(num_candidates):
        # 采样架构
        arch = searcher.sample_candidate()
        
        # 硬件成本估算
        cost = cost_estimator.estimate(arch, pruned_space)
        feasible = searcher.check_feasibility(cost)
        
        if not feasible:
            print(f"{i+1:<4} {cost.latency_ms:<12.2f} {cost.peak_dsp:<8} {'❌':<10} {'skip':<10}")
            continue
        
        # 构建模型并快速训练
        model = build_model(arch, num_classes=len(NKSID_CLASSES))
        model.to(device)
        
        # 快速代理训练
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            class_weights=class_weights,
        )
        
        history = trainer.train(epochs=proxy_epochs, verbose=False)
        accuracy = history.get('best_val_acc', 0.0)
        
        status = "✅" if feasible else "❌"
        print(f"{i+1:<4} {cost.latency_ms:<12.2f} {cost.peak_dsp:<8} {status:<10} {accuracy*100:<10.2f}%")
        
        # 记录候选
        candidates_log.append({
            'arch_id': i,
            'latency_ms': cost.latency_ms,
            'dsp': cost.peak_dsp,
            'bram': cost.peak_bram,
            'lut': cost.peak_lut,
            'accuracy': accuracy,
            'feasible': feasible,
        })
        
        # 更新最优
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_candidate = (arch, cost, accuracy)
    
    if best_candidate is None:
        raise RuntimeError("未找到可行的候选架构！")
    
    best_arch, best_cost, acc = best_candidate
    print(f"\n🏆 最优候选: Latency={best_cost.latency_ms:.2f}ms, Accuracy={acc*100:.2f}%")
    
    return best_arch, best_cost, candidates_log


def train_final_model(
    arch: ArchitectureSpec,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    epochs: int = 50,
    device: str = "cpu",
) -> tuple:
    """训练最终模型"""
    
    print_header("Step 2: 训练最终模型")
    
    # 构建模型
    model = build_model(arch, num_classes=len(NKSID_CLASSES))
    model.to(device)
    
    # 模型信息
    info = get_model_info(model, input_shape=(1, 1, 224, 224))
    print(f"\n📊 模型统计:")
    print(f"   参数量: {info['total_params']:,}")
    print(f"   模型大小: {info['param_size_mb']:.2f} MB")
    print(f"   估算MACs: {info['estimated_macs']:,}")
    
    # 训练
    print(f"\n🏋️ 开始训练 ({epochs} epochs)...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        class_weights=class_weights,
        lr=0.001,
    )
    
    history = trainer.train(epochs=epochs, verbose=True, early_stopping_patience=10)
    
    final_acc = history.get('best_val_acc', 0.0)
    print(f"\n✅ 训练完成! 最佳验证准确率: {final_acc*100:.2f}%")
    
    return model, history


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: str = "cpu",
) -> dict:
    """详细评估模型"""
    
    print_header("Step 3: 模型评估")
    
    model.eval()
    
    correct = 0
    total = 0
    class_correct = [0] * len(NKSID_CLASSES)
    class_total = [0] * len(NKSID_CLASSES)
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    overall_acc = correct / total
    
    print(f"\n📈 总体准确率: {overall_acc*100:.2f}%")
    print(f"\n📊 各类别准确率:")
    print(f"{'类别':<20} {'正确/总数':<15} {'准确率':<10}")
    print("-" * 50)
    
    for i, cls_name in enumerate(NKSID_CLASSES):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            print(f"{cls_name:<20} {class_correct[i]}/{class_total[i]:<13} {acc*100:.2f}%")
        else:
            print(f"{cls_name:<20} {'0/0':<15} {'N/A':<10}")
    
    return {
        'overall_accuracy': overall_acc,
        'class_accuracy': {
            NKSID_CLASSES[i]: class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            for i in range(len(NKSID_CLASSES))
        }
    }


def export_for_deployment(
    model: nn.Module,
    arch: ArchitectureSpec,
    output_dir: Path,
) -> dict:
    """导出部署版本"""
    
    print_header("Step 4: 导出部署版本")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    exports = {}
    
    # 1. 保存PyTorch模型
    pytorch_path = output_dir / "model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'architecture': arch.to_dict(),
        'num_classes': len(NKSID_CLASSES),
        'class_names': NKSID_CLASSES,
    }, pytorch_path)
    exports['pytorch'] = pytorch_path
    print(f"✅ PyTorch模型: {pytorch_path}")
    
    # 2. 导出ONNX
    onnx_path = output_dir / "model.onnx"
    export_to_onnx(model, onnx_path, input_shape=(1, 1, 224, 224), verbose=False)
    exports['onnx'] = onnx_path
    print(f"✅ ONNX模型: {onnx_path}")
    
    # 3. 保存架构描述
    arch_path = output_dir / "architecture.json"
    with arch_path.open('w') as f:
        json.dump(arch.to_dict(), f, indent=2)
    exports['architecture'] = arch_path
    print(f"✅ 架构描述: {arch_path}")
    
    # 4. 生成HLS配置 (stub)
    hls_config_path = output_dir / "hls_config.json"
    hls_config = {
        'target_device': 'xc7z020clg400-1',
        'clock_period_ns': 5,
        'input_shape': [1, 1, 224, 224],
        'output_classes': len(NKSID_CLASSES),
        'quantization': 'int8',
        'onnx_model': str(onnx_path),
        'notes': 'Use Vitis HLS to synthesize from ONNX model',
    }
    with hls_config_path.open('w') as f:
        json.dump(hls_config, f, indent=2)
    exports['hls_config'] = hls_config_path
    print(f"✅ HLS配置: {hls_config_path}")
    
    print("\n📦 部署流程:")
    print("   PyTorch (.pth)")
    print("      ↓")
    print("   ONNX (.onnx)")
    print("      ↓")
    print("   Vitis HLS (使用hls_config.json)")
    print("      ↓")
    print("   FPGA Bitstream")
    
    return exports


def main():
    parser = argparse.ArgumentParser(description='HW-NAS 完整端到端流程')
    parser.add_argument('--data-dir', default='data/NKSID', help='数据集路径')
    parser.add_argument('--output-dir', default='results/nas_output', help='输出目录')
    parser.add_argument('--num-candidates', type=int, default=10, help='搜索候选数')
    parser.add_argument('--proxy-epochs', type=int, default=3, help='代理训练轮数')
    parser.add_argument('--final-epochs', type=int, default=30, help='最终训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--device', default='cpu', help='设备 (cpu/cuda)')
    parser.add_argument('--fold', type=int, default=0, help='K-Fold折数')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_header("HW-NAS FPGA Sonar 完整流程")
    print(f"\n📁 数据集: {args.data_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🔧 设备: {args.device}")
    
    # ========================================
    # 准备数据
    # ========================================
    print("\n📊 加载数据集...")
    train_loader, val_loader, class_weights = create_nksid_dataloaders(
        data_dir=args.data_dir,
        image_size=224,
        batch_size=args.batch_size,
        fold=args.fold,
        use_kfold=True,
    )
    class_weights = class_weights.to(args.device)
    print(f"   训练集: {len(train_loader.dataset)} 样本")
    print(f"   验证集: {len(val_loader.dataset)} 样本")
    
    # ========================================
    # 定义硬件约束
    # ========================================
    constraints = SearchConstraints(
        max_latency_ms=50,
        max_dsp=200,
        max_bram=100,
        max_lut=50000,
    )
    
    hardware_spec = HardwareSpec(
        name="Zynq7020",
        clock_mhz=200,
        max_dsp=220,
        max_bram=140,
        max_lut=53200,
    )
    
    print(f"\n🎯 硬件目标: {hardware_spec.name}")
    print(f"   约束: Latency≤{constraints.max_latency_ms}ms, DSP≤{constraints.max_dsp}")
    
    # 创建搜索空间和估算器
    search_space = SearchSpace(SearchSpaceConfig(
        input_channels=1,
        image_size=224,
        num_classes=len(NKSID_CLASSES),
        hardware_constraints=constraints,
    ))
    
    cost_estimator = FPGACostEstimator(hardware_spec, constraints)
    
    # ========================================
    # Step 1: 架构搜索
    # ========================================
    best_arch, best_cost, candidates_log = search_best_architecture(
        search_space=search_space,
        cost_estimator=cost_estimator,
        constraints=constraints,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        num_candidates=args.num_candidates,
        proxy_epochs=args.proxy_epochs,
        device=args.device,
    )
    
    # 打印最优架构
    print_architecture(best_arch, "最优网络架构")
    print(f"\n⚡ 硬件指标:")
    print(f"   预估延迟: {best_cost.latency_ms:.2f} ms")
    print(f"   DSP使用: {best_cost.peak_dsp}")
    print(f"   BRAM使用: {best_cost.peak_bram}")
    print(f"   LUT使用: {best_cost.peak_lut}")
    
    # ========================================
    # Step 2: 训练最终模型
    # ========================================
    model, history = train_final_model(
        arch=best_arch,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        epochs=args.final_epochs,
        device=args.device,
    )
    
    # ========================================
    # Step 3: 详细评估
    # ========================================
    eval_results = evaluate_model(model, val_loader, args.device)
    
    # ========================================
    # Step 4: 导出部署版本
    # ========================================
    exports = export_for_deployment(model, best_arch, output_dir)
    
    # ========================================
    # 保存完整报告
    # ========================================
    report = {
        'timestamp': datetime.now().isoformat(),
        'hardware_target': hardware_spec.name,
        'constraints': {
            'max_latency_ms': constraints.max_latency_ms,
            'max_dsp': constraints.max_dsp,
            'max_bram': constraints.max_bram,
            'max_lut': constraints.max_lut,
        },
        'architecture': best_arch.to_dict(),
        'hardware_cost': {
            'latency_ms': best_cost.latency_ms,
            'dsp': best_cost.peak_dsp,
            'bram': best_cost.peak_bram,
            'lut': best_cost.peak_lut,
        },
        'training': {
            'final_epochs': args.final_epochs,
            'best_val_accuracy': history.get('best_val_acc', 0.0),
        },
        'evaluation': eval_results,
        'exports': {k: str(v) for k, v in exports.items()},
    }
    
    report_path = output_dir / 'report.json'
    with report_path.open('w') as f:
        json.dump(report, f, indent=2)
    
    # ========================================
    # 最终总结
    # ========================================
    print_header("🎉 完成!")
    print(f"""
📐 最优架构:
   - {len(best_arch.stages)} stages
   - 预估延迟: {best_cost.latency_ms:.2f} ms
   - DSP: {best_cost.peak_dsp}, BRAM: {best_cost.peak_bram}, LUT: {best_cost.peak_lut}

📊 性能:
   - 验证准确率: {eval_results['overall_accuracy']*100:.2f}%

📦 输出文件:
   - {exports['pytorch']}
   - {exports['onnx']}
   - {exports['architecture']}
   - {exports['hls_config']}
   - {report_path}
""")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())