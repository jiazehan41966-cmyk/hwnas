# 参考代码库分析报告

## 目录结构
```
reference/
├── FBNet/                    # Facebook Hardware-Aware NAS
│   └── mobile_cv/lut/        # LUT延迟查找表
├── HW-NAS-Bench/             # HW-NAS Benchmark
│   └── hw_nas_bench_api/     # 硬件评估API
└── TinyTNAS/                 # 约束驱动 NAS
```

---

## 1. FBNet 分析

### 关键文件
- `mobile_cv/lut/lib/lut_schema.py` - LUT 数据结构和抽象接口
- `mobile_cv/lut/lib/lut_ops.py` - 算子定义（Conv2d, Linear, MultiheadAttention等）

### 核心设计

#### 1.1 LUT 查找表架构

```python
# 核心数据结构
class OpInfo(object):
    """算子信息：算子类型 + 输入形状"""
    def __init__(self, op: OpBase, input_shapes: typing.List[TensorShape])

class LutItem(object):
    """LUT表项：OpInfo + 延迟值"""
    def __init__(self, op, input_shapes, latency)

class LutTable(object):
    """LUT表：LutItem集合，支持pickle序列化"""
    def Load(cls, file_name)  # 从文件加载
    def save(self, file_name) # 保存到文件

class LutQuery(LUTBase):
    """LUT查询器：支持镜像bias处理、重复项处理"""
    def query_op(self, op, input_shapes) -> float
```

#### 1.2 算子抽象

```python
class OpBase(abc.ABC):
    """所有算子的基类"""
    @abc.abstractmethod
    def get_nparams(self):           # 参数量
    @abc.abstractmethod
    def get_flops(self, input_shape): # FLOPs

class Conv2d(OpProperty):
    """标准卷积算子"""
    - 支持 depthwise (groups)
    - 支持 bias 镜像
    - 自动计算输出形状、FLOPs

class OpProperty(OpBase):
    """基于dict的属性存储，支持动态参数"""
    - info: dict 存储所有参数
    - __hash__ 基于 frozenset(info.items())
```

#### 1.3 与您项目的对比

| FBNet | 您的项目 | 差异 |
|---|---|---|
| pickle序列化LUT | 无LUT实现 | ⚠️ 缺失 |
| OpBase抽象算子 | 直接硬编码计算 | ⚠️ 需重构 |
| TensorShape NCHW格式 | 直接使用int | ⚠️ 建议统一 |
| 镜像bias优化 | 无 | ⚠️ 可选 |

---

## 2. HW-NAS-Bench 分析

### 关键文件
- `hw_nas_bench_api.py` - 统一API接口

### 核心设计

#### 2.1 API 设计

```python
class HWNASBenchAPI():
    def __init__(self, file_path_or_dict, search_space="nasbench201"):
        # 加载pickle文件（HW-NAS-Bench-v1_0.pickle 约10MB）
        self.HW_metrics = pickle.load(f)

    def query_by_index(self, arch_index, dataname):
        """查询架构索引对应的硬件指标"""
        # 支持的指标：
        # - edgegpu_latency/energy
        # - raspi4_latency
        # - edgetpu_latency
        # - pixel3_latency
        # - eyeriss_latency/energy/arithmetic_intensity
        # - fpga_latency/energy  ⭐ 直接支持FPGA！

    def get_net_config(self, arch_index, dataname):
        """获取架构配置"""
        # FBNet: 返回 op_idx_list + arch_str
        # NAS-Bench-201: 返回 config

    def get_op_lookup_tables(self):
        """获取算子级别的LUT表（仅FBNet）"""
```

#### 2.2 FBNet LUT 查询逻辑

```python
# 关键发现：FBNet使用字符串键来索引LUT
metric += OP_metrics_dict["ConvBlock_H{}_W{}_Cin{}_Cout{}_exp{}_kernel{}_stride{}_group{}".format(
    H, W, Cin, Cout, exp, kernel, stride, group
)]
metric += OP_metrics_dict["Skip_H{}_W{}_Cin{}_Cout{}_stride{}".format(
    H, W, Cin, Cout, stride
)]
```

#### 2.3 架构定义（FBNet空间）

```python
# FBNet固定架构结构：
stem_channel = 16
num_layer_list = [1, 4, 4, 4, 4, 4, 1]     # 7个stage
num_channel_list = [16, 24, 32, 64, 112, 184, 352]
stride_list = [1, 2, 2, 2, 1, 2, 1]

# 22个可搜索位置（arch_index长度=22）
ops_str_lookup_table = [
    "k3_e1", "k3_e1_g2", "k3_e3", "k3_e6",  # 3x3卷积，expansion=1/3/6，groups=1/2
    "k5_e1", "k5_e1_g2", "k5_e3", "k5_e6",  # 5x5卷积
    "skip"                                    # 跳跃连接
]
```

#### 2.4 与您项目的对比

| HW-NAS-Bench | 您的项目 | 差异 |
|---|---|---|
| 直接支持FPGA延迟/能耗 | 分析模型估计 | ⚠️ 精度差异 |
| Pickle加载10MB数据 | 无预存数据 | ⚠️ 缺失基准 |
| 字符串键索引LUT | - | ⚠️ 设计不同 |
| 22个搜索位置 | 可变深度/通道 | ✅ 您的更灵活 |

---

## 3. TinyTNAS 分析

### 关键文件
- `TinyTNAS.py` - 核心搜索算法
- `search.py` - 使用示例

### 核心设计

#### 3.1 搜索算法

```python
class TinyTNAS:
    def __init__(self, train_ds, val_ds, input_shape, num_class,
                 learning_rate, constraints_specs):
        # 约束规范：
        constraints_specs = {
            "ram": 20800,      # RAM (bytes)
            "flash": 258400000, # Flash (bytes)
            "macc": 2565454545 # MAC operations
        }

    def search(self, epochs=3, search_time_minute=5):
        """带时间约束的搜索"""
        # 核心逻辑：
        # 1. 初始化 k=4, c=3
        # 2. 尝试不同深度k和宽度c
        # 3. 检查约束可行性
        # 4. 如果可行，训练并记录精度
        # 5. 超时或无法改进时返回

    def ExploreDepth(self, k, current_c, current_acc, constraints_specs):
        """探索深度变化"""
        # 遍历所有可能的c值
        # 检查约束
        # 返回最佳配置
```

#### 3.2 约束检查

```python
def CheckFeasible(constraints_specs, current_specs):
    """
    constraints_specs: {"ram": 20KB, "flash": 64KB, "macc": 60K}
    current_specs: {"ram": xxx, "flash": xxx, "macc": xxx}
    """
    # 返回是否满足约束
```

#### 3.3 架构编码

```python
# TinyTNAS使用简化的(k, c)编码：
# k: 深度（层数）
# c: 通道数（宽度）
```

#### 3.4 与您项目的对比

| TinyTNAS | 您的项目 | 差异 |
|---|---|---|
| (k, c)简化编码 | 完整stage/block编码 | ✅ 您的更完整 |
| 时间约束搜索 | 固定epoch搜索 | ⚠️ 可借鉴 |
| RAM/Flash/MACC约束 | DSP/BRAM/LUT/延迟 | ✅ 概念相似 |
| 探索深度后探索宽度 | 同时搜索多维度 | ✅ 您的策略更优 |

---

## 4. 关键发现与建议

### 4.1 应该借鉴的设计

#### 优先级P0（必须）

1. **FBNet LUT架构**
   - 实现 `LutTable` 和 `LutQuery` 类
   - 支持 pickle 序列化
   - 添加算子级别的延迟查询

2. **HW-NAS-Bench FPGA数据**
   - 利用其 `fpga_latency/energy` 数据校准估计器
   - 字符串键索引方式易于扩展

3. **TinyTNAS 约束处理**
   - 硬约束检查函数
   - 可行/不可行解的分离

#### 优先级P1（建议）

1. **FBNet算子抽象**
   - `OpBase` 基类设计
   - `OpProperty` 动态属性存储
   - `get_nparams()`, `get_flops()` 统一接口

2. **HW-NAS-Bench API设计**
   - 统一的查询接口
   - 支持多个硬件平台
   - 平均硬件指标计算

#### 优先级P2（可选）

1. **FBNet 镜像bias优化**
   - 减少LUT表大小

2. **TinyTNAS 时间约束**
   - `search_time_minute` 参数

### 4.2 您项目的优势

| 方面 | 优势 |
|---|---|
| 搜索空间 | stage-based + 可变深度/通道，比FBNet更灵活 |
| 硬件估计 | 完整的DSP/BRAM/LUT估计，TinyTNAS只有RAM/Flash |
| 接口设计 | 清晰的dataclass定义（interfaces.py） |
| 可扩展性 | 模块化设计，易于添加新的估计器 |

---

## 5. 具体实现建议

### 5.1 实现LUT模块

在 `src/hwnas_fpga/hardware/` 下新增：

```python
# lookup_table.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pickle

@dataclass
class OpSpec:
    """算子规格"""
    op: str
    kernel_size: int
    in_channels: int
    out_channels: int
    stride: int
    ...

@dataclass
class LutEntry:
    """LUT条目"""
    op_spec: OpSpec
    latency_ms: float
    dsp: int
    bram: int
    lut: int

class LutTable:
    """查找表"""
    def __init__(self, entries: List[LutEntry]):
        self.entries = entries
        self._index = {entry.op_spec: entry for entry in entries}

    def query(self, op_spec: OpSpec) -> Optional[LutEntry]:
        return self._index.get(op_spec)

    @classmethod
    def load(cls, path: str) -> "LutTable":
        with open(path, "rb") as f:
            return pickle.load(f)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
```

### 5.2 扩展估计器

```python
# 在 cost.py 中添加 LUT 估计器
class LutFPGACostEstimator(FPGACostEstimator):
    """基于LUT的FPGA代价估计器"""

    def __init__(self, lut_table: LutTable, **kwargs):
        super().__init__(**kwargs)
        self.lut_table = lut_table

    def _estimate_block(self, block: ResolvedBlockSpec) -> LayerCost:
        # 先尝试LUT查询
        op_spec = OpSpec(...)
        entry = self.lut_table.query(op_spec)
        if entry:
            return self._lut_entry_to_layer_cost(entry)
        # 回退到分析模型
        return super()._estimate_block(block)
```

### 5.3 添加约束搜索器

```python
# src/hwnas_fpga/search/constrained.py
class ConstrainedSearcher:
    """约束驱动的搜索器（借鉴TinyTNAS）"""

    def __init__(self, search_space: SearchSpace,
                 cost_estimator: FPGACostEstimator,
                 constraints: SearchConstraints):
        self.search_space = search_space
        self.estimator = cost_estimator
        self.constraints = constraints
        self.feasible_solutions = []
        self.infeasible_solutions = []

    def search(self, search_time_minute: int = 60) -> SearchCandidate:
        """带时间约束的搜索"""
        # 类似TinyTNAS的逻辑
        ...
```

---

## 6. 下一步行动

| 优先级 | 任务 | 预估工作量 |
|---|---|---|
| P0 | 实现LUT模块 | 2-4小时 |
| P0 | 扩展cost.py支持LUT查询 | 1-2小时 |
| P0 | 从HW-NAS-Bench提取FPGA数据 | 1小时 |
| P1 | 实现约束搜索器 | 3-5小时 |
| P1 | 实现Pareto前沿选择器 | 2-3小时 |
| P2 | 添加时间约束支持 | 1小时 |

---

## 7. 参考资料

- FBNet LUT设计: `reference/FBNet/mobile_cv/lut/lib/lut_schema.py`
- HW-NAS-Bench API: `reference/HW-NAS-Bench/hw_nas_bench_api/hw_nas_bench_api.py`
- TinyTNAS约束搜索: `reference/TinyTNAS/TinyTNAS.py`
