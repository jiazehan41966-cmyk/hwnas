# 声呐目标检测项目审计（第一轮）

第二轮状态：完整330张train的几何/对比度/FFT和split审计已完成，见`reports/sonar_analysis_full/SONAR_DATA_ANALYSIS_FULL.md`；真实batch4 forward/loss/backward和短validation已通过，新冻结协议见`reports/baseline_sonar_frozen/BASELINE_PROTOCOL.md`。本文件以下保留第一轮的历史审计边界，不将当时的pending误作第二轮当前状态。

日期：2026-09-14。范围：当前检测分支代码审计、配置快照、原始标注基础统计及小规模验证。未修改现有模型、训练器、标签或划分，未启动训练、搜索、量化或硬件测量。工作区已有大量未提交修改，本轮仅新增文件。以下结论来自当前本地代码与文件，不代表远程作业的实时状态。

## 1. 项目目录与关键入口

| 目录/文件 | 用途 |
| --- | --- |
| `src/hwnas_fpga/models/detection.py` | Backbone、FPN Neck、Detection Head、detector 及构建函数全部在同一文件 |
| `src/hwnas_fpga/models/builder.py` | 复用 StemBlock、build_block 与基本算子，基本算子也定义于此 |
| `src/hwnas_fpga/search_space/space.py` | `baseline_architecture()` 将已有配置实例化为固定架构；本轮没有执行搜索 |
| `src/hwnas_fpga/data/detection.py` | YOLO Dataset、Dataloader、增强、坐标变换、类别均衡采样 |
| `src/hwnas_fpga/training/detection_trainer.py` | 目标分配、loss、decode、NMS、训练和验证、checkpoint |
| `src/hwnas_fpga/training/detection_metrics.py` | 旧的 AP50/AP75、Precision、Recall、F1 |
| `src/hwnas_fpga/training/detection_coco_eval.py` | 独立的 COCO 风格 AP50:95（101 点插值），未接入旧训练选模逻辑 |
| `scripts/run_detection_experiment.py` | 旧 train→valid→test 入口，默认配置为旧三类导出，使用前必须明确 config |
| `scripts/detection_controlled_study.py` | 新受控 train/diagnose 入口；固定随机源、记录哈希，仅 train/valid |
| `scripts/evaluate_detection_checkpoint.py` | checkpoint 的 valid/test 后处理扫描，含测试集最优项汇报风险 |
| `scripts/visualize_detection_boxes.py`、`scripts/diagnose_detection_errors.py` | 加载 checkpoint、推理、解码、画框/误差诊断 |
| `scripts/audit_detection_evaluation.py` | 标注几何、坐标闭环、解码及评估链审计 |
| `configs/search/*detection*.yaml` | 旧基线与多组算法控制配置，不能混为一个实验 |
| `results/detection_fixed_normalizer_20260912/configs/baseline.yaml` | 本轮选择的已有 fixed-normalizer baseline 来源 |
| `configs/baseline_sonar.yaml` | 本轮新增独立配置快照；不代表训练性能已验证 |
| `data/`、`artifacts/`、`results/`、`results_archive/` | 原始/导出数据、清单、当前与历史结果；必须区分版本和任务 |
| `run_infer.py`、`run_export.py`、`src/hwnas_fpga/deploy/` | 主要服务分类分支的推理/ONNX 导出，不是现成的检测部署链 |
| `hardware/`、`hls_lut_builder/` | 已有 FPGA 工作；本轮只辨认边界 |
| `reference/` | 外部参考项目，不是当前 detector 的训练入口 |
| `tools/sonar_analysis/`、`reports/sonar_analysis/` | 本轮独立工具与真实样本统计输出 |

仓库主体仍包含分类/NAS/FPGA 历史工程；当前检测器不是官方 YOLOv8。数据导出名中的 yolov8 只表示标注格式。

## 2. 当前实际模型

Detector：`HWNASMultiScaleDetector`。Backbone：`HWNASFeatureBackbone`；Neck：`FixedFPNNeck`；Head：`FixedDetectionHead`。

固定架构实际构建结果已保存到 `reports/sonar_analysis_audit/baseline_snapshot.json`。与来源配置构建的架构及 `DetectionLossConfig` 逐项比较一致。参数量为 **916,103**，FP32 参数张量共 **3,664,412 bytes**，不含 BN buffers、序列化开销及 optimizer，不能称为 checkpoint 文件大小。

Stem：1→32，stride 2；无额外 post-stem 下采样。8 个 stage 均深度 1：

| stage（零起始） | channels | stride | 实际 block |
| --- | --- | --- | --- |
| 0 | 16 | 1 | conv k1 e1 |
| 1 | 24 | 2 | MBConv k3 e3 |
| 2 | 32 | 2 | MBConv k3 e3 |
| 3 | 32 | 1 | skip |
| 4 | 64 | 2 | MBConv k3 e3 |
| 5 | 96 | 1 | MBConv k3 e3 |
| 6 | 160 | 2 | MBConv k3 e3 |
| 7 | 320 | 1 | MBConv k3 e3 |

输出 stage [3,5,7]，stride [8,16,32]。FPN lateral 为 1×1 Conv+BN+ReLU，最近邻上采样后相加，再用 3×3 Conv+BN+ReLU；通道均为 96。每尺度 head 为 3×3 Conv+BN+ReLU 加 1×1 预测层，每位置 3 anchors、每 anchor 7 个量（4 框参数、objectness、2 类输出）。640 输入时按结构推导输出为 [B,21,80,80]、[B,21,40,40]、[B,21,20,20]，本轮未运行模型 forward 验证。

配置中仍存在 sonar 算子候选，但实例化结果未选中它们；不能据候选名宣称模型采用了声呐专用算子。

## 3. 数据与数据流

当前配置数据根目录：`data/external_redownload_audit_20260909/roboflow_cylider2/extracted/v6_yolov8`。

本轮重新枚举：train **330**、valid **99**、test **49**，合计 **478** 张。`data.yaml` 的两类名称为 `cylider`、`manta`；保留原拼写。没有与 Figshare 或 NKSID 分类数据合并。

Annotation：`split/images/<stem>.*` 对应 `split/labels/<stem>.txt`，每行 `class cx cy w h`，后四项按原图宽高归一化，轴对齐框。加载器以目录 split 为准，val 别名映射至 valid；不执行随机划分，不以 data.yaml 中 train/val 字符串重建划分。

训练流：图像→PIL 灰度 L→双线性直接缩放 640×640（无 letterbox）→float32 /255→YOLO 框转换至输入空间→训练增强→张量与 targets→backbone→FPN→head→anchor 分配→loss→backward→optimizer。

验证流：无增强的 valid→forward→objectness sigmoid × class softmax→阈值/top-k→按类 NMS→检测指标。框映射回原图采用 x/y 独立逆缩放。

原图尺寸在本轮 32 张样本中全部为 1280×1280；全量尺寸一致性仍应由完整统计确认。样本每图恰好一个 GT，不能据此认定图像中不存在未标注目标或数据集覆盖了无目标场景。

## 4. Baseline 配置快照与复现边界

选择当前受控研究的 baseline，而非按测试表现挑选 `noobj02` 等消融变体。来源 YAML 使用 train-only anchors 和 `fixed_weights` objectness 归一化。已有 `legacy_checkpoints/baseline.pt`、`noobj02.pt` 属历史结果，不自动对应当前 loss。

| 项目 | 固定值/实际行为 |
| --- | --- |
| input | 单通道，640×640 |
| epochs | 80：与已有受控启动命令相同；来源 YAML 的 150 被 CLI 覆盖，快照明确此差异 |
| batch size | 4 |
| optimizer | AdamW，lr 0.001，weight_decay 0.0001 |
| scheduler | 无，学习率恒定；trainer 不调用 scheduler |
| AMP | 实际未实现 autocast/GradScaler；来源 YAML 的 true 不生效，快照写 false |
| augmentation | 水平翻转概率 .5；brightness .1；contrast .1；noise std .01，之后 clip [0,1] |
| sampling | train 类别均衡 WeightedRandomSampler，有放回；按图像含类计权 |
| random seed | 42；受控入口固定 Python/NumPy/Torch/CUDA/worker/sampler，确定性 warn_only，不保证跨平台逐位相同 |
| split | 使用现有 train/valid/test 目录，不重划；受控训练入口不读 test |
| loss | Smooth L1 框回归 + IoU loss，框权重5；objectness BCE+focal，gamma2/alpha.75，负样本权重.1，fixed_weights；类别 CE 权重2、类权重[1,1] |
| assignment | 正 anchor IoU .65；ignore .35；neighbor radius 1；anchors 详见快照 |
| evaluation | score .01、NMS IoU .5、pre-NMS top-k 500、max detections50；运行点在 valid 选择 |
| best checkpoint | 按 valid 的旧 map 选最佳；训练结束恢复 best state |

快照删除未使用的 search/hardware/deployment 执行段，保留现有构建器所需 `search_space` 和 constraints；没有新增搜索实现或改动架构。实际训练仍应显式传 `--epochs 80 --seed 42`，受控 CLI 有自己的默认值，配置文件不是唯一生效来源。

本轮只验证构建一致性和参数计数，没有训练、模型 forward/loss/backward/validation sanity，也没有 checkpoint 性能复验。下一轮先完成这些最小检查。`reports/sonar_analysis_audit/baseline_snapshot.json` 明确保留未测字段为 null。

## 5. 测试、推理、指标与部署

- 旧入口 `run_detection_experiment.py` 会构造所有 split 并自动测试；受控入口只 train/valid，更适合作为后续 baseline 训练入口。
- `detection_metrics.py` 及 trainer 中 `map` 默认只平均 IoU .50/.75，不是 mAP50:95；AP 仅平均存在 GT 的类别。`detection_coco_eval.py` 另有十个 IoU 阈值和101点插值实现，性能汇报应独立注明后处理、maxDets 和计算空间，不能只改字段名称。
- Precision/Recall/F1 是选定 score threshold、IoU=.5 下的操作点结果，不能混同 AP。
- checkpoint 保存 epoch、model_state_dict、optimizer_state_dict、loss_config、history；没有内嵌完整架构/数据划分，必须伴随 config、manifest 和代码哈希。
- `scripts/visualize_detection_boxes.py` 是现成检测推理/画框入口；没有发现当前检测器专用通用单图 CLI。根目录 `run_infer.py` 是分类入口。
- `src/hwnas_fpga/deploy/export.py:get_model_info` 用 numel 计 Params，Conv2d/Linear forward hooks 估算 MACs（乘加次数），FP32 参数大小估计；并非全面 FLOPs profiler。若今后以 FLOPs≈2×MACs 汇报，必须标明仅卷积/线性层，不含 BN、激活、插值、decode/NMS。
- 已有分类用 ONNX 导出路径。当前检测 checkpoint 格式和字典/多尺度输出需要独立验证，不能视为已支持。主线检测入口未发现接通的 TensorRT/TorchScript 部署流程。
- 本轮未发现可绑定当前两类 baseline checkpoint 的真实 detector latency/FPS 测量。已有 FPGA/HLS 补充表及分类部署结果不是本 detector 的全网延迟、FPS 或功耗。

## 6. 第一轮基础分析

独立入口 `tools/sonar_analysis/analyze_dataset.py`，复用 `YoloDetectionDataset` 文件发现与原像素转换函数。严格解析当前 YOLO 格式：缺失标签报错；空文件是明确负样本；错误列数、非有限数、非法类别、零面积和越界报错，不修改标签。支持路径、划分、样本数、seed、近方形阈值参数及 --help。

已分析 seed42 随机选取的32张 train 图、32个框：cylider24、manta8。归一化面积中位数0.00069374（**0.0694%**）；640输入宽/高中位数15.125/18.125 px；所有框在输入尺度上面积<32²。这里使用仓库现有 COCO 风格评价分箱，是统计约定，不是物理尺寸定义。原图面积、宽高、归一化面积同时保存。

宽高比中位数0.858；按横向>1.25、纵向<0.8、近方形[0.8,1.25]，计数3/15/14。只有 bbox shape tendency（框形状倾向），没有真实方向信息。

产物：逐框/逐图CSV、summary.json、7组PNG/SVG图与各自CSV、首版SONAR_DATA_ANALYSIS.md。分析不使用增强或加权采样；没有执行强度/对比度、FFT、声影标注、代表样本画框或全数据图像分析。

实际验证：`python -m pytest tools/sonar_analysis/test_statistics.py -q` 得到13 passed；CLI --help通过；32张真实训练图读取通过；逐框/逐图CSV行数及三种histogram总数与summary一致；7组图都有CSV和SVG；抽取一张真实图通过现有Dataset得到[1,640,640]有限张量，框宽高与分析工具的输入尺度值一致。数值、版本和验证边界保存在 `reports/sonar_analysis_audit/verification.json`。报告生成器修正了Markdown表格空行，原分析运行源码哈希保留在summary，新报告生成器哈希单独记录于verification。

## 7. 工程风险与待办

1. **多个 baseline 的语义不同**：旧三类导出、两类官方导出、anchors变体、legacy/fixed归一化不能混报。当前源快照与训练代码哈希已记录，但尚缺当前完整性能证据。
2. **配置不全等于运行协议**：旧入口 seed 不完整，AMP 配置不生效；受控命令80与 YAML150不同。旧入口两个 max-samples 参数实际取最大值同时施加给各 split。
3. **测试集选择泄漏风险**：旧 checkpoint 扫描脚本有 `reported_test_map=max(...test map...)`；不得用它选择模型/阈值。后续设计建议只参考 train；test 独立保留。
4. **标注宽松加载风险**：训练 loader 默认缺失标签视空，部分格式异常会被跳过/截取，框会裁边。本轮独立工具严格报错，但未更改训练行为。完整重复图像、近重复帧/同航次跨 split 审计尚未完成。
5. **样本代表性与尺度解释**：32张不足以证明全量比例；输入缩放使 bbox 更小。轴对齐框不支持真实方向推断，未标注声影不应补造标签。
6. **环境与证据漂移**：当前 `.venv_cuda` 已有 Pillow12.2，超出 pyproject 的<12声明；本轮未更改它。新增 matplotlib3.11.2 与 pytest8.4.2 以运行绘图和测试，未改训练依赖锁定。已有远程 RUNNING 状态文件日期为9月13日，本轮未联网确认，不能当作当前运行状态。

## 8. 后续可能修改的文件（仅定位，不授权实施）

结构候选若获后续授权：`models/detection.py`（特征输出、FPN融合）、`search_space/space.py`（变量约束）、`models/builder.py`及实际算子定义、检测配置；stride/anchors/decode的一致性同时涉及`training/detection_trainer.py`。本轮均未修改。

基于当前抽样，优先候选只有高分辨率输出/stride与P2选项，仍需完整train统计和固定baseline误差复验；多尺度融合、方向算子、大核/空洞算子尚不定案。任何新增变量的收益必须在之后的控制实验中验证。
