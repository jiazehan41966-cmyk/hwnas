# 声呐标注基础分析（第一轮）

第二轮入口：`complete_analysis.py`补充train对比度/FFT及全split完整性审计；`baseline.py`冻结并运行只用train/valid的基线。协议与真实验证结果见`reports/baseline_sonar_frozen/BASELINE_PROTOCOL.md`。

完整分析的已运行命令如下；复现时使用新输出目录，保留原32张样本结果：

```powershell
.venv_cuda/Scripts/python.exe tools/sonar_analysis/analyze_dataset.py --config configs/baseline_sonar.yaml --output reports/sonar_analysis_full --splits train --seed 42
.venv_cuda/Scripts/python.exe tools/sonar_analysis/complete_analysis.py --output reports/sonar_analysis_full --alpha 2 --epsilon 0.000001 --frequency-edges 0.3333333333333333 0.6666666666666666 --dhash-threshold 4
.venv_cuda/Scripts/python.exe tools/sonar_analysis/review_integrity.py --output reports/sonar_analysis_full --correlation-threshold 0.99
.venv_cuda/Scripts/python.exe tools/sonar_analysis/write_stage_report.py --analysis reports/sonar_analysis_full --evidence reports/SEARCH_SPACE_EVIDENCE.md
.venv_cuda/Scripts/python.exe -m pytest tools/sonar_analysis/test_statistics.py tools/sonar_analysis/test_full_stage.py -q
```

第二轮新增metric对照测试依赖`pycocotools==2.0.11`，仅作独立正确性参照，训练入口无需该包。所有test图像访问仅限完整性审计。以下第一轮说明中的“未实现contrast/FFT”仅指基础入口，不适用于第二轮补充入口。

在仓库根目录使用项目虚拟环境。依赖为现有 torch/Pillow/NumPy/PyYAML，绘图使用 matplotlib（pyproject 的 analysis optional dependency）。脚本默认只读 train；路径不硬编码，config 内相对数据路径按仓库根目录解析，CLI config/output 按当前工作目录解析。

```powershell
.venv_cuda/Scripts/python.exe tools/sonar_analysis/analyze_dataset.py --help
.venv_cuda/Scripts/python.exe tools/sonar_analysis/analyze_dataset.py --config configs/baseline_sonar.yaml --output reports/sonar_analysis_full_train --splits train --seed 42
.venv_cuda/Scripts/python.exe -m pytest tools/sonar_analysis/test_statistics.py -q
```

小样本复现：添加 `--max-images-per-split 32` 并使用新的 output 目录。使用 `--data-dir` 覆盖原始数据根目录。输出目录必须不存在或为空，避免覆盖旧证据。

需要描述全部划分时使用 `--splits train valid test`，单独指定输出目录。工具保留按 split 统计，报告的搜索建议仅由 train 形成；跨 split 汇总只能用于数据描述。不要由 test 形状分布调整结构。

逐框CSV保存原图坐标/宽高/面积、归一化面积、输入尺度宽高/面积、类别、形状和尺度分箱；逐图CSV保存目标数量、图像尺寸、源图片和标签SHA256。每张图附对应CSV，输出PNG与SVG。summary保存配置与工具哈希、命令和运行版本。

尺度分箱沿用现有检测COCO风格评价的输入空间阈值：area<32²、[32²,96²)、≥96²；并非数据集定义的物理尺寸。宽高比使用原图，避免非方形图缩放到方形后被误判。形状分箱默认横向>1.25、纵向<0.8、其余近方形，可用`--square-factor`调整，只描述框形状。

本轮不支持 mask/rotated bbox；非5列标签明确报错。缺失文件与空标签区分处理，全空选样报错。未实现contrast、FFT、声影或模型性能验证；“完整命令”仅表示对完整训练集执行目前已实现的基础统计。
