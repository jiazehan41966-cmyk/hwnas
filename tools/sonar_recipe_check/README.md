# D14 训练配方快速检查

固定 C01 的 YOLOv9-T＋quality_aware 推理结构、损失、数据划分、初始化和评估实现。
R0 复用已有 seed 42–44，复用前强制核对旧源闭包、manifest、15 份结果身份文件、环境和完整初始模型状态。
R1 为 EMA；R2 加 3 轮线性预热与余弦下降；R3 再加 mosaic 与作者几何尺度增强的受控增量。
R3 保留冻结灰度光度/翻转增强，保持 float32 拼图与固定 640 输入；无分割轮廓，关闭 Copy-Paste。

新正式运行固定 9 次（seed 42–44×R1–R3），80 轮；失败或超时停止，不自动重试、不增加候选。
第 76–80 轮逐轮保存完整 99 图预测、GT 与 SHA256 收据。
主指标为各轮 mAP50–95 的算术均值，后对三个 seed 求均值；不是预测融合，也不是 best。

有效门槛：相对 R0 平均提升至少 0.01，且 seed/图像分层配对 bootstrap 95% 区间下限严格大于 0。
2000 次抽样，随机种子 20261013，三个比较共用抽样；89 图子集只作辅助。
有效候选中，选择距最好有效候选不超过 0.005 的最简单配方；没有有效候选则保留 R0。
这是预登记的验证集开发选择，三个区间未做多重比较校正。

原始 11.66% 是历史单 seed 的 best，第 76–80 轮均值是 7.280218%；原实验也没有完整作者配方。
本任务不把差异归因于训练配方，不新增硬件资源/板端/功耗结论，不启动 72/192 搜索。

## 当前环境与执行

沿用 C01 用户自有 SSH 服务器、RTX 5090 GPU1、同一 Torch 2.7.1+cu128 环境及冻结 overlay，不安装或重建依赖。
本地使用 `.venv_cuda/Scripts/python.exe -B -m tools.sonar_recipe_check.remote <mode>`。
顺序为 `probe`、`stage`、`preflight`、独立审阅、`upload-review`、`launch`；已有调用标记禁止重复启动。
`status` 只读状态；`collect` 下载小型证据，模型 checkpoint 和逐轮预测留在远端。

远端的实际预检命令由 `supervisor.py preflight` 执行：

```text
python tools/sonar_recipe_check/train.py prepare --output RESULT
python -m unittest tests.test_sonar_recipe_check_training tests.test_sonar_recipe_check_analysis -v
python tools/sonar_recipe_check/train.py smoke --output RESULT --device cuda
python tools/sonar_recipe_check/train.py verify --output RESULT
```

独立审查必须针对真实训练 batch、验证 batch、环境与本次 execution manifest 后才能 PASS。
`supervisor.py campaign` 一次性顺序运行全部 9 次，再调用 `finalize.py` 重验归档、计算分层区间并写 D14。
它不是周期任务，不依赖新会话，不发送消息；完成/失败后退出。

## 可追溯交付

- `configs/sonar_recipe_check_20261013.yaml`：冻结机器配置。
- `results/sonar_recipe_check_20261013_01/preregistration.json`：候选结果产生前的预登记；`registration_receipt.json` 绑定配置与审计。
- `reference_audit/REFERENCE_AUDIT.md`：历史口径纠正与 R0 复用证据。
- 远端结果 `finalization/recipe_analysis.json`、`D14.md`、`D14_training_recipe.md`、`D14_selected_recipe.json`：全部完成后的真实选择。
- `finalization/completion_receipt.json`：完成核验与预登记绑定；`compatibility_metrics.csv` 保留逐 seed、终轮和 best 指标。

最终 D14 为独立版本化补充，不覆盖既有 v4 工作簿；未完成前不得写为已选择配方。
