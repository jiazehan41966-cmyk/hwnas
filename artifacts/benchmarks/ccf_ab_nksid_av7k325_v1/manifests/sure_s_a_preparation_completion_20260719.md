# SURE S-A 分阶段执行准备完成记录

## 当前结论

推荐方案 S-A 的执行基础已经完成，但尚未获得用户授权，正式 SURE 结果仍为 `0/15`，技术 smoke 也没有启动。当前不存在 `sure_formal_execution_authorization_20260719.json.txt` 活动授权。

## 已完成的安全能力

- `run_eval_protocol.py` 新增 `--max-new-units`。该字段是执行控制，不进入实验 fingerprint；已有兼容记录不计入“新单元”数量。
- 第一阶段可以使用完整的 5 folds × 3 seeds × 150 epochs 正式配置，只训练 fold 0、seed 42 一个新单元后干净退出。后续若代码、配置、环境和 source freeze 完全不变，可复用该记录并续跑，不需重训首单元。
- `scripts/run_sure_s_a_stage1_guarded.py` 只允许执行：MobileNetV2 1-epoch 技术 smoke，以及首个 150-epoch 正式单元。它明确禁止剩余 14 单元和所有 HLS、route、COM5、板级及功耗操作。
- 守护程序要求新的 source freeze、精确 SHA256、SURE 固定 commit、专用 CUDA 环境、正式开关和一次性活动授权；目录已存在、哈希不符、OOM、NaN/Inf、预测缺失或非零退出均 fail-closed。
- 首个正式单元结束后，守护程序写入中文结果卡并强制暂停；无论指标好坏都交由用户决定，不使用会掩盖负结果的自动精度淘汰线。
- 授权模板状态固定为 `NOT_AUTHORIZED_TEMPLATE`，不能触发训练。

## 关键哈希

- 统一评测入口：`e6968ffe2217fa29117f1ed5a62d1ff5c72197ba6b3486d6f1bf270da28e3817`。
- S-A 第一阶段守护程序：`d2b9de5b3bc0e0dca6ec00ab9b0f58c05c262be8b78a470c1c76dd8ba627df84`。
- 不可执行授权模板：`21cf2a20dcbfb3f9969959e0eb51e29f3b3a6e90eb23f7645d1f2dffeb81405e`。
- SURE 就绪性机器决策卡：`4894fa10a7298338e0f1e2c46fd4f82a9b788355f7c27795c2706438cab506cd`。
- 修正后的 formal readiness：`dab454f6abf2563774c6978e1d095470bc128ec7acc8b7ea4bd45c5e5ee4dfdc`。

## 验证

- SURE/readiness/provenance 定向测试：16 项通过。
- 完整项目测试：`483 passed, 1 warning, 28 subtests passed`。唯一 warning 是既有的零 warmup scheduler 测试警告，与 SURE 分阶段控制无关。
- `git diff --check` 没有内容错误，仅有既有 Windows CRLF 转换提示。
- 中文档案覆盖审计：110 份全部为中文或具备 SHA256 绑定的中文伴随档；乱码 0。

## 后续边界

只有用户明确回复批准 S-A 后，才允许把 SURE 正式开关改为 `true`、重新冻结当前源代码、生成一次性活动授权并启动技术 smoke。scratch-v2 授权不能复用。用户未授权前，不得创建活动授权，不得运行训练，也不得使用板卡。
