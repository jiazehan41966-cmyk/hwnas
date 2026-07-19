# provenance 修复、重新冻结与 scratch-v2 启动记录

## 当前结论

- provenance 固定路径覆盖缺陷已修复。
- 新增专项测试与全量回归均通过。
- 新源码冻结 `g1_20260718_v3` 已核验通过。
- scratch-v2 已于 2026-07-18 08:52（Asia/Shanghai）受控启动。
- 当前阶段只使用本机 GPU；未使用 AV7K325、COM5、Vivado/Vitis、HLS、route 或功率仪器。

## 修复内容

正式入口不再写入可覆盖的 `code_patch.diff`，而是写入：

`provenance/code_patch_<完整 SHA256>.diff`

同名文件存在时必须重新核验内容；发现损坏即 fail-closed，不允许覆盖。G1 总账新增对 `run_manifest.code_provenance.tracked_patch.path/sha256` 的独立核验，并把正式 scratch 路径切换到新的 `g1_clean_20260718/g1_mobilenet_v2_scratch_v2`。

## 测试与冻结

- provenance、G1 总账、source freeze 相关回归：`23 passed`。
- 全量回归：`478 passed, 1 warning, 28 subtests passed`。
- 唯一 warning：测试中的既有学习率调度顺序警告，与本修复无关。
- 冻结文件数：`557`。
- 冻结实时核验：`557/557 PASS`。
- 冻结 manifest SHA256：`cfbc7ec9373e762c39385d733e07682a39ef843f87c2a25100c3fb7bfb824f32`。
- 冻结归档 SHA256：`990d1dd08b1e966c4e85ff76661b4ceffb0986200a25859c8540cafe359a6ed3`。

## 独立审计复核

- 旧 scratch 再审计仍为 `FAIL`，唯一错误为 tracked patch SHA 不匹配；没有修补旧证据。
- pretrained 首次复审命令误用了方法标识 `pretrained_mobilenet_v2`，因此 15 个逐样本 method 字段检查失败。该失败文件原样保留。
- 核对历史记录后，使用正确标识 `imagenet_pretrained_mobilenet_v2` 重新输出到独立文件，状态为 `PASS`、errors 为空。该事件属于审计命令参数错误，不是 pretrained 证据损坏。

## scratch-v2 启动绑定

- run fingerprint：`16592ad47ec990244cef3cac2d100bd0929a5cac7608904b3d63f41a97ee33aa`。
- code-state SHA256：`c9309cb94d031bfbdc14c57904649fb00e040391a12049741d153d48e9645334`。
- 新 patch SHA256：`ccb0feef0dafc34a2b4fb0e2f751b698ad4a1acd9c3696fbafb6b88df5dc7280`。
- 新 patch 文件名与 SHA256 一致，文件实测哈希与 manifest 一致。
- 计划单元：5 folds × 3 seeds，共 `15`。

## 中断策略

每个单元落盘后立即核验 checkpoint、逐样本预测、freeze、patch 与旧 scratch 保全哈希。出现任一非有限值、哈希错误、非零退出、macro_f1 < 0.80，或与旧 scratch 同 fold/seed 的 macro_f1 绝对差 > 0.05 时，自动中断并生成中文事件卡。

第一次“只预检”因 Windows PowerShell 5.1 未显式指定 UTF-8，导致中文批准 JSON 解析失败；监控器按设计在训练前中断并保留中文事件卡。修复为所有 JSON 显式 `-Encoding UTF8` 后，第二次只预检通过且没有创建 run 目录。正式启动后又发现虚拟环境存在启动器与实际解释器两个 Python PID，因此增加独立进程树看门狗；若主监控器中断或异常退出，看门狗会终止完整训练进程树。

## 证据边界

截至本记录写入时，第一个 fold/seed 单元尚未完成，因此没有新的 macro_f1 可报告。G1 仍为 `PENDING 30/45`；只有 scratch-v2 15/15、独立审计和测量优先 G1 总账同时通过后才能解除冲突冻结。SURE、鲁棒性、开放集、NAS 新搜索、HLS、板级与功耗仍未授权。

