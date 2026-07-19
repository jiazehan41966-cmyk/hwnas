# provenance 不可变保存修复合同（待用户批准）

## 状态

- 状态：`PROPOSED_NOT_APPLIED`。
- 正式入口 `run_eval_protocol.py` 尚未修改。
- 本合同不批准 scratch-v2，不创建新源码冻结，也不启动训练。

## 已确认缺陷

当前 `git_code_provenance(run_dir)` 总是写入固定路径 `code_patch.diff`。该函数在旧 `run_manifest.json` 的兼容性检查之前执行，因此只要再次进入旧 run 目录，即使随后因 fingerprint 不兼容而退出，也会先改写原 patch 文件。manifest 中保存的旧 SHA 随即失去对应文件。

## 修复目标

1. patch 文件采用内容寻址文件名：`provenance/code_patch_<完整 SHA256>.diff`。
2. 文件只允许首次创建；同名文件内容不一致时 fail-closed，不得覆盖。
3. 相同 patch 再次捕获时复用同一路径，且不改变文件内容。
4. 不同 patch 捕获时生成不同路径，旧证据保持原字节和原 SHA。
5. 测量优先 G1 总账必须独立验证 `run_manifest.code_provenance.tracked_patch.path/sha256`，不得只相信 summary 自报的 `claimable=true`。
6. 所有旧 run 保持只读；不得把丢失的 scratch patch 通过猜测或当前 diff 回填。

## 拟新增测试

| 测试 | 通过条件 |
|---|---|
| 两个不同 diff 连续捕获 | 路径不同，第一次文件字节和 SHA 不变 |
| 相同 diff 重复捕获 | 路径相同，内容相同，不发生重写 |
| 内容寻址文件被破坏 | 捕获函数抛出异常并停止 |
| 旧 manifest 与当前代码不兼容 | 失败前不得改写 manifest 已绑定的任何文件 |
| G1 总账读取 scratch | patch SHA 不匹配时 G1 不得为 PASS |
| G1 总账读取完整新 run | 15 单元、patch、checkpoint、prediction 与冻结源都通过后才允许 PASS |

## 批准后实施顺序

1. 应用 `provenance_immutable_repair_20260718.patch.txt` 中的设计，不直接修改旧结果。
2. 运行定向测试和完整回归。
3. 重新生成并审计新的源码冻结；旧 `g1_20260715_v2` 保持原样。
4. 创建一次性、SHA 绑定的 scratch-v2 执行批准文件。
5. 使用新 run 名称完成 5 folds × seeds 42–44。
6. 独立核验逐样本预测、checkpoint、patch、split、data、config 和 code-state SHA。
7. 只有专项独立审计与测量优先总账一致为 PASS，才解除 `G1_CONFLICT_HOLD`。

## 停止条件

- 任一 patch、checkpoint、prediction 或源码冻结 SHA 不匹配，立即中止。
- 检测到旧结果文件被改写，立即中止并保存 incident card。
- 新 scratch 与冻结协议、归一化、fold/seed 或入口不一致，立即中止。
- 未取得用户对训练的明确批准，不得从代码修复自动进入 scratch-v2。

