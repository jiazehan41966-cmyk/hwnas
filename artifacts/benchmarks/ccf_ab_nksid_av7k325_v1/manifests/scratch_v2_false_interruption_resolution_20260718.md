# scratch-v2 成功后误报中断的处置记录

## 结论

`scratch_v2_interruption_20260718_170145.md` 是守护脚本在训练已经成功完成 15/15 单元后产生的误报，不是训练失败，也没有造成结果缺失。误报原件保留，SHA256 为 `5d1413442c322ba77d84a469a282469ac03c738af5654d75d935807c065ed2ba`，不得删除或改写。

## 证据链

- 守护状态 TSV 在误报前已经记录 fold 4、seed 44 通过及 15/15 单元齐全；其 SHA256 为 `8b3ce26833b2c7129b25e9ae4dfd8009bfb9a2a8ade8cc239fcd650546825d3e`。
- 训练 stdout 已写出完整协议摘要，SHA256 为 `fadbbd1e1ae3a5da6a39cbd034ebb787930b3ff64f59b71e346e0f3a3c3c15c1`；stderr 为空文件，SHA256 为 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。
- `protocol_summary.json` 显示 15 个预期单元全部存在且 `claimable=true`，SHA256 为 `e70fcc028b5ad16897ec6ed0c034bfd8ec8672ed028d551373e850632e1ba636`。
- 独立 scratch-v2 审计为 `PASS`，SHA256 为 `0ee6e3d982ad902762350c6d5913fe00aacecfbf00ed08e6a54b2c1cce869e3b`。
- 测量优先总账 G1 为 `PASS`，最终复审的 `status.json` SHA256 为 `c889d9c2f0e61cbea3a438c3e75e5454c6e208c7875265ce4c631835f7e63f7e`。

## 根因与修复

旧逻辑在发现 `HasExited` 后立即读取 `ExitCode`。该属性当时可能仍为 `$null`，而 PowerShell 中 `$null -ne 0` 为真，导致空退出码被误分类为“非零退出”。修复后先执行 `WaitForExit()` 和 `Refresh()`，再把 `$null`、0 和非零退出码分开处理。守护脚本退出码契约回归测试为 `PASS`，JSON SHA256 为 `3a713b9e05662f7ba31cc562faea4162fa0b9fd37a548bd3a605423d43db2b41`。

## 边界

没有重跑任何训练单元，也没有修饰或删除误报记录。该处置不授权 SURE、corruption、开放集、NAS 新搜索、HLS、route、COM5、AV7K325 板级或功耗实验。
