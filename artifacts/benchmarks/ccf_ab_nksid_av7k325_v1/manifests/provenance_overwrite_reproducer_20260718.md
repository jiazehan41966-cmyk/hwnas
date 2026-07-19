# provenance patch 覆盖缺陷最小复现

- 状态：`PASS_BUG_REPRODUCED_PATCH_NOT_APPLIED`。
- 复现位置：自动清理的隔离临时 Git 仓库。
- 正式入口修改：否。正式证据增量：0。

## 观察

- 固定路径模型连续捕获两个 diff 时使用同一路径：`True`。
- 第一次 SHA：`a0963fda9779290ec6b0c2b805a5b6ae59ff8d67a88db44e1006a4804f47e0e6`；第二次 SHA：`b2a7ba274e67aa38a95aff6554dac6907fd147effb1183e97641bae33f2666a0`。
- 第二次捕获后，第一次文件字节不再存在，证明固定 `code_patch.diff` 会破坏旧 manifest 的文件绑定。
- 内容寻址模型使用不同路径：`True`，第一次证据保持：`True`。

## 结论

缺陷可由代码结构和隔离行为复现共同支持。修复设计可阻止覆盖，但当前只作为提案，必须取得用户批准后才能修改正式入口、重新冻结和决定 scratch-v2。
