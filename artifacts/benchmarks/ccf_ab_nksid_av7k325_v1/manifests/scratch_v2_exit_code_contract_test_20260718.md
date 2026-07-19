# scratch-v2 守护脚本退出码回归测试

- 状态：**通过**（$(System.Collections.Specialized.OrderedDictionary.status)）。
- 成功子进程观测退出码：$observedZero。
- 非零子进程观测退出码：$observedSeven。
- PowerShell 中 ` $null -ne 0 ` 的结果：$nullComparisonWouldMisclassify；这解释了旧逻辑为何会在训练已成功结束后误报中断。
- 修复后的静态契约：$staticContract，包含显式 WaitForExit()、Refresh()、空值分支和整数退出码判断。
- 守护脚本 SHA256：$(System.Collections.Specialized.OrderedDictionary.wrapper_sha256)。

## 边界

本测试没有重新运行训练，没有修改 15 个 fold-seed 结果，也没有启动 SURE、HLS、route、COM5、板级或功耗实验。
