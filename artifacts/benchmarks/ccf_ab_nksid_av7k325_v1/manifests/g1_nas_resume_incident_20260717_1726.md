# G1 NAS 限定恢复启动故障卡

## 故障结论

- 发生时间：2026-07-17 17:21–17:26（Asia/Shanghai）。
- 中断等级：S2 运行故障；未进入训练。
- 触发位置：持久化 wrapper 读取机器批准 JSON 并执行 `ConvertFrom-Json`。
- 直接原因：Windows PowerShell 5.1 将无 BOM 的 UTF-8 中文字符串按本地编码读取，产生乱码并报“传入了未终止的字符串”。
- 科学数据影响：无。fold 4、seed 44 的 checkpoint、记录、逐样本预测和 protocol summary 均未生成。
- 并发与下游影响：无匹配训练进程；SURE、scratch-v2、鲁棒性、开放集、NAS 新搜索、HLS、板级和功耗实验均未启动。

## 修复与重试边界

机器 JSON 中的中文值改为 ASCII 范围内的 Unicode 转义；JSON 解析后仍恢复为中文。该修改不改变暂停清单绑定、候选、数据、划分、指标、训练协议、source freeze 或授权最大范围。

重试前必须重新确认：批准 JSON 可由 Windows PowerShell 5.1 正确解析，source freeze PASS 556/556，已有单元仍为 14/15且唯一缺失 fold 4、seed 44，计划任务保持 Disabled，匹配实验进程为 0。
