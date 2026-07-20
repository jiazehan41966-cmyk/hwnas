# rl_arch_135 跨层绑定独立审计

- 状态：`PASS`。
- 正式证据增量：`0`。

## 已通过检查

- manifest 状态、语言与正式证据增量正确。
- 生成器、全部输入和输出的 SHA/大小一致。
- 三份人类可读报告均为中文且无替换字符。
- 两幅图均含 600 dpi PNG、PDF、SVG、源 CSV 和中文 meta。
- 软件图源数据为 15 个配对单元且 NAS 胜出 0/15。
- 架构 PASS、权重未绑定、板级准确率未运行、功耗未测量且 route 资源数值正确。

本审计确认 bundle 的可追溯性和声明边界，不把历史 latency-only bitstream 提升为 formal checkpoint 部署证据。
