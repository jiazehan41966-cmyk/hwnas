# T6 陈旧交叉引用问题卡

## 问题

历史档案 `t6_hardware_collection_contract_patch_20260717.md` 的 artifact 表把 `t6_complete_network_collection_design_20260717.md` 绑定为 SHA256 `cd76632dcbb0151f513dd5bb3d171b4a111fb379c6ce406cfbb647f1db5c32a2`，但当前只读设计原件的实际 SHA256 为 `fb567a57a40a189703f66991b86ca8b081002f705eca834aafb9126b449c1f2f`。

## 影响

- 该交叉引用已陈旧，不能作为当前设计文件的有效哈希绑定。
- T6 真实完整网络样本仍为 0/100，因此没有实测 HLS/route 行因该问题被错误接纳。
- T6、F4 与 G2 状态不变，仍为 PENDING。

## 处理边界

英文历史原件保持只读，不手工篡改旧 SHA。后续在新 source freeze 下接入 canonical 硬件收集合同和正式图表 builder 时，必须由生成器重新计算并绑定当前设计、schema、auditor 与模板 SHA，同时增加“被引用文件发生变化即 fail-closed”的测试。

当前不启动 HLS、route 或板级实验；该问题不需要板卡操作。
