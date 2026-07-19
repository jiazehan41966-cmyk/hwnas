# G1 预训练完整重启记录（中文伴随档案，2026-07-16）

- 英文原件：`g1_pretrained_restart_20260716.md`；SHA256：`b35d50576e599bbb15235407f12573ffcf107bdcb9893ec431fc736ff982f981`。
- 状态：`INTERRUPTED_BY_SYSTEM_REBOOT`。
- 方法：`imagenet_pretrained_mobilenet_v2`，计划 15 个 fold-seed 单元。
- 新 fingerprint：`ff1d84701305b45488e321213320adef3f54cfd2483f38b1d8fa1719befd42b0`。
- Source freeze：PASS 556/556；manifest SHA256 `8b4de1d5bf8931c7a175cf913abd95b7a0a63848b2eaea3b2a87bc09ea2665dc`。
- 关机前 4 个记录保存在 `interrupted_20260716_0407/`，不与重启后的正式运行混合；未使用 `--force`。
- Windows 于 07:45:36 进入 Kernel API reboot，07:45:49 重启；本次 attempt 产生 0/15 原子记录，无 Python traceback 或 CUDA OOM。
- 该 attempt 只保留为中断证据，后续被持久化计划任务运行取代；再后续的预训练运行已完成并独立通过 15/15。
