# 声呐算子探索入口

本工具沿用E0+C01b/R0及原数据加载、TAL/GIoU和评估实现，对未获得稳定收益或尚未训练的候选操作做独立软件筛查。详细预登记见 `results/sonar_operator_probe_20261013_01/PREREGISTRATION.md`。它不执行正式192全空间训练、NAS或硬件实验。

既有服务器环境：Python `/home/jiajia/experiments/zcu104_full_campaign_v1/.venv/bin/python`，YOLO overlay复用当前环境规范；不安装依赖。结果在独立远端checkout内的 `results/sonar_operator_probe_20261013_01/remote`，小型状态、日志和收据回收到本地同名目录的 `transport`/`remote_snapshot`；完整checkpoint留服务器。

训练入口支持 `prepare`、`verify`、`smoke`、`worker`、`queue`、`status` 和 `summarize`，使用 `--config configs/sonar_operator_probe_20261013.yaml --output results/sonar_operator_probe_20261013_01/remote --device cuda`。调用由有限的远端监督器管理，先运行真实smoke，再根据本轮源码及审阅收据启动80轮队列。不要在已经启动的目录重复执行队列。

仅当源/config/data身份一致、smoke及审阅通过、GPU空闲时才启动完整训练。断点恢复必须显式指定相同variant/seed与 `--resume`；失败不自动重试。种子筛查分数是开发集描述性证据，不能称独立测试或硬件效率收益。

当前运行修订2保留首轮GPU占用拒绝的原namespace和本地 `remote_snapshot`。新的远端目录为 `/home/jiajia/experiments/sonar_operator_probe_20261013_01_revision2`，本地收据为 `transport_revision2`、`remote_snapshot_revision2`。`remote.py campaign` 启动一次有限调度：最多等待GPU1资源6小时，再执行完整预检与全部真实数据smoke，独立机器证据检查器通过后才进入80轮队列；整个过程不创建周期性自动化。
