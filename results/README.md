每次运行使用新目录。`run.json` 保存解析后的 OmegaConf 配置、Git commit、源码 SHA 清单和环境版本；`candidates/<arch_hash>/<record_hash>.json` 保存候选、量化、硬件上下文和真实工件摘要。

`python -m scripts.tables --run results/RUN --output results/TABLES` 从候选记录导出精度、硬件、Pareto 三张 CSV。缺测保持空值；计划记录不会产生精度或 Pareto 点。不同种子、预算、阶段分别生成前沿。

本目录的实际运行产物不提交 Git；提交源码之前，不能启动带可复现性声明的训练、量化或部署工件导出。
