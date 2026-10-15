# 迁移验收

2026-10-15，在 Windows / Python 3.13.3 / PyTorch 2.7.0+cu126 下完成验证。测试固定 CPU 两线程；没有研究任务启动。

| 检查 | 结果 | 证据范围 |
|---|---|---|
| 全套测试 | **348 passed，37.69 秒** | 单元、合同、依赖边界、纯调度和合成入口整合 |
| 原/新模型 | 216 组 state_dict 严格相等，3 组完整前向逐位相等 | 不代表已训练精度 |
| 真模型 PTQ 整合 | E1 HEAD_W8A16 与 E0 W8A8 均全节点零差异 | 临时合成图、非正式校准，软件整数参考 |
| 阶段入口 | initialized checkpoint→PTQ→valid 评价→bundle→三表；漂移拒绝 | 纯合成 train/valid，无优化器训练，无硬件测量 |
| wheel 构建 | 成功，包含 YAML、数值合同与来源清单 | 安装包完整性 |
| wheel 隔离导入 | 原仓库外构建 E1、解析配置、加载数值合同、生成67节点图 | 无旧源码包路径依赖 |
| 计划/三表 | 1296 条缺测决策，0 个 Pareto 点 | 不读取数据、不启动训练 |

本轮测试命令：

```powershell
$env:PYTHONPATH = 'E:\1\hwnas\refactor_admin_20261015\test_dependencies'
E:\1\hwnas\hwnas\.venv_cuda\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --basetemp results/pytest_final --junitxml results/pytest_final_report.xml
E:\1\hwnas\hwnas\.venv_cuda\Scripts\python.exe -m pip wheel . --no-deps --no-build-isolation --wheel-dir outputs/dist_final
```

`PYTHONPATH` 仅加入为本轮测试隔离安装的 OmegaConf 2.3.0 及其依赖；没有向旧研究环境安装或升级包。正常使用可按 README 在本项目虚拟环境执行 `pip install -e ".[dev]"`，随后不需要这项测试路径。

实际运行证据在本工作区忽略目录 `results/pytest_final_report.xml`、`results/pytest_final/`、`outputs/dist_final/`。调度测试通过 synthetic evaluator 验证 72/54/36、弃选门槛及三种子确认，不把测试分数用于论文。没有验证正式 train64 精度保持、真实 HLS/route 或板端执行与功耗。
