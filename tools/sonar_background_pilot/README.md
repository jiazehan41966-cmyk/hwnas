# 声呐背景误检三方案试验

这是原有 Dense Anchor 检测器的训练损失试验，三个独立方案分别借鉴排序、PISA 样本权重和 DMT 前景背景区分思路。没有宣称完整复现论文，也没有更改部署网络。配置、随机种子、预算和初筛门槛见 `results/sonar_background_pilot_20261010/PLAN.json`。

原项目检测指标是 mean(AP50, AP75)，使用原来的积分实现；另报缓存输出上的 COCO 101 点 AP。解码器仍然使用 score>=0.01、NMS=0.5、全图最多50框。阈值只在99张验证图上选择，整个试验不加载 test。

## Windows 独立复验

前提：已有 `.venv_cuda`，源快照由主执行者准备。不要重建环境、修改源快照或启动训练。先将工作目录设置为冻结快照：

```powershell
Set-Location 'E:\1\hwnas\hwnas\results\sonar_background_pilot_20261010\snapshot'
```

核验源码散列，预期输出 `status: PASS`：

```powershell
& 'E:/1/hwnas/hwnas/.venv_cuda/Scripts/python.exe' -m tools.sonar_background_pilot.campaign verify --campaign ..
```

运行 CPU 合成数据检查。预期四个分支完成一次有限损失、有限梯度与参数更新，四个初始化散列一致，基线损失逐值一致，并输出 `status: PASS`。这只能验证代码和环境，不证明精度提升，也不是640真实图像检查：

```powershell
& 'E:/1/hwnas/hwnas/.venv_cuda/Scripts/python.exe' -m tools.sonar_background_pilot.campaign preflight --campaign .. --synthetic
```

记录所有失败和文档与实际行为的偏差，不要修复。证据保存在上一级 `preflight_synthetic.json`。

## 服务器调用

已有解释器 `/home/jiajia/experiments/zcu104_full_campaign_v1/.venv/bin/python`，不安装包。新目录 `/home/jiajia/experiments/sonar_background_pilot_20261010` 的 `snapshot` 是唯一运行源码。数据从旧目录只读复用，预检核验 train/valid 全部图像与标签散列，正式训练每个运行再次核验。GPU 使用受当前并行任务约束；只检查占用，不停止其他任务。

从项目根目录调用 `python -m tools.sonar_background_pilot.remote probe|verify|preflight|launch|status|fetch`；preflight/launch 需 `--data-dir` 指向探测到的数据路径。`launch` 必须已有真实图像 CUDA 预检 PASS，正式队列预算固定为4分支×3种子×30轮，拒绝覆盖既有启动和运行结果。

当前尚未独立验证任何精度结果。完成后查看 `remote_results/comparison.json` 和配对运行的 `status.json`。短训练、单验证划分以及验证集上选最优轮次和阈值限制结论；不得据此声称测试集或 FPGA 部署收益。
