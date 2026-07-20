# 外部声呐数据集接入

更新时间：2026-07-05

本仓库将新增数据集作为独立、可追溯的数据源保存于
`data/external/`。原始压缩包、解压内容和本地清单均不提交到 Git；
数据源注册信息保存在
`configs/datasets/external_sonar_sources.yaml`。

## 当前本机状态

- Figshare v2 已于 2026-07-04 完整下载并通过 6/6 发布文件 MD5 校验。
- Figshare 本地原始包加解压内容共约 1.173 GiB，清单位于
  `data/external/figshare_mine_detection_v2/dataset_manifest.json`。
- 1,170 张图像均有同名 YOLO 标签文件；304 张图像含检测框，
  866 个标签文件为空。
- 共 668 个框：MILCO 437、NOMBO 231；未发现错误类别、畸形行或
  归一化坐标越界。
- Roboflow v6 已完成公开源重建：478 张原始分辨率图像、478 个 YOLO
  标签文件与 330/99/49 的 train/valid/test 划分均已落地并通过检查。
- Roboflow 本地图像约 504.4 MiB；共 478 个框，其中 `cylinder` 200、
  `cylider` 139、`manta` 139；未发现精确重复图像。
- Roboflow 本地清单位于
  `data/external/roboflow_cylider2/dataset_manifest.json`，可追溯索引位于
  `artifacts/datasets/roboflow_cylider2_v6_source_index.json`。

## 已注册数据源

### Roboflow `cylider2`

- 用户提供入口：<https://universe.roboflow.com/yeesonmin-naver-com/cylider2>
- 数据版本：共 6 个；API 下载默认解析最新发布版本，也可显式指定
  `--roboflow-version`
- 任务：object detection（目标检测）
- 项目页规模：476 张源图像；最新 v6 数据视图为 478 张图像
- v6 划分：train 330、valid 99、test 49
- 原始类别：`cylinder`、`cylider`、`manta`
- 许可：CC BY 4.0

`cylinder` 与 `cylider` 在来源中是两个不同标签。接入过程保留原标签，
不会在没有人工核验的情况下把疑似拼写错误自动合并。

Roboflow 的公开数据导出接口仍要求 API key。推荐：

```powershell
$env:ROBOFLOW_API_KEY = "<your-key>"
python scripts/fetch_external_sonar_datasets.py --dataset roboflow_cylider2
```

如需冻结指定版本：

```powershell
$env:ROBOFLOW_API_KEY = "<your-key>"
python scripts/fetch_external_sonar_datasets.py `
  --dataset roboflow_cylider2 `
  --roboflow-version 6
```

也可以先在网页中导出 YOLOv8 ZIP，再导入：

```powershell
python scripts/fetch_external_sonar_datasets.py `
  --dataset roboflow_cylider2 `
  --roboflow-zip C:\path\to\cylider2.zip
```

API key 只通过进程参数或环境变量读取，不写入数据清单。

本机已采用公开源重建模式完成接入。该模式下载公开原图，读取 v6 当前
split 与像素坐标框，并生成归一化 YOLO 标签：

```powershell
python scripts/fetch_external_sonar_datasets.py `
  --dataset roboflow_cylider2 `
  --roboflow-acquisition public-source `
  --roboflow-version 6
```

公开源重建不是 Roboflow 官方导出 ZIP 的逐字节副本：它保留原始分辨率，
没有执行 v6 的 Auto-Orient 与 1280×1280 Stretch resize。原图、类别、框和
split 已完整保留，因此适合数据审计和后续自定义预处理；若论文实验必须复刻
Roboflow v6 导出像素，则仍应使用账户 API key 获取官方导出包。

### Figshare `Side-scan sonar imaging for Mine detection`

- 页面：<https://figshare.com/articles/dataset/24574879>
- DOI：<https://doi.org/10.6084/m9.figshare.24574879.v2>
- 版本：v2
- 任务：目标检测、分类或分割
- 规模：1,170 张真实侧扫声呐图像
- 类别：NOMBO（非水雷类海底目标）与 MILCO（疑似水雷目标）
- YOLO 类别索引：`0 = MILCO`，`1 = NOMBO`
- 采集年份：2010–2021
- 许可：CC BY 4.0
- 发布文件：`2010.zip`、`2015.zip`、`2017.zip`、`2018.zip`、
  `2021.zip`、`Training.zip`，共 613,408,909 bytes

该来源可通过 Figshare 公共 API 下载，无需凭据：

```powershell
python scripts/fetch_external_sonar_datasets.py `
  --dataset figshare_mine_detection
```

下载器逐文件核验发布方提供的 MD5，另记录 SHA-256；解压时拒绝路径穿越
和符号链接条目，并审计图像/YOLO 标签配对、空标注、类别框数量和坐标范围。
可重复执行，已通过校验的压缩包与解压目录会被复用。

空 YOLO 文件的来源语义尚未由下载包本身明确说明。当前接入只记录其为空，
不会自动把对应图像标成背景类、NOMBO 或其他分类标签。

## 与当前 HW-NAS 主线的边界

当前 `run_search.py` 主线是 NKSID 单标签图像分类，类别为 8 类水下目标；
上述两个来源原生包含目标检测或混合任务标注。因此：

1. 不把外部图片直接复制进 `data/NKSID`。
2. 不把 NOMBO/MILCO 或检测框标签映射成 NKSID 的 8 类标签。
3. 不沿用 NKSID 的固定 split 文件评估外部数据。
4. 未完成转换协议、分组划分和独立审计前，不把外部数据结果与现有
   `macro_f1`（宏平均 F1）、`top1` 或板级证据合并报告。

后续若用于当前分类网络，应先确定一种可复现策略：按检测框裁剪为目标级
分类样本，或把模型扩展为检测头。两条路线必须分别冻结标签映射、去重规则、
采集年份/任务分组 split、随机种子和独立测试集。
