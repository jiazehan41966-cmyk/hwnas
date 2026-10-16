# 三套声呐数据：原始包、标签与固定视图

本目录实际保存原始 ZIP 数据，通过 Git LFS 上传；小文件直接由 Git 管理。解压后的重复图像按 `dataset_sources.json` 恢复，不重复上传。该索引逐图记录 ZIP 成员、尺寸、CRC32 和 SHA256，同时记录每个标签/配置文件与原始数据分卷的 SHA256。

| 数据 | 原始归档字节 | 保留的处理视图与划分 |
|---|---:|---|
| Cylinder2，Roboflow cylider2 v6 YOLOv8 | 232,076,741 | processed：train330 / valid99 / test49；保留原类别名 cylider、manta |
| classic SCTD 1.0 | 83,140,304 | processed：249/54/54；clean_phase3b：251/53/53；clean_phase3b_v2：249/54/54；顺序均为train/valid/test |
| UATD v3 | 4,796,321,023 | processed_yolo：train7600、test1和test2各800、带前缀merged test1600；clean_phase3b：6078/1520/1600 |

总归档 **5,111,538,068 字节**，共6个原始ZIP、9个LFS对象。UATD训练ZIP保持原字节流，按1 GiB拆为4卷；恢复器直接跨分卷seek读取，不额外拼接3.87GB完整ZIP。其他5包单独保存。所有原包整体SHA与已有来源清单核对。

UATD的merged test只是test1/test2的别名视图，不能将它们相加报告为独立样本。clean_phase3b 的两个开发样本排除及既有划分完全按保存清单重放，不重新分层。SCTD是classic 1.0，不是SCTD 3.0；其派生split不自动获得采集组独立性保证。

```powershell
git lfs install
git lfs pull
python -m scripts.restore_datasets --dataset all --verify-only
python -m scripts.restore_datasets --dataset all
```

恢复器只依赖Python标准库，支持 `--dataset Cylinder2` / `SCTD` / `UATD` 和 `--output-root F:/my_dataset_checkout`。默认恢复所有保留的训练视图，直接使用现有Git标签和配置；不调用训练、推理或划分算法，不访问外部下载站点。遇到同路径不同内容时拒绝覆盖。目标文件系统支持时，完全相同的派生图像以硬链接复用，避免重复占空间；这些链接都限于恢复输出根内，不链接到原本地数据。硬链接视图中的图片应视为只读，修改实验须另存副本。

首次恢复需要读取并验证原包，再解码ZIP压缩流；不是运行图像解码或模型。UATD图像唯一数据量约34GB，还需文件系统空间；不支持硬链接时重复视图会使用更多空间。`--verify-only` 仅校验分卷与整体SHA，不写图像。GitHub网页显示LFS指针是正常现象，必须安装LFS并拉取实际对象；脚本会明确识别未拉取的指针。

实际纳入内容：

- `raw/`：三套完整官方/来源ZIP。UATD附属OpenSLT工具ZIP也原样保留，默认不解出工具。
- `metadata/`：来源说明、原始API记录、完整性报告、类别映射、既有冻结split清单与旁车hash，原字节保留。
- 各处理视图：全部原标签、data.yaml及已有README，原字节保留。图像由离线恢复器按清单恢复。
- `dataset_sources.json`、`publication_summary.json`：可移植来源索引与纳入范围统计。

未上传重复解压副本、UATD的冗余 `phase3c_clean_phase3b.tar.zst` 传输包，或不属于这三套数据的 `figshare_mine_detection_v2`。原本地目录保持不变。本次不包含模型权重、实验结果、HLS/Vivado工程和未提交子模块修改。

来源与归属：

- Cylinder2：[Roboflow cylider2 v6](https://universe.roboflow.com/yeesonmin-naver-com/cylider2/dataset/6)，原README与data.yaml标注CC BY 4.0，随处理视图保留。
- SCTD：[MingqiangNing/SCTD](https://github.com/MingqiangNing/SCTD)，原ZIP和本地来源材料未发现明确LICENSE声明；不为它补写CC BY或项目代码许可。详见SCTD/metadata/source_investigation.md。
- UATD：[Figshare v3](https://doi.org/10.6084/m9.figshare.21331143.v3)，保留原作者、引用和CC BY 4.0元数据。OpenSLT归档内部的第三方copyright与通知保持原样，其工具许可不被数据集说明覆盖。

历史metadata中出现的 `F:\hwnas_datasets` 等绝对路径只是当时来源记录；恢复器以此处可移植索引为准，不要求新机器具有那些盘符。标签和划分没有被重新计算，test恢复仅属于字节归档，不授权将test用于结构或量化选择。
