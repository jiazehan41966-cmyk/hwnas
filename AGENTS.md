# Global research workflow for Codex

## General research rules
- For literature review, experiment design, data analysis, academic writing, paper revision, and figure/table generation, first check whether an installed skill is relevant.
- Prefer explicit skill invocation when a task maps clearly to a skill.
- Do not fabricate papers, citations, datasets, metrics, or experimental results.
- When modifying research code, preserve reproducibility: record commands, configs, random seeds, datasets, metrics, and output paths.
- Before running destructive commands, explain the impact and ask for confirmation.
- For Python research projects, prefer project-local virtual environments and keep generated outputs under results/, runs/, outputs/, or logs/.
- For academic Chinese writing, use formal, precise, and restrained language.
- For this HW-NAS / FPGA / sonar image classification project, prioritize macro_f1, top1, latency, LUT, DSP, BRAM, power, feasibility, and reproducibility.

## Project memory and terminology
- Before starting a new task in this HW-NAS repository, read `docs/PROJECT_MEMORY.md` when repository context, audit history, archive paths, or workflow boundaries matter.
- Keep project memory isolated by default: do not use preferences, conclusions, code assumptions, experimental settings, datasets, or results from other projects as reference material for this HW-NAS project unless the user explicitly asks to connect them.
- Treat the current workspace, local files, and user-provided context as the authoritative source for this project.
- In this HW-NAS project, when output includes English technical terms, provide a concise Chinese explanation at first use when it helps readability or academic clarity.
- For recurring English metrics and hardware terms, keep the original term visible and pair it with a short explanation where appropriate, for example: macro_f1（宏平均 F1 指标）, LUT（查找表资源）, DSP（数字信号处理单元）, BRAM（块 RAM）, latency（推理延迟）.

## Chinese skill aliases
- 文献综述 / 查论文 / 论文方向梳理: use literature-review or paper-lookup.
- 找论文 / DOI / arXiv / Semantic Scholar / 引文查询: use paper-lookup.
- 论文写作 / 实验设置 / 方法章节 / 论文段落: use scientific-writing.
- AI科研写作资源 / 写作工具推荐 / 写作模板 / 论文写作工具列表: use awesome-ai-research-writing.
- 实验结果分析 / 模型对比 / 消融实验 / 显著性检验: use results-analysis or statistical-analysis.
- 统计检验 / p-value / 置信区间 / 方差分析 / 相关性: use statistical-analysis.
- 画图 / 论文图 / 图表美化 / 可视化: use scientific-visualization.
- 论文自查 / 逻辑漏洞 / 审稿意见预判: use scientific-critical-thinking or citation-verification.
- 核验引用 / 防止假引用 / 检查参考文献: use citation-verification.
- PDF: use pdf.
- Word / DOCX / 报告文档: use docx.
- PPT / slides / 演示文稿: use pptx.
- 网页PPT / 横向翻页PPT / 归藏PPT / 杂志风PPT / 瑞士风PPT / Swiss Style: use guizang-ppt-skill.
- Excel / CSV / XLSX / 表格清洗: use xlsx.
