"""Render stage-one evidence from measured artifacts; never run a model or test split."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',required=True)
    p.add_argument('--evidence',required=True)
    args=p.parse_args()
    out=Path(args.analysis)
    s=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    audit=json.loads((out/'split_integrity_report.json').read_text(encoding='utf-8'))
    rows=list(csv.DictReader((out/'bbox_statistics.csv').open(encoding='utf-8')))
    freq=list(csv.DictReader((out/'frequency_statistics.csv').open(encoding='utf-8')))
    stats=s['statistics'];contrast=s['contrast'];f=s['frequency']
    widths=np.array([float(r['input_width_px']) for r in rows]);heights=np.array([float(r['input_height_px']) for r in rows])
    both2=float(np.mean((widths<16)&(heights<16)))
    either2=float(np.mean((widths<16)|(heights<16)))
    bands=['low','middle','high']
    def quant(d,percent=False):
        factor=100 if percent else 1
        return ' / '.join(f'{d[k]*factor:.6g}' for k in ('p10','median','p90'))
    ratio=stats['area_px2']['p90']/stats['area_px2']['p10']
    paired={}
    for row in freq:paired.setdefault((row['image'],row['bbox_index']),{})[row['region']]=row
    deltas={band:[float(v['target'][band])-float(v['background'][band]) for v in paired.values()
                  if all(v[r]['status']=='ok' for r in ('target','background'))] for band in bands}
    s['stride_evidence']={'fraction_either_dimension_less_than_two_stride8_cells':either2,
        'fraction_both_dimensions_less_than_two_stride8_cells':both2,'area_P90_to_P10':ratio}
    s['frequency']['paired_target_minus_background_P10_median_P90']={band:list(map(float,np.quantile(v,[.1,.5,.9]))) for band,v in deltas.items()}
    s['split_integrity']['screened_near_duplicate_pairs']=audit['near_duplicate_screening']['shortlist_pair_count']
    s['split_integrity']['filename_inferred_shared_sequence_groups']=len(audit['inferred_source_groups_across_splits'])
    (out/'summary.json').write_text(json.dumps(s,indent=2,ensure_ascii=False),encoding='utf-8')
    evidence=[
        ['High proportion of tiny objects',f"supported：{stats['size_counts']['small']}/330（{stats['size_proportions']['small']:.2%}）输入面积<32²；{either2:.2%}至少一边小于16px",'Include'],
        ['Large object-scale variation',f'unsupported：面积P90/P10={ratio:.3f}，等效边长比={np.sqrt(ratio):.3f}；不足以声称跨度很大','Do not include yet'],
        ['Strong directional structure','insufficient evidence：只有轴对齐框，orientation_ground_truth_available=false','Do not include yet'],
        ['Low target-background contrast',f"supported（限框—背景环均值定义）：C中位数{contrast['statistics']['median']:.4f}，330/330的C<1；不能称为物理弱目标分类",'Keep as experimental candidate'],
        ['Frequency evidence for larger context',f"unsupported（更大上下文需求未证明）：目标/背景低频能量{f['means']['target']['low']:.2%}/{f['means']['background']['low']:.2%}；只说明局部频谱差异",'Keep as experimental candidate'],
        ['Multiple objects / dense scene','unsupported：330张图均恰好1个GT；只能描述标注，不证明画面无其他物体','Do not include yet']]
    text=['# Search Space Evidence — 第一阶段', '',
        '证据只来自完整train；valid/test只参与独立文件完整性审计，未用于结构建议。Include表示值得纳入候选，并非已证明改造收益。', '',
        '| Evidence | Status | Search-space implication |','| --- | --- | --- |']
    text += ['| '+' | '.join(row)+' |' for row in evidence]
    text += ['', '## 变量对应与限制',
        '- Tiny objects → 最浅检测输出stride 8/4（P2开关）。优先级最高；现有stride8/16/32基线保持不变。',
        '- Low contrast → 早期特征保留/下采样位置作为实验候选；不据此添加新attention或专用SonarOp。低C可能来自框内亮暗抵消或混入背景，应先验证固定基线误差。',
        '- Frequency → 仅将现有常规3×3与5×5作为有限对照候选；频谱不提供具体kernel最优值。更大kernel和dilation暂不纳入。',
        '- 不新增复杂多尺度融合变量、方向算子或密集目标专用head变量；没有对应证据。', '',
        '## 划分独立性限制',
        f"跨split完全重复图像0组，但感知哈希筛查有{audit['near_duplicate_screening']['shortlist_pair_count']}对相关系数≥0.99的候选；文件名推断的6个采集序列组均跨train/valid/test。",
        '这些是待核实的数据相关性，不能直接等同于重复泄漏。未提供原始航次/帧关联元数据，当前无法证明valid对场景独立。进入正式架构优选前必须明确评估是同序列插值还是跨采集序列泛化；本轮不重划数据。', '',
        '依据：sonar_analysis_full/summary.json、bbox_statistics.csv、contrast_statistics.csv、frequency_statistics.csv、split_integrity_report.json。']
    Path(args.evidence).write_text('\n'.join(text)+'\n',encoding='utf-8')
    report=['# SONAR DATA ANALYSIS FULL', '', '## 1. Dataset Overview',
        '完整train：330图、330 GT；cylider234（70.91%）、manta96（29.09%）。原图均1280×1280；按原始YOLO标注逐图枚举，无增强、无WeightedRandomSampler，不依赖随机抽样；seed42仅记录。',
        '目录清点train/valid/test=330/99/49；以下形态、对比度、频率结论全部只来自train。', '',
        '## 2. Object Size Characteristics','分位数顺序均为P10 / median / P90。', '',
        '| 指标 | P10 / median / P90 |','| --- | --- |',
        f"| 640输入框宽(px) | {quant(stats['input_width_px'])} |",
        f"| 640输入框高(px) | {quant(stats['input_height_px'])} |",
        f"| 原图框面积(px²) | {quant(stats['area_px2'])} |",
        f"| 归一化框面积(%) | {quant(stats['normalized_area'],True)} |", '',
        f"输入尺度small/medium/large={stats['size_counts']}，比例={stats['size_proportions']}。",
        '分箱沿用当前COCO风格评价的网络输入面积：<32²、[32²,96²)、≥96²。不是声呐物理尺度定义，原图像素和归一化面积都已保留。',
        f"stride8下宽P10/中位/P90={quant({k:stats['input_width_px'][k]/8 for k in ('p10','median','p90')})}个单元；高={quant({k:stats['input_height_px'][k]/8 for k in ('p10','median','p90')})}个单元。",
        f"{either2:.2%}的GT至少一边小于2个stride8单元；{both2:.2%}两边均小于2个单元。这支持将P2/stride4纳入候选，不证明P2一定有效。", '',
        '| 类别 | 实例 | 输入宽P10/median/P90 | 输入高P10/median/P90 | 归一化面积(%)P10/median/P90 |',
        '| --- | --- | --- | --- | --- |']
    for name,v in stats['per_class'].items():
        report.append(f"| {name} | {v['instances']} | {quant(v['input_width_px'])} | {quant(v['input_height_px'])} | {quant(v['normalized_area'],True)} |")
    report += ['',f'面积P90/P10={ratio:.3f}，等效边长比仅{np.sqrt(ratio):.3f}；存在类间差异，尚不足以称为大范围尺度变化。每图GT数P10/median/P90=1/1/1；无空标签图，不能据此评估无目标场景误警能力。', '',
        '## 3. Shape Characteristics',
        f"宽高比P10/median/P90={quant(stats['aspect_ratio'])}；横/纵/近方形数量={stats['shape_counts']}，阈值>1.25/<0.8/其余。",
        '当前导出共478份五列bbox标签；未发现rotated bbox、polygon、mask、目标方向或shadow annotation；orientation_ground_truth_available=false。元数据中的Auto-orientation只是EXIF图像处理，不是目标朝向。',
        '仅有bbox shape tendency；不产生伪方向或声影标签。', '',
        '## 4. Intensity / Contrast Characteristics',
        f"原始图像PIL灰度/255；bbox按floor/ceil覆盖像素。目标区域T为整个bbox；背景为alpha={contrast['alpha']}扩张框减去所有GT框，裁剪到图像边界。std使用ddof=0；epsilon={contrast['epsilon']}；330/330有效。",
        f"C=abs(mu_t-mu_b)/(sigma_b+epsilon)。P10/median/P90={quant(contrast['statistics'])}；最大{contrast['statistics']['max']:.4f}；330/330的均值差小于1个背景标准差。",
        '存在明确的框—邻域灰度均值低区分度证据。但bbox不是目标mask，目标亮区与暗区可能互相抵消，因此不能称为物理弱目标比例，也不能推断分类必然困难。',
        f"仅用于样例排序的统计组：bottom25% C<{contrast['quartiles'][0]:.6f}，middle50%在两阈值间，top25% C>{contrast['quartiles'][1]:.6f}（边界相等归middle）；不赋予物理意义。",
        f"按类C分位数：cylider={quant(contrast['per_class']['cylider'])}；manta={quant(contrast['per_class']['manta'])}。",
        'contrast_statistics.csv保留mu_t、sigma_t、mu_b、sigma_b、像素数、C和分位组；image_statistics.csv另外保存全图mean/std、P1/P99和dynamic range。', '',
        '## 5. Frequency Characteristics',
        '覆盖全部330组目标/背景配对，660个有效crop。目标使用原图bbox；背景为八邻域中最近且不与任何GT重叠的同尺寸矩形，距离并列按坐标固定排序；不resize，不随机抽样。',
        '各crop去均值、二维Hann窗、FFT2、再次置DC能量为0。r=sqrt(fx²+fy²)/sqrt(0.5²+0.5²)，fx/fy单位cycles/original-pixel。low r<1/3，middle 1/3≤r<2/3，high r≥2/3；各band归一化到该crop非DC总能量。', '',
        '| 区域 | Low | Middle | High |','| --- | --- | --- | --- |']
    for region in ('target','background'):
        report.append('| '+region+' | '+' | '.join(f"{f['means'][region][band]:.2%}" for band in bands)+' |')
    report += ['', '这是每crop归一化后再均值，避免大crop支配结果。完整逐crop值、配对坐标与频带数据均保存；目标与背景原始尺寸完全匹配。',
        '目标crop的低频能量更集中，支持存在局部频谱差异；不证明依赖更大上下文，也不能推出特定最优kernel。3×3/5×5可作有限控制候选；更大kernel/dilation没有直接依据。', '',
        '## 6. Split Integrity Audit',
        '全部478图进行了文件SHA256、解码灰度像素SHA256、标签规范化哈希、stem与去Roboflow后缀stem检查；完全重复、重复标签/相同stem均为0组。test只用于此完整性审计，没有模型预测或性能评价。',
        f"64-bit dHash汉明距离≤4产生{len(audit['possible_near_duplicates'])}对跨split候选，说明全图哈希受共同背景影响，不能把这些全部算作近重复。再用64×64缩略图相关系数≥0.99筛查，保留{audit['near_duplicate_screening']['shortlist_pair_count']}对待核查。",
        '可视复核的train/valid相邻编号图具有相似底纹与中央条带，也有条纹位置变化；不能视为像素重复。6个文件名推断的采集序列组均跨split。必须核实原始采集组身份与评价目标；本轮保留原始划分，不自动删除、重标或迁移图像。',
        'split_integrity_report.json包含原始候选、筛查阈值、未决项和分组；near_duplicate_train_valid_review.png有对应索引CSV。', '',
        '## 7. Implications for NAS Search Space',
        '见 ../SEARCH_SPACE_EVIDENCE.md。当前最强证据是极小输入框支持stride4/P2候选；低均值对比度支持谨慎考察早期特征保留；频谱只支持有限常规kernel控制。方向、复杂融合、密集head及dilation暂不纳入。', '',
        '所有figures中的PNG/SVG均有同名CSV。本报告不包含任何训练结果或test性能。']
    result='\n'.join(report)+'\n'
    (out/'SONAR_DATA_ANALYSIS_FULL.md').write_text(result,encoding='utf-8')
    (out/'SONAR_DATA_ANALYSIS.md').write_text(result,encoding='utf-8')
    print('REPORTS WRITTEN',out)


if __name__=='__main__':main()
