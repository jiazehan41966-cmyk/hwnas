"""Summarize perceptual collisions and inspect likely adjacent-frame candidates."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image,ImageDraw

from analyze_dataset import write_csv,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--correlation-threshold',type=float,default=.99)
    args=p.parse_args()
    out=Path(args.output)
    path=out/'split_integrity_report.json'
    report=json.loads(path.read_text(encoding='utf-8'))
    import csv
    manifest=list(csv.DictReader((out/'split_integrity_manifest.csv').open(encoding='utf-8')))
    groups=defaultdict(list)
    for r in manifest:
        stem=re.sub(r'_bmp$','',r['source_stem'])
        match=re.match(r'(.+)_([0-9]+)$',stem)
        if match:groups[match[1]].append(dict(split=r['split'],image=r['image'],frame_number=int(match[2])))
    shared=[dict(inferred_group=k,split_counts={s:sum(r['split']==s for r in v) for s in ('train','valid','test')},
        count=len(v),items=v) for k,v in groups.items() if len({r['split'] for r in v})>1]
    shortlist=[r for r in report['possible_near_duplicates'] if r['thumbnail_correlation'] is not None and r['thumbnail_correlation']>=args.correlation_threshold]
    report['near_duplicate_screening']=dict(raw_hash_pair_count=len(report['possible_near_duplicates']),
        correlation_threshold=args.correlation_threshold,shortlist_pair_count=len(shortlist),
        limitation='Exploratory thumbnail screen, not a calibrated duplicate classifier; shared sonar backgrounds can collide',
        shortlist=shortlist)
    report['inferred_source_groups_across_splits']=shared
    report['source_group_method']='Strip .rf.export hash and _bmp; final numeric suffix treated as possible frame number. Filename heuristic only.'
    report['unresolved_cases']=[dict(kind='dHash candidate collisions',count=len(report['possible_near_duplicates'])),
        dict(kind='high thumbnail similarity pairs; review source acquisition identity',count=len(shortlist)),
        dict(kind='filename-inferred sequence groups shared across splits',count=len(shared))]
    report['annotation_inventory']['metadata_checked']=['data.yaml','README.dataset.txt','README.roboflow.txt','.hwnas_extract_manifest.json']
    report['annotation_inventory']['auto_orientation_note']='README auto-orientation refers to EXIF pixel handling, not target orientation labels.'
    report['review_tool_sha256']=sha256(__file__)
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    root=Path(report['annotation_inventory']['scope'])
    selected=sorted([r for r in shortlist if {r['split_a'],r['split_b']}=={'train','valid'}],key=lambda r:r['thumbnail_mae'])[:4]
    canvas=Image.new('RGB',(800,max(1,len(selected))*325),'white');draw=ImageDraw.Draw(canvas)
    for i,row in enumerate(selected):
        for j,key in enumerate(('image_a','image_b')):
            with Image.open(root/row[key]) as im:
                im=im.convert('RGB');im.thumbnail((390,290));canvas.paste(im,(j*400,i*325+30))
            draw.text((j*400+5,i*325+2),str(Path(row[key]).name).split('.rf.')[0],fill='black')
    canvas.save(out/'near_duplicate_train_valid_review.png')
    write_csv(out/'near_duplicate_review_index.csv',selected,fields=['image_a','image_b','split_a','split_b','hamming_distance','thumbnail_mae','thumbnail_correlation','exact_file'])
    print(json.dumps(dict(shortlist=len(shortlist),shared_inferred_groups=[{k:v for k,v in r.items() if k!='items'} for r in shared])))


if __name__=='__main__':main()
