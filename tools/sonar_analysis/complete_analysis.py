"""Complete train-only intensity/FFT evidence and all-split hash integrity audit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import hashlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from analyze_dataset import ROOT, read_labels, bbox_statistics, describe, sha256, write_csv
from hwnas_fpga.data.detection import YoloDetectionDataset


def pixel_box(box, width, height):
    x1,y1,x2,y2 = box
    return (max(0,int(np.floor(x1))),max(0,int(np.floor(y1))),
            min(width,int(np.ceil(x2))),min(height,int(np.ceil(y2))))


def contrast_region(array, box, all_boxes, alpha=2., epsilon=1e-6):
    """BBox crop is not a segmentation mask; ring excludes every GT bbox."""
    height,width = array.shape
    x1,y1,x2,y2 = pixel_box(box,width,height)
    cx,cy = (x1+x2)/2,(y1+y2)/2
    bw,bh = x2-x1,y2-y1
    ex1,ey1,ex2,ey2 = pixel_box((cx-alpha*bw/2,cy-alpha*bh/2,cx+alpha*bw/2,cy+alpha*bh/2),width,height)
    mask = np.ones((ey2-ey1,ex2-ex1),dtype=bool)
    for other in all_boxes:
        ox1,oy1,ox2,oy2 = pixel_box(other,width,height)
        ax,bx = max(ox1,ex1)-ex1,min(ox2,ex2)-ex1
        ay,by = max(oy1,ey1)-ey1,min(oy2,ey2)-ey1
        if bx>ax and by>ay:
            mask[ay:by,ax:bx] = False
    t = array[y1:y2,x1:x2].reshape(-1)
    b = array[ey1:ey2,ex1:ex2][mask]
    row = dict(target_pixels=len(t),background_pixels=len(b),target_mean=float(t.mean()),
        target_std=float(t.std()),background_mean=None,background_std=None,contrast=None,status='insufficient_background')
    if len(b)>=8:
        mt,mb,sb = float(t.mean()),float(b.mean()),float(b.std())
        row.update(background_mean=mb,background_std=sb,contrast=abs(mt-mb)/(sb+epsilon),status='ok')
    return row


def nearby_background(array, box, all_boxes):
    """Nearest valid equal-size rectangular crop, deterministic eight-neighbor search."""
    height,width = array.shape
    x1,y1,x2,y2 = pixel_box(box,width,height)
    bw,bh = x2-x1,y2-y1
    candidates = []
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if dx==dy==0:
                continue
            bx,by = x1+dx*bw,y1+dy*bh
            if bx<0 or by<0 or bx+bw>width or by+bh>height:
                continue
            candidate = (bx,by,bx+bw,by+bh)
            if any(min(candidate[2],o[2])>max(candidate[0],o[0]) and
                   min(candidate[3],o[3])>max(candidate[1],o[1]) for o in all_boxes):
                continue
            candidates.append(((dx*bw)**2+(dy*bh)**2,candidate))
    if not candidates:
        return None,None
    _,bounds = min(candidates)
    a,b,c,d = bounds
    return array[b:d,a:c],bounds


def frequency_energy(crop, edges=(1/3,2/3), epsilon=1e-12):
    """DC removed; Hann taper; radial frequency normalized by sqrt(.5²+.5²)."""
    if min(crop.shape)<4:
        return {'status':'crop_too_small','energy_total':None,'low':None,'middle':None,'high':None}
    centered = crop.astype(float)-float(crop.mean())
    window = np.outer(np.hanning(crop.shape[0]),np.hanning(crop.shape[1]))
    spectrum = np.abs(np.fft.fft2(centered*window))**2
    spectrum[0,0]=0  # Taper can reintroduce a DC term.
    fy,fx = np.meshgrid(np.fft.fftfreq(crop.shape[0]),np.fft.fftfreq(crop.shape[1]),indexing='ij')
    radius = np.hypot(fx,fy)/np.sqrt(.5**2+.5**2)
    total=float(spectrum.sum())
    if total<=epsilon:
        return {'status':'zero_ac_energy','energy_total':total,'low':None,'middle':None,'high':None}
    return dict(status='ok',energy_total=total,
        low=float(spectrum[radius<edges[0]].sum()/total),
        middle=float(spectrum[(radius>=edges[0])&(radius<edges[1])].sum()/total),
        high=float(spectrum[radius>=edges[1]].sum()/total))


def dump(path,payload):
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def hash_integrity(root, output, threshold=4):
    records=[]
    thumbnails=[]
    hashes=[]
    annotation_shapes=Counter()
    for split in ('train','valid','test'):
        ds=YoloDetectionDataset(root,split=split)
        for path in ds.samples:
            label=ds.labels_dir/(path.stem+'.txt')
            labels=read_labels(label,len(ds.class_names))
            annotation_shapes.update([5]*len(labels))
            with Image.open(path) as im:
                gray=im.convert('L')
                arr=np.asarray(gray)
                tiny=np.asarray(gray.resize((9,8),Image.Resampling.LANCZOS))
                bits=(tiny[:,1:]>tiny[:,:-1]).reshape(-1)
                dh=int.from_bytes(np.packbits(bits).tobytes(),'big')
                thumb=np.asarray(gray.resize((64,64),Image.Resampling.BILINEAR),dtype=float)/255
            normalized=json.dumps(sorted(labels),separators=(',',':'))
            records.append(dict(split=split,image=str(path.relative_to(root)),stem=path.stem,
                source_stem=path.stem.split('.rf.')[0],image_sha256=sha256(path),
                decoded_pixel_sha256=hashlib.sha256(str(arr.shape).encode()+arr.tobytes()).hexdigest(),
                label_sha256=sha256(label),normalized_label_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
                dhash64=f'{dh:016x}'))
            thumbnails.append(thumb);hashes.append(dh)
    write_csv(output/'split_integrity_manifest.csv',records)
    def groups(key,cross_only=False):
        g=defaultdict(list)
        for row in records:
            g[row[key]].append({'split':row['split'],'image':row['image']})
        return [{'key':k,'items':v,'cross_split':len({x['split'] for x in v})>1} for k,v in g.items()
                if len(v)>1 and (not cross_only or len({x['split'] for x in v})>1)]
    pairs=[]
    for i in range(len(records)):
        for j in range(i+1,len(records)):
            if records[i]['split']==records[j]['split']:
                continue
            distance=(hashes[i]^hashes[j]).bit_count()
            if distance>threshold:
                continue
            a,b=thumbnails[i],thumbnails[j]
            corr=float(np.corrcoef(a.ravel(),b.ravel())[0,1]) if a.std()>0 and b.std()>0 else None
            pairs.append(dict(image_a=records[i]['image'],image_b=records[j]['image'],
                split_a=records[i]['split'],split_b=records[j]['split'],hamming_distance=distance,
                thumbnail_mae=float(np.abs(a-b).mean()),thumbnail_correlation=corr,
                exact_file=records[i]['image_sha256']==records[j]['image_sha256']))
    write_csv(output/'possible_near_duplicates.csv',pairs,fields=['image_a','image_b','split_a','split_b','hamming_distance','thumbnail_mae','thumbnail_correlation','exact_file'])
    suffixes=Counter(p.suffix.lower() for p in root.rglob('*') if p.is_file())
    result=dict(scope='all three splits: image/label integrity only; no predictions or test metric',
        split_counts=dict(Counter(r['split'] for r in records)),
        exact_duplicates=groups('image_sha256',True),decoded_pixel_duplicates=groups('decoded_pixel_sha256',True),
        duplicate_labels=groups('normalized_label_sha256'),duplicate_stems=groups('stem'),
        duplicate_source_stems=groups('source_stem'),possible_near_duplicates=pairs,
        method={'hash':'SHA256 bytes and decoded grayscale pixels','near_duplicate':'64-bit horizontal dHash',
            'hamming_threshold':threshold,'comparison':'cross split only','confirmation':'none; perceptual collisions are not confirmed leakage'},
        annotation_inventory={'scope':str(root),'file_suffix_counts':dict(suffixes),'label_column_counts':dict(annotation_shapes),
            'orientation_ground_truth_available':False,'rotated_bbox':False,'polygon':False,'mask':False,'shadow_annotation':False,
            'limitation':'Current active YOLO export only; no claim about unprovided raw acquisition metadata.'},
        unresolved_cases=[{'kind':'perceptual_hash_pairs_need_visual_or_source_group_review','count':len(pairs)}] if pairs else [],
        caution='Identical label coordinates alone do not establish duplicate images. Do not remove or relabel data automatically.')
    dump(output/'split_integrity_report.json',result)
    return result


def plot_extra(contrast,frequencies,output,names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=output/'figures'
    def save(name,data,fields=None):
        plt.tight_layout();plt.savefig(directory/(name+'.png'),dpi=300);plt.savefig(directory/(name+'.svg'));plt.close()
        write_csv(directory/(name+'.csv'),data,fields)
    valid=[r for r in contrast if r['status']=='ok']
    counts,edges=np.histogram([r['contrast'] for r in valid],bins=24)
    plt.figure(figsize=(6,4));plt.stairs(counts,edges,fill=True);plt.xlabel('BBox–ring standardized mean contrast C');plt.ylabel('Instances')
    save('contrast_histogram',[dict(left=float(a),right=float(b),count=int(n)) for a,b,n in zip(edges[:-1],edges[1:],counts)])
    plt.figure(figsize=(6,4));active=[name for name in names if any(r['class_name']==name for r in valid)]
    plt.boxplot([[r['contrast'] for r in valid if r['class_name']==name] for name in active],tick_labels=active)
    plt.ylabel('BBox–ring contrast C');save('per_class_contrast',valid)
    plt.figure(figsize=(6,4))
    for name in active:
        rows=[r for r in valid if r['class_name']==name]
        plt.scatter([r['normalized_area'] for r in rows],[r['contrast'] for r in rows],label=name,alpha=.6)
    plt.xlabel('Normalized bbox area');plt.ylabel('Contrast C');plt.legend();save('contrast_vs_bbox_size',valid)
    aggregates=[]
    for region in ('target','background'):
        rows=[r for r in frequencies if r['region']==region and r['status']=='ok']
        data=[dict(region=region,band=band,mean=float(np.mean([r[band] for r in rows])),count=len(rows)) for band in ('low','middle','high')]
        plt.figure(figsize=(6,4));plt.bar([r['band'] for r in data],[r['mean'] for r in data]);plt.ylim(0,1)
        plt.ylabel('Mean normalized AC energy');plt.title(region+' crops');save(region+'_frequency_energy',data)
        for name in names:
            subset=[r for r in rows if r['class_name']==name]
            for band in ('low','middle','high'):
                aggregates.append(dict(region=region,class_name=name,band=band,count=len(subset),mean=float(np.mean([r[band] for r in subset])) if subset else 0.))
    plt.figure(figsize=(8,4));x=np.arange(len(names)*3)
    for offset,region in enumerate(('target','background')):
        vals=[r['mean'] for r in aggregates if r['region']==region]
        plt.bar(x+(offset-.5)*.35,vals,width=.35,label=region)
    plt.xticks(x,[name+'\n'+band for name in names for band in ('low','middle','high')]);plt.ylabel('Mean normalized AC energy');plt.legend()
    save('per_class_frequency_comparison',aggregates)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,help='Existing complete train geometry analysis directory')
    p.add_argument('--alpha',type=float,default=2.)
    p.add_argument('--epsilon',type=float,default=1e-6)
    p.add_argument('--frequency-edges',type=float,nargs=2,default=[1/3,2/3])
    p.add_argument('--dhash-threshold',type=int,default=4)
    args=p.parse_args()
    if not np.isfinite(args.alpha) or args.alpha<=1 or args.epsilon<=0:
        p.error('alpha must be >1 and epsilon positive')
    if not 0<args.frequency_edges[0]<args.frequency_edges[1]<1:
        p.error('frequency edges must satisfy 0<a<b<1')
    output=Path(args.output).resolve()
    if (output/'contrast_statistics.csv').exists():
        raise FileExistsError('Supplemental analysis already exists; use a new analysis directory')
    summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    if summary['scope']!='full_requested_splits' or set(summary['by_split'])!={'train'}:
        raise ValueError('Requires complete train-only geometry analysis')
    root=Path(summary['data_root'])
    boxes=list(csv.DictReader((output/'bbox_statistics.csv').open(encoding='utf-8')))
    images=list(csv.DictReader((output/'image_statistics.csv').open(encoding='utf-8')))
    grouped=defaultdict(list)
    for row in boxes: grouped[row['image']].append(row)
    contrasts=[];frequencies=[]
    for image_row in images:
        path=root/image_row['image']
        if sha256(path)!=image_row['image_sha256'] or sha256(root/image_row['annotation'])!=image_row['annotation_sha256']:
            raise ValueError('Source data changed since geometry analysis: '+str(path))
        with Image.open(path) as im:
            array=np.asarray(im.convert('L'),dtype=float)/255
        image_row.update(intensity_mean=float(array.mean()),intensity_std=float(array.std()),
            intensity_p01=float(np.quantile(array,.01)),intensity_p99=float(np.quantile(array,.99)),
            dynamic_range=float(array.max()-array.min()))
        records=grouped[image_row['image']]
        all_boxes=[tuple(float(r[k]) for k in ('x1','y1','x2','y2')) for r in records]
        for row,box in zip(records,all_boxes):
            common={k:row[k] for k in ('image','bbox_index','class_name')}
            c=contrast_region(array,box,all_boxes,args.alpha,args.epsilon)
            contrasts.append(dict(**common,normalized_area=float(row['normalized_area']),**c))
            a,b,c,d=pixel_box(box,array.shape[1],array.shape[0]);target=array[b:d,a:c]
            background,bounds=nearby_background(array,box,all_boxes)
            for region,crop,coords in [('target',target,(a,b,c,d)),('background',background,bounds)]:
                f=frequency_energy(crop,args.frequency_edges) if crop is not None else dict(status='no_background_crop',energy_total=None,low=None,middle=None,high=None)
                frequencies.append(dict(**common,region=region,crop_xyxy=json.dumps(coords),**f))
    valid=[r['contrast'] for r in contrasts if r['status']=='ok']
    q25,q75=map(float,np.quantile(valid,[.25,.75]))
    for row in contrasts:
        row['quantile_group']='unavailable' if row['contrast'] is None else 'bottom_25' if row['contrast']<q25 else 'top_25' if row['contrast']>q75 else 'middle_50'
    write_csv(output/'contrast_statistics.csv',contrasts)
    write_csv(output/'frequency_statistics.csv',frequencies)
    write_csv(output/'image_statistics.csv',images)
    plot_extra(contrasts,frequencies,output,summary['class_names'])
    integrity=hash_integrity(root,output,args.dhash_threshold)
    summary['contrast']={'alpha':args.alpha,'epsilon':args.epsilon,'intensity_scale':'PIL grayscale /255',
        'background':'expanded bbox ring with all GT bboxes excluded; floor/ceil boundaries; population std ddof=0',
        'valid_count':len(valid),'invalid_count':len(contrasts)-len(valid),'statistics':describe(valid),
        'quartiles':[q25,q75],'fraction_C_below_1':float(np.mean(np.array(valid)<1)),
        'C1_interpretation':'absolute difference in means less than one background standard deviation; diagnostic reference, not physical weak-target label',
        'per_class':{name:describe([r['contrast'] for r in contrasts if r['class_name']==name and r['status']=='ok']) for name in summary['class_names']}}
    summary['frequency']={'sampling':'Every train GT bbox with valid equal-size neighboring background crop; no RNG',
        'preprocessing':'original grayscale /255, subtract crop mean, separable Hann window, set DC power zero',
        'radial_normalization':'sqrt(fx^2+fy^2)/sqrt(0.5^2+0.5^2), frequencies cycles/original-pixel',
        'edges':args.frequency_edges,'energy':'abs(FFT2)^2; each band / all non-DC energy',
        'background_selection':'nearest valid eight-neighbor crop of identical dimensions excluding every GT bbox; deterministic distance then xyxy tie break',
        'status_counts':dict(Counter(r['status'] for r in frequencies)),
        'means':{region:{band:float(np.mean([r[band] for r in frequencies if r['region']==region and r['status']=='ok'])) for band in ('low','middle','high')} for region in ('target','background')}}
    summary['orientation_ground_truth_available']=False
    summary['split_integrity']={'exact_cross_split_groups':len(integrity['exact_duplicates']),
        'decoded_cross_split_groups':len(integrity['decoded_pixel_duplicates']),
        'possible_near_duplicate_pairs':len(integrity['possible_near_duplicates'])}
    summary['supplement']={'command':sys.argv,'source_sha256':sha256(__file__)}
    dump(output/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('contrast','frequency','split_integrity')},ensure_ascii=False))


if __name__=='__main__': main()
