"""Avoid Vitis 2022.2 namespace directive parsing failure in composed kernels."""
import re,shutil
from common import *
def main():
    rows=[]
    for old in json.loads((ART/'aux_grid.json').read_text()):
        if old['kind']!='chain':continue
        hw=old['H'];r=dict(old);r['retry_of']=r['point_id'];r['point_id']+='_r1';r['aux_retry']=True
        d=ART/'aux_retry_cases'/r['point_id'];shutil.copytree(ROOT/old['case_dir'],d)
        parts=['#include "numeric_v1.h"\n#include "common_defs.h"\n']
        for label,pe in [('tower',2),('head',1)]:
            source=(ART/'aux_build/representative_cases'/f'chain_{label}_{hw}'/'kernel.cpp').read_text()
            source=source.replace('#include "numeric_v1.h"','').replace('#include "common_defs.h"','')
            for constant,value in [('IN_TILE',2),('OUT_TILE',pe)]:source=re.sub(r'(static constexpr int '+constant+r' = )[^;]+;',lambda m:m[1]+str(value)+';',source)
            names=re.findall(r'static constexpr (?:int|bool) (\w+)\s*=',source)+['input_axis_t','output_axis_t']
            for name in names:source=re.sub(r'\b'+name+r'\b',label+'_'+name,source)
            source=source.replace('void closure_kernel(','void '+label+'_kernel(')
            parts.append(source)
        oldsource=(d/'kernel.cpp').read_text();top=oldsource[oldsource.rindex('void closure_kernel('):]
        top=top.replace('tower::input_axis_t','tower_input_axis_t').replace('tower::output_axis_t','tower_output_axis_t').replace('head::output_axis_t','head_output_axis_t')
        top=top.replace('tower::closure_kernel','tower_kernel').replace('head::closure_kernel','head_kernel')
        parts.append(top);(d/'kernel.cpp').write_text('\n'.join(parts))
        r['case_dir']=str(d.relative_to(ROOT));r['implementation_variant']='v1_chain_global_unique_symbols'
        r['source_files']={p.name:sha(p) for p in d.iterdir() if p.suffix in ['.h','.cpp','.tcl']};r['source_hash']=hashobj(r['source_files'])
        save(d/'measurement_key.json',r);rows.append(r)
    save(ART/'aux_retry_grid.json',rows)
if __name__=='__main__':main()
