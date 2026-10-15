"""Upgrade only the Add loader witness to consume the exported binary scale record."""
import json
import numpy as np
from build import OUT,tb
d=OUT/'representative_cases/upsample_add'
c=json.loads((d/'case.json').read_text())
np.asarray(c['nodes'][0]['numeric'],dtype='<i4').tofile(d/'add_params.bin')
(d/'tb.cpp').write_text(tb(c))
p=d/'run.tcl';s=p.read_text()
if 'add_files -tb add_params.bin' not in s:
    s=s.replace('open_solution','add_files -tb add_params.bin\nopen_solution');p.write_text(s)
