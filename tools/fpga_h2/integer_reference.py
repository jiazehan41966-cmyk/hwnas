"""Explicit V0.1 integer detector graph and shared binary parameter package."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from numeric import conv,aligned_add,quantize_real,ratio,q6
from interface import align,descriptors

ACT={'linear':0,'relu':1,'relu6':2}

def digest(data):return hashlib.sha256(data).hexdigest()

class PackageBuilder:
    def __init__(self,model,folder,contract_hashes):
        self.model=model;self.folder=Path(folder);self.nodes=[];self.weights=bytearray();self.biases=bytearray();self.records=[]
        self.edges={'image':dict(shape=[1,640,640],exp=-5)};self.hashes=contract_hashes

    def layer(self,path,source,act,out_exp):
        m=self.model.get_submodule(path);ci,h,w=self.edges[source]['shape'];ei=self.edges[source]['exp']
        wf=m.weight.detach().cpu().numpy();bf=m.bias.detach().cpu().numpy()
        ew=max(-24,int(math.ceil(math.log2(max(float(np.abs(wf).max()),1e-8)/127))))
        wq=quantize_real(wf,ew);bq=quantize_real(bf,ei+ew,bias=True)
        wo=align(len(self.weights));bo=align(len(self.biases));self.weights.extend(bytes(wo-len(self.weights)));self.biases.extend(bytes(bo-len(self.biases)))
        self.weights.extend(wq.tobytes());self.biases.extend(bq.tobytes())
        mult,shift=ratio(ei+ew,out_exp);k=m.kernel_size[0];s=m.stride[0];p=m.padding[0];co=m.out_channels
        node=dict(id=path,op='conv',inputs=[source],Cin=ci,Cout=co,H=h,W=w,kernel=k,stride=s,padding=p,groups=m.groups,input_exp=ei,weight_exp=ew,output_exp=out_exp,
                  multiplier=mult,shift=shift,activation=act,q6=q6(out_exp),weight_offset=wo,bias_offset=bo,weight_shape=list(wq.shape),bias_count=co,
                  folded_weight_hash=digest(wf.tobytes()),folded_bias_hash=digest(bf.tobytes()),weight_hash=digest(wq.tobytes()),bias_hash=digest(bq.tobytes()),layer_id=len(self.records))
        record=[node['layer_id'],ci,co,h,w,k,s,p,m.groups,ei,ew,out_exp,mult,shift,ACT[act],q6(out_exp),wo,bo,wq.size,co,1]+[0]*11
        self.records.append(record);self.nodes.append(node)
        self.edges[path]=dict(shape=[co,(h+2*p-k)//s+1,(w+2*p-k)//s+1],exp=out_exp)
        return path

    def add(self,name,a,b,out_exp,upsample=False):
        ea,eb=self.edges[a]['exp'],self.edges[b]['exp'];common=max(ea,eb)
        self.nodes.append(dict(id=name,op='upsample_add' if upsample else 'add',inputs=[a,b],input_exp_a=ea,input_exp_b=eb,common_exp=common,output_exp=out_exp,
            numeric=list(ratio(ea,common)+ratio(eb,common)+ratio(common,out_exp))))
        self.edges[name]=dict(shape=self.edges[a]['shape'],exp=out_exp)
        return name

    def build(self):
        x=self.layer('backbone.stem.conv.conv','image','relu',-5)
        x=self.layer('backbone.stages.0.0.conv',x,'relu',-5)
        features={}
        for i,blocks in enumerate(self.model.backbone.stages):
            if i==0:continue
            if i!=3:
                block=blocks[0];base=f'backbone.stages.{i}.0';skip=x
                x=self.layer(base+'.expand_conv',x,'relu6',-4)
                x=self.layer(base+'.dw_conv',x,'relu6',-5)
                x=self.layer(base+'.project_conv',x,'linear',-5)
                if block.use_residual:x=self.add(base+'.residual',x,skip,-5)
            if i in self.model.feature_indices:features[i]=x
        lateral=[]
        for j,i in enumerate(self.model.feature_indices):lateral.append(self.layer(f'neck.lateral_convs.{j}.0',features[i],'relu',-5+j))
        fused=lateral.copy()
        for j in range(len(fused)-2,-1,-1):fused[j]=self.add(f'neck.fuse.{j}',fused[j],fused[j+1],-5+j,True)
        neck=[self.layer(f'neck.output_convs.{j}.0',x,'relu',-5) for j,x in enumerate(fused)]
        outputs={}
        for j,(level,x) in enumerate(zip(self.model.levels,neck)):
            x=self.layer(f'head.towers.{j}.0',x,'relu',-5)
            outputs[level]=self.layer(f'head.towers.{j}.3',x,'linear',-4)
        self.folder.mkdir(parents=True,exist_ok=False)
        (self.folder/'weights.bin').write_bytes(self.weights);(self.folder/'biases.bin').write_bytes(self.biases)
        records=np.asarray(self.records,dtype='<i4');records.tofile(self.folder/'layers.bin')
        anchors=dict(zip(self.model.levels,self.model.anchor_sizes))
        payload=dict(version=1,architecture=self.model.candidate_architecture,inputs={'image':self.edges['image']},nodes=self.nodes,edges=self.edges,outputs=outputs,
             descriptors=[d.__dict__ for d in descriptors(self.model.levels,anchors)],**self.hashes,
             binary_hashes={f:digest((self.folder/f).read_bytes()) for f in ['weights.bin','biases.bin','layers.bin']},
             layers_record='32 little-endian int32 words; fields documented in loading contract')
        (self.folder/'graph.json').write_text(json.dumps(payload,indent=2))
        return payload

def forward(folder,image,retain=True):
    folder=Path(folder);meta=json.loads((folder/'graph.json').read_text());wb=(folder/'weights.bin').read_bytes();bb=(folder/'biases.bin').read_bytes()
    for f,h in meta['binary_hashes'].items():
        if digest((folder/f).read_bytes())!=h:raise ValueError('parameter package hash mismatch')
    records=np.fromfile(folder/'layers.bin',dtype='<i4').reshape(-1,32)
    values={'image':np.asarray(image,dtype=np.int8)}
    if list(values['image'].shape)!=[1,640,640]:raise ValueError('V0.1 input shape')
    for n in meta['nodes']:
        a=values[n['inputs'][0]]
        if n['op']=='conv':
            record=records[n['layer_id']]
            expected=[n['layer_id'],n['Cin'],n['Cout'],n['H'],n['W'],n['kernel'],n['stride'],n['padding'],n['groups'],n['input_exp'],n['weight_exp'],n['output_exp'],n['multiplier'],n['shift'],ACT[n['activation']],n['q6'],n['weight_offset'],n['bias_offset'],int(np.prod(n['weight_shape'])),n['bias_count'],1]
            if list(record[:21])!=expected:raise ValueError('JSON/binary metadata disagreement')
            # Arithmetic values originate from the very same binary record read by HLS loader.
            n=dict(n,multiplier=int(record[12]),shift=int(record[13]),output_exp=int(record[11]),activation=['linear','relu','relu6'][record[14]])
            count=int(np.prod(n['weight_shape']));w=np.frombuffer(wb,dtype=np.int8,count=count,offset=n['weight_offset']).reshape(n['weight_shape'])
            b=np.frombuffer(bb,dtype='<i4',count=n['bias_count'],offset=n['bias_offset'])
            y=conv(a,w,b,n)
        else:
            b=values[n['inputs'][1]]
            if n['op']=='upsample_add':
                if a.shape[1]!=2*b.shape[1] or a.shape[2]!=2*b.shape[2]:raise ValueError('only integer factor2')
                b=np.repeat(np.repeat(b,2,axis=1),2,axis=2)
            y=aligned_add(a,b,n['input_exp_a'],n['input_exp_b'],n['output_exp'])
        values[n['id']]=y
    return {p:values[name] for p,name in meta['outputs'].items()},values
