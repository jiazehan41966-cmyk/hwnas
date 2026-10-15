"""Exercise clipped tiles, channel tails, strides and integer corner cases."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import numpy as np
from .prepare import HERE,ROOT,GXX,constants
from .reference import make_fixture,fixture_modes

class TiledKernelTests(unittest.TestCase):
    def verify(self,op,ci,co,h,w,k=1,stride=1,pad=0,bits=8,outbits=8,activation='linear'):
        oh,ow=((h*2,w*2) if op=='nearest2x' else ((h+2*pad-k)//stride+1,(w+2*pad-k)//stride+1))
        case=dict(operator=op,input_bits=bits,output_bits=outbits,weight_bits=8 if op in ('pw1x1','conv3x3','dw3','dw5') else None,
            shape=dict(input_shapes=[[ci,h,w]],output_shape=[co,oh,ow],kernel=k,stride=stride,padding=pad,
                groups=ci if op.startswith('dw') else 1,activation=activation),mapping=dict(PE=4,SIMD=4 if op=='pw1x1' else 1,tile_rows=2,tile_cols=32))
        if op=='requant_add':case['shape']['input_shapes'].append([ci,h,w])
        conf=constants(case)
        with tempfile.TemporaryDirectory(prefix='t6_kernel_') as tmp:
            p=Path(tmp)
            shutil.copyfile(HERE/'kernel.cpp',p/'kernel.cpp');shutil.copyfile(ROOT/'tools/hardware_measurement_pilot/tb.cpp',p/'tb.cpp')
            (p/'config.h').write_text('#pragma once\n'+''.join(f'#define {key} {v}\n' for key,v in conf.items()))
            for i,mode in enumerate(fixture_modes):
                vals=make_fixture(case,mode,103+i)
                for suffix,v in vals.items():(p/f'test{i}_{suffix}.bin').write_bytes(np.ascontiguousarray(v).tobytes())
            (p/'fixtures.txt').write_text(''.join(f'test{i}\n' for i in range(len(fixture_modes))))
            env=os.environ.copy();env['PATH']=str(GXX.parent)+os.pathsep+env['PATH']
            result=subprocess.run([str(GXX),'-std=c++14','-O2','-Wno-unknown-pragmas','kernel.cpp','tb.cpp','-o','test.exe'],cwd=p,env=env,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=subprocess.run([str(p/'test.exe')],cwd=p,env=env,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            for i in range(len(fixture_modes)):self.assertEqual((p/f'test{i}_actual.bin').read_bytes(),(p/f'test{i}_expected.bin').read_bytes())
    def test_pw_channel_and_spatial_tails(self):self.verify('pw1x1',5,7,5,35,activation='relu6')
    def test_pw_w8a16(self):self.verify('pw1x1',5,7,3,37,bits=16,outbits=16)
    def test_stem_stride2_padding(self):self.verify('conv3x3',1,5,7,67,3,2,1,activation='relu')
    def test_dw3_stride2(self):self.verify('dw3',5,5,7,67,3,2,1,activation='relu6')
    def test_dw5_stride1(self):self.verify('dw5',5,5,5,35,5,1,2,activation='relu6')
    def test_nearest_partial_tile(self):self.verify('nearest2x',5,5,3,17)
    def test_add_requant_and_tail(self):self.verify('requant_add',5,5,5,35)
    def test_body8_to_head16(self):self.verify('body8_to_head16',5,5,5,35,outbits=16)
    def test_pool2(self):self.verify('maxpool',5,5,6,70,2,2,0)
    def test_pool_padding_negative_extreme(self):self.verify('maxpool',5,5,5,35,3,2,1)

if __name__=='__main__':unittest.main()
