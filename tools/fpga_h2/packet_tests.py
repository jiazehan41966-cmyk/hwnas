"""Packet software tests plus self-checking bounded/backpressured RTL testbench."""
import csv,json
from pathlib import Path
import numpy as np
from interface import pack,unpack,wide_axis_to_dma256,dma256_to_wide_axis,descriptors

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reports/fpga_h2_contract';ART=ROOT/'artifacts/fpga_h2/20260916_v1'

def main():
    folder=ART/'rtl';folder.mkdir(exist_ok=True)
    cases=[];rows=[];rng=np.random.default_rng(43)
    for c,h in [(21,20),(21,40),(21,80),(21,160),(16,17),(24,17),(32,17),(64,17),(96,17),(160,17),(320,17)]:
        x=rng.integers(-128,128,(c,h,h),dtype=np.int16).astype(np.int8);raw=x.transpose(1,2,0).copy().tobytes();packed=pack(x)
        assert np.array_equal(x,unpack(packed,x.shape))
        words=[raw[i:i+c] for i in range(0,len(raw),c)];beats=wide_axis_to_dma256(words,c,len(raw));assert dma256_to_wide_axis(beats,c,len(raw))==words
        name=f'c{c}_h{h}';cases.append(dict(name=name,channels=c,h=h,real_bytes=len(raw),padded_bytes=len(packed),native_words=h*h,dma_words=len(beats)))
        (folder/(name+'_native.hex')).write_text('\n'.join(w[::-1].hex() for w in words)+'\n')
        (folder/(name+'_dma.hex')).write_text('\n'.join(w[::-1].hex() for w in beats)+'\n')
        rows.append(dict(case=name,scope='SOFTWARE_PACKET_ROUNDTRIP',real_bytes=len(raw),padded_bytes=len(packed),mismatch_count=0,exact_match_ratio=1.0,status='PASS'))
    with (OUT/'packet_software.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (folder/'cases.json').write_text(json.dumps(cases,indent=2))
    body='''`timescale 1ns/1ps
module packet_case #(parameter C=21,H=20,parameter NAME="c21_h20")(output reg done);
localparam N=H*H, REAL=C*N, DMA=(REAL+31)/32;
reg clk=0;always #5 clk=~clk;reg rst=1;
reg [C*8-1:0] original[0:N-1];reg [255:0] expected[0:DMA-1];
reg [31:0] rng=32'h12345678;reg valid=0;reg ready=0;integer sent=0,got=0,mid=0,ticks=0;
wire sr,av,ar,al,bv,bl,e1,e2;wire[255:0] ad;wire[C*8-1:0] bd;wire[31:0] ak,astr;wire[C-1:0] bk,bstr;
axis_repack_v1 #(.IN_BYTES(C),.OUT_BYTES(32),.REAL_BYTES(REAL)) enc(clk,rst,original[sent<N?sent:0],valid,sr,sent==N-1,ad,av,ar,al,ak,astr,e1);
axis_repack_v1 #(.IN_BYTES(32),.OUT_BYTES(C),.REAL_BYTES(REAL)) dec(clk,rst,ad,av,ar,al,bd,bv,ready,bl,bk,bstr,e2);
initial begin done=0;$readmemh({NAME,"_native.hex"},original);$readmemh({NAME,"_dma.hex"},expected);repeat(5)@(negedge clk);rst=0;end
always @(posedge clk)if(!rst&&!done)begin
 ticks<=ticks+1;rng<={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};
 // Source holds VALID and payload during backpressure. Sink inserts deterministic stalls.
 if(!valid||sr)valid<=(sent+(valid&&sr)<N)&&(rng[0]||rng[4]);
 ready<=rng[1]||rng[7];
 if(valid&&sr)sent<=sent+1;
 if(av&&ar)begin
  if(ad!==expected[mid]||ak!==32'hffffffff||astr!==32'hffffffff||al!==(mid==DMA-1))$fatal(1,"DMA mismatch %s %d",NAME,mid);
  mid<=mid+1;
 end
 if(bv&&ready)begin
  if(bd!==original[got]||bl!==(got==N-1))$fatal(1,"roundtrip mismatch %s %d",NAME,got);
  got<=got+1;
  if(got==N-1)begin if(mid!=DMA)$fatal(1,"DMA count");done<=1;$display("RTL_PACKET_PASS %s native=%0d dma=%0d",NAME,N,DMA);end
 end
 if(e1||e2||enc.count>enc.CAP||dec.count>dec.CAP)$fatal(1,"frame/capacity error");
 if(ticks>2000000)$fatal(1,"timeout");
end
endmodule
module tb;
'''
    body+=f'wire [{len(cases)-1}:0] done;\n'
    for i,c in enumerate(cases):body+=f'packet_case #(.C({c["channels"]}),.H({c["h"]}),.NAME("{c["name"]}")) p{i}(done[{i}]);\n'
    body+='initial begin wait(&done);$display("ALL_RTL_PACKETS_PASS");$finish;end\nendmodule\n'
    (folder/'tb.sv').write_text(body)
    import shutil
    shutil.copyfile(Path(__file__).parent/'axis_repack_v1.sv',folder/'axis_repack_v1.sv')
    print('Packet software cases PASS',len(cases))

if __name__=='__main__':main()
