`timescale 1ns/1ps
// Finite-capacity reference gearbox. Compact bytes; only frame-end padding.
// Same module implements wide_axis_to_dma256 and dma256_to_wide_axis.
module axis_repack_v1 #(
 parameter integer IN_BYTES=21, OUT_BYTES=32, REAL_BYTES=8400,
 parameter integer CAP=IN_BYTES+OUT_BYTES
)(input wire clk,input wire rst,
 input wire [IN_BYTES*8-1:0] s_data,input wire s_valid,output wire s_ready,input wire s_last,
 output wire [OUT_BYTES*8-1:0] m_data,output wire m_valid,input wire m_ready,output wire m_last,
 output wire [OUT_BYTES-1:0] m_keep,output wire [OUT_BYTES-1:0] m_strb,output reg error);
 reg [CAP*8-1:0] buffer;
 integer count,received,i,remaining,take;
 reg ended;
 assign m_valid=(count>=OUT_BYTES)||(ended&&count>0);
 assign s_ready=!ended&&!m_valid&&(count<=CAP-IN_BYTES);
 assign m_data=buffer[OUT_BYTES*8-1:0];
 assign m_last=ended&&(count<=OUT_BYTES)&&(count>0);
 assign m_keep={OUT_BYTES{1'b1}};
 assign m_strb={OUT_BYTES{1'b1}};
 always @(posedge clk) begin
  if(rst)begin buffer<=0;count<=0;received<=0;ended<=0;error<=0;end
  else if(m_valid&&m_ready)begin
   if(m_last)begin buffer<=0;count<=0;received<=0;ended<=0;end
   else begin buffer<=buffer>>(OUT_BYTES*8);count<=count-OUT_BYTES;end
  end else if(s_valid&&s_ready)begin
   remaining=REAL_BYTES-received;take=remaining<IN_BYTES?remaining:IN_BYTES;
   if(s_last!=(remaining<=IN_BYTES))error<=1;
   for(i=0;i<IN_BYTES;i=i+1)if(i<take)buffer[(count+i)*8+:8]<=s_data[i*8+:8];
   count<=count+take;received<=received+take;
   if(remaining<=IN_BYTES)ended<=1;
  end
 end
endmodule
