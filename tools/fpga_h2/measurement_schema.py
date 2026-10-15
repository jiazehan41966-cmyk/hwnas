"""H3 schema only. CSIM wall time is never accepted as latency evidence."""
import json
from pathlib import Path
KEY=['operator','Cin','Cout','H','W','kernel','stride','activation','numeric_contract_hash','interface_contract_hash',
     'PE','SIMD','tile','buffer_profile','part','tool_version','target_clock','source_hash']
def main():
    defs={
      'CSIM':dict(type='object',required=['functional_pass'],properties={'functional_pass':{'type':'boolean'},'mismatch_count':{'type':'integer','minimum':0}},additionalProperties=False),
      'CSYNTH':dict(type='object',required=['cycles_estimate','latency_ns_estimate','resources_estimate','estimated_clock_ns'],properties={
          'cycles_estimate':{'type':'integer','minimum':0},'latency_ns_estimate':{'type':'number','minimum':0},'resources_estimate':{'$ref':'#/$defs/resources'},'estimated_clock_ns':{'type':'number','exclusiveMinimum':0}},additionalProperties=False),
      'PLACE_ROUTE':dict(type='object',required=['resources','WNS_ns'],properties={'resources':{'$ref':'#/$defs/resources'},'WNS_ns':{'type':'number'},'estimated_clock_ns':{'type':'number'}},additionalProperties=False),
      'BOARD':dict(type='object',required=['cycles_measured','latency_ns_measured'],properties={'cycles_measured':{'type':'integer'},'latency_ns_measured':{'type':'number'},'measurement_scope':{'enum':['core','DMA_end_to_end']}},additionalProperties=False)}
    defs['resources']=dict(type='object',required=['DSP','LUT','FF','BRAM18','URAM'],properties={k:{'type':'number','minimum':0} for k in ['DSP','LUT','FF','BRAM18','URAM']},additionalProperties=False)
    schema={'$schema':'https://json-schema.org/draft/2020-12/schema','title':'H3 contract-bound measurement record','type':'object','required':['key','tool_stage','value','report_sha256'],
       'properties':{'key':{'type':'object','required':KEY,'properties':{k:{'type':['number','string','object','array']} for k in KEY},'additionalProperties':False},'tool_stage':{'enum':['CSIM','CSYNTH','PLACE_ROUTE','BOARD']},'value':{'type':'object'},'report_sha256':{'type':'string','pattern':'^[0-9a-f]{64}$'}},
       'allOf':[{'if':{'properties':{'tool_stage':{'const':stage}}},'then':{'properties':{'value':{'$ref':f'#/$defs/{stage}'}}}} for stage in ['CSIM','CSYNTH','PLACE_ROUTE','BOARD']],
       '$defs':defs,'additionalProperties':False}
    (Path(__file__).resolve().parents[2]/'reports/fpga_h2_contract/H3_MEASUREMENT_SCHEMA.json').write_text(json.dumps(schema,indent=2)+'\n')
if __name__=='__main__':main()
