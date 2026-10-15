"""Generate the reviewable bounded experiment definition, before its freeze."""
from copy import deepcopy
import json
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[2]
historical=yaml.safe_load((ROOT/'configs/sonar_joint192_validation_20261008.yaml').read_text(encoding='utf-8'))
reference=deepcopy(historical['arms']['B1'])
default=dict(stage1_spatial='keep',stage2_spatial='keep',p2_op='keep',p2_inputs='adjacent',p2_fusion='sum')

def make(label, *, base=None, **changes):
    return dict(label=label,architecture=dict(base_architecture=deepcopy(base or reference),changes={**default,**changes}))

arms={
 'A00_reference':make('低通参考B1'),
 'A01_strided':make('普通步长2采样',base=historical['arms']['B0']),
 'A02_p2_identity':make('P2融合后直连',p2_op='identity'),
 'A03_p2_k5':make('P2 MBConv-k5',p2_op='mbconv_k5'),
 'A04_p2_sonarop':make('P2 SonarOp DW3+DW5',p2_op='sonarop'),
 'A05_p2_frequencylite':make('P2 FrequencyLite',p2_op='frequencylite'),
 'A06_p2_dmdc48':make('P2 DMDC48功能块',p2_op='dmdc48'),
 'A07_p2_cfem48':make('P2 CFEM48功能块',p2_op='cfem48'),
 'A08_stage1_sonarop':make('stage1 SonarOp空间操作',stage1_spatial='sonarop'),
 'A09_stage1_frequencylite':make('stage1 FrequencyLite空间操作',stage1_spatial='frequencylite'),
 'A10_stage2_sonarop':make('stage2 SonarOp空间操作',stage2_spatial='sonarop'),
 'A11_stage2_frequencylite':make('stage2 FrequencyLite空间操作',stage2_spatial='frequencylite'),
 'A12_p2_lateral_only':make('P2仅L2输入',p2_inputs='lateral_only'),
 'A13_p2_direct_p4':make('P2保留L2并新增T4直达',p2_inputs='direct_p4'),
 'A14_p2_concat_project':make('P2拼接后1x1投影',p2_fusion='concat_project'),
 'A15_p2_gated_sum':make('P2门控加权相加',p2_fusion='gated_sum'),
}
cfg=dict(version='sonar_joint192_extensions_fixed_B1_20261013_01',created_date='2026-10-13',
 source_space='sonar_backbone_neck192_v1',space_not_overwritten=True,
 scope='single-factor full-budget screening plus selected-candidate confirmation',
 reference='A00_reference',arms=arms,
 training=dict(epochs=80,input=640,batch_size=4,optimizer='AdamW',lr=.001,weight_decay=.0001,
               scheduler='none',amp=False,workers=0),
 dataset=dict(baseline='configs/baseline_sonar_frozen.yaml',train_images=330,valid_images=99,
              test_evaluated=False,test_mounted=False,label_policy_unresolved=True),
 gpu=1,concurrency=2,maximum_runs=22,
 confirmation=dict(maximum_candidates=2,seeds=[43,44],
   selection_rule='after_all16_seed42_complete; positive_best_delta_only; rank_desc_best_delta_then_endpoint_delta_then_ascending_training_MACs_then_id; maximum2; no_forced_winner',
   acceptance='mean_3seed_best>0 AND mean_3seed_endpoint>0 AND >=2/3 best_positive AND >=2/3 endpoint_positive AND both43/44_best_positive AND mean43/44_endpoint>0',
   seed42_selection_bias_retained=True,separate_report_of_seeds43_44=True),
 initialization=dict(common_tensors='unchanged_same_named_tensors_copied_from_per_seed_joint192_reference',
                     new_tensors='location_operation_semantic_seed_no_global_training_RNG_consumption',
                     concat_initialization='block_diagonal_identity_sum_no_added_BN_or_activation',
                     gated_sum='per_source_conv1_logits_softmax_over_sources_times_source_count; zero_logits_initially_equal_sum'),
 connections=dict(lateral_only=['L2'],adjacent=['L2','upsample_postop_T3'],
                  direct_p4=['L2','upsample_postop_T3','upsample_postop_T4'],
                  resize='nearest_to_exact_L2_HW',single_input='no_extra_fusion_parameters_identity',
                  higher_level_redundancy='T3_already_contains_T4_information; direct_T4_is_intentional_extra_path'),
 measurements=dict(primary='valid_mAP50_95_earliest_best',secondary='fixed_epoch80_AP',
   diagnostic=['lowcontrast_recall','extremely_small_recall','FP_per_image','background_FP_per_image','per_class_recall'],
   grouping='inherited_train_only_thresholds_full_GT_matching_before_grouping',
   cost='training_and_reparameterized_deploy_Conv_and_fixedfunctional_MACs_and_Params; nonconv_ops_exclusions_explicit',
   server_latency='shared_GPU_FP32_batch1_descriptive_not_claimable_efficiency',
   hardware='not_evaluated_no_integer_HLS_route_board_power_claim'),
 stopping=dict(failed_run='preserve_no_auto_retry',new_campaign_required_on_source_config_drift=True,
               no_full_space_search=True,no_test_or_automatic_hardware_promotion=True))
target=ROOT/'configs/sonar_joint192_extension_20261013.json'
if target.exists():
    old=json.loads(target.read_text(encoding='utf-8'))
    if old!=cfg:raise FileExistsError('Refuse overwriting a different experiment definition')
else:target.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(config=str(target),arms=len(arms),maximum_runs=cfg['maximum_runs']),ensure_ascii=False))
