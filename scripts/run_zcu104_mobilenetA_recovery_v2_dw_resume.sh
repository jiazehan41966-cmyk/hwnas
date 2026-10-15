#!/usr/bin/env bash
set -u

repo_dir="/home/jiajia/experiments/zcu104_full_campaign_v1/repo"
config_path="hls_lut_builder/configs/candidate_kernels_zcu104_mobilenetA_recovery_v2.yaml"
run_root="$repo_dir/results/zcu104_full_campaign_v1/hardware_lane/mobilenetA_full_dmdc_cfem_v2_recovered"
log_dir="$run_root/logs"
mkdir -p "$log_dir" "$run_root/status"

cases=(
  dw_conv_k3_recovery_dw_ir160_e6_14_c576_s2_lowresource_p4_cb4_ii8_main_5ns
  dw_conv_k3_recovery_dw_ir320_e3_7_c480_s1_lowresource_p4_cb4_ii8_main_5ns
  dw_conv_k3_recovery_dw_ir320_e6_7_c960_s1_lowresource_p4_cb4_ii8_main_5ns
  dw_conv_k5_recovery_dw_ir160_e3_14_c288_s2_lowresource_p4_cb4_ii8_main_5ns
  dw_conv_k5_recovery_dw_ir160_e6_14_c576_s2_lowresource_p4_cb4_ii8_main_5ns
  dw_conv_k5_recovery_dw_ir320_e3_7_c480_s1_lowresource_p4_cb4_ii8_main_5ns
  dw_conv_k5_recovery_dw_ir320_e6_7_c960_s1_lowresource_p4_cb4_ii8_main_5ns
  dw_conv_k5_recovery_dw_ir32_e6_56_c144_s2_lowresource_p4_cb4_ii8_main_5ns
  dw_conv_k5_recovery_dw_ir64_e6_28_c192_s2_lowresource_p4_cb4_ii8_main_5ns
  dw_conv_k5_recovery_dw_ir96_e3_14_c192_s1_lowresource_p4_cb4_ii8_main_5ns
  dw_conv_k5_recovery_dw_ir96_e6_14_c384_s1_lowresource_p4_cb4_ii8_main_5ns
)

printf '[%s] dw recovery resume start\n' "$(date -Iseconds)"
wait_round=0
while pgrep -u "$(id -u)" -f '/opt/Xilinx/.*/(vivado|vitis_hls)' >/dev/null 2>&1; do
  wait_round=$((wait_round + 1))
  printf '[%s] preflight wait: another jiajia Xilinx process is active (round=%s)\n' \
    "$(date -Iseconds)" "$wait_round"
  sleep 60
done

cd "$repo_dir" || exit 90
for case_name in "${cases[@]}"; do
  printf '[%s] start case=%s\n' "$(date -Iseconds)" "$case_name"
  python3 hls_lut_builder/scripts/run_synthesis.py \
    --config "$config_path" \
    --case "$case_name" \
    --downstream-check \
    --force \
    --timeout-minutes 480 \
    --downstream-timeout-minutes 360 \
    --summary-json "$run_root/status/${case_name}.resume.json" \
    > "$log_dir/${case_name}.resume.log" 2>&1
  printf '[%s] end case=%s\n' "$(date -Iseconds)" "$case_name"
done

python3 scripts/build_zcu104_mobilenetA_recovered_formal_lut.py \
  > "$log_dir/formal_lut_build_after_dw_resume.log" 2>&1
aggregate_rc=$?
printf '[%s] dw recovery resume end aggregate_rc=%s\n' "$(date -Iseconds)" "$aggregate_rc"
exit "$aggregate_rc"
