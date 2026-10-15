#!/usr/bin/env bash
set -u

repo_dir="/home/jiajia/experiments/zcu104_full_campaign_v1/repo"
config_path="hls_lut_builder/configs/candidate_kernels_zcu104_mobilenetA_recovery_v2.yaml"
run_root="$repo_dir/results/zcu104_full_campaign_v1/hardware_lane/mobilenetA_full_dmdc_cfem_v2_recovered"
log_dir="$run_root/logs"
mkdir -p "$log_dir" "$run_root/status"

dw_cases=(
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

pw_cases=(
  pw_conv_recovery_pw_ir160_proj_e6_7_576_160_lowresource_p4_cb4_ii8_main_5ns
  pw_conv_recovery_pw_ir320_proj_e6_7_960_320_lowresource_p4_cb4_ii8_main_5ns
)

run_lane() {
  local lane="$1"
  shift
  local case_name rc
  for case_name in "$@"; do
    printf '[%s] lane=%s start case=%s\n' "$(date -Iseconds)" "$lane" "$case_name"
    cd "$repo_dir" || exit 90
    python3 hls_lut_builder/scripts/run_synthesis.py \
      --config "$config_path" \
      --case "$case_name" \
      --downstream-check \
      --timeout-minutes 480 \
      --downstream-timeout-minutes 360 \
      --summary-json "$run_root/status/${case_name}.json" \
      > "$log_dir/${case_name}.log" 2>&1
    rc=$?
    printf '[%s] lane=%s end case=%s rc=%s\n' "$(date -Iseconds)" "$lane" "$case_name" "$rc"
  done
}

printf '[%s] recovery_v2 queue start\n' "$(date -Iseconds)"
wait_round=0
while pgrep -u "$(id -u)" -f '/opt/Xilinx/.*/(vivado|vitis_hls)' >/dev/null 2>&1; do
  wait_round=$((wait_round + 1))
  printf '[%s] preflight wait: another jiajia Xilinx process is active (round=%s)\n' \
    "$(date -Iseconds)" "$wait_round"
  sleep 60
done
printf '[%s] preflight gate passed: starting two recovery lanes\n' "$(date -Iseconds)"
run_lane dw "${dw_cases[@]}" > "$log_dir/lane_dw.log" 2>&1 &
dw_pid=$!
run_lane pw "${pw_cases[@]}" > "$log_dir/lane_pw.log" 2>&1 &
pw_pid=$!
wait "$dw_pid"
dw_rc=$?
wait "$pw_pid"
pw_rc=$?

cd "$repo_dir" || exit 91
python3 scripts/build_zcu104_mobilenetA_recovered_formal_lut.py \
  > "$log_dir/formal_lut_build.log" 2>&1
aggregate_rc=$?
printf '[%s] recovery_v2 queue end dw_rc=%s pw_rc=%s aggregate_rc=%s\n' \
  "$(date -Iseconds)" "$dw_rc" "$pw_rc" "$aggregate_rc"
exit "$aggregate_rc"
