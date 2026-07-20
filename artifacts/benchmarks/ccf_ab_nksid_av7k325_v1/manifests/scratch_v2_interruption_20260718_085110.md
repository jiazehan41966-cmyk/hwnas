# scratch-v2 中断记录

- 时间：2026-07-18T08:51:10.7725689+08:00
- 状态：INTERRUPTED_FAIL_CLOSED
- 已发现完整单元数：0/15
- 原因：Invalid object passed in, ':' or '}' expected. (235): {
  "schema_version": 1,
  "campaign_id": "ccf_ab_nksid_av7k325_v1",
  "authorization_status": "APPROVED_ONCE",
  "user_decision_recorded": true,
  "user_instruction_zh_cn": "搴旂敤 provenance 淇銆佽ˉ娴嬭瘯銆侀噸鏂板喕缁擄紝鍐嶅喅瀹?鎵ц scratch-v2銆?,
  "run_name": "g1_mobilenet_v2_scratch_v2",
  "method_id": "scratch_mobilenet_v2",
  "output_dir": "results/protocol/g1_clean_20260718",
  "folds": [0, 1, 2, 3, 4],
  "seeds": [42, 43, 44],
  "epochs": 150,
  "source_freeze_manifest": "artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/source_freeze/g1_20260718_v3/source_freeze_manifest.json",
  "source_freeze_manifest_sha256": "cfbc7ec9373e762c39385d733e07682a39ef843f87c2a25100c3fb7bfb824f32",
  "source_snapshot_sha256": "990d1dd08b1e966c4e85ff76661b4ceffb0986200a25859c8540cafe359a6ed3",
  "entrypoint_sha256": "8f649be022273a0bdd9633795aeba3020e401b845967eab34c21297ee31b043f",
  "measurement_ledger_sha256": "2a7b6941d1d47b0be10fa71882c3cc386ba134b22d582f645fcbbb9ebbc16b19",
  "legacy_scratch_manifest_sha256": "ba13135545258d3f8c3667782341229d1a8f942143e0e56a57f75cff07f7d8d7",
  "legacy_scratch_patch_sha256": "0e92055318fd5b96436b7852ca4be030c61f3bd1024a2913da6b0d42c86ef3d1",
  "stop_policy": {
    "minimum_macro_f1_per_unit": 0.80,
    "maximum_absolute_delta_from_legacy_same_pair": 0.05,
    "stop_on_nonfinite": true,
    "stop_on_hash_mismatch": true,
    "stop_on_nonzero_exit": true
  },
  "allow_sure": false,
  "allow_corruption": false,
  "allow_open_set": false,
  "allow_nas_search": false,
  "allow_hls_or_route": false,
  "allow_board_or_com5": false,
  "allow_power": false,
  "decision_note_zh_cn": "浠呭厑璁?fresh scratch-v2 15 鍗曞厓锛涘紓甯稿嵆鍋滐紱瀹屾垚鍚庡厛瀹¤骞舵殏鍋溿€?
}

- 运行目录：E:\1\hwnas\hwnas\results\protocol\g1_clean_20260718\g1_mobilenet_v2_scratch_v2
- stdout：E:\1\hwnas\hwnas\logs\g1_clean_20260718\scratch_v2_stdout.log
- stderr：E:\1\hwnas\hwnas\logs\g1_clean_20260718\scratch_v2_stderr.log

## 决策边界

监控器已经停止训练且禁止自动恢复。旧 scratch、SURE、corruption、开放集、NAS 搜索、HLS、route、COM5、板级与功耗实验均未获本授权，不能继续。下一步必须由用户判断。
