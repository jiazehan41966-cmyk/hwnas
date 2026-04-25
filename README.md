# HW-NAS FPGA Sonar

闈㈠悜姘翠笅澹板憪鍥惧儚鍒嗙被/璇嗗埆浠诲姟鐨勭‖浠舵劅鐭ョ缁忔灦鏋勬悳绱紙HW-NAS锛夐」鐩紝鐩爣鏄湪 FPGA 璧勬簮绾︽潫涓嬪畬鎴愪粠鎼滅储銆佽缁冨埌閮ㄧ讲鐨勯棴鐜€?
---

## 蹇€熷紑濮?
```bash
# 鏈€灏忓寲娴嬭瘯
python3 run_search.py --search-method rl --episodes 3 --train-epochs 1 --batch-size 8

# 瀹屾暣鎼滅储
python3 run_search.py --config configs/search/nksid_fpga_search_mobile_anchor_av7k325.yaml

# A1: 鎼滅储绌洪棿鍙鐜囬獙璇侊紙闅忔満閲囨牱锛屼笉璁粌锛?
python3 run_search_space_probe.py --config configs/search/nksid_fpga_search_mobile_anchor_av7k325.yaml --num-samples 200
# 鎸囧畾缁撴灉鐩綍
python3 run_search.py --config configs/search/nksid_fpga_search_mobile_anchor_av7k325.yaml --output-dir results

# 閲嶈鎼滅储寰楀埌鐨勬渶浼樻灦鏋?python3 run_retrain.py --run-dir results/<search_run_name>

# 鐢ㄦ悳绱㈠緱鍒扮殑鏈€浼樻ā鍨嬭瘑鍒浘鐗?python3 run_infer.py --checkpoint results/<run_name>/checkpoints/best_model.pt --input /path/to/image_or_dir

# 瀵煎嚭 ONNX 骞剁敓鎴?HLS 宸ョ▼楠ㄦ灦
python3 run_export.py --checkpoint results/<run_name>/checkpoints/final_best_model.pt --prepare-hls

# 棰濆瀵煎嚭 INT8 鏉冮噸閲忓寲鍖?python3 run_export.py --checkpoint results/<run_name>/checkpoints/final_best_model.pt --quantize-int8

# 浠?HLS profiling 鎶ュ憡鏋勫缓 LUT 琛?python3 run_build_lut.py --manifest configs/hardware/lut_manifest_example.yaml --output artifacts/fpga_lut.pkl
```

璇︾粏浣跨敤璇存槑锛歔docs/QUICKSTART.md](docs/QUICKSTART.md)

> Current formal search entry point: `configs/search/nksid_fpga_search_mobile_anchor_av7k325.yaml`
>
> Legacy generic `nksid_fpga_search*.yaml` configs have moved to `configs/search/legacy/`.
> The current MobileNetV2 mainline no longer treats `dw_pw_conv` as a default searchable operator.

---

## 缁撴灉钀界洏

姣忔鎼滅储閮戒細鍦?`results/<run_name>/` 涓嬬敓鎴愬畬鏁磋繍琛岀洰褰曪紝榛樿鍖呭惈锛?
```text
results/<run_name>/
鈹溾攢鈹€ config.yaml
鈹溾攢鈹€ cli_args.json
鈹溾攢鈹€ run_info.json
鈹溾攢鈹€ logs/
鈹?  鈹斺攢鈹€ console.log
鈹溾攢鈹€ results/
鈹?  鈹溾攢鈹€ baseline.json
鈹?  鈹溾攢鈹€ dataset_summary.json
鈹?  鈹溾攢鈹€ search_space_summary.json
鈹?  鈹溾攢鈹€ candidates.jsonl
鈹?  鈹溾攢鈹€ candidates.json
鈹?  鈹溾攢鈹€ candidates.csv
鈹?  鈹溾攢鈹€ pareto_front.json
鈹?  鈹溾攢鈹€ pareto_selection.json
鈹?  鈹溾攢鈹€ summary.json
鈹?  鈹斺攢鈹€ candidates/
鈹?      鈹斺攢鈹€ <arch_id>.json
鈹斺攢鈹€ checkpoints/
    鈹溾攢鈹€ search_state.json
    鈹溾攢鈹€ best_candidate.json
    鈹溾攢鈹€ best_model.pt
    鈹溾攢鈹€ final_best_model.pt   # retrain 鍚庣敓鎴?    鈹溾攢鈹€ controller_latest.pt   # RL 鎼滅储鏃剁敓鎴?    鈹斺攢鈹€ controller_best.pt     # RL 鎼滅储鏃剁敓鎴?```

---

## 椤圭洰鏂囨。

| 鏂囨。 | 璇存槑 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 鏁翠綋鎶€鏈灦鏋勪笌妯″潡杈圭晫 |
| [docs/method_design.md](docs/method_design.md) | 闂妯″瀷銆佹柟娉曡璁′笌瀹炵幇鏄犲皠 |
| [docs/project_overview.md](docs/project_overview.md) | 椤圭洰姒傝涓庨棶棰樺畾涔?|
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | 蹇€熷紑濮嬩笌浣跨敤鎸囧崡 |
| [docs/PROGRESS.md](docs/PROGRESS.md) | 瀹炵幇杩涘睍涓庡姛鑳芥€荤粨 |

---

## 椤圭洰缁撴瀯

```text
.
鈹溾攢鈹€ README.md                 # 椤圭洰璇存槑
鈹溾攢鈹€ run_search.py            # 鎼滅储鍏ュ彛鑴氭湰
鈹溾攢鈹€ run_build_lut.py         # HLS report -> LUT 鏋勫缓鍏ュ彛
鈹溾攢鈹€ configs/                  # 閰嶇疆鏂囦欢
鈹?  鈹斺攢鈹€ search/
鈹?      鈹斺攢鈹€ sonar_fpga_baseline.yaml
鈹?  鈹斺攢鈹€ hardware/
鈹?      鈹溾攢鈹€ zynq7020.yaml
鈹?      鈹溾攢鈹€ kintex7_xc7k325.yaml
鈹?      鈹斺攢鈹€ lut_manifest_example.yaml
鈹溾攢鈹€ docs/                     # 鏂囨。
鈹?  鈹溾攢鈹€ architecture.md
鈹?  鈹溾攢鈹€ method_design.md
鈹?  鈹溾攢鈹€ project_overview.md
鈹?  鈹溾攢鈹€ QUICKSTART.md
鈹?  鈹斺攢鈹€ PROGRESS.md
鈹溾攢鈹€ reference/                # 鍙傝€冧唬鐮佸簱鍒嗘瀽
鈹?  鈹溾攢鈹€ FBNet/
鈹?  鈹溾攢鈹€ HW-NAS-Bench/
鈹?  鈹溾攢鈹€ TinyTNAS/
鈹?  鈹斺攢鈹€ ANALYSIS.md
鈹斺攢鈹€ src/hwnas_fpga/
    鈹溾攢鈹€ interfaces.py         # 鎺ュ彛瀹氫箟
    鈹溾攢鈹€ search_space/         # 鎼滅储绌洪棿
    鈹?  鈹斺攢鈹€ space.py
    鈹溾攢鈹€ hardware/             # 纭欢浼拌
    鈹?  鈹溾攢鈹€ cost.py
    鈹?  鈹溾攢鈹€ lookup_table.py   # LUT鏌ユ壘琛?    鈹?  鈹溾攢鈹€ lut_pipeline.py   # profiling manifest -> LUT
    鈹?  鈹斺攢鈹€ report_parser.py
    鈹溾攢鈹€ models/               # 妯″瀷鏋勫缓
    鈹?  鈹斺攢鈹€ builder.py
    鈹溾攢鈹€ data/                 # 鏁版嵁鍔犺浇
    鈹?  鈹斺攢鈹€ dataset.py
    鈹溾攢鈹€ training/             # 璁粌涓庨噸璁?    鈹?  鈹溾攢鈹€ trainer.py
    鈹?  鈹斺攢鈹€ retrain.py
    鈹溾攢鈹€ search/               # 鎼滅储绠楁硶
    鈹?  鈹溾攢鈹€ searcher.py
    鈹?  鈹溾攢鈹€ constrained.py    # 绾︽潫鎼滅储鍣?    鈹?  鈹斺攢鈹€ pareto.py         # Pareto鍓嶆部
    鈹斺攢鈹€ deploy/               # ONNX / HLS 瀵煎嚭
        鈹溾攢鈹€ export.py
        鈹溾攢鈹€ quantization.py
        鈹溾攢鈹€ hls.py
        鈹溾攢鈹€ hls_backend.py
        鈹溾攢鈹€ report_parser.py
        鈹斺攢鈹€ inference.py
```

---

## 鏍稿績鍔熻兘

### 鉁?宸插疄鐜?
- **鎼滅储绌洪棿**: stage-based 鍙悳绱㈡灦鏋勭┖闂?- **纭欢浼拌**: 鍒嗘瀽妯″瀷 + LUT 鏌ユ壘琛?+ board profile
- **妯″瀷鏋勫缓**: ArchitectureSpec 鈫?PyTorch Model
- **璁粌璇勪及**: 瀹屾暣鐨勮缁冩祦绋?- **鎼滅储绠楁硶**: RL 鎼滅储 + 绾︽潫鍓灊 + 鐪熸 Pareto 閫変紭
- **閲嶈缁?*: best architecture 鐙珛鏈€缁堥噸璁?- **閮ㄧ讲瀵煎嚭**: ONNX 瀵煎嚭 + HLS 椤圭洰楠ㄦ灦 + report parser
- **INT8閲忓寲**: 鏉冮噸閲忓寲鍖呭鍑猴紝渚?FPGA/HLS 鍚庣浣跨敤
- **LUT profiling**: HLS report -> LUT table 鏋勫缓閾?- **瀵规瘮瀹為獙**: Fused MBConv vs 鏍囧噯 MBConv銆佹湁/鏃犳棭鏈熷壀鏋?- **Pareto浼樺寲**: 澶氱洰鏍囦紭鍖栦笌鍓嶆部鍒嗘瀽

### 鈴?寰呭疄鐜?
- 鏉冮噸鍏变韩瓒呯綉璁粌
- 鐪熷疄澹板憪鏁版嵁鍔犺浇 (MARIS/UATD)
- HLS/Vivado/Vitis 瀹為檯璋冪敤涓庢澘涓婂洖濉?
---

## LUT Profiling

鐪熷疄 FPGA profiling 鍙互閫氳繃 `run_build_lut.py` 浠?Vivado/Vitis HLS 鎶ュ憡鏋勫缓鏌ユ壘琛細

```bash
python3 run_build_lut.py \
  --manifest configs/hardware/lut_manifest_example.yaml \
  --output artifacts/fpga_lut.pkl \
  --summary-json artifacts/fpga_lut_summary.json
```

鐢熸垚鐨?`fpga_lut.pkl` 鍙互鐩存帴鎺ュ埌鎼滅储閰嶇疆锛?
```yaml
hardware:
  board: zynq7020
  lut_path: artifacts/fpga_lut.pkl
  use_dummy_lut: false
```

---

## 鏂规鍙傝€?
| 鍙傝€冨簱 | 鐢ㄩ€?|
|---|---|
| **FBNet** | stage-based 鎼滅储绌洪棿銆丩UT 鏋舵瀯璁捐 |
| **TinyTNAS** | 绾︽潫椹卞姩銆佹椂闂撮檺鍒舵悳绱?|
| **HW-NAS-Bench** | 纭欢鎸囨爣璁捐 |
| **HW-PR-NAS** | Pareto 鎺掑悕淇濇寔 |
| **DARTS** | 鍙井 NAS 鍩虹嚎锛堝弬鑰冪敤锛?|

璇﹁锛歔reference/ANALYSIS.md](reference/ANALYSIS.md)
