# T3/F5 open-set figure contract (predeclared before formal open-set runs)

- Core conclusion: under the same frozen 5-known/3-unknown manifests, CE+MSP, DMCL and PLUD may differ in known-class recognition and unknown rejection; no method ranking is asserted before all 45 units pass independent audit.
- Figure archetype: F5 is a quantitative confusion-matrix grid combining closed-set classification context and open-set detection evidence.
- Target/output: Python/matplotlib only; editable SVG primary, vector PDF and 300 dpi PNG retained.
- Final size: 180 mm x 150 mm, two rows by four columns.
- Panel map:
  - F5a-d: row-normalized 8x8 closed-set confusion matrices for Scratch MobileNetV2, ImageNet-pretrained MobileNetV2, the frozen NAS champion and SURE.
  - F5e-g: row-normalized 2x2 known-versus-unknown detection matrices for CE+MSP, DMCL and PLUD; the final panel is reserved for the shared color scale and evidence-boundary note.
- Evidence rule: the open protocol changes the identities of the five known classes by outer fold. Therefore class-specific 5+unknown matrices must not be pooled across folds. Only the fold-invariant known/unknown decision can be aggregated in F5; known-class macro-F1 and OSCRmac remain fold-seed metrics in T3/T9.
- Statistics: T3 means use 10,000 fold-stratified bootstrap draws. Pairwise known macro-F1 and OSCRmac comparisons use paired fold-seed bootstrap, exact paired sign-flip permutation and Holm correction within each metric family.
- Source data: every cell count and normalized value is retained with method, task, true label, predicted label and contributing sample count. All input audit, record and prediction hashes remain traceable.
- Reviewer risks: the known/unknown matrix does not show which known class was confused with another known class; T3 known macro-F1 supplies that complementary evidence. All folds arise from one dataset population. F5 must remain unavailable until both closed and open inputs are complete and audited.

## Frozen input binding added on 2026-07-17

- Sample manifest: `artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/manifests/open_long_tail_sample_manifest_v1.json.txt`.
- Manifest SHA256: `59878d48786129c983e976b1cf8f4fc03bda79bd9e05ec5671ab42dedc1f7a3e`.
- Independent audit: `open_long_tail_sample_manifest_v1_audit.json.txt`, SHA256 `63043398d989e10da319ab1a70bafa8204651e279bbf9cce28269ef1eb5f759f`.
- Audit result: `PASS`, 2,617/2,617 sample hashes reverified, 15/15 fold-seed memberships reconstructed, five unique fold-specific unknown-class sets, and zero errors.
- Admission rule: CE+MSP, DMCL and PLUD formal runs must bind this manifest SHA after canonical integration. This input freeze contributes zero formal result units and does not make T3 or F5 available.
