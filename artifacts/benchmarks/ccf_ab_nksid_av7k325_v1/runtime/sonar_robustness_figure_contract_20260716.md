# T4/F7/F8 figure contract (predeclared before formal corruption evaluation)

- Core conclusion: under identical NKSID outer folds and deterministic image-domain corruptions, the four audited closed-set methods may differ in the rate and uncertainty of macro-F1 degradation; no winner is asserted before the archived data exist.
- Figure archetype: F7 is a quantitative grid with the SNR response as hero evidence; F8 is an image plate plus traceability metadata and is qualitative context, not performance evidence.
- Target/output: Nature/high-impact-journal-ready Python/matplotlib output; editable SVG is primary, vector PDF and 300 dpi PNG are retained for the campaign contract.
- Final size: F7 180 mm x 82 mm, two aligned panels; F8 180 mm x 205 mm, eight class rows by seven frozen condition columns.
- Backend: Python only.
- Panel map:
  - F7a: AWGN macro-F1 at 0, 5, 10, 15 and 20 dB, with 10,000-iteration fold-stratified bootstrap 95% CI over the 15 fold-seed units.
  - F7b: speckle macro-F1 under the same grid and inference rule.
  - F8: the lowest global sample index in each of the eight NKSID classes; columns are clean, AWGN 10 dB, AWGN 0 dB, speckle 10 dB, speckle 0 dB, blur severity 5 and contrast severity 5.
- Evidence hierarchy:
  - Hero evidence: paired method curves and uncertainty in F7.
  - Validation evidence: per-unit F1-SNR AUC, worst-case macro-F1 and drop from clean in T4/T9.
  - Context only: F8 makes corruption severity visually inspectable and hash-traceable.
- Statistics: 10,000 fold-stratified paired bootstrap draws; exact paired sign-flip permutation for 15 units; Holm correction within AWGN and speckle comparison families; effect size is Cohen's dz for paired differences.
- Source data: each figure has one combined source CSV; raw per-sample predictions, achieved SNR, clipping/saturation ratio, checkpoint hashes and transformation seeds remain under the formal robustness result roots.
- Image integrity: F8 uses deterministic full-image transforms only; no local retouching, cropping, class-conditional adjustment or post-selection by model outcome is allowed. Original and rendered SHA256 values are retained.
- Reviewer risks: only 15 paired fold-seed units exist; folds share the same dataset population; confidence bands summarize protocol units and are not claims about independent acquisition sites. Clean is excluded from numerical F1-SNR AUC. PSNR/SSIM are permitted only because corruptions have an exact paired resized-clean input and do not establish real-world despeckling quality.
