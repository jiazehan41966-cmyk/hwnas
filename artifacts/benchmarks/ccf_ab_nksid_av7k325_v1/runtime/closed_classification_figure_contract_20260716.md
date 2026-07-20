# T2/F6 figure contract (predeclared before four-method completion)

- Core conclusion: under the same 5-fold x 3-seed NKSID protocol, the four audited closed-set methods may differ in macro-F1, calibration and failure-ranking quality; no winner is asserted before all 60 units and their audits exist.
- Figure archetype: F6 is a quantitative two-panel grid; risk-coverage is the hero evidence and the reliability diagram is calibration validation.
- Target/output: Python/matplotlib only; editable SVG primary, vector PDF and 300 dpi PNG retained.
- Final size: 180 mm x 82 mm.
- Panel map:
  - F6a: pooled 15-bin reliability curve for each method plus the identity line; exact bin counts remain in source CSV.
  - F6b: mean risk-coverage curve over 15 fold-seed units on a common 1%-100% coverage grid, with 10,000-iteration fold-stratified bootstrap 95% bands.
- Statistics: T2 reports fold-stratified bootstrap 95% CI over 15 protocol units. Pairwise macro-F1 differences use the same paired bootstrap, exact paired sign-flip permutation and Holm correction within the T2 macro-F1 family.
- Source data: all metrics are recomputed from archived per-sample logits and targets; every checkpoint, prediction file, record and independent audit hash is retained.
- Compute boundary: checkpoint bytes and model-state tensor elements are directly measured from archived checkpoint files. Training wall-clock/GPU-hours were not captured by a trustworthy per-unit timer and must be reported as `NOT_RECORDED`, never reconstructed from file timestamps.
- Reviewer risks: repeated seeds share each outer fold's images, and the five folds come from one dataset population. Pooled reliability bins are visual summaries; inferential comparisons remain at the paired fold-seed unit. F5 is withheld until closed- and open-set confusion evidence can be composed together.
