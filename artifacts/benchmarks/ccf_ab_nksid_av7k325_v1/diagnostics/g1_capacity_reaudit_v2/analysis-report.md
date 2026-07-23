# G1 capacity attribution re-audit

## Result

The frozen `rl_arch_135` candidate shows a large **underfitting signal under the current
recipe**, but the existing curves do **not** uniquely identify parameter capacity as the
mechanism. Across 15 complete paired fold/seed runs, its mean online,
augmentation-affected training accuracy over epochs 140-150 was 0.6824,
versus 0.9878 for scratch MobileNetV2 (paired delta
+0.3054, bootstrap 95% CI [0.3010,
0.3102]).

## What this establishes

- The gap is present during training under the frozen recipe, not only on outer validation.
- The classic pattern “near-perfect training accuracy plus weak validation” is absent from
  this logged metric.
- The NAS inner-validation optimum is usually not pinned to epoch 150, so blindly extending
  the same schedule is not the highest-information next action.

## What this does not establish

`train_acc` is computed in `model.train()` on randomly flipped, rotated, affinely transformed,
brightness/contrast-jittered, and sometimes speckle-corrupted batches. It is therefore not a
deterministic accuracy on the unaugmented training set. A shared recipe also does not control
architecture-specific optimisation difficulty. Capacity, optimiser/schedule mismatch,
regularisation strength, augmentation sensitivity, and train/eval-mode behaviour remain
confounded.

## Decision

Do not launch a full capacity sweep or distillation comparison yet. First run an inference-only
evaluation of every saved best checkpoint on its own deterministic, no-augmentation training
indices. If the large gap remains, run one micro-overfit/LR triage before selecting the capacity
sweep. The four-arm preprocessing campaign stays queued but unlaunched until a source-grouped
split contract exists.
