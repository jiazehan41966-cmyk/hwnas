# FCOS v2 shared GPU resource adapter

This package changes resource admission only. It imports the frozen v2 supervisor,
workers and train module; the existing model/config/source binding, six 30-epoch
runs, completion checks, pause markers and locks remain the execution contract.

GPU 1 is reserved. Seed pairs are fixed to 42 -> GPU 0, 43 -> GPU 2 and 44 -> GPU 3;
only previously unassigned claims are filled under the v2 claims lock. GPU 0 uses
the original idle-card rule. GPUs 2/3 allow one resident each only when GPU UUID,
boot ID, PID, all UIDs, start ticks, comm and NVIDIA process path match the policy.
Unknown processes, inconsistent observations, identity changes, low free memory,
resident memory growth relative to the fixed approved per-process memory snapshot
or resident CPU activity yield through the original pause
protocol. Shared workers and their train children use CPU nice 10.

Per-PID SM counters are unavailable. A resident's CPU counter change is only an
activity proxy and cannot prove that asynchronous GPU activity is absent. Startup
also checks aggregate GPU utilization; running monitors do not use aggregate
utilization as a resident activity measurement. The v2 20-second polling interval
is retained. First admission requires two CPU samples at least 5 seconds apart;
samples older than 60 seconds require a new sample pair. A resource veto just
before creating a train child waits and rechecks without aborting other workers.

CLI: `python -m tools.sonar_fcos_quality_shared_runtime.scheduler supervise
--config <frozen-v2-config> --policy <policy-json> --freeze-manifest <freeze-json>
--review-gate <independent-review-json>` (arguments on one shell command line).

Freeze schema: `schema_version:1`, `policy_sha256`, `config_sha256`,
`source_binding_sha256` (canonical JSON digest of the original source binding),
`runtime_sources` (repository-relative file -> SHA256). Review schema:
`status:PASS`, `policy_sha256`, `runtime_source_set_sha256` (canonical JSON digest
of `runtime_sources`), `freeze_manifest_sha256`. Root deployment performs the
separate original queue pause/lock-clearance and live resource review before
removing its stop marker and launching this adapter.
