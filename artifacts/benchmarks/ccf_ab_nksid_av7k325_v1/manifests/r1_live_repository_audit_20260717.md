# R1 live repository and license audit: 2026-07-17

## Result and boundary

All five unique external checkouts were inspected live rather than trusted solely
from the archived source-audit JSON. Every checkout remains at its registered
commit and official remote, and every tracked author-source tree has zero staged
or unstaged modifications. This proves source-version integrity only; it does not
prove a paper result has been reproduced.

## Live checkout state

| Checkout / paper IDs | Live HEAD | Official remote | Tracked modifications | Untracked state |
|---|---|---|---:|---|
| `reference/HW-PR-NAS` / `hw_pr_nas_2023` | `296c6576fbae2b277e56c704ff3b6e648ec4c2be` | `https://github.com/IHIaadj/HW-PR-NAS.git` | 0 | 3 Python bytecode cache files |
| `reference/_local/SURE` / `sure_2024` | `5ce0193bc93e73b1c7f1f53aeda8854e997011e2` | `https://github.com/Intellindust-AI-Lab/SURE.git` | 0 | 7 Python bytecode cache files |
| `reference/HARP` / `harp_2023` | `c8bffd9411917b125846429b4d6be4f21c7a7165` | `https://github.com/UCLA-VAST/HARP.git` | 0 | 3 Python bytecode cache files |
| `reference/ESDA` / `esda_2024` | `b75c8c93ca258158c06a6434f5f0f084add02ee5` | `https://github.com/CASR-HKU/ESDA.git` | 0 | clean |
| `reference/_local/Sonar-OLTR` / `dmcl_sonar_oltr_2025`, `plud_sonar_oltr_2024` | `eea8dc07ce007988150ac208cd09e00daedba2ca` | `https://github.com/gmgslinyu/Sonar-OLTR.git` | 0 | isolated archive extraction; 82 untracked files |

The untracked files are runtime/extraction by-products, not changes to tracked
author code. They are deliberately left untouched during the active source-frozen
run. Their presence must not be summarized as a clean external worktree, but it
also must not be misreported as author-source modification.

## License and archive evidence

| Repository | Live tracked license evidence | SHA256 / state |
|---|---|---|
| HW-PR-NAS | `LICENSE` | `5f89424986edb716ba4040af41626a8471120e1a625c6a463ea1b271d685fa98` / MIT verified |
| HARP | `LICENSE` | `2d633a2c625be312afeb6f660bfa12b9dd4ab8b051ee96c04f38ab91abc21912` / BSD-3-Clause verified |
| ESDA | `LICENSE` | `07fd61b75f13681e7b46355e8f70314f53ffca846fd1f02f7829cd735b2cdded` / MIT verified |
| SURE | no tracked license file | redistribution unverified; isolated only |
| Sonar-OLTR | no tracked license file | redistribution unverified; isolated only |

The isolated Sonar archive remains 6,359,379 bytes with SHA256
`4bd5158c491821bb1de3138856344949ca3ce1747f033601809d30774e7d5a61`.
Both `reference/_local/SURE` and `reference/_local/Sonar-OLTR` are ignored by the
top-level rule `.gitignore:12:/reference/_local/`, so their source is not
redistributed by this project repository.

## Superproject integration gap

The working `.gitmodules` file contains declarations for HW-PR-NAS, HARP and ESDA,
but the top-level index does not yet contain their gitlink entries. At the
superproject level the three directories are still untracked, and `git submodule
status -- reference/HW-PR-NAS reference/HARP reference/ESDA` cannot report them.

Therefore the exact state is:

- external repositories are downloaded, independently version-pinned, remote-
  verified and tracked-source clean;
- their audit cards and local execution paths are usable;
- formal superproject submodule integration is incomplete.

Do not describe this state as "submodules fully integrated". After the active G1
software chain closes, reconcile `.gitmodules` with the three gitlinks, verify
`git submodule status` reports the registered commits, create a new source freeze,
and rerun R1/T1 generation. No staging or cleanup is performed by this live audit.

## Method-specific evidence limits

- HW-PR-NAS remains `BLOCKED_OFFICIAL_CODE_INCOMPLETE`; the pinned repository lacks
  README-referenced surrogate/predictor files and cannot support an author-code
  reproduction claim.
- SURE source is present and tracked-source clean, but its missing license keeps
  redistribution unverified.
- DMCL source presence is bound to the isolated archive, but paper-to-code
  correspondence remains partial; PLUD correspondence is verified within the same
  archive.
- HARP and ESDA source integrity does not substitute for the required local HLS,
  route, board or power measurements.
