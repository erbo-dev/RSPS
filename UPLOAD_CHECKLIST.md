# Upload checklist for github.com/erbo-dev/RSPS

Do these in order. The order of commits is itself evidence, so do not
batch-commit everything at once.

## 1. Preregistrations first, in their original order

```
git add protocol/T2_PREREGISTRATION.json
git commit -m "prereg: Cho2017 bank, cohort split, feasibility criterion"

git add protocol/T3_PREREGISTRATION.json
git commit -m "prereg: analysis window, purity threshold, arms, saturation prediction"

git add protocol/T3_PREREGISTRATION_AMENDMENT.json
git commit -m "prereg amendment 1: retract the degenerate artifact-only gate, fix donor matching"

git add protocol/T3_PREREGISTRATION_AMENDMENT_2.json
git commit -m "prereg amendment 2: retract cross-scale rho comparisons, require a surrogate floor"

git add protocol/T5_PREREGISTRATION.json
git commit -m "prereg: replication criteria for the sealed subjects"
```

## 2. Then the ledger and the constants

```
git add protocol/LEDGER_CHO.md protocol/FROZEN_CONSTANTS_CHO.json
git commit -m "protocol: nine retractions with mechanisms; frozen constants for the Cho branch"
```

## 3. Then the code

```
git add notebooks_cho/
git commit -m "cho branch: thirteen stages, harm engine, run order and per-file status"
```

## 4. Then the results

```
git add results_cho/
git commit -m "results: T3c, T3de, T4b, T5"
```

## 5. Then the README

Append `README_ADDITION.md` to `README.md`, then:

```
git add README.md && git commit -m "README: within-subject branch"
```

## 6. Regenerate the checksums

Your `protocol/CHECKSUMS.md` lists sha256 of every file. Regenerate it so the
new files are covered, and commit that last.

```bash
find notebooks_cho protocol results_cho -type f | sort | \
  xargs sha256sum > protocol/CHECKSUMS_CHO.txt
git add protocol/CHECKSUMS_CHO.txt && git commit -m "checksums: Cho branch"
```

## Before you push, check three things

1. **Do not delete** the superseded stages `t1`, `t1b`, `t2`, `t2b`, `t2c`. The
   ledger cites them. Their headers say they are superseded.
2. **Do not upload the bank** `cho_bank_v3`. It is derived data, roughly 1.5 GB,
   and it is fully reproducible from `stage_t2d_rebuild_both_bands.py`. Say so
   in the README instead.
3. **Check the Cho citation** by opening doi.org/10.1093/gigascience/gix034 and
   confirming volume, issue and article id before it appears in the document.
