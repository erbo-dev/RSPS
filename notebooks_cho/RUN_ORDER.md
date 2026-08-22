# Cho2017 within-subject branch: run order and status

Every script is a single self-contained Kaggle CPU cell. Paste and run.
Superseded and retracted stages are **kept on purpose**: the retraction ledger
in `protocol/LEDGER_CHO.md` refers to them, and a ledger whose evidence has
been deleted is not a ledger.

## Run order

| # | Script | Status | Needs | Cost |
|---|---|---|---|---|
| 1 | `stage_t1_cho2017_feasibility.py` | superseded | internet | minutes |
| 2 | `stage_t1b_identify_and_gate.py` | superseded | cached .mat | minutes |
| 3 | `stage_t1c_diagnostic.py` | **current** | cached .mat | minutes |
| 4 | `stage_t2_build_cho_bank.py` | superseded, bank v1 **void** | prereg T2 | ~1.5 h, 10.6 GB |
| 5 | `stage_t2b_natural_snr.py` | **retracted** | bank v1 | minutes |
| 6 | `stage_t2c_rebuild.py` | superseded, bank v2 | prereg T2 | ~1.5 h |
| 7 | `stage_t2d_rebuild_both_bands.py` | **current**, bank v3 | prereg T2 | ~1.5 h |
| 8 | `stage_t3_decoders.py` | **current** (its gate retracted) | bank, prereg T3 | minutes |
| 9 | `stage_t3b_rho.py` | **current** | models, amendment 1 | minutes |
| 10 | `stage_t3c_surrogate_floor.py` | **current** | models, amendment 2 | minutes |
| 11 | `stage_t3de_snr_and_cohort.py` | **current** | bank v3 | minutes |
| 12 | `stage_t4b_harm_sweep.py` | **current** | bank v3, models | minutes |
| 13 | `stage_t5_locked_confirmation.py` | **current**, one-shot | bank v3, prereg T5 | minutes |
| — | `stage_resume_rebuild.py` | utility | — | as needed |
| — | `harm_engine.py` | library | — | — |

A minimal reproduction runs 3, 7, 8, 9, 10, 11, 12, 13. Stages 1, 2, 4, 5, 6
exist so the ledger can be checked, not because they need rerunning.

## Preregistration order, which is the part that matters

Commit order in git is the evidence. Each file was committed before the stage
that reads it, and each stage prints the sha256 of its preregistration and
refuses to run without it.

```
protocol/T2_PREREGISTRATION.json                 -> before stage 4
protocol/T3_PREREGISTRATION.json                 -> before stage 8
protocol/T3_PREREGISTRATION_AMENDMENT.json       -> before stage 9
protocol/T3_PREREGISTRATION_AMENDMENT_2.json     -> before stage 10
protocol/T5_PREREGISTRATION.json                 -> before stage 13
```

Both amendments were written **after** their stage had run and **before** any
harm statistic existed anywhere in the branch. Each records what changed and
why. Stage 12 is the first stage in this branch that computes a harm number.

## Session protection

`stage_t5_locked_confirmation.py` writes `T5_RESULT.json` and refuses to run
again if that file exists. A later refusal is the protection working, not a
failed run. This mirrors Stage 6 of the BNCI branch.

Kaggle discards `/kaggle/working` when a session drops and keeps it only when
the notebook was committed. Save Version, then create a Dataset from the
output, then attach that Dataset downstream. `stage_resume_rebuild.py` exists
because this lesson was learned the expensive way, and it verifies the rebuild
against recorded values before letting anything continue.

## Instrument verification

`harm_engine.py` self tests on import in stages 12 and 13. Five checks: SNR
convention to 3.55e-15, algebraic power decomposition against the actual
contaminated waveform to 2.60e-15, the vectorised path against an independent
triple loop, zero harm at lambda = 0, and a non-vacuity check that fails the
test if no harm or no interaction-required event was produced, because then the
first four checks compared zero with zero.

It was additionally cross-checked against this repository's own `harmkit`:
identical counters across four SNR grids and five seeds, and scale factors
identical to `harmkit.scale_for_snr` with a difference of exactly 0.0.
