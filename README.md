# Counterfactual Analysis of Physiological-Artifact-Induced Decision Failures in Motor-Imagery BCI

Reproducible code and results for a study of **why** real physiological interference turns a
correct brain-computer interface decision into a wrong command, and **how much** of that damage
requires the exact clean-by-artifact interaction term.

**Authors:** Bekarys Kydyrbek, Yerbossyn Shaimardan (Grade 11)
**Institution:** Nazarbayev Intellectual School of Physics and Mathematics, Nauryzbay district, Almaty, Kazakhstan
**Supervisor:** Kalima Kabylbek
**Year:** 2026

Documentation is written in Kazakh (see `docs/`). This README is in English.

---

## The question

A motor-imagery BCI reads EEG and decides *left hand* or *right hand*. Chewing contaminates the
EEG and sometimes flips a correct decision into a wrong command.

When contaminated CSP power is expanded, it splits **exactly** into three parts:

```
E[(z + u)²]  =  E[z²]  +  E[u²]  +  2·E[zu]
                 clean    artifact   cross-term
```

**Is the cross-term needed for the decision to flip?**

## The answer

**No, for about 97% of harmful events.** The cross-term is required for only **2.9%** of decision
flips (95% subject-level CI 0.7–4.0%, N = 9).

We also establish **why**: the cross-term is a **zero-mean nuisance term**, not a structured
mechanism. This is shown three independent ways (phase-randomised surrogates, required/prevented
symmetry, and a shuffle test), with a positive control proving the measurement responds to genuine
interaction.

## The obvious objection, and what we did about it

The donor and the host come from different people in unsynchronised recordings, so `E[zu] = 0`
follows algebraically. Is the result physiology, or is it arithmetic that our own design forced?

**We measured it.** In a second dataset the donor and the host come from the same person, in the
same session, on the same montage. The linear cross-term stayed at the chance floor there too.
Independence is therefore **not an artifact of the design**. Details in
[the within-subject branch](#within-subject-branch-cho2017), full numbers in `docs/`.

---

## Headline numbers

### Main branch, cross-subject (BNCI2014_001 + PhysioMotion)

| Result | Value |
|---|---|
| Interaction-required fraction, development | 0.0289 (95% CI 0.0067–0.0398, N = 9) |
| Replication on 6 unopened artifact subjects | 0.0258, preregistered criteria PASS |
| Replication on a separate recording session | 0.0321, preregistered criteria PASS |
| Phase-randomised surrogate arm | 0.0287; paired difference CI covers zero |
| Positive control, β = 0 → 1 | 0.0169 → 0.0546, monotone |
| Riemannian MDM decoder | 0.0164 (95% CI 0.0018–0.0470) |
| Artifact-only decoding accuracy | 0.504 (chance level) |
| Required / prevented symmetry, CSP | 0.987 (Wilcoxon p = 0.43) |
| Shuffle model vs observed | 2.77% median relative discrepancy |
| Closed-form prediction, zero fitted parameters | 4.1% median relative discrepancy |
| **Raw-to-result reproduction accuracy** | **4.3 × 10⁻⁷** |

### Within-subject branch (Cho2017)

| Result | Value |
|---|---|
| Feasibility gate, MI_d ≥ 3.0 | 45 of 47 (threshold 20) |
| Eligible development subjects | 19 of 32 |
| Eligible sealed subjects, opened once | 7 of 15 |
| \|rho\|, own donor vs stranger's donor | 0.0231 vs 0.0230 |
| **Paired difference** | **+0.0001, p = 0.94, n = 23** (+0.0003, p = 0.71, n = 18) |
| **Excess over a phase-randomised floor, arm A** | **−0.0001, 95% CI [−0.0004, +0.0002]** |
| Surrogate covariance preservation | 5.0 × 10⁻¹⁶ |
| Interaction-required, development | 0.0484 [0.0336, 0.0534] |
| Interaction-required, sealed group | 0.0336 [0.0205, 0.0583] |
| Primary endpoint, A minus B symmetry | +0.0942 [−0.1392, +0.1844], p = 0.47 |
| Natural SNR, 47 subjects | median −10.79 dB; 75.9% harsher than the frozen grid, 0% milder |
| Saturation correlation, new cohort | Spearman −0.514, p = 0.024, n = 19 (was −0.50, p = 0.17, n = 9) |
| Preregistered prediction, monotone −7 to −30 dB | **held**, on subjects never opened |
| Preregistered prediction, level below 0.015 at −15 dB | **refuted**, 0.0192 |
| Preregistered criteria on the sealed group | 3 of 5 passed |

---

## Within-subject branch (Cho2017)

**Data.** Cho et al. 2017, GigaScience 6(7):gix034, GigaDB 100295. 52 participants, 64 channels in
the 10-10 system, 512 Hz, BioSemi ActiveTwo. Each participant performed left and right hand motor
imagery **and** six non-task blocks, one of them jaw clenching, **inside the same session**. Same
person, same montage, same impedances, same amplifier.

**The jaw block was identified by physics, not by the publication's prose ordering**, which would
have pointed at slot 4. Slot 4 holds head movement. All five probe subjects agreed on slot 3 on
three independent signatures: maximum 40–100 Hz over 1–4 Hz ratio, spectral flatness 0.695–0.813
against 0.24–0.64 elsewhere, and a high-frequency maximum over temporal sites with a
temporal-to-frontopolar ratio of 2.0–4.4.

**What is measured, precisely.** The linear correlation of CSP projections over a 2.5 s window.
Higher-order dependence is not excluded by this. The measure is the right one nonetheless, because
the cross-term `2·E[zu]` **is** that linear correlation.

**What is not closed.** Jaw and imagery are still separate blocks. This closes the same-subject,
same-session axis. It does not close simultaneity with the task, and no open dataset we know of
provides the simultaneous case.

**Jaw clenching is not chewing.** Same muscle group, different behaviour. It is reported as a new
artifact class, not as a replication of the first.

---

## Ten retractions

Nine of our own results were withdrawn while building the second branch, and a tenth entry records
a check that fired falsely. Each is documented with what was claimed, the mechanism that made it
wrong, the check that caught it, and what survived, in
[`protocol/LEDGER_CHO.md`](protocol/LEDGER_CHO.md).

Highlights:

- **A donor bank declared void in full.** `resample_poly` pads with zeros by default; raw BioSemi
  carries a residual per-channel offset with median 135158 even after common-average referencing,
  and the anti-alias filter rang off that step into the analysis band. Caught by a contradiction:
  MI_d said 43 times the resting muscle power while purity said 1.06 times, and a broadband muscle
  artifact cannot be both.
- **A validity gate that was algebraically incapable of failing.** Folds are class balanced and a
  donor yields one prediction per fold, so artifact-only accuracy is exactly 0.5 whatever the
  prediction is. It returned 0.4999, 0.5002 and 0.4998 and "passed". Not repairable; replaced.
- **A post-hoc regularity refuted by our own preregistered test.** We observed that the saturation
  law keeps its shape and changes only its scale, ratio 3.54 with sd 0.27. Criterion C4 tested that
  on subjects nobody had opened: 2.47. Two random subgroups of one dataset differ by 32%. The claim
  was withdrawn.

The scripts that produced the void results are still in the repository, marked in their headers. A
ledger whose evidence has been deleted is not a ledger.

---

## Reproducing everything from raw data

Every number above can be regenerated from public data. Nothing in this repository depends on
private files.

### Data sources

| Dataset | Role | Access |
|---|---|---|
| BNCI2014_001 (BCI Competition IV 2a) | clean motor-imagery trials, 9 subjects | via MOABB |
| PhysioMotion (OpenNeuro ds006386) | real multichannel chewing artifacts, 30 participants | public S3 |
| Cho2017 (GigaDB 100295) | motor imagery **and** jaw clenching from the same person, 52 participants | via MOABB |

No human subjects were recruited for this study. All three datasets are public and de-identified.

### Run order, main branch

Each script in `notebooks/` is a **single self-contained Kaggle cell**.

```
stage_sminus1_rebuild_bank.py     247-donor bank from raw PhysioMotion. 1.5-3 h, resumable.
stage_s0_context.py               Clean CSP+LDA decoders, verified against the frozen ledger. ~25 min.
stage_s1_crossterm_null.py        Surrogate null control + positive control. ~40 min.
stage_s2_mdm.py                   Riemannian MDM cross-decoder falsification. ~20-60 min.
stage_s3_report.py                Tables, subject bootstrap, figures. ~5 min.
stage_s4_saturation_and_claims.py Saturation analysis, symmetry, poster claims. ~1 min.
stage_s5_artifact_only_and_model.py Artifact-only control, independence model. ~1 min.
stage_s6_law_transfer.py          Preregistered transfer of the closed-form law. ~5 min.
stage_t0_ocular_feasibility.py    Ocular donor probe. Negative result, see section 9.4.
```

### Run order, within-subject branch

`notebooks_cho/RUN_ORDER.md` gives the status of every file. A minimal reproduction runs
`t1c`, `t2d`, `t3`, `t3b`, `t3c`, `t3de`, `t4b`, `t5`. The superseded stages exist so the ledger
can be checked, not because they need rerunning.

### Environment

```
mne 1.12.1, moabb 1.5.0, numpy 2.0.2, scipy, scikit-learn, pandas, matplotlib, joblib, boto3
```

See `protocol/requirements.txt`.

---

## What makes this reproducible rather than merely open

The pipeline was rebuilt from scratch on a different library stack and reproduced the frozen values:

```
clean decoder OOF accuracy vs frozen ledger      max deviation 1.9e-07
interaction-required fraction vs frozen ledger   max deviation 4.3e-07
donor count                                      247 (expected 247)
artifact subjects                                24  (expected 24)
donor safety RMS maximum                         97.9447 µV (frozen 97.945)
```

Numerical guarantees enforced at runtime, with the run aborting on failure:

```
fast CSP engine vs actual MNE CSP                1.1e-14
covariance identity C(x+λa) decomposition        1.5e-15
surrogate covariance preservation, main branch   1.3e-15
surrogate covariance preservation, Cho branch    5.0e-16
harm engine, SNR convention                      3.55e-15
harm engine, decomposition vs actual waveform    2.60e-15
```

**The harm engine of the second branch was cross-checked against this repository's own `harmkit`**
on identical synthetic inputs: every counter matched exactly across four SNR grids and five seeds,
and the scale factors matched `harmkit.scale_for_snr` with a difference of exactly 0.0. It also
self tests on import, including a **non-vacuity check** that fails the test outright if no harm
event was produced, because the other checks would then have compared zero with zero.

---

## Guards built into the code

These are not comments. They abort the run.

- **Leakage guard.** `/kaggle/input` is scanned for filenames containing `locked`, `stage5`,
  `stage6`, `1test`, `contam`. Any hit stops execution.
- **Provenance guard.** Every donor is checked against the frozen development subject list.
- **Ledger regression guard.** Rebuilt decoders are compared against the frozen accuracy table.
- **Engine equivalence guard.** The fast analytic CSP engine is checked against real MNE CSP on
  every run; a mismatch above 1e-9 aborts.
- **Covariance identity guard.** The cross-covariance is built by an **independent formula**, and
  `C_full` is computed from the actual composed signal `x + λa`. It is never defined as
  `full − no_interaction`, which would make the check tautological.
- **Preregistration guard.** Every stage of the second branch prints the sha256 of its
  preregistration file and refuses to run without it.
- **One-shot guard.** The confirmation stages write a result file and refuse to run again if it
  exists. A later refusal is the protection working, not a failed run.

---

## Preregistration

Success criteria were written to disk **before** the held-out subjects were opened, in both
branches. Commit order in git is the evidence.

```
results/S1/S1_PREREGISTRATION.json          main branch
results/S2/S2_PREREGISTRATION.json
results/S6/S6_PREREGISTRATION.json
protocol/T2_PREREGISTRATION.json            within-subject branch
protocol/T3_PREREGISTRATION.json
protocol/T3_PREREGISTRATION_AMENDMENT.json
protocol/T3_PREREGISTRATION_AMENDMENT_2.json
protocol/T5_PREREGISTRATION.json
```

Both amendments were written **after** their stage had run and **before** any harm statistic existed
anywhere in the second branch. Each records what changed and why.

---

## Negative results

Published, not hidden.

| What failed | Evidence |
|---|---|
| Artifact-aware safety gate | Median HARM Risk@90: dirty confidence 0.3691 vs best gate 0.3698. Personalised variant improved 3/9 subjects, mean change −0.000882. |
| Explicit geometric features | Median ΔAUC = −0.00008, Wilcoxon p ≈ 0.99 against a strong baseline. |
| Synthetic EMG branch | Produced 13.8–49% harm rates, then **invalidated by our own falsification test**: the invented topographies encoded the class. All results discarded. |
| Ocular donors | 0 of 9 subjects gain 8–32 Hz energy during eye movement; the spectrum lies below 8 Hz. Measured, not assumed. |
| AI-generated experiment code | Rejected by three verifiers before execution; 10 defects found, including a tautological covariance identity check. |
| Two of five preregistered criteria on the sealed Cho group | C1 and C4 failed, with numbers. See `results_cho/T5_RESULT.json`. |

The real chewing artifact passes the same falsification test that killed the synthetic branch:
artifact-only decoding accuracy is 0.504, i.e. chance.

---

## Repository layout

```
notebooks/          main branch, nine self-contained Kaggle cells in run order
notebooks_cho/      within-subject branch, thirteen stages plus the harm engine
harmkit/            the measurement library, reusable with any decoder and donor bank
examples/           acceptance test that reproduces the nine frozen values
protocol/           frozen constants, preregistrations, checksums, the retraction ledger
results/            main-branch tables, JSON results, preregistration files
results/figures/    eleven figures
results_cho/        within-subject-branch results
logs_cho/           raw stdout of every stage, as run
docs/               documentation in Kazakh
docs_kz/            Kazakh drafts of the sections the second branch changed
```

---

## Honest limitations

Read these before citing anything.

1. **N = 9 BCI subjects** and 24 artifact-donor subjects in the main branch, plus 26 subjects with
   computed harm in the within-subject branch. Still a small cohort.
2. **Two artifact classes**, chewing and jaw clenching. Not "muscle artifacts" in general.
3. **Binary task**, decoder band 8–32 Hz.
4. **Independence of z and u is imposed by the design in the main branch. It was then tested by
   measurement.** In Cho2017 the design does not impose it, and the linear cross-term still sits at
   the chance floor: between-arm difference +0.0001 (p = 0.94, n = 23), excess over a
   phase-randomised floor −0.0001, 95% CI [−0.0004, +0.0002]. Independence is not an artifact of
   the design. **Still open:** simultaneity with the task.
5. **Millions of counterfactual pairs are not millions of experiments.** They are computational
   realisations nested inside a small number of people. The statistical unit is the subject.
6. **NO-INTERACTION ≠ ARTIFACT-ONLY.** Only the cross-term is removed.
7. **In the cross-session confirmation**, subjects 2, 5 and 7 have clean accuracy near 51%.
8. **The saturation effect.** Significant in the MDM family (ρ = −0.90, p = 0.0009, n = 9) and not
   in CSP at n = 9 (ρ = −0.50, p = 0.17). It **replicated and reached significance** on the larger
   cohort: ρ = −0.514, p = 0.024, n = 19, and −0.413, p = 0.036 pooled over 26 independent
   subjects. The coefficient agrees to the second decimal on a different dataset and a different
   artifact class. The curve's **quantitative parameters do not transfer** between cohorts; see
   ledger entry R7.
9. **The cross-decoder subject pattern does not transfer** (Spearman ρ = 0.000).
10. **In MDM the required/prevented ratio is asymmetric** (1.60). A method limitation.
11. **The independence model is post-hoc**, devised after seeing the S1 result.
12. **In the within-subject branch, arm C failed to discriminate**, and this is a defect in the
    control's design: matching SNR across arms removes the very difference the control was meant to
    detect. See ledger entry R8.
13. **Donors per host range from one to three** in the within-subject branch, capped by the 10 s
    length of the jaw block. Subjects with a single donor have noisy counters.
14. **Novelty is compositional.** Each ingredient exists in prior work. What is new is the
    combination and the quantitative calibration.

---

## Prior work

Six closest works, found by manual search, each with the difference stated.

1. Frølich L. et al. *Investigating effects of different artefact types on motor imagery BCI.*
   EMBC 2015. PMID 26736664. — Measures accuracy degradation, not per-decision flips.
2. Yin K. et al. *Pioneering EEG Motor Imagery Classification Through Counterfactual Analysis.*
   arXiv:2312.09456. — Artificial patch swapping for explainability; no artifact.
3. Zhang X., Wu D. et al. *Tiny noise, big mistakes.* National Science Review 2020.
   arXiv:2001.11569. — Deliberately optimised adversarial perturbations; ours are physiological.
4. Alfeo A.L. et al. Neural Computing and Applications 2025. DOI 10.1007/s00521-025-11300-8.
   — Minimal feature-space edits for explainability.
5. *SSDLabeler.* arXiv:2512.05500. — Semi-synthetic injection to train artifact detectors.
6. *Statistical Mixture-of-Experts for EMG artifact removal.* arXiv:2509.19385. — Goal is removal.
   Also confirms that −7 to +2 dB is the conventional SNR range for EMG-EEG work, which is why our
   frozen grid uses it. The within-subject branch measures that the natural range is harsher.

Cho et al. 2017 is a **data source**, not a competing method: it does not study decision-level harm.

---

## Data availability

- BNCI2014_001: <https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html>
- PhysioMotion ds006386: <https://doi.org/10.18112/openneuro.ds006386.v1.0.1>
  Paper: <https://doi.org/10.1038/s41597-026-07120-7>
- Cho2017: <https://doi.org/10.5524/100295>
  Paper: <https://doi.org/10.1093/gigascience/gix034>

The derived donor banks are **not** in this repository. They are large and fully reproducible from
`stage_sminus1_rebuild_bank.py` and `stage_t2d_rebuild_both_bands.py`.

## License

Code released under the MIT License. The underlying datasets remain under their own licenses.

## Contact

Open an issue in this repository.
