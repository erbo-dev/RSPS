# Counterfactual Analysis of Physiological-Artifact-Induced Decision Failures in Motor-Imagery BCI

Reproducible code and results for a study of **why** real physiological interference turns a correct
brain-computer interface decision into a wrong command, and **how much** of that damage requires the
exact clean-by-artifact interaction term.

**Authors:** Bekarys Kydyrbek, Yerbossyn Shaimardan (Grade 11)
**Institution:** Nazarbayev Intellectual School of Physics and Mathematics, Nauryzbay district, Almaty, Kazakhstan
**Supervisor:** Kalima Kabylbek
**Year:** 2026

Documentation is written in Kazakh (see `docs/`). This README is in English.

---

## The question

A motor-imagery BCI reads EEG and decides *left hand* or *right hand*. Chewing contaminates the EEG
and sometimes flips a correct decision into a wrong command.

When contaminated CSP power is expanded, it splits **exactly** into three parts:

```
E[(z + u)²]  =  E[z²]  +  E[u²]  +  2·E[zu]
                 clean    artifact   cross-term
```

**Is the cross-term needed for the decision to flip?**

## The answer

**No, for about 97% of harmful events.** The cross-term is required for only **2.9%** of decision
flips (95% subject-level CI 0.7–4.0%, N = 9).

We also establish **why**: the artifact donor and the host signal come from different people, so the
cross-term is a **zero-mean nuisance term**, not a structured mechanism. This is shown three
independent ways (phase-randomised surrogates, required/prevented symmetry, and a shuffle test),
with a positive control proving the measurement responds to genuine interaction.

---

## Headline numbers

| Result | Value |
|---|---|
| Interaction-required fraction, development | 0.0289 (95% CI 0.0067–0.0398, N=9) |
| Replication on 6 unopened artifact subjects | 0.0258, preregistered criteria PASS |
| Replication on a separate recording session | 0.0321, preregistered criteria PASS |
| Phase-randomised surrogate arm | 0.0287; paired difference CI covers zero |
| Positive control, β = 0 → 1 | 0.0169 → 0.0546, monotone |
| Riemannian MDM decoder | 0.0164 (95% CI 0.0018–0.0470) |
| Artifact-only decoding accuracy | 0.504 (chance level) |
| Required / prevented symmetry, CSP | 0.987 (Wilcoxon p = 0.43) |
| Shuffle model vs observed | 2.77% median relative discrepancy |
| Closed-form prediction, zero fitted parameters | 4.1% median relative discrepancy |
| Closed-form law, CSP: median relative error | 5.7% over 72 cells, zero fitted parameters |
| Closed-form law, MDM: preregistered transfer test | 15.0%, Spearman +0.987, criterion PASS |
| **Raw-to-result reproduction accuracy** | **4.3 × 10⁻⁷** |

---

## Reproducing everything from raw data

Every number above can be regenerated from public data. Nothing in this repository depends on
private files.

### Data sources

| Dataset | Role | Access |
|---|---|---|
| BNCI2014_001 (BCI Competition IV 2a) | clean motor-imagery trials, 9 subjects | via MOABB, downloaded automatically |
| PhysioMotion (OpenNeuro ds006386) | real multichannel chewing artifacts, 30 participants | public S3, downloaded automatically |

No human subjects were recruited for this study. Both datasets are public and de-identified.

### Run order

Each script in `notebooks/` is a **single self-contained Kaggle cell**. Paste it into a new notebook
and run.

```
notebooks/stage_sminus1_rebuild_bank.py    Rebuild the 247-donor development bank
                                           from raw PhysioMotion. 1.5-3 h, resumable.
notebooks/stage_s0_context.py              Rebuild the clean CSP+LDA decoders and verify
                                           them against the frozen ledger. ~25 min.
notebooks/stage_s1_crossterm_null.py       Surrogate null control + positive control. ~40 min.
notebooks/stage_s2_mdm.py                  Riemannian MDM cross-decoder falsification. ~20-60 min.
notebooks/stage_s3_report.py               Tables, subject bootstrap, figures. ~5 min.
notebooks/stage_s4_saturation_and_claims.py  Saturation analysis, symmetry, poster claims. ~1 min.
notebooks/stage_s5_artifact_only_and_model.py  Artifact-only control, independence model. ~1 min.
notebooks/stage_s6_law_transfer.py         Preregistered cross-family test of the closed-form law. ~10 min.
notebooks/stage_t0b_ocular_feasibility_fixed.py  Feasibility gate for a same-subject ocular donor. ~5 min.
```

S-1 produces a donor bank; attach its output to S0. S0 produces a context; attach its output to
S1 and S2. S3, S4 and S5 read the outputs of S1 and S2.

### Environment

```
mne     1.12.1
moabb   1.5.0
numpy   2.0.2
scipy, scikit-learn, pandas, matplotlib, joblib, boto3
```

See `protocol/requirements.txt`.

---

## What makes this reproducible rather than merely open

The pipeline was rebuilt from scratch on a different library stack and reproduced the frozen values:

```
clean decoder OOF accuracy vs frozen ledger    max deviation 1.9e-07
interaction-required fraction vs frozen ledger max deviation 4.3e-07
donor count                                    247 (expected 247)
artifact subjects                              24  (expected 24)
rejected episodes                              0   (expected 0)
donor safety RMS maximum                       97.9447 µV (frozen 97.945)
```

Numerical guarantees enforced at runtime, with the run aborting on failure:

```
fast CSP engine vs actual MNE CSP              1.1e-14
covariance identity C(x+λa) decomposition      1.5e-15
surrogate covariance preservation              1.3e-15
SNR convention, target -3 dB                   -3.000000000000 dB
```

---

## Guards built into the code

These are not comments. They abort the run.

- **Leakage guard.** `/kaggle/input` is scanned for filenames containing `locked`, `stage5`,
  `stage6`, `1test`, `contam`. Any hit stops execution.
- **Provenance guard.** Every donor is checked against the frozen development subject list. The six
  locked artifact subjects (12, 14, 17, 19, 23, 25) cannot enter development data.
- **Ledger regression guard.** S0 compares rebuilt decoders against the frozen accuracy table and
  reports per-subject deltas.
- **Engine equivalence guard.** The fast analytic CSP engine is checked against real MNE CSP on every
  run; a mismatch above 1e-9 aborts.
- **Covariance identity guard.** In S2 the cross-covariance is built by an **independent formula**,
  and `C_full` is computed from the actual composed signal `x + λa`. It is never defined as
  `full − no_interaction`, which would make the check tautological.
- **Clean validity gate.** Frozen before contamination: a subject enters the primary median only if
  clean decoder accuracy ≥ 0.60.

---

## Preregistration

Success criteria were written to disk **before** the held-out artifact subjects were opened:

```json
{
  "median interaction-required": "<= 0.05",
  "median no-interaction HARM recall": ">= 0.95",
  "subjects with interaction-required <= 0.05": ">= 7/9",
  "subjects with at least one HARM event": ">= 8/9"
}
```

The negative gate conclusion was frozen at the same time. See `results/S1/S1_PREREGISTRATION.json`
and `results/S2/S2_PREREGISTRATION.json`.

The confirmation stage writes a result file and **refuses to run again** if that file exists. This
prevents accidental reuse of a one-shot confirmation session.

---

## Negative results

Published, not hidden.

| What failed | Evidence |
|---|---|
| Artifact-aware safety gate | Median HARM Risk@90: dirty confidence 0.3691 vs best gate 0.3698. Personalised variant improved 3/9 subjects, mean change −0.000882. |
| Explicit geometric features | Median ΔAUC = −0.00008, Wilcoxon p ≈ 0.99 against a strong baseline. |
| Synthetic EMG branch | Produced 13.8–49% harm rates, then **invalidated by our own falsification test**: the invented topographies encoded the class, so the decoder predicted labels from the artifact alone with extreme confidence. All results discarded. |
| AI-generated experiment code | Rejected by three verifiers before execution; 10 defects found, including a tautological covariance identity check. |
| Ocular donor from the same person | BNCI2014_001 does contain simultaneous same-subject ocular artifact (52 EOG blocks, found only by reading the `.mat` directly, since MOABB silently drops runs without trials). Measured: the eye-movement block adds **no** energy in the decoder band. In 0/9 subjects does it exceed the resting block at 8–32 Hz, because ocular spectra sit below 8 Hz. The dependence axis therefore cannot be extended on open data for a mu-beta decoder, for a physical reason. |

The real chewing artifact passes the same falsification test that killed the synthetic branch:
artifact-only decoding accuracy is 0.504, i.e. chance.

---

## harmkit: the measurement as a reusable tool

The instrument is packaged so that anyone can point it at their own decoder
and their own artifact bank:

```python
from harmkit import ArtifactBank, PowerDecoder, HarmBenchmark

result = HarmBenchmark(decoder, bank, snr_db=[2, 0, -2, -4, -7]).run(
    X_clean, clean_reference_rms=4.4, subject_ids=subj)
print(result.summary())
```

Three decoder adapters are provided. `PowerDecoder` and `CovarianceDecoder`
support an exact interaction ablation; `DirectDecoder` works with any model
including neural networks, but deliberately raises rather than returning an
approximate ablation, because no exact waveform lacking only the cross-term
exists. The package ships the three controls that can invalidate a result,
including the artifact-only test that invalidated an earlier version of this
study.

`examples/reproduce_frozen_result.py` reproduces every frozen per-subject
value through the public API to better than 1e-6. See `harmkit/README.md`.

## Repository layout

```
harmkit/             the measurement packaged as a reusable library
examples/            acceptance test reproducing the frozen values
notebooks/           seven self-contained Kaggle cells, in run order
protocol/            frozen constants, requirements, checksums
results/             all output tables, JSON results, preregistration files
results/figures/     eleven figures
docs/                documentation in Kazakh, glossary, supervisor brief
```

---

## Honest limitations

Read these before citing anything.

1. **N = 9 BCI subjects**, 24 artifact-donor subjects. Small cohort.
2. **One artifact class** (chewing, 20–80 Hz component). Not "muscle artifacts" in general.
3. **Binary task**, single decoder band 8–32 Hz.
4. **Independence of z and u is imposed by the design**, not a property of nature. No public dataset
   provides simultaneous motor imagery and chewing from the same person. Our conclusion is therefore
   conditional. The positive control quantifies where reality sits: real contamination has
   |ρ| = 0.036, while the interaction becomes substantial around |ρ| = 0.133, roughly four times higher.
5. **Millions of counterfactual pairs are not millions of experiments.** They are computational
   realisations nested inside a small number of people. The statistical unit is the subject, and
   confidence intervals come from subject-level bootstrap.
6. **NO-INTERACTION ≠ ARTIFACT-ONLY.** Only the cross-term is removed; the clean contribution stays.
7. **In the cross-session confirmation**, subjects 2, 5 and 7 have clean accuracy near 51%, so their
   contribution to the claim is weak.
8. **The saturation effect** is significant in the MDM family (ρ = −0.90, p = 0.0009, n = 9) but not
   in CSP (ρ = −0.50, p = 0.17, n = 9). Pooling the two into n = 18 would be pseudoreplication,
   because both come from the same nine subjects. We report them separately.
9. **The cross-decoder subject pattern does not transfer** (Spearman ρ = 0.000). Magnitude transfers,
   per-subject vulnerability does not.
10. **In MDM the required/prevented ratio is asymmetric** (1.60). `C_x + λ²C_a` is not the covariance
    of any real signal, so the ablation may bias AIRM distances. This is a method limitation.
11. **The closed-form law is post-hoc in origin**, devised after seeing the S1 result. The
    cross-family transfer test (S6) is preregistered, but the law itself is not. In the Riemannian
    family the law leaves a systematic residual beyond counting noise (median |z| = 2.08 versus
    0.88 in CSP), consistent with the documented asymmetry of the ablation in that geometry.
12. **Novelty is compositional.** Each individual ingredient exists in prior work. What is new is the
    combination and the quantitative calibration.

---

## Prior work

Six closest works, found by manual search, each with the difference stated.

1. Frølich L. et al. *Investigating effects of different artefact types on motor imagery BCI.*
   EMBC 2015. PMID 26736664. — Measures accuracy degradation, not per-decision flips.
2. Yin K. et al. *Pioneering EEG Motor Imagery Classification Through Counterfactual Analysis.*
   arXiv:2312.09456. — Artificial patch swapping for explainability; no artifact.
3. Zhang X., Wu D. et al. *Tiny noise, big mistakes.* National Science Review 2020. arXiv:2001.11569.
   — Deliberately optimised adversarial perturbations; ours are physiological and unoptimised.
4. Alfeo A.L. et al. Neural Computing and Applications 2025. DOI 10.1007/s00521-025-11300-8.
   — Minimal feature-space edits for explainability.
5. *SSDLabeler.* arXiv:2512.05500. — Semi-synthetic injection to train artifact detectors, not to
   study decision flips.
6. *Statistical Mixture-of-Experts for EMG artifact removal.* arXiv:2509.19385. — Goal is removal.
   Also confirms that −7 to +2 dB is the conventional SNR range for EMG-EEG work, which is why our
   frozen grid uses it.

---

## Data availability

- BNCI2014_001: https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html
- PhysioMotion ds006386: https://doi.org/10.18112/openneuro.ds006386.v1.0.1
  Paper: https://doi.org/10.1038/s41597-026-07120-7

## License

Code released under the MIT License. The underlying datasets remain under their own licenses.

## Contact

Open an issue in this repository.
