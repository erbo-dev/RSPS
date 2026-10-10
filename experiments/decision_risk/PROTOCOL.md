# Exploratory decision risk experiment

This is a new exploratory extension of RSPS, not a replication or a replacement
for any previously frozen result. Protocol authored on 2026-10-10 against base
commit 05d0c24c50b1bba4a660dc74f06ab286f85609d9, before new risk outcomes.
The existing sealed groups have already been opened in the original study; this
experiment does not describe them as unseen and does not use them for tuning.

## Question

Can the observed 40–80 Hz signal estimate the artifact contribution to individual
8–32 Hz CSP powers, and does that estimate rank risky commands better than the
absolute margin of the same fixed decoder?

## Fixed data and splits

- BNCI2014_001: subjects 1–9, training session AxxT only. No AxxE downloads.
- PhysioMotion: the 24 development subjects from the original project; the
  original six excluded subjects remain excluded. Select at most the first two
  metadata rows per subject, sorted by run and episode, before signal outcomes.
- All data derived from one EEG trial stay in the same EEG fold. Artifact folds
  partition people, not donor windows. All SNR variants stay with their source.
- Six rotating EEG folds: one run for evaluation, the next for gate calibration,
  and the remaining four for fitting the clean CSP+LDA decoder. Three artifact
  folds: two thirds of donor people for the power mapping and calibration,
  the other third exclusively for evaluation.
- SNR: +2, 0, −3, −7 and −15 dB, using the original fixed 4.4 microvolt reference
  and the donor 20–80 Hz RMS. Also evaluate uncontaminated trials separately.
- Every pair adds the same scaled real donor to the decoder and high-frequency
  branches. High-frequency features must include the host signal, not the
  isolated donor. Truth, clean evaluation powers, donor identity and SNR are
  unavailable to the proposed scoring method at evaluation.

## Candidates and controls

1. Absolute dirty decoder margin: primary strong baseline.
2. Negative mean observed high-frequency power: artifact intensity baseline.
3. Logistic error risk from absolute margin and total high-frequency power:
   simple learned baseline.
4. Proposed mapping: for each fitted CSP component, estimate the low/high donor
   power ratio from training donor people. Multiply that ratio by excess observed
   high-band component power over a clean training reference. Subtract this
   estimated additive artifact power from observed decoder-band powers, with a
   floor derived from clean training data, and recompute the fixed LDA score.
   Fit a logistic command-error score using absolute margin, total high power,
   estimated score displacement, sign disagreement, estimated relative artifact
   power and the fraction of components requiring clipping. This is an estimate,
   not an exact clean reconstruction; cross-terms and mapping error remain.
5. A true-artifact-power version of candidate 4 is a diagnostic oracle, clearly
   marked unavailable in deployment. It tests whether perfect additive power
   information would help; it cannot establish the practical method works.

No neural network search, no data-driven hyperparameter sweep. Logistic C=1,
standardised inputs, fixed seed 20261010. A confidence-only learned baseline is
included to distinguish ordinary calibration from added information.

## Metrics and interpretation

- Equal empirical coverage risk curves at 50%, 70%, 90% and 100% within each
  evaluation split. These use ranks without evaluation labels; they describe
  matched coverage and are not an online threshold guarantee.
- A threshold fixed at the calibration 90% score quantile, with evaluation risk
  and actual achieved coverage, reported separately.
- Both all-command error and HARM (clean correct, dirty wrong) are reported.
  Correct rejected commands and uncontaminated rejection are reported.
- Subject-level paired differences, bootstrap confidence intervals, and paired
  tests, rather than treating pairs as independent people. Multiple candidates
  are exploratory; no statistically confirmed winner is selected by the best
  p-value. Report all candidates, donor folds and SNR levels.
- A practical candidate requires consistent improvement over BOTH margin and
  confidence-plus-high-power controls, including the stronger contamination
  level, without presenting an oracle result as deployable evidence.
- A small pilot or a result on reused public development participants cannot
  establish clinical safety, universal independence, or scientific novelty.

## Integrity checks and stop conditions

Stop on invalid EDF ranges, missing COMMON19 channels, non-finite or flat sensor
data, incorrect units, overlapping calibration/evaluation runs or donor people,
or disagreement between analytical CSP powers and direct composed waveforms.
Check donor RMS against the selected original metadata and report drift; do not
silently replace the original bank. Compare a standard clean leave-one-run-out
decoder with the frozen ledger before interpreting extension outcomes. The new
four-run decoders need not reproduce the five-run ledger.

If public sources are unreachable or no raw arrays are available, write an
explicit blocked execution report. Existing aggregate tables and synthetic
controls are not substitutes for running the proposed EEG experiment.
