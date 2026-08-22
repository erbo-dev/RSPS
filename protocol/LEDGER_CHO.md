# Retraction ledger, Cho2017 within-subject branch

Nine of our own results were retracted while building this branch. Each entry
records what was claimed, the mechanism that made it wrong, the check that
caught it, and what survived. Nothing here has been deleted from the
repository: the scripts that produced the void results are still in
`notebooks_cho/`, marked in their headers.

This is not a list of failures. It is the evidence that the procedure finds
errors, which is the only reason to believe the numbers it did not reject.

Provenance matters and is stated per entry. R1 is an error our own method
found in our own work. R2, R3, R4 and R6 are computational errors. R5, R7, R8
and R9 are defects in instrumentation added late in the project.

---

## R1. A ratio-based gate, wrong twice

**Claimed.** Jaw clenching must exceed rest in the 8-32 Hz band; threshold
ratio 1.5 in at least 4 of 5 subjects.

**Result.** 2 of 5. Gate failed.

**Mechanism.** The 8-32 Hz band holds alpha, which falls during any task, and
muscle, which rises. The ratio is the net of two opposing effects, not a
measurement of muscle.

**Evidence.** Subject 3 gave a ratio of 0.99, so the gate said "no muscle". The
same subject carried 1.42e5 of 40-100 Hz power against 1.8e4 to 2.8e4 in the
other four blocks, five to eight times higher, with the highest spectral
flatness in the sample at 0.813. They clenched hardest of anyone and the gate
did not see it.

**Replaced by.** MI_d in 40-100 Hz, where no sensorimotor rhythm lives.

**Why it matters most.** The identical error appears in section 9.4 of the
write-up, in the ocular probe, where we found it ourselves. One error, two
independent branches, caught by the procedure both times.

---

## R2. Donor bank v1 declared void

**Claimed.** The donor waveforms in bank v1 are correct.

**Mechanism.** `scipy.signal.resample_poly` defaults to `padtype='constant'`,
which pads with **zeros**. Raw BioSemi carries a residual per-channel offset
with median **135158** even after CAR19. Zero-padding creates a step of that
height at both edges, the anti-alias FIR rings off it, and the ringing lands in
20-80 Hz.

**What caught it.** A contradiction. MI_d was 43, meaning the donor carried 43
times the resting 40-100 Hz power, while purity was 0.053, meaning only 1.06
times the resting power in 20-80 Hz. For a broadband muscle artifact both
cannot be true. A second sign: a subject's three donor windows had RMS values
agreeing to within one per cent despite MI_d differing by three orders of
magnitude, so RMS was tracking the DC offset and not the muscle.

**Numerical confirmation.** With a broadband component 43 times in power,
measured purity came out 0.975 at DC = 0, 0.851 at DC = 2000 and 0.515 at
DC = 5000. `detrend` plus `padtype='line'` restores 0.975 to 0.980 at any DC.

**After the fix.** Purity median 0.914; Spearman between MI_d and purity
**+0.968**, against +0.425 in v1, which is what the physics of a broadband
muscle artifact requires.

**Survived.** The MI_d values, because they are computed by a separate function
that detrends internally. The 45/47 feasibility gate therefore stands.

---

## R3. Natural SNR, first retraction

Computed on the corrupted v1 donors, mechanism R2. Withdrawn entirely.

---

## R4. Natural SNR, second retraction

**Claimed.** Natural SNR median -9.52 dB.

**Mechanism.** In the `harmkit` convention the SNR scale is set by the
**safety band, 20-80 Hz**: `ArtifactBank.scale_for_snr` divides by that band's
RMS, and `clean_reference_rms` is documented in the same band. We took the
denominator in 20-80 Hz and the numerator in 8-32 Hz.

**Corrected value.** -10.79 dB. Shift -1.27 dB.

**Why the shift is small.** Both bands span exactly a factor of four in
frequency, 8 to 32 and 20 to 80. Under an exactly 1/f spectrum the two band
powers would be equal and the shift would be zero.

**Important.** R3 and R4 are two separate entries with two separate causes.
Merging them into one would misrepresent the procedure.

---

## R5. The artifact-only gate was degenerate

**Claimed.** Artifact-only decoding accuracy must stay below 0.55, and arm A
must not exceed arm B by more than 0.03.

**Result.** 0.4999, 0.5002 and 0.4998 across three arms. Both gates passed.

**Mechanism.** Folds are partitioned within each class, so every fold is class
balanced. A donor carries no label and produces one prediction for the whole
fold, so the fraction of matching trial labels is exactly one half whatever the
prediction is. The gate returns 0.5 **algebraically** and cannot fail. The
fourth-decimal residual is only the 199-trial subjects not dividing by five.

**Not repairable.** A donor has no class and is identical across a subject's
trials, so no statistic built on label agreement can work. Replaced by the
|rho| measurement.

**Bearing on the main work.** The 0.910 agreement figure in section 6.4 has the
same weakness. The "honest note" there was correct, and more strongly correct
than it was written: in both branches this measure is not independent evidence.

---

## R6. A six-subject check was too small

**Claimed.** Moving from bank v2 to v3 drifted purity by at most 0.021, over
six subjects.

**What happened.** The full recount over 32 subjects showed subject 31 moving
by 0.111, five times larger. The checking instrument itself was badly chosen: a
six-subject sample cannot estimate a spread.

**Consequence.** The cohort moved from 18 to 19. Subject 31 entered, nobody
left. The threshold was not moved.

---

## R7. The post-hoc factor 3.54 was refuted

**Claimed**, on the development cohort: the saturation law keeps its shape and
changes only its scale; the ratio to BNCI is 3.54 with sd 0.27 across all eight
frozen levels.

**Test.** Criterion C4 was written into `T5_PREREGISTRATION.json` before the
sealed subjects were opened: the ratio must fall in [2.65, 4.43].

**Result.** 2.47. Criterion failed.

**Reading.** The scale is not a property of the dataset. Two random subgroups
of the same Cho2017 differ by 32 per cent. The slope drifts too: BNCI +0.1390,
Cho development +0.1299, Cho locked +0.1123 per dB.

**Survived.** The monotone character, confirmed from -7 to -30 dB, outside the
range the law was derived on, on subjects nobody had opened.

**Lesson.** 3.54 was a post-hoc pattern from one cohort. C4 was written for
exactly this purpose and it worked.

---

## R8. Arm C cannot discriminate, by construction

**Claimed.** Arm C, the host's own resting block as donor, answers whether the
effect is muscle or background.

**What happened.** The A minus C symmetry difference was -0.1951, 95% CI
[-0.4484, +0.0917], covering zero.

**Mechanism.** SNR was matched across arms, and the interaction-required
fraction is driven mainly by the margin density near zero and by scale. The
control removed, by matching, the very difference it was built to detect.

**Consequence.** Arm C cannot be cited as evidence that the effect is muscle
rather than background. That assurance comes from the purity threshold applied
at the input instead.

---

## R9. Criterion C1 was brittle by construction

**What happened.** The locked median was 0.0336 and the development interval's
lower bound was also 0.0336. The comparison ran at full float precision while
both numbers round to four decimals in print. The criterion failed by less than
its own reported precision.

**Mechanism.** Comparing a point estimate against an interval bound was the
wrong test. Intervals should have been compared: the locked interval
[0.0205, 0.0583] fully contains the development interval [0.0336, 0.0534].

**Reported as.** Both readings. Strict form: **failed**. Interval overlap:
**fully consistent**.

---

# Procedural deviations

1. **The purity threshold of 0.90** was chosen after the purity distribution of
   the confirmatory subjects was visible. Mitigations: the threshold is
   justified physically and independently of that distribution; the 15 sealed
   subjects were untouched; results are reported at 0.75, 0.90 and 0.95, giving
   23, 19 and 13 subjects.
2. **Stage T3b ran on 23 subjects** where the frozen eligibility rule leaves 18;
   the filter checked purity only. Both are published and the conclusion is
   identical: +0.0001 (p = 0.94) and +0.0003 (p = 0.71).
3. **`clean_reference_rms` is per subject**, where the BNCI branch uses one
   scalar, 4.4, for the whole dataset. Cho amplitudes differ between people.
4. **The CAR check was made relative.** `harmkit.validate` uses an absolute
   tolerance of 1e-3, calibrated for microvolts: the PhysioMotion bank peaked at
   97.9 uV. Raw Cho units are about 1e4 larger. Relative residual: median
   6.77e-08, max 1.60e-07, which is float32 epsilon. CAR is exact by
   construction, since subtracting the channel mean gives a sum of exactly zero
   and every later operation is linear and identical across channels; only the
   float32 cast breaks it.
