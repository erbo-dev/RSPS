# Exploratory follow-up after the first pilot

The first pilot found small average risk reductions but deterioration at −15 dB
for the deployable estimated-power score. Those results remain unchanged and
are not treated as successful confirmation.

This second stage is exploratory and reuses the same public development people;
its result cannot be called independent confirmation. Before its outcomes, fix:

- The same EEG and donor-person folds, source signals, SNR grid and reference.
- Keep clean-trained CSP filters fixed. Compare the original clean LDA with
  a shrinkage LDA trained on real-donor-contaminated training-run log powers.
- Equal total representation of clean and contaminated training examples;
  all five SNR values and all training donor people are represented equally.
  Repetition is sample weighting, not independent evidence.
- Second candidate: apply the training-only high-to-low component-power
  estimate to training and evaluation observed powers, then train the identical
  shrinkage LDA on these estimated-corrected augmented features. No true donor
  powers, source IDs, SNR values or clean evaluation powers enter prediction.
- All calibration and evaluation signals remain excluded from decoder fitting;
  evaluation artifact people remain excluded from augmentation and mapping.
- Original vs augmented vs estimated-corrected augmented: report full-coverage
  and matched-coverage command error, clean accuracy, calibrated-threshold
  coverage, per-SNR results, and subject-level paired differences. HARM uses
  the original clean-correct trials as a fixed denominator across decoders.
- No architecture search or hyperparameter search. All three methods reported.
- Augmentation itself is an established technique, not a claim of novelty.
  A positive combined result would require broader validation and a novelty
  review; it is a candidate extension only.
