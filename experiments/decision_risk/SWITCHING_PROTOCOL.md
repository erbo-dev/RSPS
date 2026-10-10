# Exploratory switching follow-up

After the augmentation stage, real-artifact training reduced contaminated
command error but reduced clean accuracy. Keep both previous outputs unchanged.
This stage tests one fixed mechanism, without tuning its threshold on outcomes:

- Fit original and real-artifact-augmented decoders exactly as in stage two.
- On clean decoder-fitting runs only, compute the mean observed high-frequency
  power across the eight fixed CSP projections and its empirical 95th percentile.
- Use the original decoder at or below that threshold and the augmented decoder
  above it. Both scores depend only on the currently observed signal and fitted
  training parameters. No source identity, clean evaluation signal, SNR or label
  is available to the switch.
- A corresponding corrected-augmented switch is a second explicitly reported
  comparison, not a retrospectively selected winner.
- Evaluate clean accuracy, contaminated error at 100% and matched 90% coverage,
  stronger contamination, all participants and the pre-existing eligible cohort.
- Confidence values from two LDA models are not guaranteed equally calibrated.
  Full-coverage command error is therefore the primary result for the switch;
  matched-coverage rejection and fixed-threshold coverage are secondary.

This is the third exploratory stage on reused public development data, following
observed earlier outcomes. It cannot supply an independent confirmatory p-value.
An amplitude-based switch between trained classifiers is not claimed to be a
new scientific invention. A larger donor cohort and a separate-session transfer
test must evaluate the recipe before a strong applied claim.
