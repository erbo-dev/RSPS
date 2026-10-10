# Calibrated choice between two fixed decoders

The fixed high95 switch helped on the initial strong-artifact grid but increased
error at +12 and +20 dB in the frozen-recipe challenge. This fifth exploratory
stage is defined after those outcomes and uses the same public sources. It is
not independent confirmation and will not establish scientific novelty.

Keep the CSP/original LDA and real-artifact-augmented LDA fits unchanged. The
augmented LDA is trained on the original five SNR values only. On the next-run
calibration partition, generate the fixed nine SNR values already declared in
STRESS_PROTOCOL.md, using training donor people only. Include unaugmented
calibration trials with repeated-sample weight that balances their aggregate
weight with all contaminated calibration variants. Repetition never creates
independent observations.

On cases where the two decoders disagree, fit StandardScaler + logistic
regression (C=1, max_iter=1000, seed=20261010). Target 1 means the augmented
decoder is correct; in this binary-class disagreement one decoder is correct.
Use exactly three observable features: absolute original margin, absolute
augmented margin, log of mean high-frequency power divided by its clean-fitting
median. Use probability >0.5 to choose augmented, otherwise original. On
agreement the command is unchanged. If calibration disagreement has a single
target class, use its constant target. No source ID, SNR, clean evaluation
counterfactual or evaluation label enters features. No feature/hyperparameter
search is made.

Primary summaries: full-coverage errors on the original five-SNR grid, separately
on the four weaker values and on unaugmented test trials. Report every SNR and
HARM with the common original clean-correct reference. Check both unchanged
decoder error vectors against saved stage-three outcomes. Record the selected
fraction and the fallback frequency. The host test run and artifact people are
excluded from every fitting and calibration stage. A further independent
transfer test is required regardless of this pilot's outcome.
