# harmkit

Measure how often a physiological artifact turns a **correct** brain-computer
interface decision into a **wrong command**, and how much of that damage needs
the exact clean-by-artifact interaction rather than plain addition.

Most artifact tooling answers "is an artifact present". That is not the
question a BCI user cares about. They care whether this particular
interference changed the command that was sent.

## Install

```bash
pip install numpy scipy
# then put the repository root on your path, or copy harmkit/ next to your code
```

## The measurement

A trial counts as a harmful event only if the decoder was **right** on the
clean signal and **wrong** after contamination. The same neural activity
appears on both sides, so the artifact is the only thing that changed.

For power-of-projection decoders the contaminated power splits exactly:

```
E[(z + u)²]  =  E[z²]  +  E[u²]  +  2·E[zu]
                 clean    artifact   interaction
```

Removing only `2·E[zu]` and re-deciding tells you whether the interaction was
needed for the flip.

## Minimal example

```python
from harmkit import ArtifactBank, PowerDecoder, HarmBenchmark

bank = ArtifactBank(
    decoder_band=donors_8_32,     # (n_donors, n_channels, n_times)
    safety_band=donors_broadband, # same donors, wider band, sets the SNR scale
    donor_id=donor_names,
    subject_id=donor_person,      # who each donor came from
)

def score_fn(log_power):          # (..., n_components) -> signed score
    flat = log_power.reshape(-1, log_power.shape[-1])
    return lda.decision_function(flat).reshape(log_power.shape[:-1])

decoder = PowerDecoder(csp.filters_[:8], score_fn, y_true=labels_01)

result = HarmBenchmark(decoder, bank, snr_db=[2, 0, -2, -4, -7]).run(
    X_clean,
    clean_reference_rms=4.4,      # from training trials only
    subject_ids=subject_of_trial,
    keep_margins=True,
)
print(result.summary())
```

```
Decision-level harm benchmark
============================================
eligible counterfactual pairs : 239,096
harmful events (HARM)         : 86,957   (36.37% of eligible)
interaction-required          : 2,514
interaction-prevented         : 2,482
interaction-required fraction : 0.028911
  subject-level median 95% CI : 0.028911 [0.006686, 0.039841]  (N=9 subjects)
symmetry (prevented/required) : 0.987
```

## Decoder support

| Adapter | Applies to | Exact ablation |
|---|---|---|
| `PowerDecoder` | CSP+LDA, FBCSP, log-variance pipelines | yes |
| `CovarianceDecoder` | MDM and other Riemannian methods | yes |
| `DirectDecoder` | anything, including neural networks | **no** |

`DirectDecoder` deliberately raises `AblationNotAvailable` rather than
returning an approximation. There is no exact way to synthesise a waveform
that lacks only the cross-term, and presenting an approximate ablation as
exact would misstate what was measured.

For `CovarianceDecoder`, note that `C_x + λ²C_a` is not the covariance of any
real signal while the full matrix is. On a curved manifold that can bias
distances. Check `result.symmetry_ratio` before trusting the ablation there.

## Run the controls first

```python
from harmkit import artifact_only_control, phase_randomised_bank, alignment_sweep
```

**`artifact_only_control`** — feed the decoder the artifact with the neural
signal removed. Accuracy must sit at chance. This control invalidated an
earlier version of the study behind this package: artifacts built from
invented spatial topographies produced striking harm rates and then turned
out to predict the class label on their own. Real recorded artifacts pass at
0.504.

**`phase_randomised_bank`** — a null bank whose donors keep their covariance
to machine precision, so projected artifact power and the SNR scaling are
untouched and only the interaction realisation changes. If your result
matches the null, the interaction carries no pair-specific information, which
is an explanation rather than a failure, but you have to say so.

**`alignment_sweep`** — inject dependence deliberately and confirm the
measurement reacts. A null result means nothing until you have shown the
instrument responds when there is something to detect.

## Statistics

`subject_bootstrap` resamples **subjects**, not rows. A sweep produces
millions of rows, but they are computational realisations nested inside a
handful of people. Bootstrapping rows gives an interval roughly an order of
magnitude too narrow, and it is the easiest way to overstate a result here.

## Closed-form prediction

`predict_interaction_required(m, I)` predicts the fraction from three
measured quantities with **no fitted parameters**: the density of the margin
at zero, the mean negative part of the interaction effect, and the overall
harm rate. On the reference dataset it reaches 3.6% relative error for
CSP+LDA and, under a preregistered transfer test, 15% for Riemannian MDM.

The law was devised after seeing the null result, so its origin is post-hoc;
only the cross-family transfer test was preregistered. In the Riemannian
family it leaves a systematic residual beyond counting noise.

## Verify the install

```bash
python examples/reproduce_frozen_result.py path/to/context
```

Reproduces the frozen per-subject values to better than 1e-6.

## Limits worth knowing

- Donors and host trials must share the same montage in the same order.
- One scalar scales each donor; channels are never normalised individually,
  because that would invent a topography nobody recorded.
- Freeze the SNR grid before looking at results. `-7` to `+2` dB is the
  conventional range for EMG-in-EEG work.
- No-interaction is **not** artifact-only. The clean contribution stays.

## License

MIT. See `../LICENSE`.
