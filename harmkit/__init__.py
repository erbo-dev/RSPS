"""harmkit — decision-level measurement of artifact-induced BCI failures.

Most artifact work asks whether an artifact is present. This package asks a
different question: did this particular interference turn a *correct*
decision into a wrong command, and how much of that damage required the
exact clean-by-artifact interaction rather than plain addition.

Quick start
-----------
>>> from harmkit import ArtifactBank, PowerDecoder, HarmBenchmark
>>> bank = ArtifactBank(decoder_band, safety_band, donor_id, donor_subject)
>>> dec = PowerDecoder(csp.filters_[:8], score_fn, y_true)
>>> res = HarmBenchmark(dec, bank).run(X_clean, clean_reference_rms=4.4,
...                                    subject_ids=subj, keep_margins=True)
>>> print(res.summary())

Before believing any number this produces, run the two controls in
``harmkit.controls``. The first checks that your artifacts do not encode the
class label, which is the failure mode that invalidates the whole design.
"""

from .bank import ArtifactBank
from .benchmark import (HarmBenchmark, HarmResult, predict_interaction_required,
                        subject_bootstrap)
from .decoders import (AblationNotAvailable, CovarianceDecoder, DirectDecoder,
                       PowerDecoder)
from .controls import artifact_only_control, phase_randomised_bank, alignment_sweep

__version__ = "0.1.0"
__all__ = [
    "ArtifactBank", "HarmBenchmark", "HarmResult",
    "PowerDecoder", "CovarianceDecoder", "DirectDecoder", "AblationNotAvailable",
    "predict_interaction_required", "subject_bootstrap",
    "artifact_only_control", "phase_randomised_bank", "alignment_sweep",
]
