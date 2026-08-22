"""Controls that can invalidate a result. Run them before believing anything.

Each function here exists because of a specific way this method can go
wrong, and in one case because it already did.
"""

from __future__ import annotations

import numpy as np

from .bank import ArtifactBank

__all__ = ["artifact_only_control", "phase_randomised_bank", "alignment_sweep"]


def artifact_only_control(power_decoder, bank, lam, y_true):
    """Does the artifact alone reveal the class label?

    Feed the decoder the scaled artifact with the neural signal removed
    entirely. Accuracy against the true labels should sit at chance. If it
    does not, the artifacts carry class information and every harm number
    you have measured is contaminated by it.

    This control killed an earlier version of this study: artifacts built
    from invented spatial topographies produced striking harm rates, and
    then turned out to predict the label on their own with extreme
    confidence. Real recorded artifacts pass at 0.504.

    Returns
    -------
    dict with ``accuracy`` (should be ≈0.5), ``positive_fraction`` (the
    directional push, which can legitimately be far from 0.5), and
    ``verdict``.
    """
    score = power_decoder.artifact_only_margin(bank, lam)   # (n_donors, n_snr)
    pred_pos = score > 0
    y = np.asarray(y_true)
    n_pos = float(np.mean(y == 1))
    acc = float(np.mean(pred_pos) * n_pos + np.mean(~pred_pos) * (1 - n_pos))
    pos_frac = float(np.mean(pred_pos))
    dev = abs(acc - 0.5)
    return dict(
        accuracy=acc, positive_fraction=pos_frac, deviation_from_chance=dev,
        verdict=("PASS: artifact carries no class information" if dev <= 0.05
                 else "FAIL: artifact predicts the label; results are invalid"),
        note=("A positive_fraction far from 0.5 is fine and expected: the "
              "artifact pushes the decoder in a fixed direction. That is not "
              "the same as carrying class information."),
    )


def phase_randomised_bank(bank, rng=None):
    """A null bank whose donors keep their covariance exactly.

    One random phase sequence is applied across *all* channels of a donor,
    so every cross-spectrum, and therefore the whole covariance matrix, is
    preserved to machine precision. Projected artifact power is unchanged
    and the SNR scaling is untouched; only the realisation of the
    cross-term differs.

    Comparing the real bank against this one separates "the interaction is
    small because of something specific to these pairings" from "the
    interaction is small because independent sources have a zero-mean
    product".
    """
    rng = np.random.default_rng(rng)

    def randomise(A):
        Aw = np.fft.rfft(A, axis=-1)
        phi = rng.uniform(0, 2 * np.pi, size=(A.shape[0], 1, Aw.shape[-1]))
        phi[..., 0] = 0.0
        if A.shape[-1] % 2 == 0:
            phi[..., -1] = 0.0
        return np.ascontiguousarray(np.fft.irfft(Aw * np.exp(1j * phi),
                                                 n=A.shape[-1], axis=-1))

    return ArtifactBank(randomise(bank.decoder_band), bank.safety_band,
                        np.array([f"surrogate_{d}" for d in bank.donor_id], dtype=object),
                        bank.subject_id)


def alignment_sweep(bank, clean_trial, betas=(0.0, 0.25, 0.5, 1.0)):
    """Positive control: banks with dependence deliberately injected.

    Yields ``(beta, bank)`` where the donor becomes ``a + beta * x``. A null
    result is only worth reporting once you have shown that the measurement
    reacts when genuine dependence is present, and by how much.

    The safety band is left untouched, so the SNR reference does not move
    with beta and the sweep changes only the coupling.
    """
    x = np.asarray(clean_trial, dtype=np.float64)
    for beta in betas:
        yield float(beta), ArtifactBank(
            bank.decoder_band + beta * x[None, :, :], bank.safety_band,
            np.array([f"{d}_b{beta}" for d in bank.donor_id], dtype=object),
            bank.subject_id)
