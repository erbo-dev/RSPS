"""Decoder adapters.

The central difficulty of this benchmark is not measuring harm. It is
removing *only* the interaction term and leaving everything else intact.
That requires an algebraic handle on how the decoder turns a signal into a
decision, and different decoder families give different handles.

Three adapters are provided:

``PowerDecoder``
    For decoders whose decision is a function of the power of a fixed set
    of spatial projections (CSP + LDA, FBCSP, log-variance pipelines).
    Contaminated power splits exactly:
    ``E[(z+u)^2] = E[z^2] + E[u^2] + 2 E[zu]``.
    The ablation removes ``2 E[zu]`` and nothing else. Exact and fast.

``CovarianceDecoder``
    For decoders whose decision is a function of the spatial covariance
    matrix (MDM and other Riemannian methods). The identity
    ``C(x + λa) = C_x + λ² C_a + λ (C_xa + C_xaᵀ)`` is exact, so the
    ablation is exact too.

    Caveat, and it is a real one: ``C_x + λ² C_a`` is not the covariance of
    any real signal, whereas the full matrix is. On a curved manifold this
    can bias distances. Check the required/prevented symmetry before
    trusting an ablation in this mode.

``DirectDecoder``
    For anything else, including neural networks. Contamination is applied
    to the waveform and the decoder is called as-is. Harm is measured
    exactly; the interaction ablation is **not available**, because there is
    no exact way to synthesise a waveform lacking only the cross-term.
    Reporting an approximate ablation here as if it were exact is the
    single easiest way to fool yourself with this method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["PowerDecoder", "CovarianceDecoder", "DirectDecoder", "AblationNotAvailable"]


class AblationNotAvailable(RuntimeError):
    """Raised when an exact interaction ablation cannot be constructed."""


class BaseDecoder(ABC):
    supports_ablation = True

    @abstractmethod
    def margins(self, clean, bank, lam, trial_idx):
        """Return ``(m_no_interaction, m_full)``, both shape (n_trials, n_donors, n_snr).

        ``trial_idx`` indexes the rows of ``clean`` back into the full trial
        array the decoder was constructed with, so that labels stay aligned
        when the benchmark processes trials in blocks.

        A margin is positive exactly when the decision is correct, so its
        sign carries the outcome and its magnitude carries the distance to
        the decision boundary.
        """

    @abstractmethod
    def clean_margin(self, clean):
        """Signed margin on all uncontaminated trials, shape (n_trials,)."""


# ----------------------------------------------------------------------
class PowerDecoder(BaseDecoder):
    """Exact ablation for power-of-projection decoders.

    Parameters
    ----------
    filters : ndarray, shape (n_components, n_channels)
        Spatial filters, already fitted and frozen.
    score_fn : callable
        Maps log-power features of shape ``(..., n_components)`` to a signed
        score whose sign gives the predicted class (positive = class 1).
    y_true : ndarray of {0, 1}, shape (n_trials,)
    """

    def __init__(self, filters, score_fn, y_true):
        self.W = np.asarray(filters, dtype=np.float64)
        self.score_fn = score_fn
        self.y_sign = np.where(np.asarray(y_true) == 1, 1.0, -1.0)

    def _terms(self, clean, art):
        z = np.einsum("kc,nct->nkt", self.W, clean, optimize=True)
        u = np.einsum("kc,dct->dkt", self.W, art, optimize=True)
        T = z.shape[-1]
        p_clean = np.mean(z ** 2, axis=-1)
        p_art = np.mean(u ** 2, axis=-1)
        cross = np.einsum("nkt,dkt->ndk", z, u, optimize=True) / T
        return p_clean, p_art, cross

    def clean_margin(self, clean):
        z = np.einsum("kc,nct->nkt", self.W, clean, optimize=True)
        return self.y_sign * self.score_fn(np.log(np.maximum(np.mean(z ** 2, -1), 1e-300)))

    def margins(self, clean, bank, lam, trial_idx):
        p_clean, p_art, cross = self._terms(clean, bank.decoder_band)
        pc = p_clean[:, None, None, :]
        pa = p_art[None, :, None, :]
        cr = cross[:, :, None, :]
        lm = lam[None, :, :, None]
        ys = self.y_sign[np.asarray(trial_idx)][:, None, None]
        no_int = ys * self.score_fn(np.log(np.maximum(pc + lm ** 2 * pa, 1e-300)))
        full = ys * self.score_fn(
            np.log(np.maximum(pc + 2.0 * lm * cr + lm ** 2 * pa, 1e-300))
        )
        return no_int, full

    def artifact_only_margin(self, bank, lam):
        """Decision on the scaled artifact alone, with the clean signal removed.

        This is *not* the same as the no-interaction condition, and confusing
        the two is a common error. Use it as a control: if the decoder can
        recover the class label from the artifact alone, the artifacts encode
        the label and the whole study is invalid.
        """
        u = np.einsum("kc,dct->dkt", self.W, bank.decoder_band, optimize=True)
        p_art = np.mean(u ** 2, axis=-1)
        return self.score_fn(np.log(np.maximum((lam[:, :, None] ** 2) * p_art[:, None, :], 1e-300)))


# ----------------------------------------------------------------------
class CovarianceDecoder(BaseDecoder):
    """Exact ablation for covariance-based decoders.

    Parameters
    ----------
    score_fn : callable
        Maps a stack of covariance matrices ``(n, p, p)`` to signed scores
        ``(n,)``, positive for class 1.
    projector : ndarray or None, shape (n_channels, p)
        Optional rank reduction. Common-average referencing leaves an exact
        null direction; projecting it out before any matrix logarithm is
        cleaner than propping the matrix up with a large ridge.
    """

    def __init__(self, score_fn, y_true, projector=None, ridge_rel=1e-6):
        self.score_fn = score_fn
        self.y_sign = np.where(np.asarray(y_true) == 1, 1.0, -1.0)
        self.V = None if projector is None else np.asarray(projector, dtype=np.float64)
        self.ridge_rel = float(ridge_rel)
        self._eps = None

    def _proj(self, C):
        if self.V is None:
            return C
        return np.einsum("ci,...cs,sj->...ij", self.V, C, self.V, optimize=True)

    @staticmethod
    def _cov(X):
        Xc = X - X.mean(-1, keepdims=True)
        return np.einsum("nct,nst->ncs", Xc, Xc, optimize=True) / (X.shape[-1] - 1)

    @staticmethod
    def _cross_cov(X, A):
        Xc = X - X.mean(-1, keepdims=True)
        Ac = A - A.mean(-1, keepdims=True)
        return np.einsum("nct,dst->ndcs", Xc, Ac, optimize=True) / (X.shape[-1] - 1)

    def set_ridge_from(self, clean_train):
        """Fix the ridge from clean *training* trials only, then freeze it."""
        C = self._proj(self._cov(clean_train))
        p = C.shape[-1]
        self._eps = self.ridge_rel * float(np.mean(np.trace(C, axis1=-2, axis2=-1)) / p)
        return self._eps

    def _spd(self, C):
        if self._eps is None:
            raise RuntimeError("call set_ridge_from(clean_train) before scoring")
        C = 0.5 * (C + np.swapaxes(C, -1, -2))
        return C + self._eps * np.eye(C.shape[-1])

    def clean_margin(self, clean):
        C = self._spd(self._proj(self._cov(clean)))
        return self.y_sign * self.score_fn(C)

    def margins(self, clean, bank, lam, trial_idx):
        Cx = self._proj(self._cov(clean))
        Ca = self._proj(self._cov(bank.decoder_band))
        Cxa = self._proj(self._cross_cov(clean, bank.decoder_band))
        Csym = Cxa + np.swapaxes(Cxa, -1, -2)

        n, d, s = clean.shape[0], len(bank), lam.shape[1]
        lm = lam[None, :, :, None, None]
        base = Cx[:, None, None] + lm ** 2 * Ca[None, :, None]
        shp = (n * d * s, Cx.shape[-1], Cx.shape[-1])
        ys = np.repeat(self.y_sign[np.asarray(trial_idx)], d * s)
        m_no = (ys * self.score_fn(self._spd(base.reshape(shp)))).reshape(n, d, s)
        m_fu = (ys * self.score_fn(
            self._spd((base + lm * Csym[:, :, None]).reshape(shp)))).reshape(n, d, s)
        return m_no, m_fu

    def identity_error(self, clean, bank, lam_scalar=1.0, n_check=4):
        """Verify the covariance identity against directly composed signals.

        Built deliberately as an *independent* check: the full covariance is
        computed from the actual waveform ``x + λa``, never as
        ``no_interaction + cross``, which would make the test circular.
        """
        worst = 0.0
        A = bank.decoder_band
        for i in range(min(n_check, clean.shape[0])):
            for j in range(min(n_check, len(bank))):
                mix = (clean[i] + lam_scalar * A[j])[None]
                direct = self._proj(self._cov(mix))[0]
                Cx = self._proj(self._cov(clean[i][None]))[0]
                Ca = self._proj(self._cov(A[j][None]))[0]
                Cxa = self._proj(self._cross_cov(clean[i][None], A[j][None]))[0, 0]
                asm = Cx + lam_scalar ** 2 * Ca + lam_scalar * (Cxa + Cxa.T)
                worst = max(worst, float(np.max(np.abs(direct - asm)) / np.max(np.abs(direct))))
        return worst


# ----------------------------------------------------------------------
class DirectDecoder(BaseDecoder):
    """Any decoder at all, via waveform contamination. No exact ablation.

    Parameters
    ----------
    predict_score : callable
        Maps raw epochs ``(n, n_channels, n_times)`` to signed scores ``(n,)``.
    """

    supports_ablation = False

    def __init__(self, predict_score, y_true):
        self.predict_score = predict_score
        self.y_sign = np.where(np.asarray(y_true) == 1, 1.0, -1.0)

    def clean_margin(self, clean):
        return self.y_sign * np.asarray(self.predict_score(clean)).reshape(-1)

    def full_margin(self, clean, bank, lam, trial_idx):
        """Margin under full contamination. Shape (n_trials, n_donors, n_snr)."""
        n, d, s = clean.shape[0], len(bank), lam.shape[1]
        ys = self.y_sign[np.asarray(trial_idx)]
        out = np.empty((n, d, s), dtype=np.float64)
        for i in range(n):
            for k in range(s):
                mix = clean[i][None] + lam[:, k][:, None, None] * bank.decoder_band
                out[i, :, k] = ys[i] * np.asarray(
                    self.predict_score(mix)).reshape(-1)
        return out

    def margins(self, clean, bank, lam, trial_idx):
        raise AblationNotAvailable(
            "An exact interaction ablation needs an algebraic handle on the "
            "decoder. Use PowerDecoder for power-of-projection decoders or "
            "CovarianceDecoder for covariance-based ones. With DirectDecoder "
            "you can measure harm rates, but reporting an approximate ablation "
            "as exact would misstate what was measured."
        )
