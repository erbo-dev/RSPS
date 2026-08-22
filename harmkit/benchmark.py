"""The benchmark itself, its result object, and the closed-form law."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decoders import AblationNotAvailable, DirectDecoder

__all__ = ["HarmBenchmark", "HarmResult", "predict_interaction_required",
           "subject_bootstrap"]


# ----------------------------------------------------------------------
def subject_bootstrap(values, n_boot=20000, seed=0, stat=np.median):
    """Confidence interval with the *subject* as the resampling unit.

    This is not a stylistic choice. A contamination sweep produces millions
    of rows, but they are computational realisations nested inside a handful
    of people. Bootstrapping rows would report an interval an order of
    magnitude too narrow.
    """
    v = np.asarray([x for x in np.ravel(values) if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return dict(n=0, point=np.nan, lo=np.nan, hi=np.nan)
    rng = np.random.default_rng(seed)
    draws = stat(v[rng.integers(0, v.size, (n_boot, v.size))], axis=1)
    return dict(n=int(v.size), point=float(stat(v)),
                lo=float(np.quantile(draws, 0.025)),
                hi=float(np.quantile(draws, 0.975)))


# ----------------------------------------------------------------------
def predict_interaction_required(m, interaction, bandwidth_q=0.02):
    """Closed-form prediction of the interaction-required fraction.

    Writing ``m`` for the no-interaction margin and ``I`` for the exact
    effect of the cross-term on the score, a flip needs the interaction
    exactly when ``0 < m < -I``. If ``I`` is independent of ``m`` and small
    relative to the spread of ``m``, that probability is
    ``f_m(0) · E[(-I)^+]``, and dividing by the overall harm rate ``P(m<0)``
    gives the fraction.

    There are no free parameters: all three quantities are measured.

    Returns ``(prediction, diagnostics)``.
    """
    m = np.asarray(m, dtype=np.float64).ravel()
    interaction = np.asarray(interaction, dtype=np.float64).ravel()
    p_neg = float(np.mean(m < 0))
    if p_neg <= 0:
        return np.nan, {"p_neg": p_neg, "reason": "no harmful events"}
    h = float(np.quantile(np.abs(m), bandwidth_q))
    if h <= 0:
        return np.nan, {"h": h, "reason": "degenerate bandwidth"}
    f0 = float(np.mean(np.abs(m) < h) / (2.0 * h))
    e_neg = float(np.mean(np.maximum(-interaction, 0.0)))
    return f0 * e_neg / p_neg, {"f_m_0": f0, "E_neg_I": e_neg, "P_m_neg": p_neg,
                                "bandwidth": h}


# ----------------------------------------------------------------------
@dataclass
class HarmResult:
    """Outcome of one benchmark run. All counts are exact, not sampled."""

    n_eligible: int
    full_harm: int
    shared_harm: int
    interaction_required: int
    interaction_prevented: int
    per_subject: dict
    by_snr: dict
    snr_db: np.ndarray
    ablation_available: bool
    margins: dict = None

    # -- headline numbers ------------------------------------------------
    @property
    def harm_rate(self) -> float:
        return self.full_harm / self.n_eligible if self.n_eligible else np.nan

    @property
    def interaction_required_fraction(self) -> float:
        return self.interaction_required / self.full_harm if self.full_harm else np.nan

    @property
    def symmetry_ratio(self) -> float:
        """prevented / required.

        A cross-term that is a zero-mean nuisance breaks as many decisions as
        it rescues, so this sits near 1. A ratio far from 1 means either real
        structure or a biased ablation, and the two look identical here, so
        do not read it as evidence on its own.
        """
        if not self.interaction_required:
            return np.nan
        return self.interaction_prevented / self.interaction_required

    # -- statistics ------------------------------------------------------
    def subject_interval(self, seed=0):
        vals = [v["interaction_required_fraction"] for v in self.per_subject.values()]
        return subject_bootstrap(vals, seed=seed)

    def predicted_fraction(self, bandwidth_q=0.02):
        if self.margins is None:
            raise RuntimeError("run the benchmark with keep_margins=True")
        return predict_interaction_required(
            self.margins["no_interaction"], self.margins["interaction"], bandwidth_q)

    # -- reporting -------------------------------------------------------
    def summary(self) -> str:
        lines = ["Decision-level harm benchmark", "=" * 44,
                 f"eligible counterfactual pairs : {self.n_eligible:,}",
                 f"harmful events (HARM)         : {self.full_harm:,}"
                 f"   ({self.harm_rate*100:.2f}% of eligible)"]
        if self.ablation_available:
            ci = self.subject_interval()
            lines += [
                f"interaction-required          : {self.interaction_required:,}",
                f"interaction-prevented         : {self.interaction_prevented:,}",
                f"interaction-required fraction : {self.interaction_required_fraction:.6f}",
                f"  subject-level median 95% CI : {ci['point']:.6f} "
                f"[{ci['lo']:.6f}, {ci['hi']:.6f}]  (N={ci['n']} subjects)",
                f"symmetry (prevented/required) : {self.symmetry_ratio:.3f}",
            ]
        else:
            lines.append("interaction ablation          : not available for this decoder")
        return "\n".join(lines)

    def __str__(self):
        return self.summary()


# ----------------------------------------------------------------------
class HarmBenchmark:
    """Measure how often a physiological artifact flips a correct decision.

    The unit of analysis is a single decision, not an accuracy average. A
    trial counts only if the decoder got it right on the clean signal and
    wrong after contamination, so the same neural activity appears on both
    sides of the comparison and the artifact is the only thing that changed.

    Parameters
    ----------
    decoder : PowerDecoder, CovarianceDecoder or DirectDecoder
    bank : ArtifactBank
    snr_db : sequence of float
        Contamination levels. ``-7`` to ``+2`` dB is the conventional range
        for EMG-in-EEG work; picking a grid after seeing results is the
        fastest way to produce a number you cannot defend.

    Example
    -------
    >>> bench = HarmBenchmark(decoder, bank, snr_db=[2, 0, -2, -4, -7])
    >>> result = bench.run(X_clean, clean_reference_rms=4.4,
    ...                    subject_ids=subj, keep_margins=True)
    >>> print(result.summary())
    """

    def __init__(self, decoder, bank, snr_db=(2, 0, -2, -3, -4, -5, -6, -7)):
        self.decoder = decoder
        self.bank = bank
        self.snr_db = np.asarray(snr_db, dtype=np.float64)
        bank.validate()

    # ------------------------------------------------------------------
    def run(self, clean, clean_reference_rms, subject_ids=None,
            keep_margins=False, block_size=64):
        """Run the sweep.

        Parameters
        ----------
        clean : ndarray, shape (n_trials, n_channels, n_times)
        clean_reference_rms : float
            RMS of the clean signal in the safety band, defining 0 dB.
            Compute it from *training* trials only.
        subject_ids : sequence, optional
            One entry per trial. Needed for subject-level intervals; without
            it every trial is treated as its own subject, which will make the
            confidence interval far too narrow.
        """
        clean = np.asarray(clean, dtype=np.float64)
        if clean.shape[1] != self.bank.n_channels:
            raise ValueError(
                f"channel mismatch: trials have {clean.shape[1]}, "
                f"bank has {self.bank.n_channels}. The two must share the same "
                "montage in the same order."
            )
        if clean.shape[2] != self.bank.n_times:
            raise ValueError(
                f"length mismatch: trials {clean.shape[2]}, bank {self.bank.n_times}"
            )
        if subject_ids is None:
            subject_ids = np.arange(clean.shape[0])
        subject_ids = np.asarray(subject_ids)

        lam = self.bank.scale_for_snr(clean_reference_rms, self.snr_db)
        eligible = self.decoder.clean_margin(clean) > 0
        n_elig_trials = int(eligible.sum())
        if n_elig_trials == 0:
            raise ValueError(
                "the decoder is wrong on every clean trial, so there is no "
                "correct decision left to break"
            )

        n_snr = len(self.snr_db)
        totals = dict(full=0, shared=0, required=0, prevented=0)
        per_subject = {}
        by_snr = {j: dict(full=0, required=0) for j in range(n_snr)}
        keep_m, keep_i = [], []

        direct = isinstance(self.decoder, DirectDecoder)
        idx = np.flatnonzero(eligible)

        for start in range(0, len(idx), block_size):
            sel = idx[start:start + block_size]
            Xb = clean[sel]
            if direct:
                m_full = self.decoder.full_margin(Xb, self.bank, lam, sel)
                m_no = None
            else:
                m_no, m_full = self.decoder.margins(Xb, self.bank, lam, sel)

            harm = m_full < 0
            totals["full"] += int(harm.sum())
            if m_no is not None:
                no_harm = m_no < 0
                totals["shared"] += int((harm & no_harm).sum())
                totals["required"] += int((harm & ~no_harm).sum())
                totals["prevented"] += int((~harm & no_harm).sum())
                if keep_margins:
                    keep_m.append(m_no.ravel().astype(np.float32))
                    keep_i.append((m_full - m_no).ravel().astype(np.float32))

            for j in range(n_snr):
                by_snr[j]["full"] += int(harm[:, :, j].sum())
                if m_no is not None:
                    by_snr[j]["required"] += int((harm & ~no_harm)[:, :, j].sum())

            for s in np.unique(subject_ids[sel]):
                mask = subject_ids[sel] == s
                d = per_subject.setdefault(
                    s, dict(full=0, required=0, prevented=0, n_trials=0))
                d["n_trials"] += int(mask.sum())
                d["full"] += int(harm[mask].sum())
                if m_no is not None:
                    d["required"] += int((harm & ~no_harm)[mask].sum())
                    d["prevented"] += int((~harm & no_harm)[mask].sum())

        for d in per_subject.values():
            d["interaction_required_fraction"] = (
                d["required"] / d["full"] if d["full"] else np.nan)

        by_snr_out = {
            float(self.snr_db[j]): dict(
                full_harm=by_snr[j]["full"],
                interaction_required=by_snr[j]["required"],
                fraction=(by_snr[j]["required"] / by_snr[j]["full"]
                          if by_snr[j]["full"] else np.nan))
            for j in range(n_snr)
        }

        margins = None
        if keep_margins and keep_m:
            margins = dict(no_interaction=np.concatenate(keep_m),
                           interaction=np.concatenate(keep_i))

        return HarmResult(
            n_eligible=n_elig_trials * len(self.bank) * n_snr,
            full_harm=totals["full"], shared_harm=totals["shared"],
            interaction_required=totals["required"],
            interaction_prevented=totals["prevented"],
            per_subject=per_subject, by_snr=by_snr_out, snr_db=self.snr_db,
            ablation_available=not direct, margins=margins,
        )
