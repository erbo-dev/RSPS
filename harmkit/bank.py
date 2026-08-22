"""Artifact donor bank with a frozen SNR-scaling convention."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ArtifactBank:
    """A bank of real artifact donors, held in two frequency bands.

    Parameters
    ----------
    decoder_band : ndarray, shape (n_donors, n_channels, n_times)
        Donors filtered to the decoder's own passband. These enter the
        contaminated signal that the decoder actually sees.
    safety_band : ndarray, shape (n_donors, n_channels, n_times)
        The same donors in a broader band, used only to define the SNR
        reference. Keeping this separate matters: scaling on the decoder
        band alone would make the SNR meaningless for artifacts whose
        energy mostly falls outside it.
    donor_id : sequence of str
        Stable identifier per donor, e.g. ``"P07_R03_CHEW02"``.
    subject_id : sequence
        Which person each donor came from. Required, because donors from
        the same person are not independent observations and the
        statistics must respect that.

    Notes
    -----
    Donors must already be re-referenced the same way as the host EEG
    (typically CAR over the shared channel set). ``validate`` checks this.
    """

    decoder_band: np.ndarray
    safety_band: np.ndarray
    donor_id: np.ndarray
    subject_id: np.ndarray
    _safety_rms: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.decoder_band = np.asarray(self.decoder_band, dtype=np.float64)
        self.safety_band = np.asarray(self.safety_band, dtype=np.float64)
        self.donor_id = np.asarray(self.donor_id, dtype=object)
        self.subject_id = np.asarray(self.subject_id)
        self._safety_rms = np.sqrt(np.mean(self.safety_band ** 2, axis=(1, 2)))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.decoder_band.shape[0]

    @property
    def n_channels(self) -> int:
        return self.decoder_band.shape[1]

    @property
    def n_times(self) -> int:
        return self.decoder_band.shape[2]

    @property
    def safety_rms(self) -> np.ndarray:
        """RMS of each donor in the safety band. Defines the SNR scale."""
        return self._safety_rms

    # ------------------------------------------------------------------
    def validate(self, car_tol: float = 1e-3) -> dict:
        """Check the invariants that silently break contamination studies.

        Returns a dict of check names to booleans and raises on hard failures.
        """
        checks = {}
        if self.decoder_band.shape != self.safety_band.shape:
            raise ValueError(
                f"band shape mismatch: {self.decoder_band.shape} vs "
                f"{self.safety_band.shape}"
            )
        if len(self.donor_id) != len(self):
            raise ValueError("donor_id length does not match the number of donors")
        if len(self.subject_id) != len(self):
            raise ValueError("subject_id length does not match the number of donors")
        if len(set(map(str, self.donor_id))) != len(self):
            raise ValueError("donor_id values are not unique")

        checks["finite"] = bool(
            np.isfinite(self.decoder_band).all() and np.isfinite(self.safety_band).all()
        )
        checks["positive_rms"] = bool(np.all(self._safety_rms > 0))
        car_err = float(max(np.max(np.abs(self.decoder_band.sum(axis=1))),
                            np.max(np.abs(self.safety_band.sum(axis=1)))))
        checks["common_average_referenced"] = bool(car_err < car_tol)
        checks["car_residual"] = car_err
        checks["n_donors"] = len(self)
        checks["n_subjects"] = int(len(np.unique(self.subject_id)))

        if not checks["finite"]:
            raise ValueError("bank contains non-finite values")
        if not checks["positive_rms"]:
            raise ValueError("bank contains a donor with zero energy")
        return checks

    # ------------------------------------------------------------------
    def scale_for_snr(self, clean_reference_rms: float, snr_db) -> np.ndarray:
        """Scalar multiplier per (donor, SNR level).

        One scalar per donor. Channels are never normalised individually,
        so the donor's spatial topography is preserved exactly. This is the
        whole point: a per-channel normalisation would invent a topography
        that no one recorded.

        Returns
        -------
        ndarray, shape (n_donors, n_snr)
        """
        snr_db = np.atleast_1d(np.asarray(snr_db, dtype=np.float64))
        return float(clean_reference_rms) / (
            self._safety_rms[:, None] * 10.0 ** (snr_db[None, :] / 20.0)
        )

    # ------------------------------------------------------------------
    def subset(self, mask) -> "ArtifactBank":
        """Take a subset, e.g. to hold out donor subjects for confirmation."""
        mask = np.asarray(mask)
        return ArtifactBank(
            decoder_band=self.decoder_band[mask],
            safety_band=self.safety_band[mask],
            donor_id=self.donor_id[mask],
            subject_id=self.subject_id[mask],
        )

    def split_by_subject(self, holdout_subjects):
        """Split into (development, holdout) by donor subject.

        Use this *before* looking at any result, and do not merge the two
        afterwards. Holding out donor subjects rather than donors is the
        point: donors from the same person share anatomy and montage.
        """
        holdout_subjects = set(np.atleast_1d(holdout_subjects).tolist())
        held = np.isin(self.subject_id, list(holdout_subjects))
        return self.subset(~held), self.subset(held)

    # ------------------------------------------------------------------
    @classmethod
    def from_npy(cls, decoder_path, safety_path, donor_id, subject_id):
        return cls(np.load(decoder_path), np.load(safety_path), donor_id, subject_id)
