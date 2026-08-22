# REPO STATUS: CURRENT. Produces bank v3 with decoder and safety bands.
# Run order, retractions and full context: notebooks_cho/RUN_ORDER.md

"""
STAGE T2d  ---  rebuild the bank in the harmkit band convention.

Why v2 is not usable for T4
---------------------------
Reading harmkit/bank.py showed that a donor is held in TWO copies:

    decoder_band   the donor filtered to 8-32 Hz. THIS is what gets added to
                   the trial. It is the signal the decoder actually sees.
    safety_band    the same donor filtered to 20-80 Hz. Used ONLY to define
                   the SNR scale, through ArtifactBank.scale_for_snr, which
                   divides by _safety_rms.

and clean_reference_rms is documented as the RMS of the CLEAN signal in the
SAFETY band, not in the decoder band. The frozen BNCI value 4.4 is that.

Stage T2c stored only the 20-80 Hz copy of each donor, on the mistaken belief
that 20-80 Hz was what gets injected. So v2 has the SNR reference and not the
injectable waveform. T4 cannot run on it. This is a rebuild, and the reason is
recorded rather than quietly fixed.

The same mistake makes the natural-SNR figure wrong: it used the clean RMS in
8-32 Hz where the convention calls for 20-80 Hz. Background EEG is far weaker
at 20-80, so the ratio moves. The old median of -9.52 dB is RETRACTED and is
recomputed in Stage T3d from this bank. That is the second retraction of that
number; the first was the zero-padded resample.

What is deliberately preserved
------------------------------
Donor windows stay on the frozen 3 s grid, and MI_d and purity are computed
exactly as in v2, so the frozen purity threshold of 0.90 and the 18-subject
cohort remain valid. The injectable copies are the central 625 samples of each
3 s window, matching the frozen 0.5-3.0 s analysis window. Nothing that was
already frozen is redefined.

Band-pass is applied to the WHOLE block before windowing. An 8 Hz lower edge
needs roughly 412 samples of transient at 250 Hz, which does not fit inside a
750-sample window. Filtering the 2500-sample block first and cutting after is
the same lesson as T1b, applied before it can bite.

Verification
------------
Recomputed MI_d, purity and residual DC are checked against values recorded
from the v2 run. A silent drift here would move the cohort.

Output: cho_bank_v3. v2 is superseded, not deleted.
Cost: about 1.5 hours and 10.6 GB. This is the last download.

WHEN IT FINISHES: Save Version, then New Dataset from the output. Skipping
that is why this is the third download.
"""

import gc
import glob
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

OUT = "/kaggle/working"
BANK = f"{OUT}/cho_bank_v3"
os.makedirs(BANK, exist_ok=True)

# ------------------------------------------------------------------ frozen
SF_NATIVE, SF_TARGET, NOTCH_HZ = 512.0, 250.0, 60.0
DEC_BAND, SAF_BAND = (8.0, 32.0), (20.0, 80.0)
DONOR_WIN_S, JAW_SLOT = 3.0, 3          # unchanged, so purity stays comparable
W0, NWIN = 0.5, 625                     # frozen analysis window 0.5-3.0 s
HF_BAND, MAINS, MAINS_HW, NPERSEG = (40.0, 100.0), [60.0, 120.0, 180.0], 2.0, 512
COMMON19 = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
            "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP3", "CP1", "CP2", "CP4", "Pz", "POz"]
PILOT = [1, 2, 3, 4, 5]
DEV = [6, 7, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 25, 26, 27, 29, 31,
       32, 33, 34, 35, 37, 38, 39, 40, 41, 44, 47, 49, 50]
LOCKED = [8, 15, 16, 17, 24, 28, 30, 36, 42, 43, 45, 46, 48, 51, 52]
ALL = sorted(PILOT + DEV + LOCKED)

# From the v2 run. A drift here would move the frozen cohort.
LEDGER = {6: (78822.5, 28.54, 0.904), 9: (252117.6, 633.36, 0.995),
          14: (89701.7, 342.84, 0.988), 34: (189966.4, 9.88, 0.396),
          42: (191137.9, 5639.05, 1.000), 45: (199496.3, 1.53, 0.174)}

print("=" * 78)
print("STAGE T2d  rebuild with decoder band and safety band")
print("=" * 78)

try:
    import mne
    from moabb.datasets import Cho2017
    from scipy.io import loadmat
    from scipy.signal import detrend, resample_poly, welch
except ImportError:
    os.system(f"{sys.executable} -m pip install -q mne moabb")
    import mne
    from moabb.datasets import Cho2017
    from scipy.io import loadmat
    from scipy.signal import detrend, resample_poly, welch
mne.set_log_level("ERROR")

ds = Cho2017()
raw = None
for a in ds.get_data(subjects=[ALL[0]]).values():
    for b in a.values():
        for r in b.values():
            raw = r
            break
        break
    break
CH_ALL = [c for c in raw.ch_names if c.upper() not in ("STIM", "STI 014")]
IDX19 = np.array([CH_ALL.index(c) for c in COMMON19])
del raw
gc.collect()


def unwrap(st, f):
    v = st[f]
    while isinstance(v, np.ndarray) and v.dtype == object and v.size == 1:
        v = v.ravel()[0]
    return v


def orient(a):
    a = np.asarray(a, dtype=np.float64)
    return a if a.shape[0] == len(CH_ALL) else a.T


def car19_clean(a):
    """Subset, CAR over those 19, detrend. Detrend is linear, so the CAR
    identity sum-over-channels == 0 survives it, and so does every later
    filter. harmkit.validate checks that identity, so it must hold."""
    x = a[IDX19]
    x = x - x.mean(axis=0, keepdims=True)
    return detrend(x, axis=-1, type="linear")


def to_target(x):
    g = int(round(SF_NATIVE / np.gcd(int(SF_NATIVE), int(SF_TARGET))))
    up = int(round(SF_TARGET * g / SF_NATIVE))
    return resample_poly(x, up, g, axis=-1, padtype="line")


def bp(x, band):
    y = mne.filter.notch_filter(x, SF_TARGET, [NOTCH_HZ], verbose=False)
    return mne.filter.filter_data(y, SF_TARGET, band[0], band[1],
                                  pad="reflect_limited", verbose=False)


def band_power(x, sf, band):
    x = detrend(np.asarray(x, float), axis=-1, type="linear")
    nps = int(min(NPERSEG, x.shape[-1]))
    f, Pxx = welch(x, fs=sf, nperseg=nps, noverlap=nps // 2, axis=-1)
    m = (f >= band[0]) & (f <= band[1])
    for m0 in MAINS:
        m &= ~((f > m0 - MAINS_HW) & (f < m0 + MAINS_HW))
    return np.trapezoid(Pxx[..., m], f[m], axis=-1) if m.any() else np.zeros(x.shape[0])


def mat_idx(cell, k):
    try:
        v = np.asarray(np.asarray(cell).ravel()[k]).ravel()
        return set(int(i) - 1 for i in v if np.isfinite(i))
    except Exception:
        return set()


def rms(z):
    return np.sqrt((np.asarray(z, float) ** 2).mean(axis=(-2, -1)))


def trials_both_bands(stream, ev, n_tr):
    """Whole 7 s trials in the decoder band, plus the safety-band RMS.

    The safety-band waveform is not stored: only its RMS over the analysis
    window is needed, because that is what defines clean_reference_rms.
    """
    x = to_target(car19_clean(orient(stream)))
    per = int(round(x.shape[-1] / n_tr))
    if per * n_tr != x.shape[-1]:
        raise ValueError(f"{x.shape[-1]} does not divide into {n_tr} trials")
    tr = x.reshape(19, n_tr, per).transpose(1, 0, 2).copy()

    e = np.asarray(ev).ravel().astype(np.int64)
    e = e[: n_tr * int(round(SF_NATIVE * per / SF_TARGET))].reshape(n_tr, -1)
    on = np.array([int(np.flatnonzero(r)[0] * SF_TARGET / SF_NATIVE)
                   if np.flatnonzero(r).size else -1 for r in e])
    if (on < 0).any():
        raise ValueError("a trial carries no cue marker")
    a0 = int(np.median(on)) + int(round(W0 * SF_TARGET))

    dec = np.empty((n_tr, 19, per), np.float32)
    saf_rms = np.empty(n_tr)
    for i in range(n_tr):
        dec[i] = bp(tr[i], DEC_BAND)
        saf_rms[i] = float(rms(bp(tr[i], SAF_BAND)[:, a0:a0 + NWIN]))
    return dec, saf_rms, on, per


need = [s for s in ALL if not os.path.exists(f"{BANK}/sub-{s:02d}.npz")]
print(f"rebuilding {len(need)} subjects into {BANK}\n")
print(f"{'subj':>5} {'grp':>7} {'residDC':>9} {'MI_d max':>9} {'purity':>7} "
      f"{'CAR res':>9} {'trials':>7}")

drift, dcs, purs = [], [], []
for s in need:
    grp = "pilot" if s in PILOT else ("dev" if s in DEV else "locked")
    path = None
    try:
        p = ds.data_path(s)
        path = p[0] if isinstance(p, (list, tuple)) else p
        m = loadmat(path, squeeze_me=False, struct_as_record=True)
        root = m["eeg"].ravel()[0]
        sf = float(np.asarray(unwrap(root, "srate")).ravel()[0])
        n_tr = int(np.asarray(unwrap(root, "n_imagery_trials")).ravel()[0])
        ev = np.asarray(unwrap(root, "imagery_event")).ravel()

        jaw_raw = orient(list(unwrap(root, "noise").ravel())[JAW_SLOT])
        rest_raw = orient(unwrap(root, "rest"))
        j19 = jaw_raw[IDX19]
        resid = float(np.median(np.abs((j19 - j19.mean(0, keepdims=True)).mean(-1))))

        XL, safL, on, per = trials_both_bands(unwrap(root, "imagery_left"), ev, n_tr)
        XR, safR, _, _ = trials_both_bands(unwrap(root, "imagery_right"), ev, n_tr)

        bad = unwrap(root, "bad_trial_indices")
        v_l, v_r = mat_idx(bad["bad_trial_idx_voltage"], 0), mat_idx(bad["bad_trial_idx_voltage"], 1)
        e_l, e_r = mat_idx(bad["bad_trial_idx_mi"], 0), mat_idx(bad["bad_trial_idx_mi"], 1)
        k_l = [i for i in range(n_tr) if i not in v_l and i not in e_l]
        k_r = [i for i in range(n_tr) if i not in v_r and i not in e_r]

        rest_hf = float(band_power(car19_clean(rest_raw), sf, HF_BAND).mean())

        def donors(rw):
            """MI_d on the frozen 3 s grid at native rate, waveforms in both
            bands, filtered on the whole block and cut afterwards."""
            xn = car19_clean(rw)                       # 512 Hz, for MI_d only
            L = int(DONOR_WIN_S * sf)
            n_w = int(xn.shape[-1] // L)
            MId = [float(band_power(xn[:, k * L:(k + 1) * L], sf, HF_BAND).mean()
                         / max(rest_hf, 1e-300)) for k in range(n_w)]

            xt = to_target(xn)                          # 250 Hz, whole block
            d_all, s_all = bp(xt, DEC_BAND), bp(xt, SAF_BAND)
            Lt = int(DONOR_WIN_S * SF_TARGET)           # 750
            c = (Lt - NWIN) // 2                        # centre 625 of each window
            D, S = [], []
            for k in range(n_w):
                a = k * Lt + c
                D.append(d_all[:, a:a + NWIN].astype(np.float32))
                S.append(s_all[:, a:a + NWIN].astype(np.float32))
            return np.stack(D), np.stack(S), np.asarray(MId)

        jaw_dec, jaw_saf, jaw_mi = donors(jaw_raw)
        rest_dec, rest_saf, rest_mi = donors(rest_raw)

        # purity, defined exactly as in v2: safety band, own rest as background
        pj = rms(jaw_saf) ** 2
        pr_ = float(np.median(rms(rest_saf)) ** 2)
        purity = np.clip(1.0 - pr_ / np.maximum(pj, 1e-300), 0, 1)

        car_res = float(max(np.max(np.abs(jaw_dec.sum(1))),
                            np.max(np.abs(jaw_saf.sum(1)))))

        np.savez_compressed(
            f"{BANK}/sub-{s:02d}.npz",
            trials_L=XL, trials_R=XR, keep_L=np.array(k_l), keep_R=np.array(k_r),
            clean_saf_rms_L=safL, clean_saf_rms_R=safR,
            emg_L=np.array(sorted(e_l)), emg_R=np.array(sorted(e_r)),
            jaw_dec=jaw_dec, jaw_saf=jaw_saf, jaw_saf_rms=rms(jaw_saf),
            jaw_mi=jaw_mi, jaw_purity=purity,
            rest_dec=rest_dec, rest_saf=rest_saf, rest_saf_rms=rms(rest_saf),
            rest_mi=rest_mi, rest_hf_power=rest_hf,
            cue_onset=on, samples_per_trial=per, residual_dc=resid,
            car_residual=car_res, subject=s, group=grp,
            supersedes="cho_bank_v2, which stored only the 20-80 Hz donor copy")

        dcs.append(resid); purs += list(purity)
        if s in LEDGER:
            ld, lm, lp = LEDGER[s]
            drift.append((s, abs(resid - ld), abs(float(jaw_mi.max()) - lm),
                          abs(float(np.median(purity)) - lp)))
        print(f"{s:5d} {grp:>7} {resid:9.1f} {jaw_mi.max():9.2f} "
              f"{np.median(purity):7.3f} {car_res:9.2e} {len(k_l)+len(k_r):7d}")
        del m, root, XL, XR, jaw_dec, jaw_saf, rest_dec, rest_saf, jaw_raw, rest_raw
    except Exception as e:
        print(f"{s:5d} {grp:>7}  FAILED  {type(e).__name__}: {e}")
    finally:
        if path and os.path.exists(path):
            os.remove(path)
        gc.collect()

print("\n" + "=" * 78)
print("VERIFICATION against the v2 ledger")
print("=" * 78)
if drift:
    print(f"  {'subj':>5} {'d residDC':>11} {'d MI_d':>10} {'d purity':>10}")
    for s, a, b, c in drift:
        print(f"  {s:5d} {a:11.4f} {b:10.4f} {c:10.4f}")
    worst = max(c for _, _, _, c in drift)
    print(f"\n  worst purity drift: {worst:.4f}")
    if worst > 0.005:
        print("""
  PURITY DRIFTED. The frozen 0.90 threshold selects donors by this number, so
  a drift moves the cohort and the 18 eligible subjects are no longer the same
  18. Stop here and find the cause before running anything else.
""")
    else:
        print("  Purity reproduces v2. The frozen threshold and cohort still hold.")
else:
    print("  no ledger subjects rebuilt in this pass")

if dcs:
    print(f"\n  residual DC median {np.median(dcs):.1f}   (v2 gave 135158.0)")
    print(f"  purity median      {np.median(purs):.3f}   (v2 gave 0.914)")

print(f"""
{'=' * 78}
WHAT THIS BANK NOW CONTAINS, AND WHAT IT MEANS

  jaw_dec, rest_dec    8-32 Hz. THIS is what T4 adds to a trial.
  jaw_saf, rest_saf    20-80 Hz. Used only to set the SNR scale.
  jaw_saf_rms          the denominator of harmkit's scale_for_snr
  clean_saf_rms_L/R    per trial, the safety-band RMS over the frozen
                       0.5-3.0 s window. T4 builds clean_reference_rms from
                       the TRAINING trials of each fold out of these, never
                       from the test trials.

  The natural-SNR figure of -9.52 dB is retracted. It used the clean RMS in
  8-32 Hz where the convention calls for 20-80 Hz. Stage T3d recomputes it
  from clean_saf_rms and jaw_saf_rms. Record both the retraction and the
  mechanism in the ledger; this is the second retraction of that number.

  Nothing measured in T3b or T3c is affected. Correlation is scale invariant
  and both bands were stated explicitly there, so the |rho| result stands.

SAVE VERSION NOW, THEN NEW DATASET FROM THE OUTPUT.
{'=' * 78}""")

with open(f"{OUT}/T2d_RESULT.json", "w") as f:
    json.dump(dict(stage="T2d", supersedes="cho_bank_v2",
                   decoder_band=DEC_BAND, safety_band=SAF_BAND,
                   donor_window_s=DONOR_WIN_S, stored_samples=NWIN,
                   residual_dc_median=float(np.median(dcs)) if dcs else None,
                   purity_median=float(np.median(purs)) if purs else None,
                   ledger_drift=[dict(subject=s, d_dc=a, d_mid=b, d_purity=c)
                                 for s, a, b, c in drift]),
              f, indent=2, default=float)
print(f"written: {OUT}/T2d_RESULT.json")
