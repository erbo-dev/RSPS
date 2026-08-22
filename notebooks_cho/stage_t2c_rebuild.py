# REPO STATUS: SUPERSEDED by t2d. Produced bank v2. See ledger R4.
# Run order, retractions and full context: notebooks_cho/RUN_ORDER.md

"""
STAGE T2c  ---  rebuild the bank correctly, and store the whole trial.

Why v1 is void
--------------
Stage T2 wrote jaw_D and rest_D through this path:

    w = x19[:, a:a+1536]      # CAR19 applied, per-channel DC still present
    y = to_target(w)          # resample_poly, no detrend

scipy.signal.resample_poly defaults to padtype='constant', which pads with
ZEROS.  Raw BioSemi carries per-channel DC in the hundreds of thousands and
CAR19 removes only the component common to the 19 channels, so a large
residual step sits at both edges of every window.  The anti-alias FIR rings
off that step and the ringing spreads across the band, including 20-80 Hz.

Signature in the v1 output, which is how this was caught: the three jaw
windows of a subject had RMS agreeing to within one per cent even when their
MI_d differed by three orders of magnitude.  Donor RMS was tracking the DC
offset, not the muscle.

Confirmed numerically.  With a broadband component 43x in power over
background, measured purity 1 - P_rest/P_jaw comes out at
    DC =    0  ->  0.975
    DC = 2000  ->  0.851
    DC = 5000  ->  0.515
and detrending with padtype='line' restores 0.975-0.980 at every DC.  The
mechanism is established.  Whether it accounts for the full collapse to 0.053
is NOT established, because the residual DC after CAR19 was never measured.
This script measures it.  If the fix does not restore purity, the diagnosis
was wrong and the script says so instead of proceeding.

What v1 numbers survive
-----------------------
MI_d survives: band_power detrends internally, so the feasibility gate of
45/47 stands, as does the cohort split and the frame/cue verification.
Everything derived from jaw_D or rest_D is void: natural SNR, purity, and the
purity columns of the threshold table.

What changes besides the fix
----------------------------
The whole 7 s trial is stored, band-passed once with the 2 s blank periods
acting as guards, so Stage T3 can cut any imagery window without refiltering
and without another download.  That removes the epoch-window degree of
freedom from the analysis stage instead of carrying it forward.

Output goes to cho_bank_v2.  v1 is not overwritten; it is superseded and the
reason is recorded.

Run as one Kaggle CPU cell.  Expect roughly one to two hours and 10.6 GB.
"""

import gc
import hashlib
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

PREREG_PATH = "/kaggle/input/datasets/erbosynshaimardan/t2-preregistration/T2_PREREGISTRATION.json"
OUT_DIR = "/kaggle/working/cho_bank_v2"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 78)
print("STAGE T2c  rebuild with the DC fix, store the whole trial")
print("=" * 78)

if not os.path.exists(PREREG_PATH):
    for alt in ("./T2_PREREGISTRATION.json", "/kaggle/working/T2_PREREGISTRATION.json"):
        if os.path.exists(alt):
            PREREG_PATH = alt
            break
    else:
        raise SystemExit("STOP. T2_PREREGISTRATION.json not found.")

with open(PREREG_PATH, "rb") as f:
    pb = f.read()
PREREG_SHA = hashlib.sha256(pb).hexdigest()
P = json.loads(pb.decode("utf-8"))
print(f"preregistration sha256 : {PREREG_SHA}")
if PREREG_SHA != "23b32a592ec64b3d5de84d890b0174a9a148bee79616171b95c30c2d92710898":
    print("  WARNING: this differs from the sha recorded at the Stage T2 run.")
    print("  The freeze must not have changed. Investigate before trusting anything.")

C = P["constants"]
SF_NATIVE, SF_TARGET = float(C["sfreq_native_hz"]), float(C["sfreq_target_hz"])
NOTCH_HZ = float(C["notch_hz"])
DEC_BAND, DON_BAND = tuple(C["decoder_band_hz"]), tuple(C["donor_band_hz"])
DONOR_WIN_S = float(C["epoch_s"])          # 3 s donor windows, unchanged
JAW_SLOT = int(P["block_identification"]["jaw_slot"])
MI_D_ACCEPT = 3.0
PILOT, DEV, LOCKED = set(P["pilot_exclusion"]["subjects"]), set(P["cohort"]["development"]), set(P["cohort"]["locked"])
ALL_SUBJECTS = sorted(PILOT | DEV | LOCKED)

HF_BAND, MAINS, MAINS_HW, NPERSEG = (40.0, 100.0), [60.0, 120.0, 180.0], 2.0, 512
COMMON19 = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
            "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP3", "CP1", "CP2", "CP4", "Pz", "POz"]

try:
    import mne
    from moabb.datasets import Cho2017
    from scipy.io import loadmat
    from scipy.signal import detrend, resample_poly, welch
    from scipy.stats import spearmanr
except ImportError:
    os.system(f"{sys.executable} -m pip install -q mne moabb")
    import mne
    from moabb.datasets import Cho2017
    from scipy.io import loadmat
    from scipy.signal import detrend, resample_poly, welch
    from scipy.stats import spearmanr

mne.set_log_level("ERROR")
ds = Cho2017()

raw_mi = None
for a in ds.get_data(subjects=[ALL_SUBJECTS[0]]).values():
    for b in a.values():
        for r in b.values():
            raw_mi = r
            break
        break
    break
CH_ALL = [c for c in raw_mi.ch_names if c.upper() not in ("STIM", "STI 014")]
IDX19 = np.array([CH_ALL.index(c) for c in COMMON19])
del raw_mi
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
    """Subset to COMMON19, CAR over those 19, then remove per-channel DC and drift.

    The detrend is the fix. Without it the residual per-channel offset survives
    CAR and turns every subsequent resample into a step response.
    """
    x = a[IDX19]
    x = x - x.mean(axis=0, keepdims=True)
    return detrend(x, axis=-1, type="linear")


def to_target(x):
    g = int(round(SF_NATIVE / np.gcd(int(SF_NATIVE), int(SF_TARGET))))
    up = int(round(SF_TARGET * g / SF_NATIVE))
    return resample_poly(x, up, g, axis=-1, padtype="line")   # never zero-pad


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


def full_trials(stream68, ev, n_tr):
    """Whole 7 s trials, band-passed once, no window cut. T3 cuts what it likes."""
    x = car19_clean(orient(stream68))
    x = to_target(x)
    per = int(round(x.shape[-1] / n_tr))
    if per * n_tr != x.shape[-1]:
        raise ValueError(f"{x.shape[-1]} does not divide into {n_tr} trials")
    tr = x.reshape(19, n_tr, per).transpose(1, 0, 2).copy()
    out = np.empty((n_tr, 19, per), np.float32)
    for i in range(n_tr):
        t = mne.filter.notch_filter(tr[i], SF_TARGET, [NOTCH_HZ], verbose=False)
        out[i] = mne.filter.filter_data(t, SF_TARGET, DEC_BAND[0], DEC_BAND[1],
                                        pad="reflect_limited", verbose=False)
    e = np.asarray(ev).ravel().astype(np.int64)
    e = e[: n_tr * int(round(SF_NATIVE * per / SF_TARGET))].reshape(n_tr, -1)
    on = np.array([int(np.flatnonzero(r)[0] * SF_TARGET / SF_NATIVE)
                   if np.flatnonzero(r).size else -1 for r in e])
    if (on < 0).any():
        raise ValueError("a trial carries no cue marker")
    return out, on, per


print(f"\n{'subj':>5} {'grp':>7} {'resid DC':>10} {'MI_d max':>10} "
      f"{'purity med':>11} {'SNRnat med':>11} {'trials':>7}")

summary, all_pur, all_mid = [], [], []
for s in ALL_SUBJECTS:
    npz, path = f"{OUT_DIR}/sub-{s:02d}.npz", None
    grp = "pilot" if s in PILOT else ("dev" if s in DEV else "locked")
    if os.path.exists(npz):
        d = np.load(npz)
        all_pur += list(d["jaw_purity"]); all_mid += list(d["jaw_mi"])
        print(f"{s:5d} {grp:>7}  cached")
        continue
    try:
        p = ds.data_path(s)
        path = p[0] if isinstance(p, (list, tuple)) else p
        m = loadmat(path, squeeze_me=False, struct_as_record=True)
        root = m["eeg"].ravel()[0]
        sf = float(np.asarray(unwrap(root, "srate")).ravel()[0])
        n_tr = int(np.asarray(unwrap(root, "n_imagery_trials")).ravel()[0])
        ev = np.asarray(unwrap(root, "imagery_event")).ravel()

        # ---- the measurement that closes the diagnosis --------------
        jaw_raw = orient(list(unwrap(root, "noise").ravel())[JAW_SLOT])
        j19 = jaw_raw[IDX19]
        resid_dc = float(np.median(np.abs((j19 - j19.mean(0, keepdims=True)).mean(-1))))

        XL, on, per = full_trials(unwrap(root, "imagery_left"), ev, n_tr)
        XR, _, _ = full_trials(unwrap(root, "imagery_right"), ev, n_tr)

        bad = unwrap(root, "bad_trial_indices")
        v_l, v_r = mat_idx(bad["bad_trial_idx_voltage"], 0), mat_idx(bad["bad_trial_idx_voltage"], 1)
        e_l, e_r = mat_idx(bad["bad_trial_idx_mi"], 0), mat_idx(bad["bad_trial_idx_mi"], 1)
        k_l = [i for i in range(n_tr) if i not in v_l and i not in e_l]
        k_r = [i for i in range(n_tr) if i not in v_r and i not in e_r]

        rest_raw = orient(unwrap(root, "rest"))
        rest_hf = float(band_power(car19_clean(rest_raw), sf, HF_BAND).mean())

        def donors(raw):
            x = car19_clean(raw)
            nw, L = int(x.shape[-1] // (DONOR_WIN_S * sf)), int(DONOR_WIN_S * sf)
            D, MId = [], []
            for k in range(nw):
                w = x[:, k * L:(k + 1) * L]
                MId.append(float(band_power(w, sf, HF_BAND).mean() / max(rest_hf, 1e-300)))
                y = to_target(w)
                y = mne.filter.notch_filter(y, SF_TARGET, [NOTCH_HZ], verbose=False)
                D.append(mne.filter.filter_data(y, SF_TARGET, DON_BAND[0], DON_BAND[1],
                                                pad="reflect_limited", verbose=False).astype(np.float32))
            return np.stack(D), np.asarray(MId)

        jaw_D, jaw_mi = donors(jaw_raw)
        rest_D, rest_mi = donors(rest_raw)

        def rms(z):
            return np.sqrt((np.asarray(z, float) ** 2).mean(axis=(-2, -1)))

        pj, pr_ = rms(jaw_D) ** 2, float(np.median(rms(rest_D)) ** 2)
        purity = np.clip(1.0 - pr_ / np.maximum(pj, 1e-300), 0, 1)

        n_ep = int(round(3.0 * SF_TARGET))
        a0 = int(np.median(on))
        clean_rms = float(np.median(rms(np.concatenate(
            [XL[k_l][:, :, a0:a0 + n_ep], XR[k_r][:, :, a0:a0 + n_ep]], 0))))
        snr_nat = 20 * np.log10(clean_rms / np.maximum(rms(jaw_D), 1e-300))

        np.savez_compressed(
            npz, trials_L=XL, trials_R=XR, keep_L=np.array(k_l), keep_R=np.array(k_r),
            emg_L=np.array(sorted(e_l)), emg_R=np.array(sorted(e_r)),
            jaw_D=jaw_D, jaw_mi=jaw_mi, jaw_purity=purity, jaw_snr_nat_db=snr_nat,
            rest_D=rest_D, rest_mi=rest_mi, rest_hf_power=rest_hf,
            cue_onset=on, samples_per_trial=per, residual_dc=resid_dc,
            clean_rms_3s=clean_rms, subject=s, group=grp,
            prereg_sha=PREREG_SHA, supersedes="cho_bank v1, void: resample_poly zero-pad")

        all_pur += list(purity); all_mid += list(jaw_mi)
        summary.append(dict(subject=s, group=grp, residual_dc=resid_dc,
                            mi_d_max=float(jaw_mi.max()), purity_med=float(np.median(purity)),
                            snr_nat_med=float(np.median(snr_nat)),
                            n_trials=len(k_l) + len(k_r)))
        print(f"{s:5d} {grp:>7} {resid_dc:10.1f} {jaw_mi.max():10.2f} "
              f"{np.median(purity):11.3f} {np.median(snr_nat):11.2f} {len(k_l)+len(k_r):7d}")
        del m, root, XL, XR, jaw_D, rest_D, jaw_raw, rest_raw
    except Exception as e:
        print(f"{s:5d} {grp:>7}  FAILED  {type(e).__name__}: {e}")
        summary.append(dict(subject=s, group=grp, failed=f"{type(e).__name__}: {e}"))
    finally:
        if path and os.path.exists(path):
            os.remove(path)
        gc.collect()

# ======================================================================
print("\n" + "=" * 78)
print("FALSIFICATION CHECK on the v1 diagnosis")
print("=" * 78)
pur, mid = np.array(all_pur), np.array(all_mid)
ok = np.isfinite(pur) & np.isfinite(mid)
rho, pv = spearmanr(mid[ok], pur[ok]) if ok.sum() > 2 else (np.nan, np.nan)
dc = [r["residual_dc"] for r in summary if "residual_dc" in r]
print(f"  residual per-channel DC after CAR19 : median {np.median(dc):.1f}" if dc else "")
print(f"  purity  : median {np.median(pur):.3f}  (v1 gave 0.053)")
print(f"  Spearman(MI_d, purity) = {rho:+.3f}, p = {pv:.2g}  (v1 gave +0.425)")

if np.median(pur) > 0.5 and rho > 0.6:
    print("""
  DIAGNOSIS CONFIRMED. The v1 collapse was the zero-padded resample. Purity is
  now consistent with MI_d, which is what physics requires of a broadband
  muscle artifact. v1 is void and the reason is recorded.
""")
else:
    print("""
  DIAGNOSIS NOT CONFIRMED. The fix did not restore purity, so the zero-padded
  resample was not the whole story, or not the story at all. Do NOT proceed to
  T3. Plot one jaw window and one rest window for a high-MI_d subject and look
  at them before writing another line of code.
""")

print("""
Whatever happens above, the natural-SNR line of the v1 run stays retracted.
Recompute it from this bank and report the retraction in the ledger alongside
the mechanism, the numerical confirmation, and the check that caught it.

Next: T3. Freeze, in one file, before any decoder accuracy exists:
  the imagery window cut from the stored 7 s trial
  one MI_d and purity threshold pair
  the arm A / B / C rules with donor counts matched at 3 per host
  the artifact-only validity gate at 0.55
""")

with open("/kaggle/working/T2c_RESULT.json", "w") as f:
    json.dump(dict(stage="T2c", prereg_sha256=PREREG_SHA, supersedes="cho_bank v1",
                   purity_median=float(np.median(pur)) if pur.size else None,
                   spearman_mid_purity=float(rho) if np.isfinite(rho) else None,
                   residual_dc_median=float(np.median(dc)) if dc else None,
                   subjects=summary), f, indent=2, default=float)
print("written: /kaggle/working/T2c_RESULT.json")
