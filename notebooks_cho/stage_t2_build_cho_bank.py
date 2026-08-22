# REPO STATUS: SUPERSEDED. Produced bank v1, which is VOID. See ledger R2.
# Run order, retractions and full context: notebooks_cho/RUN_ORDER.md

"""
STAGE T2  ---  build the Cho2017 bank: jaw donors, rest donors, MI epochs.

Reads T2_PREREGISTRATION.json, prints its sha256, and refuses to run without
it.  The freeze has to be a file with a hash, not a comment in a script.

What this stage produces, per subject, as one npz:
    mi_X, mi_y          clean motor-imagery epochs, COMMON19, CAR19, 8-32 Hz
    mi_emg_X, mi_emg_y  trials the original authors flagged as EMG-contaminated
    jaw_D, jaw_mi       jaw-clench donor windows, 20-80 Hz, and their MI_d
    rest_D, rest_mi     resting donor windows, same pipeline, arm C
    diagnostics

What this stage does NOT produce: any harm statistic, any contamination, any
decoder accuracy.  Building the bank and reading the result on the same pass
is how a preregistration turns into decoration.

Three things the earlier stages forced into this design.

1.  imagery_left is (68, 358400) = 100 trials x 7 s at 512 Hz.  It is trial
    segmented and concatenated, which is what the MOABB warning about zero
    buffers was telling us.  Filtering it as if it were continuous puts an
    edge artefact on every trial boundary.  So each 7 s trial is filtered on
    its own, with padding, and only then is the middle 3 s cut out.  The 2 s
    of blank screen on each side of the imagery period absorbs the transient.

2.  Band power is measured by Welch PSD, never by a time-domain band-pass on
    a 10 s block.  That mistake produced the impossible T1b table.  The 20-80
    Hz donor filter is safe on 10 s because a 20 Hz lower edge needs a short
    kernel; a 0.5 Hz lower edge does not, and is not used anywhere.

3.  52 x 204 MB will not survive a session.  Each .mat is deleted after its
    npz is written, and a subject whose npz already exists is skipped, so the
    stage resumes after a disconnect.

Run as one Kaggle CPU cell.  Expect roughly one to two hours for all 52.
"""

import gc
import hashlib
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

PREREG_PATH = "/kaggle/input/datasets/erbosynshaimardan/t2-preregistration/T2_PREREGISTRATION.json"   # or upload next to the notebook
OUT_DIR = "/kaggle/working/cho_bank"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 78)
print("STAGE T2  build the Cho2017 bank")
print("=" * 78)

# ----------------------------------------------------------------------
# The freeze
# ----------------------------------------------------------------------
if not os.path.exists(PREREG_PATH):
    for alt in ("./T2_PREREGISTRATION.json", "/kaggle/working/T2_PREREGISTRATION.json"):
        if os.path.exists(alt):
            PREREG_PATH = alt
            break
    else:
        raise SystemExit(
            "STOP. T2_PREREGISTRATION.json not found.\n"
            "Commit it to the repository first, then attach it to the notebook.\n"
            "Running the bank build before the freeze exists defeats the point.")

with open(PREREG_PATH, "rb") as f:
    prereg_bytes = f.read()
PREREG_SHA = hashlib.sha256(prereg_bytes).hexdigest()
P = json.loads(prereg_bytes.decode("utf-8"))
print(f"preregistration sha256 : {PREREG_SHA}")
print(f"  record this in protocol/CHECKSUMS.md before reading any result below")

C = P["constants"]
SEED = P["seed"]
SF_NATIVE = float(C["sfreq_native_hz"])
SF_TARGET = float(C["sfreq_target_hz"])
NOTCH_HZ = float(C["notch_hz"])
EPOCH_S = float(C["epoch_s"])
DEC_BAND = tuple(C["decoder_band_hz"])
DON_BAND = tuple(C["donor_band_hz"])
JAW_SLOT = int(P["block_identification"]["jaw_slot"])
MI_D_ACCEPT = float(P["primary_criterion"]["window_accept"].split(">=")[-1])
MIN_SUBJECTS = int("".join(ch for ch in P["primary_criterion"]["feasibility_PASS"]
                           if ch.isdigit())[:2])
PILOT = set(P["pilot_exclusion"]["subjects"])
DEV = set(P["cohort"]["development"])
LOCKED = set(P["cohort"]["locked"])
ALL_SUBJECTS = sorted(PILOT | DEV | LOCKED)

HF_BAND, MAINS, MAINS_HW, NPERSEG = (40.0, 100.0), [60.0, 120.0, 180.0], 2.0, 512

COMMON19 = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
            "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP3", "CP1", "CP2", "CP4", "Pz", "POz"]

print(f"jaw slot {JAW_SLOT}, epoch {EPOCH_S} s, notch {NOTCH_HZ} Hz, "
      f"MI_d accept >= {MI_D_ACCEPT}, feasibility needs >= {MIN_SUBJECTS} subjects")
print(f"pilot {sorted(PILOT)}")
print(f"development {len(DEV)}, locked {len(LOCKED)}")

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
print(f"channels {len(CH_ALL)}, COMMON19 indices max {IDX19.max()}")


# ----------------------------------------------------------------------
def unwrap(struct, field):
    v = struct[field]
    while isinstance(v, np.ndarray) and v.dtype == object and v.size == 1:
        v = v.ravel()[0]
    return v


def orient(a):
    a = np.asarray(a, dtype=np.float64)
    return a if a.shape[0] == len(CH_ALL) else a.T


def car19(a):
    """Subset to COMMON19, then average-reference over exactly those 19."""
    x = a[IDX19]
    return x - x.mean(axis=0, keepdims=True)


def to_target(x):
    g = int(round(SF_NATIVE / np.gcd(int(SF_NATIVE), int(SF_TARGET))))
    up = int(round(SF_TARGET * g / SF_NATIVE))
    return resample_poly(x, up, g, axis=-1)


def band_power(x, sf, band):
    """Welch band power per channel, mains bins dropped, no filter designed."""
    x = detrend(x, axis=-1, type="linear")
    nps = int(min(NPERSEG, x.shape[-1]))
    f, Pxx = welch(x, fs=sf, nperseg=nps, noverlap=nps // 2, axis=-1)
    m = (f >= band[0]) & (f <= band[1])
    for m0 in MAINS:
        m &= ~((f > m0 - MAINS_HW) & (f < m0 + MAINS_HW))
    return np.trapezoid(Pxx[..., m], f[m], axis=-1) if m.any() else np.zeros(x.shape[0])


def mat_indices(cell, k):
    """MATLAB 1-based bad-trial indices out of a (1,2) cell, possibly empty."""
    try:
        v = np.asarray(cell).ravel()[k]
        v = np.asarray(v).ravel()
        return set(int(i) - 1 for i in v if np.isfinite(i))
    except Exception:
        return set()


def cut_epochs(stream68, event_row, n_trials, tag):
    """Reshape into trials, filter each trial alone, then cut the imagery window.

    stream68 is (68, 100*3584) at 512 Hz, already trial-segmented by the file.
    Filtering it as one stream would ring at every 7 s boundary.
    """
    x = car19(orient(stream68))                      # (19, N) at 512 Hz
    x = to_target(x)                                 # (19, N') at 250 Hz
    per = int(round(x.shape[-1] / n_trials))
    if per * n_trials != x.shape[-1]:
        raise ValueError(f"{tag}: {x.shape[-1]} not divisible into {n_trials} trials")
    trials = x.reshape(19, n_trials, per).transpose(1, 0, 2).copy()   # (T,19,per)

    ev = np.asarray(event_row).ravel().astype(np.int64)
    ev = ev[: n_trials * int(round(SF_NATIVE * per / SF_TARGET))]
    ev = ev.reshape(n_trials, -1)
    onsets = []
    for row in ev:
        nz = np.flatnonzero(row)
        onsets.append(int(nz[0] * SF_TARGET / SF_NATIVE) if nz.size else -1)
    onsets = np.array(onsets)
    if (onsets < 0).any():
        raise ValueError(f"{tag}: {int((onsets < 0).sum())} trials carry no cue marker")

    n_ep = int(round(EPOCH_S * SF_TARGET))
    out = np.empty((n_trials, 19, n_ep), dtype=np.float32)
    for i in range(n_trials):
        t = mne.filter.notch_filter(trials[i], SF_TARGET, [NOTCH_HZ], verbose=False)
        t = mne.filter.filter_data(t, SF_TARGET, DEC_BAND[0], DEC_BAND[1],
                                   pad="reflect_limited", verbose=False)
        a = onsets[i]
        if a + n_ep > per:                            # clamp, then report it
            a = per - n_ep
        out[i] = t[:, a:a + n_ep]
    return out, onsets, per


# ======================================================================
print("\n" + "-" * 78)
print("building")
print("-" * 78)
print(f"{'subj':>5} {'group':>6} {'MI ok':>6} {'EMG':>5} {'volt':>5} "
      f"{'jaw win':>8} {'MI_d max':>9} {'rest win':>9} {'cue s':>7}")

summary = []
for s in ALL_SUBJECTS:
    npz = f"{OUT_DIR}/sub-{s:02d}.npz"
    group = "pilot" if s in PILOT else ("dev" if s in DEV else "locked")
    if os.path.exists(npz):
        d = np.load(npz)
        summary.append(dict(subject=s, group=group, cached=True,
                            mi_d_max=float(d["jaw_mi"].max()) if d["jaw_mi"].size else 0.0,
                            n_jaw=int(d["jaw_mi"].size),
                            n_accepted=int((d["jaw_mi"] >= MI_D_ACCEPT).sum()),
                            n_mi=int(d["mi_y"].size), n_emg=int(d["mi_emg_y"].size)))
        print(f"{s:5d} {group:>6}  cached")
        continue

    path = None
    try:
        p = ds.data_path(s)
        path = p[0] if isinstance(p, (list, tuple)) else p
        m = loadmat(path, squeeze_me=False, struct_as_record=True)
        root = m["eeg"].ravel()[0]
        sf = float(np.asarray(unwrap(root, "srate")).ravel()[0])
        if abs(sf - SF_NATIVE) > 1e-6:
            raise ValueError(f"srate {sf}, expected {SF_NATIVE}")

        n_tr = int(np.asarray(unwrap(root, "n_imagery_trials")).ravel()[0])
        ev = np.asarray(unwrap(root, "imagery_event")).ravel()
        if s == ALL_SUBJECTS[0]:
            print(f"      [first subject] frame = "
                  f"{np.asarray(unwrap(root, 'frame')).ravel().tolist()}, "
                  f"n_imagery_trials = {n_tr}, nonzero event samples = {int((ev != 0).sum())}")

        XL, oL, per = cut_epochs(unwrap(root, "imagery_left"), ev, n_tr, "left")
        XR, oR, _ = cut_epochs(unwrap(root, "imagery_right"), ev, n_tr, "right")

        bad = unwrap(root, "bad_trial_indices")
        v_l, v_r = mat_indices(bad["bad_trial_idx_voltage"], 0), mat_indices(bad["bad_trial_idx_voltage"], 1)
        e_l, e_r = mat_indices(bad["bad_trial_idx_mi"], 0), mat_indices(bad["bad_trial_idx_mi"], 1)

        keep_l = [i for i in range(n_tr) if i not in v_l and i not in e_l]
        keep_r = [i for i in range(n_tr) if i not in v_r and i not in e_r]
        emg_l = [i for i in sorted(e_l) if i not in v_l and i < n_tr]
        emg_r = [i for i in sorted(e_r) if i not in v_r and i < n_tr]

        mi_X = np.concatenate([XL[keep_l], XR[keep_r]], 0)
        mi_y = np.concatenate([np.zeros(len(keep_l), np.int8), np.ones(len(keep_r), np.int8)])
        emg_X = np.concatenate([XL[emg_l], XR[emg_r]], 0) if (emg_l or emg_r) else np.zeros((0, 19, XL.shape[-1]), np.float32)
        emg_y = np.concatenate([np.zeros(len(emg_l), np.int8), np.ones(len(emg_r), np.int8)])

        # ---- donors -------------------------------------------------
        noise = unwrap(root, "noise")
        blocks = list(noise.ravel()) if noise.dtype == object else list(np.asarray(noise))
        jaw_raw = orient(blocks[JAW_SLOT])
        rest_raw = orient(unwrap(root, "rest"))

        rest_hf = float(band_power(car19(rest_raw), sf, HF_BAND).mean())

        def make_donors(raw):
            x19 = car19(raw)
            n_w = int(x19.shape[-1] // (EPOCH_S * sf))
            D, MId = [], []
            for k in range(n_w):
                a = int(k * EPOCH_S * sf)
                w = x19[:, a:a + int(EPOCH_S * sf)]
                MId.append(float(band_power(w, sf, HF_BAND).mean() / max(rest_hf, 1e-300)))
                y = to_target(w)
                y = mne.filter.notch_filter(y, SF_TARGET, [NOTCH_HZ], verbose=False)
                y = mne.filter.filter_data(y, SF_TARGET, DON_BAND[0], DON_BAND[1],
                                           pad="reflect_limited", verbose=False)
                D.append(y.astype(np.float32))
            return (np.stack(D) if D else np.zeros((0, 19, 1), np.float32),
                    np.asarray(MId, np.float64))

        jaw_D, jaw_mi = make_donors(jaw_raw)
        rest_D, rest_mi = make_donors(rest_raw)

        np.savez_compressed(
            npz, mi_X=mi_X, mi_y=mi_y, mi_emg_X=emg_X, mi_emg_y=emg_y,
            jaw_D=jaw_D, jaw_mi=jaw_mi, rest_D=rest_D, rest_mi=rest_mi,
            rest_hf_power=rest_hf, cue_onset_samples=oL, samples_per_trial=per,
            subject=s, group=group, prereg_sha=PREREG_SHA)

        n_acc = int((jaw_mi >= MI_D_ACCEPT).sum())
        summary.append(dict(subject=s, group=group, cached=False,
                            mi_d_max=float(jaw_mi.max()) if jaw_mi.size else 0.0,
                            n_jaw=int(jaw_mi.size), n_accepted=n_acc,
                            n_mi=int(mi_y.size), n_emg=int(emg_y.size),
                            n_bad_voltage=len(v_l | v_r), n_rest=int(rest_mi.size),
                            cue_s=float(np.median(oL) / SF_TARGET)))
        print(f"{s:5d} {group:>6} {mi_y.size:6d} {emg_y.size:5d} {len(v_l | v_r):5d} "
              f"{jaw_mi.size:8d} {jaw_mi.max():9.2f} {rest_mi.size:9d} "
              f"{np.median(oL) / SF_TARGET:7.2f}")

        del m, root, XL, XR, mi_X, emg_X, jaw_D, rest_D, jaw_raw, rest_raw
    except Exception as e:
        print(f"{s:5d} {group:>6}  FAILED  {type(e).__name__}: {e}")
        summary.append(dict(subject=s, group=group, failed=f"{type(e).__name__}: {e}"))
    finally:
        if path and os.path.exists(path):
            os.remove(path)          # 52 x 204 MB will not fit, delete as we go
        gc.collect()

# ======================================================================
print("\n" + "=" * 78)
print("PREREGISTERED FEASIBILITY GATE, confirmatory subjects only")
print("=" * 78)

conf = [r for r in summary if r["group"] != "pilot" and "failed" not in r]
pilot = [r for r in summary if r["group"] == "pilot" and "failed" not in r]
acc = [r for r in conf if r.get("n_accepted", 0) >= 1]
passed = len(acc) >= MIN_SUBJECTS

print(f"  confirmatory subjects processed : {len(conf)}/47")
print(f"  with >= 1 window at MI_d >= {MI_D_ACCEPT}  : {len(acc)}  "
      f"(need >= {MIN_SUBJECTS})")
print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
if conf:
    v = np.array([r.get("mi_d_max", 0.0) for r in conf])
    print(f"  MI_d max per subject: median {np.median(v):.2f}, "
          f"range {v.min():.2f} to {v.max():.2f}")
print(f"  pilot, reported separately and excluded: "
      f"{[round(r.get('mi_d_max', 0.0), 2) for r in pilot]}")

with open("/kaggle/working/T2_RESULT.json", "w") as f:
    json.dump(dict(stage="T2", prereg_sha256=PREREG_SHA, seed=SEED,
                   mi_d_accept=MI_D_ACCEPT, min_subjects=MIN_SUBJECTS,
                   n_confirmatory=len(conf), n_accepted=len(acc),
                   verdict="PASS" if passed else "FAIL",
                   subjects=summary), f, indent=2, default=float)

print(f"""
written: /kaggle/working/T2_RESULT.json and {OUT_DIR}/sub-XX.npz

Next, and in this order:
  publish {OUT_DIR} as a Kaggle Dataset so the 10.6 GB download never repeats
  commit T2_RESULT.json and the preregistration sha256 to protocol/
  only then write Stage T3, which trains the decoders and freezes the arm
  A/B/C analysis rules

Do not open the locked subjects {sorted(LOCKED)} until the development
analysis has been run once and written to disk.
""")
