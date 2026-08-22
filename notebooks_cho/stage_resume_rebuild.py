# REPO STATUS: UTILITY. Rebuilds the bank and models after a lost Kaggle session, with a ledger regression check.
# Run order, retractions and full context: notebooks_cho/RUN_ORDER.md

"""
STAGE RESUME  ---  rebuild everything T3c needs after a Kaggle session died.

A dropped session wipes /kaggle/working. Anything under /kaggle/input survives.
This script takes inventory, rebuilds only what is missing, and verifies the
rebuild against values already recorded, so a silent drift cannot slip in.

Rebuilds, in order:
    cho_bank_v2/sub-XX.npz   Stage T2c logic, with the DC fix
    T3_MODELS.npz            frozen per-fold CSP and LDA
    T3_RESULT.json           clean accuracy, eligibility, the artifact-only figure

Then run stage_t3c_surrogate_floor.py, unchanged.

Every constant below is copied from the frozen protocol and must not be edited.
If a rebuilt number disagrees with the ledger, the script says so instead of
carrying on: that is the whole point of running a rebuild against a ledger.

Cost: about one to two hours and 10.6 GB if the bank has to come back from
GigaDB, minutes if it is already attached as a dataset.

READ THIS BEFORE YOU START, or you will be here a third time
------------------------------------------------------------
When the bank finishes, press Save Version (Save and Run All), wait for the
commit, then New Dataset from the notebook output. Attach that dataset to
every later notebook. /kaggle/working only survives a session if the notebook
was committed. That is why this file exists.
"""

import gc
import glob
import hashlib
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

OUT = "/kaggle/working"
BANK = f"{OUT}/cho_bank_v2"
os.makedirs(BANK, exist_ok=True)

# ---------------------------------------------------------------- frozen
SF_NATIVE, SF_TARGET, NOTCH_HZ = 512.0, 250.0, 60.0
DEC_BAND, DON_BAND = (8.0, 32.0), (20.0, 80.0)
DONOR_WIN_S, JAW_SLOT = 3.0, 3
HF_BAND, MAINS, MAINS_HW, NPERSEG = (40.0, 100.0), [60.0, 120.0, 180.0], 2.0, 512
W0, NWIN, NFOLD = 0.5, 625, 5
PURITY, ACC_MIN, SEED = 0.90, 0.60, 20260813
COMMON19 = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
            "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP3", "CP1", "CP2", "CP4", "Pz", "POz"]

PILOT = [1, 2, 3, 4, 5]
DEV = [6, 7, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 25, 26, 27, 29, 31,
       32, 33, 34, 35, 37, 38, 39, 40, 41, 44, 47, 49, 50]
LOCKED = [8, 15, 16, 17, 24, 28, 30, 36, 42, 43, 45, 46, 48, 51, 52]
ALL = sorted(PILOT + DEV + LOCKED)

# Recorded earlier in this project. A rebuild must reproduce these.
LEDGER_ACC = {6: 0.6450, 9: 0.7292, 10: 0.8300, 14: 0.9800, 20: 0.7250,
              23: 0.8550, 34: 0.4300, 38: 0.4600, 41: 0.9196, 47: 0.6450}
LEDGER_N_ELIGIBLE = 18
LEDGER_ACC_MEDIAN = 0.6825
LEDGER_PURITY_MEDIAN = 0.914
LEDGER_RESID_DC_MEDIAN = 135158.0
LEDGER_PREREG_T2 = "23b32a592ec64b3d5de84d890b0174a9a148bee79616171b95c30c2d92710898"

print("=" * 78)
print("STAGE RESUME  rebuild after a lost session")
print("=" * 78)

# ---------------------------------------------------------------- inventory
def find(name):
    """Locate a file anywhere Kaggle might have mounted it."""
    for p in ([f"{OUT}/{name}", f"./{name}"]
              + sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))):
        if os.path.exists(p):
            return p
    return None


def find_bank():
    if len(glob.glob(f"{BANK}/sub-*.npz")) >= len(ALL):
        return BANK
    for d in sorted(glob.glob("/kaggle/input/**/", recursive=True)):
        if len(glob.glob(f"{d}/sub-*.npz")) >= len(ALL):
            return d.rstrip("/")
    return None


print("\ninventory")
src_bank = find_bank()
have_models = find("T3_MODELS.npz")
have_t3res = find("T3_RESULT.json")
prereg2 = find("T2_PREREGISTRATION.json")
amend2 = find("T3_PREREGISTRATION_AMENDMENT_2.json")

print(f"  cho_bank_v2 (52 npz)              : {src_bank or 'MISSING, will rebuild'}")
print(f"  T3_MODELS.npz                     : {have_models or 'MISSING, will rebuild'}")
print(f"  T3_RESULT.json                    : {have_t3res or 'MISSING, will rebuild'}")
print(f"  T2_PREREGISTRATION.json           : {prereg2 or 'not found'}")
print(f"  T3_PREREGISTRATION_AMENDMENT_2    : {amend2 or 'not found, T3c will refuse to run'}")
print(f"  partial bank in working           : {len(glob.glob(f'{BANK}/sub-*.npz'))} npz")

if prereg2:
    sha = hashlib.sha256(open(prereg2, "rb").read()).hexdigest()
    ok = sha == LEDGER_PREREG_T2
    print(f"  T2 preregistration sha256         : {sha[:16]}...  "
          f"{'matches ledger' if ok else 'DOES NOT MATCH LEDGER'}")
    if not ok:
        print("    The freeze must not have changed. Stop and investigate before")
        print("    trusting anything rebuilt below.")
else:
    print("  subject split falls back to the lists embedded in this file;")
    print("  attach the preregistration so the freeze is verified, not assumed")

if src_bank and src_bank != BANK:
    BANK = src_bank
    print(f"\nusing attached bank at {BANK}, no download needed")

try:
    import mne
    from mne.decoding import CSP
    from moabb.datasets import Cho2017
    from scipy.io import loadmat
    from scipy.signal import detrend, resample_poly, welch
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
except ImportError:
    os.system(f"{sys.executable} -m pip install -q mne moabb scikit-learn")
    import mne
    from mne.decoding import CSP
    from moabb.datasets import Cho2017
    from scipy.io import loadmat
    from scipy.signal import detrend, resample_poly, welch
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
mne.set_log_level("ERROR")
rng = np.random.default_rng(SEED)

# ============================================================ PART 1: bank
need = [s for s in ALL if not os.path.exists(f"{BANK}/sub-{s:02d}.npz")]
if need:
    if not os.access(BANK, os.W_OK):
        BANK = f"{OUT}/cho_bank_v2"
        os.makedirs(BANK, exist_ok=True)
        need = [s for s in ALL if not os.path.exists(f"{BANK}/sub-{s:02d}.npz")]
    print(f"\n{'-' * 78}\nPART 1  rebuilding {len(need)} subjects into {BANK}\n{'-' * 78}")

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
        """Subset, CAR over those 19, then detrend. The detrend IS the v2 fix."""
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

    def full_trials(stream, ev, n_tr):
        x = to_target(car19_clean(orient(stream)))
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

    dcs, purs = [], []
    print(f"{'subj':>5} {'grp':>7} {'resid DC':>10} {'MI_d max':>10} {'purity':>8} {'trials':>7}")
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
            j19 = jaw_raw[IDX19]
            resid = float(np.median(np.abs((j19 - j19.mean(0, keepdims=True)).mean(-1))))

            XL, on, per = full_trials(unwrap(root, "imagery_left"), ev, n_tr)
            XR, _, _ = full_trials(unwrap(root, "imagery_right"), ev, n_tr)

            bad = unwrap(root, "bad_trial_indices")
            v_l, v_r = mat_idx(bad["bad_trial_idx_voltage"], 0), mat_idx(bad["bad_trial_idx_voltage"], 1)
            e_l, e_r = mat_idx(bad["bad_trial_idx_mi"], 0), mat_idx(bad["bad_trial_idx_mi"], 1)
            k_l = [i for i in range(n_tr) if i not in v_l and i not in e_l]
            k_r = [i for i in range(n_tr) if i not in v_r and i not in e_r]

            rest_raw = orient(unwrap(root, "rest"))
            rest_hf = float(band_power(car19_clean(rest_raw), sf, HF_BAND).mean())

            def donors(rw):
                x = car19_clean(rw)
                L = int(DONOR_WIN_S * sf)
                D, MId = [], []
                for k in range(int(x.shape[-1] // L)):
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

            n_ep, a0 = int(round(3.0 * SF_TARGET)), int(np.median(on))
            crms = float(np.median(rms(np.concatenate(
                [XL[k_l][:, :, a0:a0 + n_ep], XR[k_r][:, :, a0:a0 + n_ep]], 0))))
            snr = 20 * np.log10(crms / np.maximum(rms(jaw_D), 1e-300))

            np.savez_compressed(
                f"{BANK}/sub-{s:02d}.npz", trials_L=XL, trials_R=XR,
                keep_L=np.array(k_l), keep_R=np.array(k_r),
                emg_L=np.array(sorted(e_l)), emg_R=np.array(sorted(e_r)),
                jaw_D=jaw_D, jaw_mi=jaw_mi, jaw_purity=purity, jaw_snr_nat_db=snr,
                rest_D=rest_D, rest_mi=rest_mi, rest_hf_power=rest_hf,
                cue_onset=on, samples_per_trial=per, residual_dc=resid,
                clean_rms_3s=crms, subject=s, group=grp)
            dcs.append(resid); purs += list(purity)
            print(f"{s:5d} {grp:>7} {resid:10.1f} {jaw_mi.max():10.2f} "
                  f"{np.median(purity):8.3f} {len(k_l)+len(k_r):7d}")
            del m, root, XL, XR, jaw_D, rest_D, jaw_raw, rest_raw
        except Exception as e:
            print(f"{s:5d} {grp:>7}  FAILED  {type(e).__name__}: {e}")
        finally:
            if path and os.path.exists(path):
                os.remove(path)
            gc.collect()

    if dcs:
        print(f"\n  residual DC median {np.median(dcs):.1f}  "
              f"(ledger {LEDGER_RESID_DC_MEDIAN:.1f})")
        print(f"  purity median      {np.median(purs):.3f}  "
              f"(ledger {LEDGER_PURITY_MEDIAN:.3f})")
        if abs(np.median(purs) - LEDGER_PURITY_MEDIAN) > 0.02:
            print("  WARNING: purity drifted from the ledger. Do not proceed.")
else:
    print("\nPART 1  bank complete, nothing to rebuild")

# ========================================================== PART 2: models
if have_models and have_t3res:
    print("\nPART 2  T3 models already present, nothing to rebuild")
else:
    print(f"\n{'-' * 78}\nPART 2  refitting the frozen fold models\n{'-' * 78}")

    def load(f):
        d = np.load(f, allow_pickle=True)
        if str(d["group"]) != "dev":
            return None
        a0 = int(np.median(d["cue_onset"])) + int(round(W0 * SF_TARGET))
        XL = d["trials_L"][d["keep_L"]][:, :, a0:a0 + NWIN]
        XR = d["trials_R"][d["keep_R"]][:, :, a0:a0 + NWIN]
        return dict(s=int(d["subject"]),
                    X=np.concatenate([XL, XR], 0).astype(np.float64),
                    y=np.concatenate([np.zeros(len(XL), int), np.ones(len(XR), int)]),
                    n_pure=int((np.asarray(d["jaw_purity"], float) >= PURITY).sum()))

    def folds_of(y, k):
        f = np.empty(y.size, int)
        for cls in (0, 1):
            idx = np.flatnonzero(y == cls)
            f[idx] = np.minimum((np.arange(idx.size) * k) // idx.size, k - 1)
        return f

    dev = [x for x in (load(f) for f in sorted(glob.glob(f"{BANK}/sub-*.npz"))) if x]
    print(f"development subjects loaded: {len(dev)}")
    print(f"{'subj':>5} {'clean acc':>10} {'ledger':>9} {'delta':>10} {'elig':>6}")

    store, table, drift = {}, [], []
    for d in dev:
        y, X = d["y"], d["X"]
        fo = folds_of(y, NFOLD)
        oof, per = np.zeros(y.size), []
        for k in range(NFOLD):
            tr = fo != k
            csp = CSP(n_components=8, reg=None, log=True, norm_trace=False,
                      transform_into="average_power").fit(X[tr], y[tr])
            lda = LDA(solver="lsqr", shrinkage="auto").fit(csp.transform(X[tr]), y[tr])
            te = ~tr
            oof[te] = (csp.transform(X[te]) @ lda.coef_.T).ravel() + float(lda.intercept_[0])
            per.append((csp, lda))
        acc = float(np.mean((oof > 0).astype(int) == y))
        store[d["s"]] = dict(folds=fo, oof=oof, per=per)
        table.append(dict(subject=d["s"], n=int(y.size), clean_acc=acc,
                          eligible=bool(acc >= ACC_MIN), n_pure=d["n_pure"]))
        led = LEDGER_ACC.get(d["s"])
        if led is not None:
            drift.append(abs(acc - led))
        print(f"{d['s']:5d} {acc:10.4f} {led if led is not None else '':>9} "
              f"{(acc - led) if led is not None else float('nan'):+10.6f} "
              f"{'yes' if acc >= ACC_MIN else 'NO':>6}")

    accs = np.array([t["clean_acc"] for t in table])
    elig = [t for t in table if t["eligible"] and t["n_pure"] >= 1]
    print(f"\n  median clean accuracy {np.median(accs):.4f}  "
          f"(ledger {LEDGER_ACC_MEDIAN:.4f})")
    print(f"  eligible              {len(elig)}  (ledger {LEDGER_N_ELIGIBLE})")
    print(f"  max drift vs ledger   {max(drift) if drift else float('nan'):.6f}")

    if drift and max(drift) > 1e-6:
        print("""
  REBUILD DOES NOT MATCH THE LEDGER. Something in the environment changed:
  a library version, a filter default, a random state. Do not run T3c on this.
  Print the library versions and compare them with protocol/requirements.txt
  before going further.
""")
    elif len(elig) != LEDGER_N_ELIGIBLE:
        print("\n  Eligible count differs from the ledger. Investigate before T3c.")
    else:
        print("\n  Rebuild reproduces the ledger exactly. Safe to run T3c.")

    np.savez_compressed(
        f"{OUT}/T3_MODELS.npz",
        **{f"folds_{s}": m["folds"] for s, m in store.items()},
        **{f"oof_{s}": m["oof"] for s, m in store.items()},
        **{f"W_{s}_{k}": m["per"][k][0].filters_[:8]
           for s, m in store.items() for k in range(NFOLD)},
        **{f"coef_{s}_{k}": m["per"][k][1].coef_.ravel()
           for s, m in store.items() for k in range(NFOLD)},
        **{f"icpt_{s}_{k}": np.atleast_1d(m["per"][k][1].intercept_)
           for s, m in store.items() for k in range(NFOLD)})
    with open(f"{OUT}/T3_RESULT.json", "w") as f:
        json.dump(dict(stage="T3", rebuilt=True, window_s=[W0, 3.0], n_folds=NFOLD,
                       purity_threshold=PURITY, clean_acc_min=ACC_MIN,
                       subjects=table, n_eligible=len(elig)), f, indent=2, default=float)
    print(f"  written: {OUT}/T3_MODELS.npz and {OUT}/T3_RESULT.json")

print(f"""
{'=' * 78}
DO THIS NOW, BEFORE RUNNING ANYTHING ELSE

  1. Save Version, choose Save and Run All, and wait for the commit to finish.
  2. From the finished version, New Dataset out of the notebook output.
  3. Attach that dataset to the T3c notebook and to every notebook after it.

/kaggle/working is discarded when a session drops and is kept only when the
notebook was committed. Skipping this means rebuilding 10.6 GB a third time.

Then run stage_t3c_surrogate_floor.py, unchanged, with:
  the new dataset attached
  T3_PREREGISTRATION_AMENDMENT_2.json attached
Kaggle mounts a dataset at /kaggle/input/<slug>/, not
/kaggle/input/datasets/<user>/<slug>/. The T3c script searches for the file
itself, so leave its path discovery alone.
{'=' * 78}""")
