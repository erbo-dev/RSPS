# REPO STATUS: CURRENT for decoders and eligibility. Its artifact-only gate is RETRACTED, ledger R5.
# Run order, retractions and full context: notebooks_cho/RUN_ORDER.md

"""
STAGE T3  ---  decoders, eligibility, and the validity gates.  No HARM yet.

Reads T3_PREREGISTRATION.json, prints its sha256, refuses to run without it.

What this stage does
--------------------
1. Cuts the frozen imagery window 0.5-3.0 s from the 7 s trials stored by T2c.
2. Fits CSP+LDA in five contiguous folds per subject and SAVES every fold
   model, so that in T4 each trial's decision is taken by the model that did
   not see it. That is the Cho equivalent of the leave-one-run-out discipline
   of the BNCI branch, and it has to be frozen here rather than refitted later.
3. Applies the frozen eligibility rule, clean out-of-fold accuracy >= 0.60.
4. Runs the two artifact-only validity gates.

What it deliberately does not do
--------------------------------
No contamination, no counterfactual pair, no HARM. Those are Stage T4. The
decoders have to exist and the gates have to pass before any of that is
allowed to be computed, otherwise the gates stop being gates.

The gate that the BNCI branch could never run
---------------------------------------------
In the cross-subject regime the donor came from a stranger, so it could not
carry the host's own decodable background. Here it can: the jaw block is the
same person, the same session, the same electrodes, and the decoder was fitted
on that person. So artifact-only accuracy is compared BETWEEN arms. If feeding
the subject's own donor beats feeding a stranger's donor by more than 0.03,
arm A is measuring the subject's background rather than the muscle, and arm A
is withdrawn. This is the same falsification test that killed the synthetic
EMG branch in section 9.3, now pointed at the new design.

Development subjects only. The 15 locked subjects stay sealed.

Run as one Kaggle CPU cell with cho_bank_v2 attached. Minutes, not hours.
"""

import glob
import hashlib
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

PREREG = "/kaggle/input/datasets/erbosynshaimardan/t3-preregistration/T3_PREREGISTRATION.json"
BANK = "/kaggle/working/cho_bank_v2"
OUT = "/kaggle/working"

for alt in ("./T3_PREREGISTRATION.json", f"{OUT}/T3_PREREGISTRATION.json"):
    if not os.path.exists(PREREG) and os.path.exists(alt):
        PREREG = alt
if not os.path.exists(PREREG):
    raise SystemExit("STOP. T3_PREREGISTRATION.json not found. Commit and attach it first.")
if not glob.glob(f"{BANK}/sub-*.npz"):
    for a in glob.glob("/kaggle/input/*/cho_bank_v2") + glob.glob("/kaggle/input/*"):
        if glob.glob(f"{a}/sub-*.npz"):
            BANK = a
            break
if not glob.glob(f"{BANK}/sub-*.npz"):
    raise SystemExit("STOP. cho_bank_v2 not found. Attach the Stage T2c output.")

with open(PREREG, "rb") as f:
    pb = f.read()
SHA = hashlib.sha256(pb).hexdigest()
P = json.loads(pb.decode("utf-8"))
print("=" * 78)
print("STAGE T3  decoders, eligibility, validity gates")
print(f"preregistration sha256 : {SHA}")
print(f"bank                   : {BANK}")
print("=" * 78)

SF = 250.0
W0, W1 = P["analysis_window"]["imagery_window_s"]
NWIN = int(P["analysis_window"]["length_samples_at_250hz"])
NFOLD = 5
PURITY = float(P["donor_purity_threshold"]["primary"])
PURITY_SENS = P["donor_purity_threshold"]["sensitivity_analysis"]
ACC_MIN = float(P["eligibility"]["clean_accuracy_min"])
GATE_ONLY = float(P["validity_gates_run_before_any_endpoint_is_read"]["artifact_only_pooled"]["threshold"])
GATE_DELTA = 0.03
SEED = int(P["seed"])
DEV = set(json.loads(open(glob.glob("**/T2_PREREGISTRATION.json", recursive=True)[0]).read()
                     )["cohort"]["development"]) if glob.glob("**/T2_PREREGISTRATION.json", recursive=True) else None

print(f"window {W0}-{W1} s ({NWIN} samples), folds {NFOLD}, purity >= {PURITY}, "
      f"clean acc >= {ACC_MIN}")

try:
    import mne
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from scipy.stats import wilcoxon
except ImportError:
    os.system(f"{sys.executable} -m pip install -q mne scikit-learn")
    import mne
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from scipy.stats import wilcoxon
mne.set_log_level("ERROR")

rng = np.random.default_rng(SEED)


def load(f):
    d = np.load(f, allow_pickle=True)
    s, grp = int(d["subject"]), str(d["group"])
    a0 = int(np.median(d["cue_onset"])) + int(round(W0 * SF))
    XL = d["trials_L"][d["keep_L"]][:, :, a0:a0 + NWIN]
    XR = d["trials_R"][d["keep_R"]][:, :, a0:a0 + NWIN]
    X = np.concatenate([XL, XR], 0).astype(np.float64)
    y = np.concatenate([np.zeros(len(XL), int), np.ones(len(XR), int)])
    c = int(round((750 - NWIN) / 2))                 # centre cut of the donor
    return dict(s=s, grp=grp, X=X, y=y,
                jaw=d["jaw_D"][:, :, c:c + NWIN].astype(np.float64),
                jaw_pur=np.asarray(d["jaw_purity"], float),
                rest=d["rest_D"][:, :, c:c + NWIN].astype(np.float64),
                nat=np.asarray(d["jaw_snr_nat_db"], float))


def contiguous_folds(y, k):
    """Contiguous blocks within each class, so folds respect time order."""
    f = np.empty(y.size, int)
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        f[idx] = np.minimum((np.arange(idx.size) * k) // idx.size, k - 1)
    return f


def fit_fold(X, y, tr):
    csp = CSP(n_components=8, reg=None, log=True, norm_trace=False,
              transform_into="average_power")
    F = csp.fit_transform(X[tr], y[tr])
    lda = LDA(solver="lsqr", shrinkage="auto").fit(F, y[tr])
    return csp, lda


def signed_score(csp, lda, X):
    return (csp.transform(X) @ lda.coef_.T).ravel() + float(lda.intercept_[0])


FILES = sorted(glob.glob(f"{BANK}/sub-*.npz"))
subs = [load(f) for f in FILES]
dev = [d for d in subs if d["grp"] == "dev"]
pilot = [d for d in subs if d["grp"] == "pilot"]
print(f"loaded {len(subs)} subjects: dev {len(dev)}, pilot {len(pilot)}, "
      f"locked sealed {len([d for d in subs if d['grp'] == 'locked'])}")

# ----------------------------------------------------------------------
print("\n" + "-" * 78)
print("DECODERS, development only")
print("-" * 78)
print(f"{'subj':>5} {'trials':>7} {'clean acc':>10} {'eligible':>9} "
      f"{'jaw>=pur':>9} {'nat SNR':>8}")

models, table = {}, []
for d in dev:
    X, y = d["X"], d["y"]
    folds = contiguous_folds(y, NFOLD)
    oof = np.zeros(y.size)
    per_fold = []
    for k in range(NFOLD):
        tr, te = folds != k, folds == k
        csp, lda = fit_fold(X, y, tr)
        oof[te] = signed_score(csp, lda, X[te])
        per_fold.append((csp, lda))
    acc = float(np.mean((oof > 0).astype(int) == y))
    n_pure = int((d["jaw_pur"] >= PURITY).sum())
    models[d["s"]] = dict(folds=folds, per_fold=per_fold, oof=oof)
    table.append(dict(subject=d["s"], n=int(y.size), clean_acc=acc,
                      eligible=bool(acc >= ACC_MIN), n_pure=n_pure,
                      nat_snr=float(np.median(d["nat"]))))
    print(f"{d['s']:5d} {y.size:7d} {acc:10.4f} {'yes' if acc >= ACC_MIN else 'NO':>9} "
          f"{n_pure:9d} {np.median(d['nat']):8.2f}")

elig = [t for t in table if t["eligible"] and t["n_pure"] >= 1]
accs = np.array([t["clean_acc"] for t in table])
print(f"\n  clean accuracy: median {np.median(accs):.4f}, "
      f"range {accs.min():.4f} to {accs.max():.4f}")
print(f"  eligible (acc >= {ACC_MIN}) and >= 1 donor at purity >= {PURITY}: "
      f"{len(elig)}/{len(table)}")
for p in PURITY_SENS:
    n = sum(1 for t, d in zip(table, dev)
            if t["eligible"] and (d["jaw_pur"] >= p).sum() >= 1)
    print(f"    sensitivity, purity >= {p:.2f}: {n} subjects")

if len(elig) < 10:
    print("\n  WARNING: fewer than 10 eligible subjects. Report it, do not "
          "loosen the frozen thresholds to recover cohort size.")

# ----------------------------------------------------------------------
print("\n" + "-" * 78)
print("VALIDITY GATE, artifact-only decoding")
print("-" * 78)


def artifact_only(d, donors):
    """Decoder fed the donor alone. Pooled over trial-donor pairs.

    The donor carries no label, so its prediction is compared against the
    labels of the trials it would have been paired with. With three donors
    per host this is meaningless per subject and only the pooled figure and
    the between-arm contrast are read.
    """
    folds, per = models[d["s"]]["folds"], models[d["s"]]["per_fold"]
    hits = tot = 0
    right = []
    for k in range(NFOLD):
        te = folds == k
        if not te.any():
            continue
        csp, lda = per[k]
        pred = (signed_score(csp, lda, donors) > 0).astype(int)
        right.append(pred.mean())
        for p in pred:
            hits += int((d["y"][te] == p).sum())
            tot += int(te.sum())
    return hits, tot, float(np.mean(right))


eligible_ids = {t["subject"] for t in elig}
work = [d for d in dev if d["s"] in eligible_ids]
if not work:
    raise SystemExit("STOP. No eligible subject. Nothing can be gated.")

hA = tA = hB = tB = hC = tC = 0
bias = []
for d in work:
    pure = d["jaw"][d["jaw_pur"] >= PURITY]
    if pure.size == 0:
        continue
    h, t, r = artifact_only(d, pure)
    hA += h; tA += t; bias.append(dict(subject=d["s"], frac_right=r))

    pool = np.concatenate([o["jaw"][o["jaw_pur"] >= PURITY]
                           for o in work if o["s"] != d["s"] and (o["jaw_pur"] >= PURITY).any()])
    pick = pool[rng.choice(pool.shape[0], size=pure.shape[0], replace=False)]
    h, t, _ = artifact_only(d, pick)
    hB += h; tB += t

    rpick = d["rest"][rng.choice(d["rest"].shape[0], size=pure.shape[0], replace=False)]
    h, t, _ = artifact_only(d, rpick)
    hC += h; tC += t

accA, accB, accC = hA / max(tA, 1), hB / max(tB, 1), hC / max(tC, 1)
print(f"  arm A, own jaw donor      : {accA:.4f}   ({tA:,} pairs)")
print(f"  arm B, stranger's donor   : {accB:.4f}   ({tB:,} pairs)")
print(f"  arm C, own resting donor  : {accC:.4f}   ({tC:,} pairs)")
print(f"  A - B                     : {accA - accB:+.4f}   (allowed {GATE_DELTA:+.2f})")

g1 = accA < GATE_ONLY and accB < GATE_ONLY
g2 = (accA - accB) <= GATE_DELTA
print(f"\n  gate 1, pooled artifact-only < {GATE_ONLY}      : {'PASS' if g1 else 'FAIL'}")
print(f"  gate 2, arm A does not beat arm B by > {GATE_DELTA} : {'PASS' if g2 else 'FAIL'}")

fr = np.array([b["frac_right"] for b in bias])
print(f"\n  directional bias, fraction predicting class 1: median {np.median(fr):.3f}, "
      f"range {fr.min():.3f} to {fr.max():.3f}")
print("  reported only. Section 6.4 already showed this figure is a mechanical")
print("  consequence of a shared bias direction and is not independent evidence.")

if g1 and g2:
    print("""
  BOTH GATES PASS. Arm A is measuring the muscle, not the subject's own
  decodable background, and T4 may proceed. Freeze this output before running
  it: the fold models above are the decoders T4 must use, unchanged.
""")
else:
    print("""
  A GATE FAILED. Arm A is void. Withdraw it the way the synthetic EMG branch
  was withdrawn in section 9.3, publish the numbers, and fall back to arms B
  and C only. Do not weaken the threshold.
""")

np.savez_compressed(
    f"{OUT}/T3_MODELS.npz", prereg_sha=SHA,
    **{f"folds_{s}": m["folds"] for s, m in models.items()},
    **{f"oof_{s}": m["oof"] for s, m in models.items()},
    **{f"W_{s}_{k}": m["per_fold"][k][0].filters_[:8]
       for s, m in models.items() for k in range(NFOLD)},
    **{f"coef_{s}_{k}": m["per_fold"][k][1].coef_.ravel()
       for s, m in models.items() for k in range(NFOLD)},
    **{f"icpt_{s}_{k}": np.atleast_1d(m["per_fold"][k][1].intercept_)
       for s, m in models.items() for k in range(NFOLD)})

with open(f"{OUT}/T3_RESULT.json", "w") as f:
    json.dump(dict(stage="T3", prereg_sha256=SHA, window_s=[W0, W1], n_folds=NFOLD,
                   purity_threshold=PURITY, clean_acc_min=ACC_MIN,
                   subjects=table, n_eligible=len(elig),
                   artifact_only=dict(arm_A=accA, arm_B=accB, arm_C=accC,
                                      A_minus_B=accA - accB,
                                      gate_pooled=bool(g1), gate_between=bool(g2)),
                   directional_bias=bias), f, indent=2, default=float)
print(f"written: {OUT}/T3_RESULT.json and {OUT}/T3_MODELS.npz")
