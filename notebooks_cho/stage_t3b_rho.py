# REPO STATUS: CURRENT. First |rho| measurement, arms A/B/C.
# Run order, retractions and full context: notebooks_cho/RUN_ORDER.md

"""
STAGE T3b  ---  the dependence measurement that the artifact-only gate could
not make.

The T3 gate returned 0.4999, 0.5002 and 0.4998 in three arms because it is
algebraically pinned to one half: folds are class balanced, a donor yields one
prediction per fold, so the fraction of matching labels is exactly half
whatever the prediction is. The gate is retracted in the amendment. It cannot
be repaired, because a donor carries no class and is the same for every trial
of a subject, so nothing built on label agreement can work.

What replaces it is the quantity the whole Cho branch was built to obtain.

    rho = corr( z(t), u(t) )   per CSP component, per trial-donor pair

z is the clean trial projected through the frozen spatial filter, u is the
donor through the same filter. Correlation is scale invariant, so this needs
no contamination, no SNR sweep and no harm statistic. It is computed with the
fold model that did not see the trial, loaded from T3_MODELS.npz and never
refitted.

Two numbers already in the paper give it meaning:

    0.036   the value in the cross-subject regime, where independence is
            imposed by construction
    0.133   the level at which the cross-term became visible in the positive
            control of section 6.2

Limitation 10.3.4 states that |rho| for one person recorded at one sitting is
unknown. This measures it.

Reading rules were frozen in T3_PREREGISTRATION_AMENDMENT.json before this ran:
arm A within 0.02 of arm B means the within-subject design adds no dependence
and that is the finding; arm A approaching 0.133 means the dependence axis is
genuinely open.

Still no HARM. That is Stage T4.

Run as one Kaggle CPU cell with cho_bank_v2 and T3_MODELS.npz available.
"""

import glob
import hashlib
import json
import os

import numpy as np

BANK, OUT = "/kaggle/working/cho_bank_v2", "/kaggle/working"
MODELS = f"{OUT}/T3_MODELS.npz"
AMEND = "/kaggle/input/datasets/erbosynshaimardan/t3-preregistration-v2/T3_PREREGISTRATION_AMENDMENT.json"

for a in glob.glob("/kaggle/input/*/cho_bank_v2") + glob.glob("/kaggle/input/*"):
    if not glob.glob(f"{BANK}/sub-*.npz") and glob.glob(f"{a}/sub-*.npz"):
        BANK = a
for a in glob.glob("/kaggle/input/*/T3_MODELS.npz"):
    if not os.path.exists(MODELS):
        MODELS = a
if not os.path.exists(MODELS):
    raise SystemExit("STOP. T3_MODELS.npz not found. The frozen fold models are required.")

ap = ([AMEND] + glob.glob(f"**/{AMEND}", recursive=True) + glob.glob(f"/kaggle/input/*/{AMEND}"))
ap = next((p for p in ap if os.path.exists(p)), None)
if ap is None:
    raise SystemExit(f"STOP. {AMEND} not found. Commit the amendment before running this.")
AMEND_SHA = hashlib.sha256(open(ap, "rb").read()).hexdigest()

SF, W0, NWIN, NFOLD, PURITY, SEED, REDRAW = 250.0, 0.5, 625, 5, 0.90, 20260813, 25
REF_INDEP, REF_VISIBLE = 0.036, 0.133

print("=" * 78)
print("STAGE T3b  |rho| between clean and donor CSP projections")
print(f"amendment sha256 : {AMEND_SHA}")
print(f"reference points : independent regime {REF_INDEP}, "
      f"cross-term visible from {REF_VISIBLE}")
print("=" * 78)

M = np.load(MODELS, allow_pickle=True)
rng = np.random.default_rng(SEED)


def load(f):
    d = np.load(f, allow_pickle=True)
    s = int(d["subject"])
    if f"folds_{s}" not in M:
        return None                      # not a development subject, or sealed
    a0 = int(np.median(d["cue_onset"])) + int(round(W0 * SF))
    XL = d["trials_L"][d["keep_L"]][:, :, a0:a0 + NWIN]
    XR = d["trials_R"][d["keep_R"]][:, :, a0:a0 + NWIN]
    c = int(round((750 - NWIN) / 2))
    pur = np.asarray(d["jaw_purity"], float)
    return dict(s=s,
                X=np.concatenate([XL, XR], 0).astype(np.float64),
                jaw=d["jaw_D"][:, :, c:c + NWIN].astype(np.float64)[pur >= PURITY],
                rest=d["rest_D"][:, :, c:c + NWIN].astype(np.float64),
                folds=np.asarray(M[f"folds_{s}"]))


def rho_matrix(W, X, A):
    """|corr(z, u)| for every trial, donor and CSP component.

    Returns shape (n_trials, n_donors, n_components). Scale invariant, so the
    SNR grid is irrelevant here.
    """
    z = np.einsum("kc,nct->nkt", W, X, optimize=True)
    u = np.einsum("kc,dct->dkt", W, A, optimize=True)
    z = z - z.mean(-1, keepdims=True)
    u = u - u.mean(-1, keepdims=True)
    nz = np.linalg.norm(z, axis=-1)
    nu = np.linalg.norm(u, axis=-1)
    num = np.einsum("nkt,dkt->ndk", z, u, optimize=True)
    den = np.maximum(nz[:, None, :] * nu[None, :, :], 1e-300)
    return np.abs(num / den)


def arm_rho(sub, donors):
    """Pool |rho| over folds, each trial handled by the model that skipped it."""
    out = []
    for k in range(NFOLD):
        te = sub["folds"] == k
        if not te.any() or donors.shape[0] == 0:
            continue
        W = np.asarray(M[f"W_{sub['s']}_{k}"])
        out.append(rho_matrix(W, sub["X"][te], donors).reshape(-1, W.shape[0]))
    return np.concatenate(out, 0) if out else np.zeros((0, 8))


subs = [x for x in (load(f) for f in sorted(glob.glob(f"{BANK}/sub-*.npz"))) if x]
work = [d for d in subs if d["jaw"].shape[0] >= 1]
print(f"development subjects with frozen models {len(subs)}, "
      f"with >= 1 donor at purity >= {PURITY}: {len(work)}")
print(f"donor counts per host: {[d['jaw'].shape[0] for d in work]}")

print("\n" + "-" * 78)
print(f"{'subj':>5} {'n_i':>4} {'|rho| A':>9} {'|rho| B':>9} {'|rho| C':>9} "
      f"{'A - B':>8}")
print("-" * 78)

rows = []
for d in work:
    n_i = d["jaw"].shape[0]

    rA = arm_rho(d, d["jaw"])

    pool = [o["jaw"] for o in work if o["s"] != d["s"]]
    pool = np.concatenate(pool, 0) if pool else np.zeros((0, 19, NWIN))
    bs = []
    for _ in range(REDRAW):
        if pool.shape[0] < n_i:
            break
        pick = pool[rng.choice(pool.shape[0], n_i, replace=False)]
        bs.append(float(np.median(arm_rho(d, pick))))
    rB = float(np.median(bs)) if bs else np.nan

    cs = []
    for _ in range(REDRAW):
        pick = d["rest"][rng.choice(d["rest"].shape[0], n_i, replace=False)]
        cs.append(float(np.median(arm_rho(d, pick))))
    rC = float(np.median(cs)) if cs else np.nan

    mA = float(np.median(rA)) if rA.size else np.nan
    rows.append(dict(subject=d["s"], n_donors=n_i, rho_A=mA, rho_B=rB, rho_C=rC,
                     rho_A_per_component=[float(v) for v in np.median(rA, 0)]
                     if rA.size else []))
    print(f"{d['s']:5d} {n_i:4d} {mA:9.4f} {rB:9.4f} {rC:9.4f} {mA - rB:+8.4f}")

A = np.array([r["rho_A"] for r in rows])
B = np.array([r["rho_B"] for r in rows])
C = np.array([r["rho_C"] for r in rows])
ok = np.isfinite(A) & np.isfinite(B) & np.isfinite(C)

print("\n" + "=" * 78)
print("RESULT")
print("=" * 78)
for nm, v in [("arm A, own jaw donor", A), ("arm B, stranger's jaw", B),
              ("arm C, own resting", C)]:
    print(f"  {nm:>24}: median {np.median(v[ok]):.4f}   "
          f"range {v[ok].min():.4f} to {v[ok].max():.4f}")

dAB = float(np.median(A[ok] - B[ok]))
print(f"\n  paired A - B, median : {dAB:+.4f}")
if ok.sum() >= 6:
    from scipy.stats import wilcoxon
    try:
        st, pv = wilcoxon(A[ok], B[ok])
        print(f"  Wilcoxon A vs B      : p = {pv:.4g}, n = {int(ok.sum())}")
    except ValueError as e:
        print(f"  Wilcoxon unavailable : {e}")

print(f"\n  for comparison, already in the paper:")
print(f"    cross-subject regime, independence imposed : {REF_INDEP}")
print(f"    positive control, cross-term became visible: {REF_VISIBLE}")

mA = float(np.median(A[ok]))
if abs(dAB) <= 0.02:
    verdict = "A_equals_B"
    print(f"""
  READING, per the frozen rule: arm A sits within 0.02 of arm B. Recording the
  donor from the same person in the same session introduces NO measurable
  dependence between the clean signal and the artifact. Arm A adds nothing
  beyond a new artifact class.

  This is a null result and it is published as the headline of the branch, not
  buried. It converts limitation 10.3.4 from an unknown into a measurement:
  |rho| for one person at one sitting is {mA:.4f}, against {REF_INDEP} in the
  cross-subject regime and {REF_VISIBLE} where the cross-term became visible.
  The independence assumption is not an artefact of the design after all.
""")
elif mA >= REF_VISIBLE:
    verdict = "A_reaches_visible"
    print(f"""
  READING: arm A reaches {mA:.4f}, at or above the {REF_VISIBLE} level where the
  positive control showed the cross-term becoming visible. The dependence axis
  is genuinely open and the primary endpoint in T4 is informative.
""")
else:
    verdict = "A_above_B_below_visible"
    print(f"""
  READING: arm A exceeds arm B by {dAB:+.4f} but stays below {REF_VISIBLE}. The
  within-subject design moves the dependence in the expected direction without
  reaching the level where the positive control saw the cross-term matter.
  Report the magnitude; do not describe it as opening the dependence axis.
""")

print("""  Caveat that goes in the text: z is filtered 8-32 Hz and u 20-80 Hz, so
  this correlation is carried by the 20-32 Hz overlap. That is the frozen band
  convention of the study and it is stated, not absorbed.

  Also state: jaw and imagery are still separate blocks. This closes the
  same-subject same-session axis. It does not close simultaneity with the task,
  and no open dataset does.
""")

with open(f"{OUT}/T3b_RHO.json", "w") as f:
    json.dump(dict(stage="T3b", amendment_sha256=AMEND_SHA, purity=PURITY,
                   redraws=REDRAW, reference_independent=REF_INDEP,
                   reference_visible=REF_VISIBLE, verdict=verdict,
                   median_A=mA, median_B=float(np.median(B[ok])),
                   median_C=float(np.median(C[ok])), paired_A_minus_B=dAB,
                   subjects=rows), f, indent=2, default=float)
print(f"written: {OUT}/T3b_RHO.json")
