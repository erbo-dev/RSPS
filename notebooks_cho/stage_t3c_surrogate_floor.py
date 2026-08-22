# REPO STATUS: CURRENT. Surrogate-calibrated |rho|. Main result of the branch.
# Run order, retractions and full context: notebooks_cho/RUN_ORDER.md

"""
STAGE T3c  ---  the calibrated zero for |rho|.

T3b measured |rho| between the clean CSP projection z and the donor projection
u in three arms and got 0.0231, 0.0230 and 0.0307. The A versus B comparison
is self-calibrating and stands. The absolute values are not interpretable and
the arm C excess, present in 23 of 23 subjects, is not evidence of anything
until a floor exists.

Why a floor is needed
---------------------
The correlation numerator lives only in the 20-32 Hz overlap of the 8-32 Hz
clean band and the 20-80 Hz donor band, while the denominator uses each
signal's full norm. A donor with a 1/f tilt places more of its norm inside the
overlap than a flat broadband donor, so it sits at a higher |rho| without
being any more dependent. Independent synthetic signals over 625 samples give
0.0169 for a flat donor and 0.0272 for a 1/f donor, a gap of 0.010 against the
0.008 observed between arms A and C. The arm C excess is spectral, not causal.

The instrument
--------------
Phase randomisation, the same control as section 6.1: one common random phase
set applied across all 19 channels, which preserves the donor covariance
exactly and therefore its spectrum and its spatial structure, while destroying
any specific temporal relationship with the host. |rho| against a surrogate is
the chance floor for that exact donor, that window and that band.

The reported quantity becomes

    excess = median |rho| observed  -  median |rho| surrogate

per arm, per subject. Only the excess may be compared with anything outside
this branch. The raw |rho| may not.

Reading rules were frozen in T3_PREREGISTRATION_AMENDMENT_2.json before this
ran. No threshold is set: with 18 subjects a null is as consistent with low
power as with equivalence, so intervals are reported, not verdicts.

Primary cohort is the 18 subjects passing the frozen clean-accuracy rule of
0.60. T3b ran on 23 because it filtered on purity only; both are reported and
that discrepancy is recorded in the amendment.

Still no HARM. That is Stage T4.

Run as one Kaggle CPU cell with cho_bank_v2, T3_MODELS.npz and T3_RESULT.json.
"""

import glob
import hashlib
import json
import os

import numpy as np

BANK, OUT = "/kaggle/working/cho_bank_v2", "/kaggle/working"
MODELS, T3RES = f"{OUT}/T3_MODELS.npz", f"{OUT}/T3_RESULT.json"
AMEND = "/kaggle/input/datasets/erbosynshaimardan/fafdsfwrgfsfg/T3_PREREGISTRATION_AMENDMENT_2.json"

for a in glob.glob("/kaggle/input/*/cho_bank_v2") + glob.glob("/kaggle/input/*"):
    if not glob.glob(f"{BANK}/sub-*.npz") and glob.glob(f"{a}/sub-*.npz"):
        BANK = a
for pat, var in ((f"/kaggle/input/*/T3_MODELS.npz", "MODELS"),
                 (f"/kaggle/input/*/T3_RESULT.json", "T3RES")):
    hit = glob.glob(pat)
    if hit and not os.path.exists(locals()[var]):
        globals()[var] = hit[0]
ap = next((p for p in [AMEND] + glob.glob(f"**/{AMEND}", recursive=True)
           + glob.glob(f"/kaggle/input/*/{AMEND}") if os.path.exists(p)), None)
if ap is None:
    raise SystemExit(f"STOP. {AMEND} not found. Commit the amendment before running this.")
if not os.path.exists(MODELS):
    raise SystemExit("STOP. T3_MODELS.npz not found. The frozen fold models are required.")

AMEND_SHA = hashlib.sha256(open(ap, "rb").read()).hexdigest()
SF, W0, NWIN, NFOLD, PURITY, SEED = 250.0, 0.5, 625, 5, 0.90, 20260813
REDRAW, N_SURR = 25, 5          # surrogate banks, matching section 6.1

print("=" * 78)
print("STAGE T3c  surrogate floor for |rho|")
print(f"amendment sha256 : {AMEND_SHA}")
print("=" * 78)

M = np.load(MODELS, allow_pickle=True)
rng = np.random.default_rng(SEED)

ELIG = None
if os.path.exists(T3RES):
    R = json.loads(open(T3RES).read())
    ELIG = {t["subject"] for t in R["subjects"] if t["eligible"]}
    print(f"eligible by the frozen clean-accuracy rule: {len(ELIG)} subjects")
else:
    print("T3_RESULT.json not found, the eligible split cannot be applied; "
          "all subjects reported together and this must be stated")


def phase_randomise(A):
    """Common random phase across all 19 channels. Covariance preserved exactly.

    Applying the SAME phase set to every channel is what makes this exact: the
    cross-spectra keep their magnitudes and their relative phases, so the 19x19
    covariance is untouched while the temporal realisation is destroyed.
    """
    F = np.fft.rfft(A, axis=-1)
    n = F.shape[-1]
    ph = rng.uniform(0, 2 * np.pi, size=(A.shape[0], 1, n))
    ph[..., 0] = 0.0
    if A.shape[-1] % 2 == 0:
        ph[..., -1] = 0.0
    return np.fft.irfft(F * np.exp(1j * ph), n=A.shape[-1], axis=-1)


def load(f):
    d = np.load(f, allow_pickle=True)
    s = int(d["subject"])
    if f"folds_{s}" not in M:
        return None
    a0 = int(np.median(d["cue_onset"])) + int(round(W0 * SF))
    XL = d["trials_L"][d["keep_L"]][:, :, a0:a0 + NWIN]
    XR = d["trials_R"][d["keep_R"]][:, :, a0:a0 + NWIN]
    c = int(round((750 - NWIN) / 2))
    pur = np.asarray(d["jaw_purity"], float)
    return dict(s=s, X=np.concatenate([XL, XR], 0).astype(np.float64),
                jaw=d["jaw_D"][:, :, c:c + NWIN].astype(np.float64)[pur >= PURITY],
                rest=d["rest_D"][:, :, c:c + NWIN].astype(np.float64),
                folds=np.asarray(M[f"folds_{s}"]))


def rho_matrix(W, X, A):
    z = np.einsum("kc,nct->nkt", W, X, optimize=True)
    u = np.einsum("kc,dct->dkt", W, A, optimize=True)
    z = z - z.mean(-1, keepdims=True)
    u = u - u.mean(-1, keepdims=True)
    num = np.einsum("nkt,dkt->ndk", z, u, optimize=True)
    den = np.maximum(np.linalg.norm(z, axis=-1)[:, None, :]
                     * np.linalg.norm(u, axis=-1)[None, :, :], 1e-300)
    return np.abs(num / den)


def arm(sub, donors):
    out = []
    for k in range(NFOLD):
        te = sub["folds"] == k
        if not te.any() or donors.shape[0] == 0:
            continue
        out.append(rho_matrix(np.asarray(M[f"W_{sub['s']}_{k}"]),
                              sub["X"][te], donors).ravel())
    return float(np.median(np.concatenate(out))) if out else np.nan


def surrogate_floor(sub, donors):
    """Median |rho| against N_SURR independent phase-randomised banks."""
    return float(np.median([arm(sub, phase_randomise(donors)) for _ in range(N_SURR)]))


subs = [x for x in (load(f) for f in sorted(glob.glob(f"{BANK}/sub-*.npz"))) if x]
work = [d for d in subs if d["jaw"].shape[0] >= 1]

# covariance-preservation check, the guarantee section 6.1 reports as 1.3e-15
chk = work[0]["jaw"]
c0 = np.cov(chk.reshape(-1, chk.shape[-1])[:19])
c1 = np.cov(phase_randomise(chk).reshape(-1, chk.shape[-1])[:19])
err = float(np.max(np.abs(c0 - c1)) / max(np.max(np.abs(c0)), 1e-300))
print(f"surrogate covariance preservation, max relative error : {err:.2e}")
if err > 1e-10:
    raise SystemExit("STOP. Phase randomisation is not preserving covariance. "
                     "The floor would then differ from the observed arm for the "
                     "wrong reason and nothing below would mean anything.")

print(f"\nsubjects with >= 1 donor at purity >= {PURITY}: {len(work)}")
print("-" * 78)
print(f"{'subj':>5} {'elig':>5} {'n_i':>4} | {'A obs':>7} {'A surr':>7} {'A exc':>8} "
      f"| {'B exc':>8} {'C obs':>7} {'C surr':>7} {'C exc':>8}")
print("-" * 78)

rows = []
for d in work:
    n_i = d["jaw"].shape[0]
    Ao, As = arm(d, d["jaw"]), surrogate_floor(d, d["jaw"])

    pool = [o["jaw"] for o in work if o["s"] != d["s"]]
    pool = np.concatenate(pool, 0) if pool else np.zeros((0, 19, NWIN))
    Bo, Bs = [], []
    for _ in range(REDRAW):
        if pool.shape[0] < n_i:
            break
        p = pool[rng.choice(pool.shape[0], n_i, replace=False)]
        Bo.append(arm(d, p)); Bs.append(arm(d, phase_randomise(p)))
    Bo = float(np.median(Bo)) if Bo else np.nan
    Bs = float(np.median(Bs)) if Bs else np.nan

    Co, Cs = [], []
    for _ in range(REDRAW):
        p = d["rest"][rng.choice(d["rest"].shape[0], n_i, replace=False)]
        Co.append(arm(d, p)); Cs.append(arm(d, phase_randomise(p)))
    Co, Cs = float(np.median(Co)), float(np.median(Cs))

    e = bool(ELIG is None or d["s"] in ELIG)
    rows.append(dict(subject=d["s"], eligible=e, n_donors=n_i,
                     A_obs=Ao, A_surr=As, A_exc=Ao - As,
                     B_obs=Bo, B_surr=Bs, B_exc=Bo - Bs,
                     C_obs=Co, C_surr=Cs, C_exc=Co - Cs))
    print(f"{d['s']:5d} {'y' if e else 'n':>5} {n_i:4d} | {Ao:7.4f} {As:7.4f} "
          f"{Ao - As:+8.4f} | {Bo - Bs:+8.4f} {Co:7.4f} {Cs:7.4f} {Co - Cs:+8.4f}")


def boot(v, n=20000):
    v = np.asarray([x for x in v if np.isfinite(x)])
    if v.size == 0:
        return np.nan, np.nan, np.nan
    dr = np.median(v[rng.integers(0, v.size, (n, v.size))], axis=1)
    return float(np.median(v)), float(np.quantile(dr, .025)), float(np.quantile(dr, .975))


print("\n" + "=" * 78)
print("RESULT")
print("=" * 78)
for label, sel in (("primary, 18 eligible", [r for r in rows if r["eligible"]]),
                   ("supplement, all", rows)):
    print(f"\n  {label}  (n = {len(sel)})")
    for a in "ABC":
        m, lo, hi = boot([r[f"{a}_exc"] for r in sel])
        o, _, _ = boot([r[f"{a}_obs"] for r in sel])
        s, _, _ = boot([r[f"{a}_surr"] for r in sel])
        tag = "" if lo <= 0 <= hi else "   <- interval excludes zero"
        print(f"    arm {a}: observed {o:.4f}  floor {s:.4f}  "
              f"excess {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]{tag}")
    ab = [r["A_exc"] - r["B_exc"] for r in sel]
    m, lo, hi = boot(ab)
    print(f"    A excess minus B excess: {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    if len(sel) >= 6:
        from scipy.stats import wilcoxon
        try:
            print(f"    Wilcoxon on that difference: p = "
                  f"{wilcoxon([x for x in ab if np.isfinite(x)]).pvalue:.4f}")
        except ValueError as ex:
            print(f"    Wilcoxon unavailable: {ex}")

print("""
  How to read this, per the frozen rule:

  All three excesses within noise of zero
      no arm carries dependence. The independence assumption of the original
      work holds in the within-subject regime too, and that is the finding.
      Publish it as the headline of this branch.

  Arm A excess above arm B excess
      within-subject dependence exists and the difference quantifies it.

  Arm C floor above arms A and B while its excess is not
      confirms the arm C gap seen in 23 of 23 subjects was the spectral floor
      of a tilted donor, not dependence, and clears arm C for use as the
      background control in T4.

  No threshold is set. With 18 subjects a null here is as consistent with low
  power as with equivalence, so the intervals are the result, not a verdict.

  Only the excess may be compared with anything outside this branch. The raw
  |rho| may not: 0.036 came from 4 s BNCI epochs with a different donor
  spectrum, and 0.133 came from a positive control with clean signal mixed in
  on purpose. Both are retracted from the reading rule.
""")

with open(f"{OUT}/T3c_SURROGATE_FLOOR.json", "w") as f:
    json.dump(dict(stage="T3c", amendment_sha256=AMEND_SHA, purity=PURITY,
                   n_surrogate_banks=N_SURR, redraws=REDRAW,
                   covariance_preservation_rel_error=err, subjects=rows),
              f, indent=2, default=float)
print(f"written: {OUT}/T3c_SURROGATE_FLOOR.json")
