# REPO STATUS: CURRENT. Three-arm harm sweep. Harm engine embedded and self-testing.
# Run order, retractions and full context: notebooks_cho/RUN_ORDER.md

"""
STAGE T4b  ---  the harm sweep. Three arms, extended grid, natural point.

This is the first stage in the Cho branch that computes a harm statistic. It
was held back until the instrument was verified twice: a synthetic self test
with a brute-force reference, and a cross-implementation check against this
project's own harmkit that matched every counter exactly across four SNR
grids and five seeds.

Arms, per the frozen rules
--------------------------
A  own jaw donors that pass purity >= 0.90
B  the same number of jaw donors drawn from OTHER eligible subjects, 25 redraws
C  the same number of the host's own resting donors, 25 redraws

Matching the donor count is not a nicety. With one to three donors per host,
an unmatched arm B would differ from arm A in counting noise before it
differed in anything else.

SNR
---
The eight frozen levels are kept unchanged so the BNCI comparison stays exact.
Five harsher levels are added because the measured natural SNR has median
-10.79 dB with 75.9 per cent of donor windows harsher than the frozen grid and
none milder, so the frozen grid does not cover the regime people generate.
A natural point is added per host: the median natural SNR of that host's own
jaw donors, applied identically in all three arms so the arms stay comparable.

clean_reference_rms
-------------------
Built per fold from the TRAINING trials only, as the docstring of
HarmBenchmark.run requires. Taking it from all trials would leak the test
fold's amplitude into the scaling of the artifact added to that same fold.

The prediction, frozen in T3_PREREGISTRATION.json before any of this ran
-----------------------------------------------------------------------
  1. interaction-required at -15 dB is below 0.015
  2. interaction-required decreases monotonically from -7 dB through -30 dB

Both are extrapolations outside the +2 to -7 dB range the saturation law of
section 7 was derived on. A confirmed out-of-range prediction is worth more
than any in-range fit. A refuted one is published as a bound on the law's
domain, not deleted.

Primary endpoint: symmetry ratio, prevented over required, arm A versus arm B,
paired Wilcoxon over subjects.

Development subjects only. The 15 locked subjects stay sealed for Stage T5.

SELF CONTAINED. The verified harm engine is embedded below, so only
cho_bank_v3, T3_MODELS.npz and T3_RESULT.json need attaching. The engine runs
its own self test at startup and refuses to continue if it fails.

Run as one Kaggle CPU cell. Minutes, not hours.
"""

import glob
import json
import os
import sys

import numpy as np

BANK, OUT = "/kaggle/working/cho_bank_v3", "/kaggle/working"
for a in glob.glob("/kaggle/input/**/", recursive=True):
    if not glob.glob(f"{BANK}/sub-*.npz") and glob.glob(f"{a}/sub-*.npz"):
        BANK = a.rstrip("/")
for pat in ("T3_MODELS.npz", "T3_RESULT.json"):
    if not os.path.exists(f"{OUT}/{pat}"):
        hit = glob.glob(f"/kaggle/input/**/{pat}", recursive=True)
        if hit:
            os.system(f"cp '{hit[0]}' '{OUT}/{pat}'")
sys.path.insert(0, OUT)

if not glob.glob(f"{BANK}/sub-*.npz"):
    raise SystemExit("STOP. cho_bank_v3 not found.")
for f in ("T3_MODELS.npz", "T3_RESULT.json"):
    if not os.path.exists(f"{OUT}/{f}"):
        raise SystemExit(f"STOP. {f} not found. T4b must not refit anything.")



# ======================================================================
# EMBEDDED HARM ENGINE
# Cross-checked against this project's own harmkit on identical synthetic
# inputs: every counter matched exactly across four SNR grids and five seeds,
# and the scale factors matched harmkit.scale_for_snr to 0.0 exactly.
# It self tests below and stops the stage if anything drifts.
# ======================================================================


import numpy as np

__all__ = ["scale_factors", "harm_counts", "harm_counts_naive", "self_test"]


def scale_factors(A, clean_ref_rms, snr_db):
    """lam[d, s] so that 20 log10(clean_ref_rms / rms(lam * a_d)) == snr_db[s]."""
    r = np.sqrt((np.asarray(A, float) ** 2).mean(axis=(-2, -1)))
    return clean_ref_rms / (r[:, None] * 10.0 ** (np.asarray(snr_db, float)[None, :] / 20.0))


def _margins(W, coef, icpt, X, A, lam, ysign):
    z = np.einsum("kc,nct->nkt", W, X, optimize=True)
    u = np.einsum("kc,dct->dkt", W, A, optimize=True)
    T = z.shape[-1]
    Px = np.mean(z ** 2, -1)                                   # (n, k)
    Pa = np.mean(u ** 2, -1)                                   # (d, k)
    cr = np.einsum("nkt,dkt->ndk", z, u, optimize=True) / T     # (n, d, k)

    pc = Px[:, None, None, :]
    pa = Pa[None, :, None, :]
    cx = cr[:, :, None, :]
    lm = lam[None, :, :, None]

    P_no = pc + lm ** 2 * pa
    P_fu = pc + 2.0 * lm * cx + lm ** 2 * pa
    ys = ysign[:, None, None]

    def m(P):
        return ys * ((np.log(np.maximum(P, 1e-300)) @ coef) + icpt)

    return m(P_no), m(P_fu), ys.squeeze() * ((np.log(np.maximum(Px, 1e-300)) @ coef) + icpt), P_fu


def harm_counts(W, coef, icpt, X, y, A, lam):
    """Vectorised. Returns counts and the per-cell margins."""
    ysign = np.where(np.asarray(y) == 1, 1.0, -1.0)
    m_no, m_fu, m_cl, P_fu = _margins(W, coef, icpt, X, A, lam, ysign)
    elig = (m_cl > 0)[:, None, None]
    harm = elig & (m_fu < 0)
    return dict(eligible=int(elig.sum() * m_fu.shape[1] * m_fu.shape[2]),
                full_harm=int(harm.sum()),
                interaction_required=int((harm & (m_no > 0)).sum()),
                interaction_prevented=int((elig & (m_no < 0) & (m_fu > 0)).sum()),
                m_no=m_no, m_full=m_fu, m_clean=m_cl, P_full=P_fu)


def harm_counts_naive(W, coef, icpt, X, y, A, lam):
    """Independent triple loop. Slow on purpose. Ground truth for the fast path."""
    n, d, s = X.shape[0], A.shape[0], lam.shape[1]
    ysign = np.where(np.asarray(y) == 1, 1.0, -1.0)
    hi = req = prev = elig_cnt = 0
    for i in range(n):
        z = W @ X[i]
        f_cl = np.log(np.maximum((z ** 2).mean(-1), 1e-300))
        if ysign[i] * (f_cl @ coef + icpt) <= 0:
            continue
        elig_cnt += d * s
        for j in range(d):
            u_raw = W @ A[j]
            for k in range(s):
                L = lam[j, k]
                u = L * u_raw
                p_no = (z ** 2).mean(-1) + (u ** 2).mean(-1)
                p_fu = ((z + u) ** 2).mean(-1)          # from the waveform itself
                mn = ysign[i] * (np.log(np.maximum(p_no, 1e-300)) @ coef + icpt)
                mf = ysign[i] * (np.log(np.maximum(p_fu, 1e-300)) @ coef + icpt)
                if mf < 0:
                    hi += 1
                    if mn > 0:
                        req += 1
                if mn < 0 and mf > 0:
                    prev += 1
    return dict(eligible=elig_cnt, full_harm=hi,
                interaction_required=req, interaction_prevented=prev)


def self_test(seed=0, verbose=True):
    """Four checks. Any failure means the engine must not be used."""
    rng = np.random.default_rng(seed)
    n, d, T, C, K = 200, 12, 625, 19, 8
    # This grid is chosen so the cross-term actually decides some outcomes.
    # Under harsh contamination every flip lands far past the boundary, the
    # ablation never changes an outcome, and checks 3 and 4 compare zero with
    # zero. A grid that cannot produce interaction-required events cannot test
    # the ablation, whatever it prints.
    snr = np.array([20.0, 16.0, 14.0, 12.0, 8.0, 0.0])

    X = rng.standard_normal((n, C, T))
    y = rng.integers(0, 2, n)
    X[y == 1, :4] *= 1.18                     # weak, so margins sit near the boundary
    A = rng.standard_normal((d, C, T)) * 3.0
    W = rng.standard_normal((K, C))

    # A real fitted LDA, not random coefficients. With random coefficients the
    # margins land far from zero, nothing ever flips, and check 3 compares two
    # zeros, which is exactly the kind of vacuous pass this file exists to catch.
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    zc = np.einsum("kc,nct->nkt", W, X)
    F = np.log(np.maximum(np.mean(zc ** 2, -1), 1e-300))
    _l = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(F, y)
    coef, icpt = _l.coef_.ravel(), float(_l.intercept_[0])

    ref = float(np.median(np.sqrt((X ** 2).mean(axis=(-2, -1)))))
    lam = scale_factors(A, ref, snr)
    out, res = {}, True

    # 1. SNR convention actually achieved
    got = 20 * np.log10(ref / np.sqrt(((lam[:, :, None, None] * A[:, None]) ** 2)
                                      .mean(axis=(-2, -1))))
    e1 = float(np.max(np.abs(got - snr[None, :])))
    out["snr_convention"] = e1
    res &= e1 < 1e-10

    # 2. algebraic P_full equals the power of the actual contaminated waveform
    f = harm_counts(W, coef, icpt, X, y, A, lam)
    z = np.einsum("kc,nct->nkt", W, X)
    u = np.einsum("kc,dct->dkt", W, A)
    direct = np.mean((z[:, None, None] + lam[None, :, :, None, None]
                      * u[None, :, None]) ** 2, -1)
    e2 = float(np.max(np.abs(f["P_full"] - direct) / np.maximum(np.abs(direct), 1e-300)))
    out["decomposition_vs_waveform"] = e2
    res &= e2 < 1e-10

    # 3. fast path equals the independent loop, exactly
    nv = harm_counts_naive(W, coef, icpt, X, y, A, lam)
    same = all(f[k] == nv[k] for k in
               ("eligible", "full_harm", "interaction_required", "interaction_prevented"))
    out["fast_vs_naive"] = dict(fast={k: f[k] for k in nv}, naive=nv, identical=same)
    res &= same

    # 4. lam = 0 must leave every decision untouched
    z0 = harm_counts(W, coef, icpt, X, y, A, np.zeros((d, 1)))
    out["lambda_zero_harm"] = z0["full_harm"]
    res &= z0["full_harm"] == 0

    # 5. non-vacuity. If nothing flipped, checks 3 and 4 compared zero with zero
    # and prove nothing. A test that cannot fail is not a test.
    nonvac = f["full_harm"] > 0 and f["interaction_required"] > 0
    out["non_vacuous"] = dict(full_harm=f["full_harm"],
                              interaction_required=f["interaction_required"],
                              ok=bool(nonvac))
    res &= nonvac

    if verbose:
        print("harm engine self test")
        print(f"  1 SNR convention, max error          : {e1:.2e}   "
              f"{'OK' if e1 < 1e-10 else 'FAIL'}")
        print(f"  2 decomposition vs waveform, max rel : {e2:.2e}   "
              f"{'OK' if e2 < 1e-10 else 'FAIL'}")
        print(f"  3 fast vs independent loop           : {out['fast_vs_naive']['fast']}")
        print(f"                                         {nv}   "
              f"{'IDENTICAL' if same else 'MISMATCH'}")
        print(f"  4 lambda = 0 gives zero harm         : {z0['full_harm']}   "
              f"{'OK' if z0['full_harm'] == 0 else 'FAIL'}")
        print(f"  5 non-vacuity, harm and required > 0 : "
              f"{f['full_harm']} / {f['interaction_required']}   "
              f"{'OK' if nonvac else 'FAIL, checks 3 and 4 proved nothing'}")
        if f["full_harm"]:
            print(f"      interaction-required fraction    : "
                  f"{f['interaction_required'] / f['full_harm']:.4f}")
        print(f"  VERDICT: {'PASS' if res else 'FAIL'}")
    return res, out

_ok, _det = self_test(verbose=True)
if not _ok:
    raise SystemExit("STOP. The embedded engine failed its self test. "
                     "No harm number from this run would mean anything.")
print()

SF, W0, NWIN, NFOLD = 250.0, 0.5, 625, 5
PURITY, ACC_MIN, SEED, REDRAW = 0.90, 0.60, 20260813, 25
GRID_FROZEN = [2.0, 0.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0]
GRID_EXT = [-10.0, -15.0, -20.0, -25.0, -30.0]
GRID = GRID_FROZEN + GRID_EXT
PRED_AT, PRED_BELOW = -15.0, 0.015

rng = np.random.default_rng(SEED)
M = np.load(f"{OUT}/T3_MODELS.npz", allow_pickle=True)
ACC = {t["subject"]: t["clean_acc"]
       for t in json.loads(open(f"{OUT}/T3_RESULT.json").read())["subjects"]}

print("=" * 80)
print("STAGE T4b  harm sweep, three arms")
print(f"grid: {len(GRID_FROZEN)} frozen + {len(GRID_EXT)} extension + 1 natural point")
print("=" * 80)


def load(f):
    d = np.load(f, allow_pickle=True)
    s = int(d["subject"])
    if str(d["group"]) != "dev" or f"folds_{s}" not in M:
        return None
    a0 = int(np.median(d["cue_onset"])) + int(round(W0 * SF))
    XL = d["trials_L"][d["keep_L"]][:, :, a0:a0 + NWIN]
    XR = d["trials_R"][d["keep_R"]][:, :, a0:a0 + NWIN]
    saf = np.concatenate([np.asarray(d["clean_saf_rms_L"])[d["keep_L"]],
                          np.asarray(d["clean_saf_rms_R"])[d["keep_R"]]])
    pur = np.asarray(d["jaw_purity"], float)
    keep = pur >= PURITY
    return dict(s=s, X=np.concatenate([XL, XR], 0).astype(np.float64),
                y=np.concatenate([np.zeros(len(XL), int), np.ones(len(XR), int)]),
                saf=saf, folds=np.asarray(M[f"folds_{s}"]),
                jaw=np.asarray(d["jaw_dec"], float)[keep],
                jaw_saf_rms=np.asarray(d["jaw_saf_rms"], float)[keep],
                rest=np.asarray(d["rest_dec"], float),
                rest_saf_rms=np.asarray(d["rest_saf_rms"], float),
                acc=ACC.get(s, np.nan))


subs = [x for x in (load(f) for f in sorted(glob.glob(f"{BANK}/sub-*.npz"))) if x]
work = [d for d in subs if d["acc"] >= ACC_MIN and d["jaw"].shape[0] >= 1]
print(f"development loaded {len(subs)}, eligible {len(work)}, "
      f"donors per host {[d['jaw'].shape[0] for d in work]}")
if not work:
    raise SystemExit("STOP. No eligible subject.")


def per_snr(res, n_snr):
    """Counts split by SNR level, from the margins the engine returns."""
    elig = (res["m_clean"] > 0)[:, None, None]
    harm = elig & (res["m_full"] < 0)
    req = harm & (res["m_no"] > 0)
    prev = elig & (res["m_no"] < 0) & (res["m_full"] > 0)
    return [dict(full=int(harm[:, :, j].sum()), req=int(req[:, :, j].sum()),
                 prev=int(prev[:, :, j].sum())) for j in range(n_snr)]


def run_arm(d, donors, saf_rms, grid):
    """Sum over folds. Each fold uses the model that never saw its trials,
    and a clean reference built from that fold's TRAINING trials only."""
    tot = dict(full=0, req=0, prev=0)
    by = [dict(full=0, req=0, prev=0) for _ in grid]
    for k in range(NFOLD):
        te = d["folds"] == k
        if not te.any() or donors.shape[0] == 0:
            continue
        ref = float(np.median(d["saf"][~te]))          # training trials only
        lam = float(ref) / (saf_rms[:, None] * 10.0 ** (np.asarray(grid)[None, :] / 20.0))
        r = harm_counts(np.asarray(M[f"W_{d['s']}_{k}"]),
                           np.asarray(M[f"coef_{d['s']}_{k}"]),
                           float(np.asarray(M[f"icpt_{d['s']}_{k}"]).ravel()[0]),
                           d["X"][te], d["y"][te], donors, lam)
        tot["full"] += r["full_harm"]
        tot["req"] += r["interaction_required"]
        tot["prev"] += r["interaction_prevented"]
        for j, c in enumerate(per_snr(r, len(grid))):
            for kk in ("full", "req", "prev"):
                by[j][kk] += c[kk]
    return tot, by


def frac(t):
    return t["req"] / t["full"] if t["full"] else np.nan


def sym(t):
    return t["prev"] / t["req"] if t["req"] else np.nan


print(f"\n{'subj':>5} {'n_i':>4} {'natSNR':>7} | {'A frac':>8} {'A sym':>7} "
      f"| {'B frac':>8} {'B sym':>7} | {'C frac':>8} {'C sym':>7}")
print("-" * 80)

rows, bySNR_A = [], [dict(full=0, req=0) for _ in GRID]
for d in work:
    n_i = d["jaw"].shape[0]
    ref_all = float(np.median(d["saf"]))
    nat = float(np.median(20.0 * np.log10(ref_all / np.maximum(d["jaw_saf_rms"], 1e-300))))
    grid = GRID + [nat]                       # natural point last, same in all arms

    tA, byA = run_arm(d, d["jaw"], d["jaw_saf_rms"], grid)
    for j in range(len(GRID)):
        bySNR_A[j]["full"] += byA[j]["full"]
        bySNR_A[j]["req"] += byA[j]["req"]

    pool = [(o["jaw"], o["jaw_saf_rms"]) for o in work if o["s"] != d["s"]]
    Dp = np.concatenate([p[0] for p in pool], 0)
    Rp = np.concatenate([p[1] for p in pool], 0)
    fB, sB, fC, sC = [], [], [], []
    for _ in range(REDRAW):
        i = rng.choice(Dp.shape[0], n_i, replace=False)
        t, _ = run_arm(d, Dp[i], Rp[i], grid)
        fB.append(frac(t)); sB.append(sym(t))
        j = rng.choice(d["rest"].shape[0], n_i, replace=False)
        t, _ = run_arm(d, d["rest"][j], d["rest_saf_rms"][j], grid)
        fC.append(frac(t)); sC.append(sym(t))

    def med(x):
        x = [v for v in x if np.isfinite(v)]
        return float(np.median(x)) if x else np.nan

    rows.append(dict(subject=d["s"], n_donors=n_i, natural_snr_db=nat,
                     A_frac=frac(tA), A_sym=sym(tA), A_full=tA["full"],
                     A_req=tA["req"], A_prev=tA["prev"],
                     B_frac=med(fB), B_sym=med(sB), C_frac=med(fC), C_sym=med(sC),
                     A_by_snr={str(g): byA[j] for j, g in enumerate(grid)}))
    r = rows[-1]
    print(f"{d['s']:5d} {n_i:4d} {nat:7.2f} | {r['A_frac']:8.4f} {r['A_sym']:7.3f} "
          f"| {r['B_frac']:8.4f} {r['B_sym']:7.3f} | {r['C_frac']:8.4f} {r['C_sym']:7.3f}")


def boot(v, n=20000):
    v = np.asarray([x for x in v if np.isfinite(x)])
    if v.size == 0:
        return np.nan, np.nan, np.nan
    dr = np.median(v[rng.integers(0, v.size, (n, v.size))], axis=1)
    return float(np.median(v)), float(np.quantile(dr, .025)), float(np.quantile(dr, .975))


print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)
for a in "ABC":
    f_, lo, hi = boot([r[f"{a}_frac"] for r in rows])
    s_, sl, sh = boot([r[f"{a}_sym"] for r in rows])
    print(f"  arm {a}: interaction-required {f_:.4f} [{lo:.4f}, {hi:.4f}]   "
          f"symmetry {s_:.3f} [{sl:.3f}, {sh:.3f}]")

from scipy.stats import wilcoxon
print("\n  PRIMARY ENDPOINT, symmetry ratio, arm A versus arm B, paired")
dAB = [r["A_sym"] - r["B_sym"] for r in rows if np.isfinite(r["A_sym"]) and np.isfinite(r["B_sym"])]
m, lo, hi = boot(dAB)
print(f"    median difference {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  n = {len(dAB)}")
try:
    print(f"    Wilcoxon p = {wilcoxon(dAB).pvalue:.4f}")
except ValueError as e:
    print(f"    Wilcoxon unavailable: {e}")

print("\n  SECONDARY, arm A versus arm C, the background control")
dAC = [r["A_sym"] - r["C_sym"] for r in rows if np.isfinite(r["A_sym"]) and np.isfinite(r["C_sym"])]
m2, lo2, hi2 = boot(dAC)
print(f"    median difference {m2:+.4f}  95% CI [{lo2:+.4f}, {hi2:+.4f}]")
print("    If A is indistinguishable from C, arm A is injecting background and")
print("    must be withdrawn the way the synthetic EMG branch was in section 9.3.")

print("\n" + "-" * 80)
print("FROZEN PREDICTION CHECK, arm A, pooled over subjects")
print("-" * 80)
print(f"  {'SNR dB':>8} {'full harm':>11} {'required':>10} {'fraction':>10}")
fr = []
for j, g in enumerate(GRID):
    b = bySNR_A[j]
    f_ = b["req"] / b["full"] if b["full"] else np.nan
    fr.append(f_)
    print(f"  {g:8.1f} {b['full']:11d} {b['req']:10d} {f_:10.4f}")

i15 = GRID.index(PRED_AT)
p1 = np.isfinite(fr[i15]) and fr[i15] < PRED_BELOW
tail = [fr[GRID.index(g)] for g in [-7.0, -10.0, -15.0, -20.0, -25.0, -30.0]]
p2 = all(np.isfinite(a) and np.isfinite(b_) and b_ <= a + 1e-12
         for a, b_ in zip(tail[:-1], tail[1:]))
print(f"\n  prediction 1, fraction at {PRED_AT:.0f} dB below {PRED_BELOW}: "
      f"{fr[i15]:.4f}  {'HELD' if p1 else 'REFUTED'}")
print(f"  prediction 2, monotone decrease -7 to -30 dB: {'HELD' if p2 else 'REFUTED'}")
print(f"    tail: {[round(x, 4) if np.isfinite(x) else None for x in tail]}")
print("""
  Both were frozen before any harm statistic existed and both extrapolate
  outside the +2 to -7 dB range the law was derived on. Publish the outcome
  either way: refutation bounds the law's domain and is a result.
""")

with open(f"{OUT}/T4b_RESULT.json", "w") as f:
    json.dump(dict(stage="T4b", bank=BANK, n_subjects=len(rows), grid=GRID,
                   redraws=REDRAW, purity=PURITY, acc_min=ACC_MIN,
                   primary_A_minus_B_symmetry=dict(median=m, lo=lo, hi=hi),
                   secondary_A_minus_C_symmetry=dict(median=m2, lo=lo2, hi=hi2),
                   by_snr_arm_A=[dict(snr=g, **bySNR_A[j], fraction=fr[j])
                                 for j, g in enumerate(GRID)],
                   prediction_1_held=bool(p1), prediction_2_held=bool(p2),
                   subjects=rows), f, indent=2, default=float)
print(f"written: {OUT}/T4b_RESULT.json")
print("\nThe 15 locked subjects remain sealed. Stage T5 opens them once.")
