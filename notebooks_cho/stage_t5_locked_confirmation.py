# REPO STATUS: CURRENT. One-shot confirmation on the 15 sealed subjects. Refuses to rerun.
# Run order, retractions and full context: notebooks_cho/RUN_ORDER.md

"""
STAGE T5  ---  the locked subjects, opened once.

Fifteen subjects were sealed in T2_PREREGISTRATION.json before any analysis and
have been touched by nothing since. This stage opens them, fits their decoders
by the frozen procedure, runs the same three arms, and evaluates the criteria
frozen in T5_PREREGISTRATION.json.

Two protections are in the code, not in good intentions.

  The stage writes T5_RESULT.json and refuses to run if that file already
  exists. A later refusal is the protection working, not a failed experiment.
  This is the same rule as Stage 6 of the BNCI branch.

  The preregistration is located, hashed and printed before anything is loaded.
  Without it the stage stops, because opening sealed subjects without frozen
  criteria turns a confirmation into a second development run.

Decoders could not be fitted in advance without touching sealed data, so they
are fitted here by exactly the development procedure: five contiguous folds per
class, CSP with eight components, LDA with lsqr and automatic shrinkage.

C4 is the interesting one. Development found the saturation law keeps its shape
across datasets and changes only its scale: log-log slopes +0.1390 per dB for
BNCI against +0.1299 for Cho, log-curve correlation 0.9866, and a ratio of 3.54
with sd 0.27 across all eight frozen levels. That was observed, not predicted.
C4 tests it on subjects nobody has looked at.

The embedded engine below was cross-checked against this project's own harmkit
on identical synthetic inputs: every counter matched exactly across four SNR
grids and five seeds. It self tests at startup and stops the stage if it drifts.

Run as one Kaggle CPU cell with cho_bank_v3 and T5_PREREGISTRATION.json
attached. Minutes.
"""

import numpy as np

# ======================================================================
# EMBEDDED HARM ENGINE, verified against harmkit
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

_ok, _ = self_test(verbose=True)
if not _ok:
    raise SystemExit("STOP. Embedded engine failed its self test.")
print()

import glob
import hashlib
import json
import os


BANK, OUT = "/kaggle/working/cho_bank_v3", "/kaggle/working"
for a in glob.glob("/kaggle/input/**/", recursive=True):
    if not glob.glob(f"{BANK}/sub-*.npz") and glob.glob(f"{a}/sub-*.npz"):
        BANK = a.rstrip("/")
if not glob.glob(f"{BANK}/sub-*.npz"):
    raise SystemExit("STOP. cho_bank_v3 not found.")

PRE = next((p for p in [f"{OUT}/T5_PREREGISTRATION.json", "./T5_PREREGISTRATION.json"]
            + glob.glob("/kaggle/input/**/T5_PREREGISTRATION.json", recursive=True)
            if os.path.exists(p)), None)
if PRE is None:
    raise SystemExit("STOP. T5_PREREGISTRATION.json not found. Commit it BEFORE opening "
                     "the locked subjects, or this stops being a confirmation.")
PB = open(PRE, "rb").read()
SHA = hashlib.sha256(PB).hexdigest()
P = json.loads(PB.decode("utf-8"))

# --- one-shot protection, the same rule as Stage 6 of the BNCI branch -------
RESULT = f"{OUT}/T5_RESULT.json"
if os.path.exists(RESULT):
    raise SystemExit(
        "STOP. T5_RESULT.json already exists. The locked subjects are opened ONCE.\n"
        "This refusal is the protection working, not a failed run. If you truly need\n"
        "to rerun, delete the file deliberately and record that you did so.")

SF, W0, NWIN, NFOLD = 250.0, 0.5, 625, 5
PURITY, ACC_MIN, SEED, REDRAW = 0.90, 0.60, int(P["seed"]), 25
GRID_FROZEN = [2.0, 0.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0]
GRID = GRID_FROZEN + [-10.0, -15.0, -20.0, -25.0, -30.0]
LOCKED = set(P["cohort"]["locked"])
DEV_CI = P["criteria"]["C1_effect_size"]["test"]
CI_LO, CI_HI = P["development_result_these_criteria_are_written_against"]["arm_A_interaction_required_CI"]
BNCI = np.array(P["bnci_reference_curve"]["fraction"])
RATIO_LO, RATIO_HI, SPREAD_MAX = 2.65, 4.43, 0.25
MIN_ELIG = 6

rng = np.random.default_rng(SEED)
print("=" * 80)
print("STAGE T5  one-shot confirmation on the 15 locked subjects")
print(f"preregistration sha256 : {SHA}")
print("=" * 80)

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import wilcoxon
import mne
mne.set_log_level("ERROR")


def folds_of(y, k):
    f = np.empty(y.size, int)
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        f[idx] = np.minimum((np.arange(idx.size) * k) // idx.size, k - 1)
    return f


def load(f):
    d = np.load(f, allow_pickle=True)
    s = int(d["subject"])
    if s not in LOCKED:
        return None
    a0 = int(np.median(d["cue_onset"])) + int(round(W0 * SF))
    XL = d["trials_L"][d["keep_L"]][:, :, a0:a0 + NWIN]
    XR = d["trials_R"][d["keep_R"]][:, :, a0:a0 + NWIN]
    pur = np.asarray(d["jaw_purity"], float)
    keep = pur >= PURITY
    return dict(s=s, X=np.concatenate([XL, XR], 0).astype(np.float64),
                y=np.concatenate([np.zeros(len(XL), int), np.ones(len(XR), int)]),
                saf=np.concatenate([np.asarray(d["clean_saf_rms_L"])[d["keep_L"]],
                                    np.asarray(d["clean_saf_rms_R"])[d["keep_R"]]]),
                jaw=np.asarray(d["jaw_dec"], float)[keep],
                jaw_saf_rms=np.asarray(d["jaw_saf_rms"], float)[keep],
                rest=np.asarray(d["rest_dec"], float),
                rest_saf_rms=np.asarray(d["rest_saf_rms"], float))


subs = [x for x in (load(f) for f in sorted(glob.glob(f"{BANK}/sub-*.npz"))) if x]
print(f"locked subjects loaded: {len(subs)}\n")
print(f"{'subj':>5} {'trials':>7} {'clean acc':>10} {'n pure':>7} {'eligible':>9}")

for d in subs:
    d["folds"] = folds_of(d["y"], NFOLD)
    oof, per = np.zeros(d["y"].size), []
    for k in range(NFOLD):
        tr = d["folds"] != k
        csp = CSP(n_components=8, reg=None, log=True, norm_trace=False,
                  transform_into="average_power").fit(d["X"][tr], d["y"][tr])
        lda = LDA(solver="lsqr", shrinkage="auto").fit(csp.transform(d["X"][tr]), d["y"][tr])
        te = ~tr
        oof[te] = (csp.transform(d["X"][te]) @ lda.coef_.T).ravel() + float(lda.intercept_[0])
        per.append((csp.filters_[:8], lda.coef_.ravel(), float(lda.intercept_[0])))
    d["acc"] = float(np.mean((oof > 0).astype(int) == d["y"]))
    d["models"] = per
    d["elig"] = d["acc"] >= ACC_MIN and d["jaw"].shape[0] >= 1
    print(f"{d['s']:5d} {d['y'].size:7d} {d['acc']:10.4f} {d['jaw'].shape[0]:7d} "
          f"{'yes' if d['elig'] else 'NO':>9}")

work = [d for d in subs if d["elig"]]
print(f"\neligible: {len(work)}/15   (C0 needs >= {MIN_ELIG})")


def per_snr(res, n):
    e = (res["m_clean"] > 0)[:, None, None]
    h = e & (res["m_full"] < 0)
    r = h & (res["m_no"] > 0)
    p_ = e & (res["m_no"] < 0) & (res["m_full"] > 0)
    return [dict(full=int(h[:, :, j].sum()), req=int(r[:, :, j].sum()),
                 prev=int(p_[:, :, j].sum())) for j in range(n)]


def run_arm(d, donors, saf_rms, grid):
    tot = dict(full=0, req=0, prev=0)
    by = [dict(full=0, req=0, prev=0) for _ in grid]
    for k in range(NFOLD):
        te = d["folds"] == k
        if not te.any() or donors.shape[0] == 0:
            continue
        ref = float(np.median(d["saf"][~te]))
        lam = ref / (saf_rms[:, None] * 10.0 ** (np.asarray(grid)[None, :] / 20.0))
        W, c, i0 = d["models"][k]
        r = harm_counts(W, c, i0, d["X"][te], d["y"][te], donors, lam)
        tot["full"] += r["full_harm"]; tot["req"] += r["interaction_required"]
        tot["prev"] += r["interaction_prevented"]
        for j, cc in enumerate(per_snr(r, len(grid))):
            for kk in ("full", "req", "prev"):
                by[j][kk] += cc[kk]
    return tot, by


frac = lambda t: t["req"] / t["full"] if t["full"] else np.nan
sym = lambda t: t["prev"] / t["req"] if t["req"] else np.nan

if len(work) < MIN_ELIG:
    print("\nC0 FAILED. Underpowered. Every criterion below is NOT EVALUABLE.")

print(f"\n{'subj':>5} {'n_i':>4} {'natSNR':>7} | {'A frac':>8} {'A sym':>7} | {'B sym':>7}")
rows, bySNR = [], [dict(full=0, req=0) for _ in GRID]
for d in work:
    n_i = d["jaw"].shape[0]
    nat = float(np.median(20.0 * np.log10(float(np.median(d["saf"]))
                                          / np.maximum(d["jaw_saf_rms"], 1e-300))))
    grid = GRID + [nat]
    tA, byA = run_arm(d, d["jaw"], d["jaw_saf_rms"], grid)
    for j in range(len(GRID)):
        bySNR[j]["full"] += byA[j]["full"]; bySNR[j]["req"] += byA[j]["req"]
    Dp = np.concatenate([o["jaw"] for o in work if o["s"] != d["s"]], 0)
    Rp = np.concatenate([o["jaw_saf_rms"] for o in work if o["s"] != d["s"]], 0)
    sB = []
    for _ in range(REDRAW):
        i = rng.choice(Dp.shape[0], n_i, replace=False)
        t, _ = run_arm(d, Dp[i], Rp[i], grid)
        sB.append(sym(t))
    sB = [v for v in sB if np.isfinite(v)]
    rows.append(dict(subject=d["s"], n_donors=n_i, natural_snr_db=nat, acc=d["acc"],
                     A_frac=frac(tA), A_sym=sym(tA), A_full=tA["full"], A_req=tA["req"],
                     B_sym=float(np.median(sB)) if sB else np.nan))
    r = rows[-1]
    print(f"{d['s']:5d} {n_i:4d} {nat:7.2f} | {r['A_frac']:8.4f} {r['A_sym']:7.3f} "
          f"| {r['B_sym']:7.3f}")


def boot(v, n=20000):
    v = np.asarray([x for x in v if np.isfinite(x)])
    if v.size == 0:
        return np.nan, np.nan, np.nan
    dr = np.median(v[rng.integers(0, v.size, (n, v.size))], axis=1)
    return float(np.median(v)), float(np.quantile(dr, .025)), float(np.quantile(dr, .975))


print("\n" + "=" * 80)
print("PREREGISTERED CRITERIA")
print("=" * 80)
ev = len(work) >= MIN_ELIG
c0 = ev
fA, flo, fhi = boot([r["A_frac"] for r in rows])
c1 = ev and np.isfinite(fA) and CI_LO <= fA <= CI_HI
dAB = [r["A_sym"] - r["B_sym"] for r in rows
       if np.isfinite(r["A_sym"]) and np.isfinite(r["B_sym"])]
m, lo, hi = boot(dAB)
c2 = ev and np.isfinite(m) and lo <= 0 <= hi

fr = [b["req"] / b["full"] if b["full"] else np.nan for b in bySNR]
tail_i = [GRID.index(g) for g in (-7.0, -10.0, -15.0, -20.0, -25.0, -30.0)]
tail = [fr[i] for i in tail_i]
c3 = ev and all(np.isfinite(a) and np.isfinite(b) and b <= a + 1e-12
                for a, b in zip(tail[:-1], tail[1:]))
ratio = np.array(fr[:8]) / BNCI
rm, rs = float(np.median(ratio)), float(np.std(ratio, ddof=1))
c4 = ev and RATIO_LO <= rm <= RATIO_HI and (rs / max(rm, 1e-12)) < SPREAD_MAX

print(f"  C0 power, eligible >= {MIN_ELIG}                : {len(work)}/15   "
      f"{'PASS' if c0 else 'FAIL'}")
print(f"  C1 arm A fraction in [{CI_LO:.4f}, {CI_HI:.4f}] : {fA:.4f} "
      f"[{flo:.4f}, {fhi:.4f}]   {'PASS' if c1 else 'FAIL' if ev else 'n/a'}")
print(f"  C2 A-B symmetry interval covers zero     : {m:+.4f} [{lo:+.4f}, {hi:+.4f}]   "
      f"{'PASS' if c2 else 'FAIL' if ev else 'n/a'}")
if len(dAB) >= 6:
    try:
        print(f"       Wilcoxon p = {wilcoxon(dAB).pvalue:.4f}")
    except ValueError as e:
        print(f"       Wilcoxon unavailable: {e}")
print(f"  C3 monotone -7 to -30 dB                 : "
      f"{[round(x, 4) if np.isfinite(x) else None for x in tail]}   "
      f"{'PASS' if c3 else 'FAIL' if ev else 'n/a'}")
print(f"  C4 Cho/BNCI ratio, median in [{RATIO_LO}, {RATIO_HI}]   : "
      f"{rm:.2f}, sd/median {rs/max(rm,1e-12):.3f}   "
      f"{'PASS' if c4 else 'FAIL' if ev else 'n/a'}")
print(f"\n  {'SNR':>6} {'locked':>9} {'BNCI':>9} {'ratio':>7}")
for g, a, b, r_ in zip(GRID_FROZEN, fr[:8], BNCI, ratio):
    print(f"  {g:+6.1f} {a:9.4f} {b:9.4f} {r_:7.2f}")

print(f"""
  OVERALL: {'ALL PREREGISTERED CRITERIA PASSED' if all([c0, c1, c2, c3, c4]) else 'NOT ALL CRITERIA PASSED'}

  Publish every line above whatever it says. A failed criterion here is a
  measurement about how far the development result generalises, and the file
  that defines these criteria was committed before these subjects were opened.
""")

json.dump(dict(stage="T5", prereg_sha256=SHA, bank=BANK, n_locked=len(subs),
               n_eligible=len(work), grid=GRID,
               arm_A_fraction=dict(median=fA, lo=flo, hi=fhi),
               A_minus_B_symmetry=dict(median=m, lo=lo, hi=hi),
               by_snr=[dict(snr=g, **bySNR[j], fraction=fr[j])
                       for j, g in enumerate(GRID)],
               cho_over_bnci=dict(median=rm, sd=rs, values=[float(x) for x in ratio]),
               criteria=dict(C0=bool(c0), C1=bool(c1), C2=bool(c2),
                             C3=bool(c3), C4=bool(c4)),
               all_passed=bool(all([c0, c1, c2, c3, c4])),
               subjects=rows,
               all_locked_accuracy=[dict(subject=d["s"], acc=d["acc"],
                                         n_pure=int(d["jaw"].shape[0]),
                                         eligible=bool(d["elig"])) for d in subs]),
          open(RESULT, "w"), indent=2, default=float)
print(f"written: {RESULT}   (this stage will now refuse to run again)")
