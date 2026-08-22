"""Decision-level harm engine with exact interaction ablation, plus self test."""

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


if __name__ == "__main__":
    ok, _ = self_test()
    raise SystemExit(0 if ok else 1)
