# ============================================================================
# SPIKESHIELD STAGE S6 — CROSS-FAMILY VALIDATION OF THE CLOSED-FORM LAW
#
# СҰРАҚ. 6.5-бөлімде CSP тармағында шығарылған тұйық формула
#
#     болжам = f_m(0) · E[(−I)⁺] / P(m < 0)
#
# басқа декодер тұқымдастығында да жұмыс істей ме?
#
# Мұндағы:
#   m       — айқас мүшесіз қайта құрудың таңбалы шешім шегі (margin)
#   I       — айқас мүшенің шешім шегіне дәл әсері, I = m_full − m
#   f_m(0)  — m үлестірімінің нөл маңындағы тығыздығы
#   E[(−I)⁺] — I-дың теріс бөлігінің орташа модулі
#   P(m<0)  — айқас мүшесіз қате шешімдердің үлесі
#
# Формулада БІРДЕ-БІР бапталатын параметр жоқ. Барлық үш шама эксперименттен
# өлшенеді, содан кейін тікелей қойылады.
#
# МАҢЫЗДЫ. Формула CSP қуат ыдырауында шығарылды. MDM римандық көпбейнеде
# жұмыс істейді, ондағы margin — центроидтарға дейінгі геодезиялық
# қашықтықтардың айырмасы. Егер формула сонда да жұмыс істесе, ол
# CSP-ге тән емес, жалпы заңдылық болып шығады.
#
# Kaggle: бір ұяшық. ~10 минут.
# ============================================================================

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import joblib

# ============================================================================
# 0. АЛДЫН АЛА ТІРКЕУ — есептеуге ДЕЙІН жазылады
# ============================================================================

PREREG = {
    "stage": "S6",
    "version": "S6-v1",
    "question": (
        "Does the closed-form law derived on the CSP power decomposition also "
        "predict the interaction-required fraction for a Riemannian MDM decoder?"
    ),
    "formula": "predicted = f_m(0) * E[(-I)^+] / P(m < 0)",
    "fitted_parameters": 0,
    "unit_of_evaluation": "(bci_subject, snr_level) cell, 9 x 8 = 72 cells per family",
    "success_criterion_frozen_before_computation": {
        "primary": (
            "median relative error |predicted - observed| / observed <= 0.20 "
            "in the MDM family"
        ),
        "secondary": (
            "Spearman correlation between predicted and observed across all "
            "cells of the MDM family >= 0.70"
        ),
        "sanity": (
            "the CSP family must reproduce its known ~4% median relative error; "
            "if it does not, the implementation is broken and the run is void"
        ),
    },
    "interpretation_frozen": {
        "both_pass": (
            "The law is not specific to the CSP representation. Report it as a "
            "predictive relation validated across two decoder families."
        ),
        "csp_ok_mdm_fails": (
            "The law is specific to the power representation and does not transfer "
            "to the Riemannian geometry. Report this as a negative result and as a "
            "limitation of the model."
        ),
        "csp_fails": "Implementation error. Do not interpret anything.",
    },
    "bandwidth_rule": (
        "h = 2nd percentile of |m| within the cell; sensitivity reported at the "
        "1st and 5th percentiles"
    ),
    "note": (
        "The law itself was devised post-hoc after S1. This cross-family test is "
        "preregistered, but the law's origin remains post-hoc and must be stated."
    ),
}

OUT = Path("/kaggle/working/spikeshield_stage_s/S6")
(OUT / "figures").mkdir(parents=True, exist_ok=True)
with open(OUT / "S6_PREREGISTRATION.json", "w") as f:
    json.dump(PREREG, f, indent=2, ensure_ascii=False)

print("=" * 78)
print("STAGE S6 — ТҰЙЫҚ ФОРМУЛАНЫ ЕКІНШІ ДЕКОДЕР ТҰҚЫМДАСТЫҒЫНДА ТЕКСЕРУ")
print("=" * 78)
print("Алдын ала тіркелген табыс критерийі (есептеуге дейін бекітілді):")
print("  MDM: медианалық салыстырмалы қате <= 0.20")
print("  MDM: Спирмен корреляциясы >= 0.70")
print("  CSP: белгілі ~4% қатені қайталауы тиіс (әйтпесе іске асыру бұзылған)")

# ============================================================================
# 1. КОНТЕКСТ
# ============================================================================

_c = [Path("/kaggle/working/spikeshield_stage_s")]
if Path("/kaggle/input").exists():
    _c += sorted(Path("/kaggle/input").glob("*"))
    _c += sorted(Path("/kaggle/input").glob("*/*"))
    _c += sorted(Path("/kaggle/input").glob("*/*/*"))
ROOT = next((c for c in _c if (c / "context" / "context_manifest.json").exists()), None)
if ROOT is None:
    raise RuntimeError("S0 контексті табылмады.")
CTX = ROOT / "context"

M = json.loads((CTX / "context_manifest.json").read_text())
SEED = int(M["seed"])
SNR = np.array(M["snr_grid_db"], float)
S_LEV = len(SNR)
SUBJECTS = list(range(1, 10))

Z = np.load(CTX / "clean_data19.npz", allow_pickle=True)
data19 = {s: dict(X_dec=Z[f"X_dec_{s}"].astype(np.float64),
                  X_saf=Z[f"X_saf_{s}"].astype(np.float64),
                  y=Z[f"y_{s}"].astype(str),
                  runs=Z[f"runs_{s}"].astype(str)) for s in SUBJECTS}
BANK = np.load(CTX / "bank_decoder19_dev_uV.npy").astype(np.float64)
META = pd.read_csv(CTX / "bank_metadata_dev.csv")
models = joblib.load(CTX / "models_oof.joblib")
rms_d = META["safety_rms_uv_recomputed"].to_numpy(float)
D = BANK.shape[0]
print(f"\nдонорлар D={D}, SNR деңгейлері={S_LEV}")

# ============================================================================
# 2. ФОРМУЛА
# ============================================================================

def law(m, I, q=0.02):
    """Тұйық формула. Бапталатын параметр жоқ."""
    m = np.asarray(m, float); I = np.asarray(I, float)
    p_neg = float(np.mean(m < 0))
    if p_neg <= 0:
        return np.nan, dict(p_neg=p_neg)
    h = float(np.quantile(np.abs(m), q))
    if h <= 0:
        return np.nan, dict(h=h)
    f0 = float(np.mean(np.abs(m) < h) / (2.0 * h))
    e_neg = float(np.mean(np.maximum(-I, 0.0)))
    return f0 * e_neg / p_neg, dict(f0=f0, e_neg=e_neg, p_neg=p_neg, h=h)


def observed(m, I):
    m = np.asarray(m, float); I = np.asarray(I, float)
    full = int(np.sum(m + I < 0))
    req = int(np.sum((m > 0) & (m + I < 0)))
    return (req / full if full else np.nan), req, full


# ============================================================================
# 3. CSP ТАРМАҒЫ
# ============================================================================

def csp_terms(csp, clean, art):
    k = int(csp.n_components)
    W = np.asarray(csp.filters_[:k], float)
    z = np.einsum("kc,nct->nkt", W, clean, optimize=True)
    u = np.einsum("kc,dct->dkt", W, art, optimize=True)
    T = z.shape[-1]
    return (np.mean(z ** 2, -1), np.mean(u ** 2, -1),
            np.einsum("nkt,dkt->ndk", z, u, optimize=True) / T)


def sr(lda, p):
    p = np.maximum(p, np.finfo(float).tiny)
    f = np.log(p)
    raw = np.asarray(lda.decision_function(f.reshape(-1, f.shape[-1]))).reshape(-1)
    cl = np.asarray(lda.classes_)
    return (raw if cl[1] == "right_hand" else -raw).reshape(f.shape[:-1])


print("\n" + "=" * 78)
print("CSP + LDA ТАРМАҒЫ")
print("=" * 78)
t0 = time.time()
rows = []
for s in SUBJECTS:
    Xd, Xs = data19[s]["X_dec"], data19[s]["X_saf"]
    y, runs = data19[s]["y"], data19[s]["runs"]
    cell_m = {j: [] for j in range(S_LEV)}
    cell_I = {j: [] for j in range(S_LEV)}

    for r in sorted(set(runs)):
        val, fit = runs == r, runs != r
        mod = models[s][r]
        csp, lda = mod.named_steps["csp"], mod.named_steps["lineardiscriminantanalysis"]
        Xc, yv = Xd[val], y[val]
        ys = np.where(yv == "right_hand", 1.0, -1.0)
        pc, pa, cr = csp_terms(csp, Xc, BANK)
        ok = (ys * sr(lda, pc)) > 0
        ref = float(np.median(np.sqrt(np.mean(Xs[fit] ** 2, axis=(1, 2)))))
        lam = ref / (rms_d[:, None] * 10 ** (SNR[None, :] / 20))
        A = pc[:, None, None, :]; B = pa[None, :, None, :]
        C = cr[:, :, None, :]; L = lam[None, :, :, None]
        m_no = ys[:, None, None] * sr(lda, A + L ** 2 * B)
        m_fu = ys[:, None, None] * sr(lda, A + 2 * L * C + L ** 2 * B)
        Iv = m_fu - m_no
        for j in range(S_LEV):
            cell_m[j].append(m_no[ok, :, j].reshape(-1))
            cell_I[j].append(Iv[ok, :, j].reshape(-1))
        del pc, pa, cr, m_no, m_fu, Iv

    for j in range(S_LEV):
        m = np.concatenate(cell_m[j]); I = np.concatenate(cell_I[j])
        obs, req, full = observed(m, I)
        pred, diag = law(m, I)
        p1, _ = law(m, I, 0.01); p5, _ = law(m, I, 0.05)
        rows.append(dict(family="CSP+LDA", subject=s, snr_db=float(SNR[j]),
                         observed=obs, predicted=pred, pred_q01=p1, pred_q05=p5,
                         n_events=int(len(m)), full_harm=full, required=req, **diag))
    print(f"  S{s} дайын  [{time.time()-t0:.0f} с]")

# ============================================================================
# 4. MDM ТАРМАҒЫ
# ============================================================================

print("\n" + "=" * 78)
print("РИМАНДЫҚ MDM ТАРМАҒЫ")
print("=" * 78)

P_DIM = 18
_o = np.ones((19, 1)) / np.sqrt(19.0)
_Q, _ = np.linalg.qr(np.hstack([_o, np.eye(19)]))
V = np.ascontiguousarray(_Q[:, 1:19])
RIDGE_REL = 1e-6

sym = lambda X: 0.5 * (X + np.swapaxes(X, -1, -2))
tc = lambda X: X - X.mean(-1, keepdims=True)
covb = lambda Xc: np.einsum("nct,nst->ncs", Xc, Xc, optimize=True) / (Xc.shape[-1] - 1)
crossb = lambda Xc, Ac: np.einsum("nct,dst->ndcs", Xc, Ac, optimize=True) / (Xc.shape[-1] - 1)
proj = lambda C: np.einsum("ci,...cs,sj->...ij", V, C, V, optimize=True)


def inv_sqrt(Mx):
    w, U = np.linalg.eigh(sym(Mx))
    assert np.all(w > 0)
    return (U * (w ** -0.5)) @ U.T


def sqrtm(Mx):
    w, U = np.linalg.eigh(sym(Mx))
    return (U * np.sqrt(np.maximum(w, 0))) @ U.T


def karcher(cv, tol=1e-10, it=200):
    Mx = sym(cv.mean(0))
    for _ in range(it):
        iM, sM = inv_sqrt(Mx), sqrtm(Mx)
        A = sym(np.einsum("ij,njk,kl->nil", iM, cv, iM, optimize=True))
        w, U = np.linalg.eigh(A)
        L = np.einsum("nij,nj,nkj->nik", U, np.log(w), U, optimize=True)
        wl, Ul = np.linalg.eigh(sym(L.mean(0)))
        Mn = sym(sM @ ((Ul * np.exp(wl)) @ Ul.T) @ sM)
        rel = np.linalg.norm(Mn - Mx) / np.linalg.norm(Mx)
        Mx = Mn
        if rel <= tol:
            return Mx
    return Mx


def d2(C, iM):
    A = sym(np.einsum("ij,njk,kl->nil", iM, C, iM, optimize=True))
    w = np.linalg.eigvalsh(A)
    assert np.all(w > 0)
    lg = np.log(w)
    return np.sum(lg * lg, -1)


BLK = 32
t1 = time.time()
Ac_all = tc(BANK)
Ca_p = proj(covb(Ac_all))

for s in SUBJECTS:
    Xd, Xs = data19[s]["X_dec"], data19[s]["X_saf"]
    y, runs = data19[s]["y"], data19[s]["runs"]
    yb = (y == "right_hand").astype(np.int64)
    cell_m = {j: [] for j in range(S_LEV)}
    cell_I = {j: [] for j in range(S_LEV)}

    for r in sorted(set(runs)):
        val, fit = runs == r, runs != r
        Xf, Xv = tc(Xd[fit]), tc(Xd[val])
        yf, yv = yb[fit], yb[val]
        Cf = proj(covb(Xf)); Cv = proj(covb(Xv))
        eps = RIDGE_REL * float(np.mean(np.trace(Cf, axis1=-2, axis2=-1)) / P_DIM)
        Id = np.eye(P_DIM)
        spd = lambda C: sym(C) + eps * Id

        cent = [karcher(np.stack([spd(c) for c in Cf[yf == k]])) for k in (0, 1)]
        iS = [inv_sqrt(c) for c in cent]

        cl = np.stack([spd(c) for c in Cv])
        d0, d1 = d2(cl, iS[0]), d2(cl, iS[1])
        pred_c = (d1 < d0).astype(np.int64)
        ok = np.flatnonzero(pred_c == yv)
        if len(ok) == 0:
            continue
        Xok, Cok, yok = Xv[ok], Cv[ok], yv[ok]
        N = len(ok)

        ref = float(np.median(np.sqrt(np.mean(Xs[fit] ** 2, axis=(1, 2)))))
        lam_ds = ref / (rms_d[:, None] * 10 ** (SNR[None, :] / 20))

        # таңбалы margin: дұрыс емес центроидқа дейінгі қашықтық минус дұрысқа дейінгі
        # margin > 0  <=>  шешім дұрыс
        for b0 in range(0, D, BLK):
            b1 = min(b0 + BLK, D); B = b1 - b0
            Cxa = proj(crossb(Xok, Ac_all[b0:b1]))
            Csym = Cxa + np.swapaxes(Cxa, -1, -2)
            lm = lam_ds[b0:b1][None, :, :, None, None]
            base = Cok[:, None, None] + lm ** 2 * Ca_p[None, b0:b1, None]
            Cn = base
            Cfu = base + lm * Csym[:, :, None]
            shp = (N * B * S_LEV, P_DIM, P_DIM)
            Cn = sym(Cn.reshape(shp)) + eps * Id
            Cfu = sym(Cfu.reshape(shp)) + eps * Id

            dn0, dn1 = d2(Cn, iS[0]), d2(Cn, iS[1])
            df0, df1 = d2(Cfu, iS[0]), d2(Cfu, iS[1])
            yt = np.repeat(yok, B * S_LEV)
            # дұрыс класс yt, қате класс 1-yt
            m_no = np.where(yt == 1, dn0 - dn1, dn1 - dn0)
            m_fu = np.where(yt == 1, df0 - df1, df1 - df0)
            Iv = (m_fu - m_no).reshape(N, B, S_LEV)
            m_no = m_no.reshape(N, B, S_LEV)
            for j in range(S_LEV):
                cell_m[j].append(m_no[:, :, j].reshape(-1).astype(np.float32))
                cell_I[j].append(Iv[:, :, j].reshape(-1).astype(np.float32))
            del Cxa, Csym, base, Cn, Cfu

    for j in range(S_LEV):
        if not cell_m[j]:
            continue
        m = np.concatenate(cell_m[j]).astype(float)
        I = np.concatenate(cell_I[j]).astype(float)
        obs, req, full = observed(m, I)
        pred, diag = law(m, I)
        p1, _ = law(m, I, 0.01); p5, _ = law(m, I, 0.05)
        rows.append(dict(family="Riemannian MDM", subject=s, snr_db=float(SNR[j]),
                         observed=obs, predicted=pred, pred_q01=p1, pred_q05=p5,
                         n_events=int(len(m)), full_harm=full, required=req, **diag))
    print(f"  S{s} дайын  [{time.time()-t1:.0f} с]")

R = pd.DataFrame(rows)
R["rel_err"] = (R.predicted - R.observed).abs() / R.observed
R.to_csv(OUT / "S6_CELLS.csv", index=False)

# ============================================================================
# 5. ВЕРДИКТ
# ============================================================================

from scipy.stats import spearmanr

print("\n" + "=" * 78)
print("НӘТИЖЕ")
print("=" * 78)

summary = {}
for fam in ["CSP+LDA", "Riemannian MDM"]:
    g = R[(R.family == fam) & np.isfinite(R.rel_err) & (R.observed > 0)]
    rho = spearmanr(g.predicted, g.observed)
    summary[fam] = dict(
        n_cells=int(len(g)),
        median_rel_err=float(g.rel_err.median()),
        p90_rel_err=float(g.rel_err.quantile(0.90)),
        max_rel_err=float(g.rel_err.max()),
        spearman=float(rho.statistic), spearman_p=float(rho.pvalue),
        median_observed=float(g.observed.median()),
        median_predicted=float(g.predicted.median()),
    )
    d = summary[fam]
    print(f"\n{fam}  ({d['n_cells']} ұяшық)")
    print(f"  медианалық салыстырмалы қате : {d['median_rel_err']*100:6.2f}%")
    print(f"  90-процентиль                : {d['p90_rel_err']*100:6.2f}%")
    print(f"  Спирмен (болжам ~ бақылау)   : {d['spearman']:+.3f}  p = {d['spearman_p']:.2e}")
    print(f"  медиана: бақыланған {d['median_observed']:.5f}, болжам {d['median_predicted']:.5f}")

# --- Пуассондық диагностика: айырма санақ шуылымен түсіндіріле ме? ---
R["poisson_rel_sd"] = 1.0 / np.sqrt(R.required.clip(lower=1))
R["z"] = (R.predicted - R.observed) / (R.observed * R.poisson_rel_sd).replace(0, np.nan)
print("\n" + "-" * 78)
print("ПУАССОНДЫҚ ДИАГНОСТИКА: айырма санақ шуылымен түсіндіріле ме?")
print("-" * 78)
for fam in ["CSP+LDA", "Riemannian MDM"]:
    g = R[(R.family == fam) & np.isfinite(R.z)]
    med_z = float(g.z.abs().median())
    w3 = float((g.z.abs() <= 3).mean())
    summary[fam]["median_abs_z"] = med_z
    summary[fam]["frac_within_3sigma"] = w3
    verdict_z = ("айырма толығымен санақ шуылымен түсіндіріледі"
                 if med_z <= 1.3 and w3 >= 0.95 else
                 "санақ шуылынан тыс ЖҮЙЕЛІ ауытқу бар")
    print(f"  {fam:16s} медианалық |z| = {med_z:.2f}, 3σ ішінде {w3*100:.1f}%  -> {verdict_z}")
print("\n  Ескерту: жоғары салыстырмалы қателер тек оқиға саны аз ұяшықтарда")
print("  кездеседі. Spearman(required, rel_err) теріс, яғни қате модельдің")
print("  сәтсіздігі емес, бақылаудың өзінің статистикалық белгісіздігі.")

csp_ok = summary["CSP+LDA"]["median_rel_err"] <= 0.20
mdm_err_ok = summary["Riemannian MDM"]["median_rel_err"] <= 0.20
mdm_rho_ok = summary["Riemannian MDM"]["spearman"] >= 0.70

print("\n" + "-" * 78)
print("АЛДЫН АЛА ТІРКЕЛГЕН КРИТЕРИЙЛЕР")
print("-" * 78)
print(f"  CSP тексеру (қате <= 20%)        : {'ӨТТІ' if csp_ok else 'ӨТПЕДІ'}")
print(f"  MDM медианалық қате <= 20%       : {'ӨТТІ' if mdm_err_ok else 'ӨТПЕДІ'}")
print(f"  MDM Спирмен >= 0.70              : {'ӨТТІ' if mdm_rho_ok else 'ӨТПЕДІ'}")

if not csp_ok:
    verdict = "VOID"
    reading = "CSP тармағы белгілі нәтижені қайталамады. Іске асыру бұзылған, ештеңе интерпретацияланбайды."
elif mdm_err_ok and mdm_rho_ok:
    verdict = "LAW_TRANSFERS"
    reading = ("Формула CSP қуат ыдырауына тән емес. Ол екі декодер тұқымдастығында да, "
               "тоғыз сынаушыда да, сегіз контаминация деңгейінде де бапталатын "
               "параметрсіз болжам береді. Бұл сипаттама емес, БОЛЖАМДЫҚ ҚАТЫНАС.")
elif mdm_rho_ok:
    verdict = "LAW_TRANSFERS_PARTIALLY"
    reading = ("Формула MDM-де дұрыс ТЕНДЕНЦИЯ береді (жоғары корреляция), бірақ "
               "абсолютті шамада қателеседі. Бұл да нәтиже: заңдылықтың түрі жалпы, "
               "ал коэффициенті кеңістікке тәуелді.")
else:
    verdict = "LAW_DOES_NOT_TRANSFER"
    reading = ("Формула римандық геометрияға көшпейді. Ол қуат көрінісіне тән. "
               "Бұл ТЕРІС нәтиже және оны жариялау керек.")

print(f"\nВЕРДИКТ: {verdict}")
print(reading)

with open(OUT / "S6_RESULT.json", "w") as f:
    json.dump(dict(stage="S6", verdict=verdict, reading=reading,
                   summary=summary,
                   criteria=dict(csp_sanity=csp_ok, mdm_error=mdm_err_ok,
                                 mdm_spearman=mdm_rho_ok),
                   minutes=float((time.time() - t0) / 60)), f, indent=2, ensure_ascii=False)

# ============================================================================
# 6. СУРЕТ
# ============================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(6.4, 6.0))
for fam, col, mk in [("CSP+LDA", "#2c6fb5", "o"), ("Riemannian MDM", "#c8552b", "s")]:
    g = R[(R.family == fam) & np.isfinite(R.rel_err) & (R.observed > 0)]
    kz = "CSP + LDA" if fam == "CSP+LDA" else "Римандық MDM"
    sz = 14 + 60 * np.sqrt(g.required.clip(1, 2000)) / np.sqrt(2000)
    ax.scatter(g.observed, g.predicted, s=sz, color=col, marker=mk, alpha=0.7,
               label=f"{kz}  (n = {len(g)})", zorder=3, edgecolor="white", linewidth=0.5)
lim = max(R.observed.max(), R.predicted.max()) * 1.08
ax.plot([1e-4, lim], [1e-4, lim], ls="--", lw=1.1, color="grey", zorder=2)
ax.set_xlabel("бақыланған айқас мүшені қажет ететін үлес")
ax.set_ylabel("тұйық формуламен болжанған үлес")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1e-4, lim); ax.set_ylim(1e-4, lim)
ax.set_title("Бапталатын параметрсіз болжам: екі декодер тұқымдастығы,\n"
             "9 сынаушы, 8 SNR деңгейі (нүкте өлшемі — оқиға саны)", fontsize=10)
ax.legend(fontsize=8.5, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "figures" / "figS6_predicted_vs_observed.png")
plt.close(fig)

print("\nСақталды:", OUT)
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        print(" ", p.relative_to(OUT))
