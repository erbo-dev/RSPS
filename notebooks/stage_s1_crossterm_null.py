# ============================================================================
# SPIKESHIELD STAGE S1 — CROSS-TERM NULL CALIBRATION
#
# Вопрос: наблюдаемая доля interaction-required HARM отличается от того,
# что даёт заведомо независимая контаминация той же спектральной и
# пространственной структуры?
#
# Ветка OBSERVED : реальные доноры (воспроизводит логику Stage 3F).
# Ветка NULL     : фазово-рандомизированные суррогаты доноров.
#                  Общий набор фаз по всем 19 каналам сохраняет
#                  ковариационную матрицу донора ТОЧНО, поэтому p_art
#                  и SNR-масштабирование не меняются вообще.
#                  Меняется только реализация перекрёстного члена.
# Ветка POSCTRL  : донор с принудительным выравниванием a' = a + beta*x.
#                  Проверка, что измеритель вообще реагирует.
#
# Данные: только 0train, только 247 development-доноров.
# Kaggle: один cell после S0. ~30-50 минут.
# ============================================================================

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================================
# 0. КОНТЕКСТ
# ============================================================================

CTX_CANDIDATES = [Path("/kaggle/working/spikeshield_stage_s/context")]
if Path("/kaggle/input").exists():
    CTX_CANDIDATES += sorted(Path("/kaggle/input").glob("*/context"))
    CTX_CANDIDATES += sorted(Path("/kaggle/input").glob("*/*/context"))

CTX = next((c for c in CTX_CANDIDATES if (c / "context_manifest.json").exists()), None)
if CTX is None:
    raise RuntimeError(
        "Контекст S0 не найден. Запустите stage_s0_context.py или приложите его вывод "
        "как Kaggle Dataset."
    )

OUT = Path("/kaggle/working/spikeshield_stage_s/S1")
OUT.mkdir(parents=True, exist_ok=True)

with open(CTX / "context_manifest.json") as f:
    MANIFEST = json.load(f)

SEED = int(MANIFEST["seed"])
SUBJECTS = list(range(1, 10))
SNR_GRID_DB = np.array(MANIFEST["snr_grid_db"], dtype=np.float64)
N_TIMES = int(MANIFEST["n_times"])
COMMON19 = MANIFEST["common19"]
FORBIDDEN = set(MANIFEST["forbidden_artifact_subjects"])

print("=" * 78)
print("STAGE S1 — CROSS-TERM NULL CALIBRATION")
print("=" * 78)
print(f"context: {CTX}")
print(f"SNR grid: {SNR_GRID_DB.tolist()}")

# ============================================================================
# 1. ПРЕДРЕГИСТРАЦИЯ (пишется на диск ДО вычисления результатов)
# ============================================================================

N_NULL_REPEATS = 5           # число независимых суррогатных банков
POSCTRL_BETAS = [0.0, 0.25, 0.5, 1.0]
POSCTRL_N_TRIALS = 8
POSCTRL_N_DONORS = 20

PREREG = {
    "stage": "S1",
    "version": "S1-v1",
    "primary_statistic": "interaction_required_harm_fraction, per BCI subject",
    "arms": ["observed", f"null x{N_NULL_REPEATS} (shared-phase randomized donors)",
             "positive_control (a' = a + beta*x)"],
    "null_construction": (
        "rfft along time; one random phase vector shared by all 19 channels; "
        "irfft. Preserves the donor covariance matrix exactly, hence p_art "
        "and the frozen SNR scaling are unchanged."
    ),
    "decision_rule_frozen_before_results": {
        "null_matches_observed": (
            "95% subject-level intervals of observed and null medians overlap -> "
            "the cross-term carries no pair-specific information; report the "
            "interaction-secondary finding as a quantitative magnitude statement."
        ),
        "observed_above_null": "alignment structure exists; describe separately.",
        "observed_below_null": "systematic anti-alignment; describe separately.",
        "positive_control_must_increase": (
            "interaction_required must increase monotonically in beta; "
            "otherwise the measurement instrument is broken and S1 is void."
        ),
    },
    "data_scope": "BNCI2014_001 0train only; 247 development CHEW donors only.",
    "seed": SEED,
    "n_null_repeats": N_NULL_REPEATS,
    "posctrl_betas": POSCTRL_BETAS,
}
with open(OUT / "S1_PREREGISTRATION.json", "w") as f:
    json.dump(PREREG, f, indent=2)
print("\nprereg written:", OUT / "S1_PREREGISTRATION.json")

# ============================================================================
# 2. ЗАГРУЗКА
# ============================================================================

import joblib

Z = np.load(CTX / "clean_data19.npz", allow_pickle=True)
data19 = {
    s: dict(
        X_dec=Z[f"X_dec_{s}"].astype(np.float64),
        X_saf=Z[f"X_saf_{s}"].astype(np.float64),
        y=Z[f"y_{s}"].astype(str),
        runs=Z[f"runs_{s}"].astype(str),
    )
    for s in SUBJECTS
}
BANK_DEC = np.load(CTX / "bank_decoder19_dev_uV.npy").astype(np.float64)
BANK_SAF = np.load(CTX / "bank_safety19_dev_uV.npy").astype(np.float64)
META = pd.read_csv(CTX / "bank_metadata_dev.csv")
models_oof = joblib.load(CTX / "models_oof.joblib")

donor_safety_rms = META["safety_rms_uv_recomputed"].to_numpy(dtype=np.float64)
artifact_subject = META["artifact_subject"].to_numpy(dtype=np.int64)
donor_index = META["donor_index"].to_numpy(dtype=np.int64)

assert not (set(artifact_subject) & FORBIDDEN), "locked artifact subject leaked"
D = BANK_DEC.shape[0]
S = len(SNR_GRID_DB)
print(f"\ndonors D={D}, SNR levels S={S}, subjects={len(SUBJECTS)}")

# ============================================================================
# 3. ЯДРО: точное разложение мощности CSP
# ============================================================================


def csp_terms(csp, clean, art):
    k = int(csp.n_components)
    W = np.asarray(csp.filters_[:k], dtype=np.float64)
    z = np.einsum("kc,nct->nkt", W, clean, optimize=True)
    u = np.einsum("kc,dct->dkt", W, art, optimize=True)
    T = z.shape[-1]
    p_clean = np.mean(z ** 2, axis=-1)                       # (N,K)
    p_art = np.mean(u ** 2, axis=-1)                         # (D,K)
    cross = np.einsum("nkt,dkt->ndk", z, u, optimize=True) / T   # (N,D,K)
    rho = cross / np.maximum(np.sqrt(p_clean[:, None, :] * p_art[None, :, :]), 1e-300)
    return p_clean, p_art, cross, rho


def score_right_from_power(lda, power):
    power = np.maximum(power, np.finfo(np.float64).tiny)
    feats = np.log(power)
    flat = feats.reshape(-1, feats.shape[-1])
    raw = np.asarray(lda.decision_function(flat)).reshape(-1)
    cl = np.asarray(lda.classes_)
    if cl[1] == "right_hand":
        sr = raw
    elif cl[0] == "right_hand":
        sr = -raw
    else:
        raise RuntimeError(f"unexpected classes {cl}")
    return sr.reshape(feats.shape[:-1])


def harm_counts(lda, p_clean, p_art, cross, lam, clean_correct, y_sign, clean_margin):
    """
    p_clean (N,K), p_art (D,K), cross (N,D,K), lam (D,S)
    возвращает словарь счётчиков и массив (N,D,S) флагов
    """
    pc = p_clean[:, None, None, :]
    pa = p_art[None, :, None, :]
    cr = cross[:, :, None, :]
    lm = lam[None, :, :, None]

    full_power = pc + 2.0 * lm * cr + (lm ** 2) * pa
    noint_power = pc + (lm ** 2) * pa

    full_sr = score_right_from_power(lda, full_power)
    noint_sr = score_right_from_power(lda, noint_power)

    full_m = y_sign[:, None, None] * full_sr
    noint_m = y_sign[:, None, None] * noint_sr

    elig = clean_correct[:, None, None]
    full_harm = elig & (full_m < 0.0)
    noint_harm = elig & (noint_m < 0.0)
    shared = full_harm & noint_harm
    required = full_harm & (~noint_harm)
    prevented = (~full_harm) & noint_harm & elig

    full_disp = full_m - clean_margin[:, None, None]
    noint_disp = noint_m - clean_margin[:, None, None]

    return dict(
        full_harm=full_harm, noint_harm=noint_harm, shared=shared,
        required=required, prevented=prevented,
        full_disp=full_disp, noint_disp=noint_disp,
    )


def phase_randomize_shared(A, rng):
    """
    A: (D, C, T). Один общий набор фаз на все каналы каждого донора.
    Точно сохраняет ковариационную матрицу каждого донора.
    """
    Aw = np.fft.rfft(A, axis=-1)
    n_freq = Aw.shape[-1]
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(A.shape[0], 1, n_freq))
    phi[..., 0] = 0.0
    if A.shape[-1] % 2 == 0:
        phi[..., -1] = 0.0          # частота Найквиста должна остаться вещественной
    out = np.fft.irfft(Aw * np.exp(1j * phi), n=A.shape[-1], axis=-1)
    return np.ascontiguousarray(out)


def cov_batch(A):
    Ac = A - A.mean(axis=-1, keepdims=True)
    T = A.shape[-1]
    return np.einsum("dct,dst->dcs", Ac, Ac, optimize=True) / (T - 1)


# --- проверка суррогата: ковариация обязана сохраниться точно ---
rng_check = np.random.default_rng(SEED)
_sub = BANK_DEC[:12]
_sur = phase_randomize_shared(_sub, rng_check)
_c0, _c1 = cov_batch(_sub), cov_batch(_sur)
_cov_err = float(np.max(np.abs(_c1 - _c0)) / np.max(np.abs(_c0)))
_rms_err = float(
    np.max(np.abs(np.sqrt(np.mean(_sur ** 2, axis=(1, 2)))
                  - np.sqrt(np.mean(_sub ** 2, axis=(1, 2)))))
)
print(f"\n[surrogate guard] max relative covariance error : {_cov_err:.3e}")
print(f"[surrogate guard] max absolute RMS error        : {_rms_err:.3e} µV")
if _cov_err > 1e-9:
    raise RuntimeError("phase-randomized surrogate does not preserve donor covariance")
print("[surrogate guard] PASS")


# ============================================================================
# 4. ОСНОВНОЙ ПРОГОН: OBSERVED + NULL
# ============================================================================

rng = np.random.default_rng(SEED)
SURROGATE_BANKS = [phase_randomize_shared(BANK_DEC, rng) for _ in range(N_NULL_REPEATS)]

subject_rows = []
snr_rows = []
rho_rows = []
artsubj_rows = []

for s in SUBJECTS:
    X_dec = data19[s]["X_dec"]
    X_saf = data19[s]["X_saf"]
    y = data19[s]["y"]
    runs = data19[s]["runs"]
    uruns = sorted(set(runs))

    acc = {arm: dict(full=0, noint=0, shared=0, required=0, prevented=0,
                     fd=[], nd=[])
           for arm in ["observed"] + [f"null{r}" for r in range(N_NULL_REPEATS)]}
    acc_snr = {arm: {j: dict(full=0, required=0) for j in range(S)} for arm in acc}
    acc_art = {arm: {} for arm in acc}
    rho_pool = {arm: [] for arm in acc}

    for r in uruns:
        val = runs == r
        fit = ~val
        model = models_oof[s][r]
        csp = model.named_steps["csp"]
        lda = model.named_steps["lineardiscriminantanalysis"]

        Xc = X_dec[val]
        yv = y[val]
        y_sign = np.where(yv == "right_hand", 1.0, -1.0)

        Zc = csp.transform(Xc)
        raw = np.asarray(lda.decision_function(Zc)).reshape(-1)
        cl = np.asarray(lda.classes_)
        sr_clean = raw if cl[1] == "right_hand" else -raw
        clean_margin = y_sign * sr_clean
        clean_correct = clean_margin > 0.0

        # опорный RMS чистой ЭЭГ берётся ТОЛЬКО с обучающих ранов фолда
        clean_ref_rms = float(np.median(np.sqrt(np.mean(X_saf[fit] ** 2, axis=(1, 2)))))
        lam = clean_ref_rms / (
            donor_safety_rms[:, None] * (10.0 ** (SNR_GRID_DB[None, :] / 20.0))
        )

        p_clean, p_art, cross_obs, rho_obs = csp_terms(csp, Xc, BANK_DEC)

        for arm_i, arm in enumerate(acc.keys()):
            if arm == "observed":
                cross, rho = cross_obs, rho_obs
                p_art_arm = p_art
            else:
                k = int(arm.replace("null", ""))
                p_clean_chk, p_art_arm, cross, rho = csp_terms(
                    csp, Xc, SURROGATE_BANKS[k]
                )
                # чистая часть обязана быть идентичной
                if not np.array_equal(p_clean_chk, p_clean):
                    raise RuntimeError(f"S{s} {arm}: p_clean changed between arms")
                # p_art обязан совпасть с наблюдаемым до машинной точности,
                # иначе суррогат изменил ковариацию донора и SNR-механику
                pe = float(np.max(np.abs(p_art_arm - p_art)) / max(np.max(p_art), 1e-300))
                if pe > 1e-9:
                    raise RuntimeError(f"S{s} {arm}: p_art drift {pe:.3e}")

            out = harm_counts(lda, p_clean, p_art_arm, cross, lam,
                              clean_correct, y_sign, clean_margin)

            acc[arm]["full"] += int(out["full_harm"].sum())
            acc[arm]["noint"] += int(out["noint_harm"].sum())
            acc[arm]["shared"] += int(out["shared"].sum())
            acc[arm]["required"] += int(out["required"].sum())
            acc[arm]["prevented"] += int(out["prevented"].sum())

            el = clean_correct
            acc[arm]["fd"].append(out["full_disp"][el].reshape(-1).astype(np.float32))
            acc[arm]["nd"].append(out["noint_disp"][el].reshape(-1).astype(np.float32))

            fh = out["full_harm"]
            rq = out["required"]
            for j in range(S):
                acc_snr[arm][j]["full"] += int(fh[:, :, j].sum())
                acc_snr[arm][j]["required"] += int(rq[:, :, j].sum())

            fh_d = fh.sum(axis=(0, 2))
            rq_d = rq.sum(axis=(0, 2))
            for a_sub in np.unique(artifact_subject):
                m = artifact_subject == a_sub
                d = acc_art[arm].setdefault(int(a_sub), dict(full=0, required=0))
                d["full"] += int(fh_d[m].sum())
                d["required"] += int(rq_d[m].sum())

            rho_pool[arm].append(rho.reshape(-1).astype(np.float32))

        del p_clean, p_art, cross_obs, rho_obs

    # --- сводка по субъекту ---
    for arm, a in acc.items():
        full = a["full"]
        frac = a["required"] / full if full else np.nan
        recall = a["shared"] / full if full else np.nan
        fd = np.concatenate(a["fd"]).astype(np.float64)
        nd = np.concatenate(a["nd"]).astype(np.float64)
        sst = float(np.sum((fd - fd.mean()) ** 2))
        r2 = float(1.0 - np.sum((fd - nd) ** 2) / sst) if sst > 0 else np.nan

        subject_rows.append(dict(
            bci_subject=s, arm=arm,
            full_harm=full, noint_harm=a["noint"], shared_harm=a["shared"],
            interaction_required_harm=a["required"],
            interaction_prevented_harm=a["prevented"],
            no_interaction_harm_recall=recall,
            interaction_required_harm_fraction=frac,
            no_interaction_displacement_r2=r2,
        ))

        rp = np.concatenate(rho_pool[arm]).astype(np.float64)
        rho_rows.append(dict(
            bci_subject=s, arm=arm,
            rho_mean=float(rp.mean()), rho_sd=float(rp.std(ddof=0)),
            abs_rho_median=float(np.median(np.abs(rp))),
            abs_rho_p95=float(np.quantile(np.abs(rp), 0.95)),
            abs_rho_max=float(np.abs(rp).max()),
            n=int(rp.size),
        ))

        for j in range(S):
            f_ = acc_snr[arm][j]["full"]
            r_ = acc_snr[arm][j]["required"]
            snr_rows.append(dict(
                bci_subject=s, arm=arm, snr_db=float(SNR_GRID_DB[j]),
                full_harm=f_, interaction_required_harm=r_,
                interaction_required_harm_fraction=(r_ / f_ if f_ else np.nan),
            ))

        for a_sub, d in sorted(acc_art[arm].items()):
            artsubj_rows.append(dict(
                bci_subject=s, arm=arm, artifact_subject=a_sub,
                full_harm=d["full"], interaction_required_harm=d["required"],
                interaction_required_harm_fraction=(
                    d["required"] / d["full"] if d["full"] else np.nan
                ),
            ))

    obs = [r for r in subject_rows if r["bci_subject"] == s and r["arm"] == "observed"][0]
    nulls = [r["interaction_required_harm_fraction"] for r in subject_rows
             if r["bci_subject"] == s and r["arm"].startswith("null")]
    print(f"  S{s}  observed IR={obs['interaction_required_harm_fraction']:.6f}   "
          f"null mean={np.mean(nulls):.6f} (sd {np.std(nulls):.6f})   "
          f"HARM={obs['full_harm']}")

S1_SUBJECT = pd.DataFrame(subject_rows)
S1_BY_SNR = pd.DataFrame(snr_rows)
S1_RHO = pd.DataFrame(rho_rows)
S1_BY_ARTSUBJ = pd.DataFrame(artsubj_rows)


# ============================================================================
# 5. ПОЗИТИВНЫЙ КОНТРОЛЬ
# ============================================================================

print("\n" + "=" * 78)
print("POSITIVE CONTROL  a' = a + beta * x")
print("=" * 78)

pc_rows = []
rng_pc = np.random.default_rng(SEED + 1)

for s in SUBJECTS:
    X_dec = data19[s]["X_dec"]
    X_saf = data19[s]["X_saf"]
    y = data19[s]["y"]
    runs = data19[s]["runs"]
    r = sorted(set(runs))[0]
    val, fit = runs == r, runs != r

    model = models_oof[s][r]
    csp = model.named_steps["csp"]
    lda = model.named_steps["lineardiscriminantanalysis"]

    Xc_all, Xs_all, yv = X_dec[val], X_saf[val], y[val]
    y_sign_all = np.where(yv == "right_hand", 1.0, -1.0)
    Zc = csp.transform(Xc_all)
    raw = np.asarray(lda.decision_function(Zc)).reshape(-1)
    cl = np.asarray(lda.classes_)
    sr_clean = raw if cl[1] == "right_hand" else -raw
    cm_all = y_sign_all * sr_clean
    ok = np.flatnonzero(cm_all > 0)
    if len(ok) == 0:
        continue
    sel_t = rng_pc.choice(ok, size=min(POSCTRL_N_TRIALS, len(ok)), replace=False)
    sel_d = rng_pc.choice(D, size=min(POSCTRL_N_DONORS, D), replace=False)

    clean_ref_rms = float(np.median(np.sqrt(np.mean(X_saf[fit] ** 2, axis=(1, 2)))))

    for beta in POSCTRL_BETAS:
        full_h = req_h = 0
        rhos = []
        for ti in sel_t:
            x_dec = Xc_all[ti]
            x_saf = Xs_all[ti]
            ysgn = y_sign_all[ti]

            a_dec = BANK_DEC[sel_d] + beta * x_dec[None, :, :]
            a_saf = BANK_SAF[sel_d] + beta * x_saf[None, :, :]
            a_rms = np.sqrt(np.mean(a_saf ** 2, axis=(1, 2)))
            lam = clean_ref_rms / (
                a_rms[:, None] * (10.0 ** (SNR_GRID_DB[None, :] / 20.0))
            )

            p_clean, p_art, cross, rho = csp_terms(
                csp, x_dec[None, :, :], a_dec
            )
            rhos.append(rho.reshape(-1))

            pc_ = p_clean[:, None, None, :]
            pa_ = p_art[None, :, None, :]
            cr_ = cross[:, :, None, :]
            lm_ = lam[None, :, :, None]

            fs = score_right_from_power(lda, pc_ + 2 * lm_ * cr_ + lm_ ** 2 * pa_)
            ns = score_right_from_power(lda, pc_ + lm_ ** 2 * pa_)
            fm, nm = ysgn * fs, ysgn * ns
            fh = fm < 0.0
            nh = nm < 0.0
            full_h += int(fh.sum())
            req_h += int((fh & (~nh)).sum())

        rr = np.concatenate(rhos)
        pc_rows.append(dict(
            bci_subject=s, beta=beta, full_harm=full_h,
            interaction_required_harm=req_h,
            interaction_required_harm_fraction=(req_h / full_h if full_h else np.nan),
            abs_rho_median=float(np.median(np.abs(rr))),
            n_trials=int(len(sel_t)), n_donors=int(len(sel_d)),
        ))

S1_POSCTRL = pd.DataFrame(pc_rows)
pc_summary = (
    S1_POSCTRL.groupby("beta")[
        ["interaction_required_harm_fraction", "abs_rho_median"]
    ]
    .median()
    .reset_index()
)
print(pc_summary.to_string(index=False))

monotone = bool(
    np.all(np.diff(pc_summary["interaction_required_harm_fraction"].to_numpy()) >= -1e-9)
)
print(f"\npositive control monotone in beta: {monotone}")
if not monotone:
    print("WARN: измеритель не реагирует монотонно. S1 НЕ интерпретируется научно.")


# ============================================================================
# 6. СРАВНЕНИЕ OBSERVED vs NULL НА УРОВНЕ СУБЪЕКТОВ
# ============================================================================

def subject_bootstrap_median(values, n_boot=20000, seed=SEED):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    rng_b = np.random.default_rng(seed)
    idx = rng_b.integers(0, len(v), size=(n_boot, len(v)))
    meds = np.median(v[idx], axis=1)
    return float(np.median(v)), float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


obs_vals = (
    S1_SUBJECT[S1_SUBJECT.arm == "observed"]
    .sort_values("bci_subject")["interaction_required_harm_fraction"].to_numpy()
)
null_by_subject = (
    S1_SUBJECT[S1_SUBJECT.arm.str.startswith("null")]
    .groupby("bci_subject")["interaction_required_harm_fraction"].mean()
    .sort_index().to_numpy()
)
paired_delta = obs_vals - null_by_subject

o_med, o_lo, o_hi = subject_bootstrap_median(obs_vals)
n_med, n_lo, n_hi = subject_bootstrap_median(null_by_subject, seed=SEED + 7)
d_med, d_lo, d_hi = subject_bootstrap_median(paired_delta, seed=SEED + 13)

overlap = not (o_hi < n_lo or n_hi < o_lo)
delta_covers_zero = (d_lo <= 0.0 <= d_hi)

print("\n" + "=" * 78)
print("PRIMARY COMPARISON")
print("=" * 78)
print(f"observed  median IR = {o_med:.6f}   95% CI [{o_lo:.6f}, {o_hi:.6f}]")
print(f"null      median IR = {n_med:.6f}   95% CI [{n_lo:.6f}, {n_hi:.6f}]")
print(f"paired delta median = {d_med:+.6f}  95% CI [{d_lo:+.6f}, {d_hi:+.6f}]")
print(f"intervals overlap        : {overlap}")
print(f"paired delta covers zero : {delta_covers_zero}")

if delta_covers_zero:
    verdict = "NULL_MATCHES_OBSERVED"
    reading = (
        "Перекрёстный член не несёт информации, специфичной для пары трайл-донор. "
        "Наблюдаемая доля interaction-required объясняется амплитудой независимой "
        "флуктуации. Формулируйте вывод как количественное утверждение о величине, "
        "а не как открытие структуры."
    )
elif d_lo > 0.0:
    verdict = "OBSERVED_ABOVE_NULL"
    reading = "Существует структура выравнивания сверх независимой флуктуации."
else:
    verdict = "OBSERVED_BELOW_NULL"
    reading = "Систематическое анти-выравнивание относительно независимой ветки."

print(f"\nVERDICT: {verdict}")
print(reading)

# теоретическая шкала rho при независимости, для контекста
BW = 32.0 - 8.0
DUR = (N_TIMES - 1) / 250.0
T_EFF = 2.0 * BW * DUR
theo_sd = 1.0 / np.sqrt(T_EFF)
emp_sd = float(S1_RHO[S1_RHO.arm.str.startswith("null")]["rho_sd"].median())
print(f"\nrho sd: theoretical (T_eff={T_EFF:.0f}) = {theo_sd:.4f} | empirical null = {emp_sd:.4f}")


# ============================================================================
# 7. СОХРАНЕНИЕ
# ============================================================================

S1_SUBJECT.to_csv(OUT / "S1_SUBJECT_SUMMARY.csv", index=False)
S1_BY_SNR.to_csv(OUT / "S1_BY_SNR.csv", index=False)
S1_RHO.to_csv(OUT / "S1_RHO_SUMMARY.csv", index=False)
S1_BY_ARTSUBJ.to_csv(OUT / "S1_BY_ARTIFACT_SUBJECT.csv", index=False)
S1_POSCTRL.to_csv(OUT / "S1_POSITIVE_CONTROL.csv", index=False)

result = dict(
    stage="S1",
    verdict=verdict,
    reading=reading,
    observed=dict(median=o_med, ci95=[o_lo, o_hi], per_subject=obs_vals.tolist()),
    null=dict(median=n_med, ci95=[n_lo, n_hi], per_subject=null_by_subject.tolist()),
    paired_delta=dict(median=d_med, ci95=[d_lo, d_hi]),
    intervals_overlap=bool(overlap),
    paired_delta_covers_zero=bool(delta_covers_zero),
    positive_control_monotone=monotone,
    positive_control=pc_summary.to_dict(orient="records"),
    rho_sd_theoretical=float(theo_sd),
    rho_sd_empirical_null=emp_sd,
    n_null_repeats=N_NULL_REPEATS,
)
with open(OUT / "S1_RESULT.json", "w") as f:
    json.dump(result, f, indent=2)

print("\n" + "=" * 78)
print("STAGE S1: COMPLETE")
print("=" * 78)
for p in sorted(OUT.iterdir()):
    print(f"  {p.name}")
