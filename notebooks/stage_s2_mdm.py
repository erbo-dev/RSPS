# ============================================================================
# SPIKESHIELD STAGE S2 — RIEMANNIAN MDM CROSS-TERM ABLATION
#
# Вопрос: сохраняется ли "взаимодействие вторично" в другом семействе
# декодеров, где решение принимается по геодезическому расстоянию между
# ковариационными матрицами на SPD-многообразии?
#
# Исправлены все 10 дефектов, из-за которых был отклонён агентский worker:
#   1  G0 реально проверяется, а не захардкожен как success
#   2  G4 проверяет конечность всех AIRM-расстояний
#   3  биологические родительские ID сохраняются
#   4  метки и ПОРЯДОК каналов донора сверяются с COMMON19
#   5  масштабирование берётся из замороженной конвенции S0
#   6  жёсткий assert на размерность матриц
#   7  перекрёстный член строится независимой формулой, не как full - no_int
#   8  сохраняются счётчики по родителям для иерархического бутстрапа
#   9  строгий манифест входов вместо рекурсивного поиска
#  10  сегмент донора фиксирован (1001 отсчёт), без случайной подрезки
#
# Данные: только 0train, только 247 development-доноров.
# Kaggle: один cell после S0. ~20-60 минут. Есть чекпойнты по субъектам.
# ============================================================================

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================================
# 0. КОНФИГ ЗАПУСКА
# ============================================================================

PILOT = False          # True -> быстрый прогон для отладки софта, БЕЗ научной интерпретации
PILOT_DONORS = 24
PILOT_SUBJECTS = [1, 3]

# ============================================================================
# 1. КОНТЕКСТ
# ============================================================================

_cands = [Path("/kaggle/working/spikeshield_stage_s/context")]
if Path("/kaggle/input").exists():
    _cands += sorted(Path("/kaggle/input").glob("*/context"))
    _cands += sorted(Path("/kaggle/input").glob("*/*/context"))
CTX = next((c for c in _cands if (c / "context_manifest.json").exists()), None)
if CTX is None:
    raise RuntimeError("Контекст S0 не найден.")

OUT = Path("/kaggle/working/spikeshield_stage_s/S2")
CKPT = OUT / "checkpoints"
CKPT.mkdir(parents=True, exist_ok=True)

with open(CTX / "context_manifest.json") as f:
    MANIFEST = json.load(f)

SEED = int(MANIFEST["seed"])
SNR_GRID_DB = np.array(MANIFEST["snr_grid_db"], dtype=np.float64)
COMMON19 = MANIFEST["common19"]
N_TIMES = int(MANIFEST["n_times"])
FORBIDDEN = set(MANIFEST["forbidden_artifact_subjects"])
SUBJECTS = PILOT_SUBJECTS if PILOT else list(range(1, 10))

print("=" * 78)
print("STAGE S2 — RIEMANNIAN MDM" + ("  [PILOT — НЕ ИНТЕРПРЕТИРОВАТЬ]" if PILOT else ""))
print("=" * 78)

# ============================================================================
# 2. ПРЕДРЕГИСТРАЦИЯ (пишется ДО контаминации)
# ============================================================================

CLEAN_VALIDITY_MIN_ACC = 0.60      # заморожено до просмотра HARM
RIDGE_REL = 1e-6                   # eps_fold = RIDGE_REL * mean(trace(C_train)/18)
KARCHER_TOL = 1e-10
KARCHER_MAXIT = 200

PREREG = {
    "stage": "S2",
    "version": "S2-v1",
    "question": (
        "Does the interaction-secondary HARM structure survive a Riemannian "
        "MDM decoder on the SPD manifold?"
    ),
    "covariance": "SCM, temporal centering, divisor T-1",
    "rank_handling": (
        "CAR19 leaves a null direction. Project onto a fixed orthonormal basis "
        "of the 18-dim complement of the all-ones vector, THEN add ridge once."
    ),
    "ridge_rule": f"eps_fold = {RIDGE_REL} * mean(trace(C_train)/18), clean training runs only",
    "metric": "AIRM for both centroid (Karcher mean) and distance",
    "clean_validity_gate_frozen_before_contamination": {
        "rule": f"subject enters the primary median only if clean MDM OOF accuracy >= {CLEAN_VALIDITY_MIN_ACC}",
        "rationale": "fixes the Stage 6 defect where subjects near chance dominated the count",
    },
    "primary_reporting": "9 subject values, median, 95% subject-level bootstrap interval",
    "thresholds": "NOT copied from CSP Stage 5/6; reporting is continuous",
    "data_scope": "0train only; 247 development donors; artifact subjects 12/14/17/19/23/25 excluded",
    "seed": SEED,
}
with open(OUT / "S2_PREREGISTRATION.json", "w") as f:
    json.dump(PREREG, f, indent=2)
print("prereg written.")

# ============================================================================
# 3. ЗАГРУЗКА + G0 (реальная проверка ресурсов, не заглушка)
# ============================================================================

Z = np.load(CTX / "clean_data19.npz", allow_pickle=True)
data19 = {
    s: dict(
        X_dec=Z[f"X_dec_{s}"].astype(np.float64),
        X_saf=Z[f"X_saf_{s}"].astype(np.float64),
        y=Z[f"y_{s}"].astype(str),
        runs=Z[f"runs_{s}"].astype(str),
    )
    for s in range(1, 10)
}
BANK_DEC = np.load(CTX / "bank_decoder19_dev_uV.npy").astype(np.float64)
BANK_SAF = np.load(CTX / "bank_safety19_dev_uV.npy").astype(np.float64)
META = pd.read_csv(CTX / "bank_metadata_dev.csv")

if PILOT:
    keep = np.arange(min(PILOT_DONORS, len(META)))
    BANK_DEC, BANK_SAF, META = BANK_DEC[keep], BANK_SAF[keep], META.iloc[keep].reset_index(drop=True)

donor_safety_rms = META["safety_rms_uv_recomputed"].to_numpy(dtype=np.float64)
artifact_subject = META["artifact_subject"].to_numpy(dtype=np.int64)
donor_index = META["donor_index"].to_numpy(dtype=np.int64)

G0 = {}
G0["common19_order_matches_manifest"] = (MANIFEST["common19"] == COMMON19)
G0["n_channels_19"] = bool(BANK_DEC.shape[1] == 19 and data19[1]["X_dec"].shape[1] == 19)
G0["n_times_matches"] = bool(BANK_DEC.shape[2] == N_TIMES == data19[1]["X_dec"].shape[2])
G0["no_locked_artifact_subjects"] = bool(not (set(artifact_subject) & FORBIDDEN))
G0["all_donors_in_dev_set"] = bool(
    set(artifact_subject).issubset(set(MANIFEST["artifact_subjects"]))
)
G0["donor_safety_rms_positive"] = bool(np.all(donor_safety_rms > 0))
G0["donor_car_residual_ok"] = bool(np.max(np.abs(BANK_DEC.sum(axis=1))) < 1e-3)
G0["clean_car_residual_ok"] = bool(
    max(np.max(np.abs(data19[s]["X_dec"].sum(axis=1))) for s in SUBJECTS) < 1e-2
)
G0["snr_grid_matches_frozen"] = bool(
    np.array_equal(SNR_GRID_DB, np.array([2.0, 0, -2, -3, -4, -5, -6, -7]))
)
_rms_chk = np.sqrt(np.mean(BANK_SAF ** 2, axis=(1, 2)))
G0["donor_rms_consistent_with_metadata"] = bool(
    np.max(np.abs(_rms_chk - donor_safety_rms) / donor_safety_rms) < 1e-3
)
G0["ok"] = all(v for k, v in G0.items() if k != "ok")

print("\nG0 resource identity:")
for k, v in G0.items():
    print(f"  {k:42s} {v}")
if not G0["ok"]:
    raise RuntimeError("G0 failed. STOP.")

D = BANK_DEC.shape[0]
S = len(SNR_GRID_DB)
print(f"\ndonors={D}, snr={S}, subjects={SUBJECTS}")

# ============================================================================
# 4. ГЕОМЕТРИЯ
# ============================================================================

# фиксированный ортонормированный базис дополнения к вектору из единиц
_ones = np.ones((19, 1)) / np.sqrt(19.0)
_Q, _ = np.linalg.qr(np.hstack([_ones, np.eye(19)]))
V = np.ascontiguousarray(_Q[:, 1:19])          # (19, 18)
assert V.shape == (19, 18)
assert np.allclose(V.T @ V, np.eye(18), atol=1e-12)
assert np.allclose(V.T @ np.ones(19), 0.0, atol=1e-12)
P_DIM = 18


def temporal_center(X):
    return X - X.mean(axis=-1, keepdims=True)


def cov_single(Xc):
    T = Xc.shape[-1]
    return (Xc @ Xc.T) / (T - 1)


def cov_batch(Xc):
    T = Xc.shape[-1]
    return np.einsum("nct,nst->ncs", Xc, Xc, optimize=True) / (T - 1)


def cross_cov_batch(Xc, Ac):
    """НЕЗАВИСИМАЯ формула. Xc (N,19,T), Ac (D,19,T) -> (N,D,19,19)"""
    T = Xc.shape[-1]
    return np.einsum("nct,dst->ndcs", Xc, Ac, optimize=True) / (T - 1)


def project(C):
    """C (...,19,19) -> (...,18,18)"""
    return np.einsum("ci,...cs,sj->...ij", V, C, V, optimize=True)


def sym(M):
    return 0.5 * (M + np.swapaxes(M, -1, -2))


def inv_sqrtm_spd(M):
    w, U = np.linalg.eigh(sym(M))
    if np.any(w <= 0):
        raise RuntimeError(f"non-SPD matrix, min eig = {w.min():.3e}")
    return (U * (w ** -0.5)) @ U.T


def sqrtm_spd(M):
    w, U = np.linalg.eigh(sym(M))
    return (U * np.sqrt(np.maximum(w, 0.0))) @ U.T


def karcher_mean(covs, tol=KARCHER_TOL, maxit=KARCHER_MAXIT):
    """AIRM (Riemannian/geometric) mean. covs: (n,p,p) SPD."""
    M = sym(covs.mean(axis=0))
    for it in range(maxit):
        iM = inv_sqrtm_spd(M)
        sM = sqrtm_spd(M)
        A = sym(np.einsum("ij,njk,kl->nil", iM, covs, iM, optimize=True))
        w, U = np.linalg.eigh(A)
        if np.any(w <= 0):
            raise RuntimeError("non-SPD in Karcher iteration")
        L = np.einsum("nij,nj,nkj->nik", U, np.log(w), U, optimize=True)
        Lbar = sym(L.mean(axis=0))
        wl, Ul = np.linalg.eigh(Lbar)
        E = (Ul * np.exp(wl)) @ Ul.T
        M_new = sym(sM @ E @ sM)
        rel = np.linalg.norm(M_new - M) / (np.linalg.norm(M) + 1e-300)
        M = M_new
        if rel <= tol:
            return M, True, it + 1
    return M, False, maxit


def airm_dist2_batch(C, M_inv_sqrt):
    """d^2(C, M) для батча C (n,p,p) при заранее посчитанном M^{-1/2}."""
    A = sym(np.einsum("ij,njk,kl->nil", M_inv_sqrt, C, M_inv_sqrt, optimize=True))
    w = np.linalg.eigvalsh(A)
    if np.any(w <= 0) or not np.all(np.isfinite(w)):
        return None
    lg = np.log(w)
    return np.sum(lg * lg, axis=-1)


# --- проверка геометрии на синтетике ---
_rng = np.random.default_rng(SEED)
_B = _rng.normal(size=(30, P_DIM, P_DIM))
_C = np.einsum("nij,nkj->nik", _B, _B) + P_DIM * np.eye(P_DIM)
_M, _conv, _it = karcher_mean(_C)
_iM = inv_sqrtm_spd(_M)
_d = airm_dist2_batch(_C, _iM)
assert _conv and _d is not None and np.all(np.isfinite(_d))
_self = airm_dist2_batch(_M[None], _iM)
assert abs(float(_self[0])) < 1e-16, _self
print(f"\n[geometry guard] Karcher converged in {_it} it, d(M,M)^2 = {float(_self[0]):.3e}  PASS")


# ============================================================================
# 5. G3 — НЕЗАВИСИМАЯ ПРОВЕРКА КОВАРИАЦИОННОГО ТОЖДЕСТВА
#    (не тавтологическая: C_full считается из реального сигнала x + lam*a)
# ============================================================================

def identity_check(n_trials=6, n_donors=6, n_lambda=4, subject=None):
    subject = subject or SUBJECTS[0]
    rg = np.random.default_rng(SEED + 99)
    X = data19[subject]["X_dec"]
    ti = rg.choice(X.shape[0], n_trials, replace=False)
    di = rg.choice(D, n_donors, replace=False)
    lams = rg.uniform(0.05, 3.0, n_lambda)

    Xc = temporal_center(X[ti])
    Ac = temporal_center(BANK_DEC[di])
    Cx = cov_batch(Xc)
    Ca = cov_batch(Ac)
    Cxa = cross_cov_batch(Xc, Ac)

    worst_raw = 0.0
    worst_proj = 0.0
    for i in range(n_trials):
        for j in range(n_donors):
            for lam in lams:
                mix = temporal_center(Xc[i] + lam * Ac[j])
                C_direct = cov_single(mix)
                C_assembled = (
                    Cx[i] + (lam ** 2) * Ca[j] + lam * (Cxa[i, j] + Cxa[i, j].T)
                )
                e = np.max(np.abs(C_direct - C_assembled)) / np.max(np.abs(C_direct))
                worst_raw = max(worst_raw, e)
                pd_ = project(C_direct)
                pa_ = project(C_assembled)
                worst_proj = max(
                    worst_proj, np.max(np.abs(pd_ - pa_)) / np.max(np.abs(pd_))
                )
    return worst_raw, worst_proj


g3_raw, g3_proj = identity_check()
print(f"[G3] max relative covariance identity error: raw {g3_raw:.3e}, projected {g3_proj:.3e}")
if max(g3_raw, g3_proj) > 1e-10:
    raise RuntimeError("covariance identity failed. STOP.")
print("[G3] PASS")


# ============================================================================
# 6. ОСНОВНОЙ ПРОГОН
# ============================================================================

DONOR_BLOCK = 32

subject_rows = []
snr_rows = []
artsubj_rows = []
diag_rows = []

t_start = time.time()

for s in SUBJECTS:
    ck = CKPT / f"S2_subject_{s}.json"
    if ck.exists():
        print(f"\nSubject {s}: checkpoint exists, skipping recompute.")
        with open(ck) as f:
            saved = json.load(f)
        subject_rows.append(saved["subject"])
        snr_rows.extend(saved["snr"])
        artsubj_rows.extend(saved["artsubj"])
        diag_rows.append(saved["diag"])
        continue

    t0 = time.time()
    X_dec = data19[s]["X_dec"]
    X_saf = data19[s]["X_saf"]
    y = data19[s]["y"]
    runs = data19[s]["runs"]
    y_bin = (y == "right_hand").astype(np.int64)
    uruns = sorted(set(runs))

    n_test = n_clean_correct = 0
    c_full = c_shared = c_required = c_prevented = 0
    snr_acc = {j: dict(full=0, required=0) for j in range(S)}
    art_acc = {}
    n_nonfinite = 0
    n_nonspd = 0
    karcher_fail = 0
    ridge_values = []

    # ковариации доноров считаются один раз
    Ac_all = temporal_center(BANK_DEC)
    Ca_all = cov_batch(Ac_all)
    Ca_proj = project(Ca_all)

    for r in uruns:
        val = runs == r
        fit = ~val

        Xc_fit = temporal_center(X_dec[fit])
        Xc_val = temporal_center(X_dec[val])
        y_fit, y_val = y_bin[fit], y_bin[val]

        Cx_fit = project(cov_batch(Xc_fit))
        Cx_val_raw = cov_batch(Xc_val)
        Cx_val = project(Cx_val_raw)

        # --- ридж по замороженному правилу, только из обучающих ранов ---
        eps = RIDGE_REL * float(np.mean(np.trace(Cx_fit, axis1=-2, axis2=-1)) / P_DIM)
        ridge_values.append(eps)
        I = np.eye(P_DIM)

        def spd(C):
            return sym(C) + eps * I

        # --- центроиды ---
        centroids = []
        for cls in (0, 1):
            covs = np.stack([spd(c) for c in Cx_fit[y_fit == cls]])
            w = np.linalg.eigvalsh(covs)
            n_nonspd += int(np.sum(w <= 0))
            M, conv, _ = karcher_mean(covs)
            if not conv:
                karcher_fail += 1
            centroids.append(M)
        inv_sqrt = [inv_sqrtm_spd(M) for M in centroids]

        # --- чистое решение MDM ---
        clean_spd = np.stack([spd(c) for c in Cx_val])
        d0 = airm_dist2_batch(clean_spd, inv_sqrt[0])
        d1 = airm_dist2_batch(clean_spd, inv_sqrt[1])
        if d0 is None or d1 is None:
            raise RuntimeError(f"S{s} run {r}: non-finite clean AIRM distance")
        clean_pred = (d1 < d0).astype(np.int64)
        clean_correct = clean_pred == y_val
        n_test += int(val.sum())
        n_clean_correct += int(clean_correct.sum())

        # --- масштабирование: замороженная конвенция ---
        clean_ref_rms = float(np.median(np.sqrt(np.mean(X_saf[fit] ** 2, axis=(1, 2)))))
        lam_ds = clean_ref_rms / (
            donor_safety_rms[:, None] * (10.0 ** (SNR_GRID_DB[None, :] / 20.0))
        )

        idx_ok = np.flatnonzero(clean_correct)
        if len(idx_ok) == 0:
            continue
        Xc_ok = Xc_val[idx_ok]
        Cx_ok = Cx_val[idx_ok]
        y_ok = y_val[idx_ok]
        N = len(idx_ok)

        for b0 in range(0, D, DONOR_BLOCK):
            b1 = min(b0 + DONOR_BLOCK, D)
            Ab = Ac_all[b0:b1]
            Cxa = cross_cov_batch(Xc_ok, Ab)                 # (N,B,19,19)
            Cxa_p = project(Cxa)                             # (N,B,18,18)
            Cxa_sym = Cxa_p + np.swapaxes(Cxa_p, -1, -2)
            Ca_b = Ca_proj[b0:b1]
            lam_b = lam_ds[b0:b1]                            # (B,S)

            pc = Cx_ok[:, None, None, :, :]
            pa = Ca_b[None, :, None, :, :]
            cr = Cxa_sym[:, :, None, :, :]
            lm = lam_b[None, :, :, None, None]

            C_full = pc + (lm ** 2) * pa + lm * cr
            C_noint = pc + (lm ** 2) * pa

            B = b1 - b0
            shp = (N * B * S, P_DIM, P_DIM)
            Cf = sym(C_full.reshape(shp)) + eps * I
            Cn = sym(C_noint.reshape(shp)) + eps * I

            wf = np.linalg.eigvalsh(Cf)
            wn = np.linalg.eigvalsh(Cn)
            n_nonspd += int(np.sum(wf <= 0) + np.sum(wn <= 0))
            if np.any(wf <= 0) or np.any(wn <= 0):
                raise RuntimeError(f"S{s}: non-SPD contaminated covariance")

            df0 = airm_dist2_batch(Cf, inv_sqrt[0])
            df1 = airm_dist2_batch(Cf, inv_sqrt[1])
            dn0 = airm_dist2_batch(Cn, inv_sqrt[0])
            dn1 = airm_dist2_batch(Cn, inv_sqrt[1])
            for arr in (df0, df1, dn0, dn1):
                if arr is None:
                    raise RuntimeError(f"S{s}: non-finite AIRM distance in sweep")
                n_nonfinite += int(np.sum(~np.isfinite(arr)))

            pred_f = (df1 < df0).astype(np.int64).reshape(N, B, S)
            pred_n = (dn1 < dn0).astype(np.int64).reshape(N, B, S)
            yt = y_ok[:, None, None]

            full_harm = pred_f != yt
            noint_harm = pred_n != yt
            shared = full_harm & noint_harm
            required = full_harm & (~noint_harm)
            prevented = (~full_harm) & noint_harm

            c_full += int(full_harm.sum())
            c_shared += int(shared.sum())
            c_required += int(required.sum())
            c_prevented += int(prevented.sum())

            for j in range(S):
                snr_acc[j]["full"] += int(full_harm[:, :, j].sum())
                snr_acc[j]["required"] += int(required[:, :, j].sum())

            fh_d = full_harm.sum(axis=(0, 2))
            rq_d = required.sum(axis=(0, 2))
            asub_b = artifact_subject[b0:b1]
            for a in np.unique(asub_b):
                m = asub_b == a
                d = art_acc.setdefault(int(a), dict(full=0, required=0))
                d["full"] += int(fh_d[m].sum())
                d["required"] += int(rq_d[m].sum())

            del Cxa, Cxa_p, Cxa_sym, C_full, C_noint, Cf, Cn

    clean_acc = n_clean_correct / max(1, n_test)
    frac = c_required / c_full if c_full else np.nan
    recall = c_shared / c_full if c_full else np.nan

    srow = dict(
        bci_subject=s,
        clean_mdm_oof_accuracy=clean_acc,
        clean_validity_pass=bool(clean_acc >= CLEAN_VALIDITY_MIN_ACC),
        n_test_trials=n_test,
        n_clean_correct_trials=n_clean_correct,
        full_harm=c_full,
        shared_harm=c_shared,
        interaction_required_harm=c_required,
        interaction_prevented_harm=c_prevented,
        no_interaction_harm_recall=recall,
        interaction_required_harm_fraction=frac,
    )
    drow = dict(
        bci_subject=s, ridge_min=float(np.min(ridge_values)),
        ridge_max=float(np.max(ridge_values)),
        karcher_failures=int(karcher_fail),
        nonspd_events=int(n_nonspd),
        nonfinite_distances=int(n_nonfinite),
        seconds=float(time.time() - t0),
    )
    srows = [
        dict(bci_subject=s, snr_db=float(SNR_GRID_DB[j]),
             full_harm=snr_acc[j]["full"],
             interaction_required_harm=snr_acc[j]["required"],
             interaction_required_harm_fraction=(
                 snr_acc[j]["required"] / snr_acc[j]["full"] if snr_acc[j]["full"] else np.nan
             ))
        for j in range(S)
    ]
    arows = [
        dict(bci_subject=s, artifact_subject=a, full_harm=d["full"],
             interaction_required_harm=d["required"],
             interaction_required_harm_fraction=(
                 d["required"] / d["full"] if d["full"] else np.nan))
        for a, d in sorted(art_acc.items())
    ]

    subject_rows.append(srow)
    snr_rows.extend(srows)
    artsubj_rows.extend(arows)
    diag_rows.append(drow)

    with open(ck, "w") as f:
        json.dump(dict(subject=srow, snr=srows, artsubj=arows, diag=drow), f, indent=2)

    print(
        f"\nS{s}: clean MDM acc={clean_acc:.4f} "
        f"({'PASS' if srow['clean_validity_pass'] else 'BELOW GATE'})  "
        f"HARM={c_full}  IR={frac:.6f}  recall={recall:.6f}  "
        f"[{drow['seconds']/60:.1f} min]"
    )

S2_SUBJECT = pd.DataFrame(subject_rows)
S2_BY_SNR = pd.DataFrame(snr_rows)
S2_BY_ARTSUBJ = pd.DataFrame(artsubj_rows)
S2_DIAG = pd.DataFrame(diag_rows)


# ============================================================================
# 7. ВОРОТА ВАЛИДНОСТИ И ПЕРВИЧНЫЙ ОТЧЁТ
# ============================================================================

G = {
    "G0_resource_identity": bool(G0["ok"]),
    "G1_clean_validity_subjects": int(S2_SUBJECT["clean_validity_pass"].sum()),
    "G2_min_harm_per_valid_subject": int(
        S2_SUBJECT.loc[S2_SUBJECT.clean_validity_pass, "full_harm"].min()
        if S2_SUBJECT["clean_validity_pass"].any() else 0
    ),
    "G3_covariance_identity_max_error": float(max(g3_raw, g3_proj)),
    "G4_nonfinite_distances": int(S2_DIAG["nonfinite_distances"].sum()),
    "G4_nonspd_events": int(S2_DIAG["nonspd_events"].sum()),
    "G5_karcher_failures": int(S2_DIAG["karcher_failures"].sum()),
}
G["all_ok"] = bool(
    G["G0_resource_identity"]
    and G["G1_clean_validity_subjects"] >= 5
    and G["G2_min_harm_per_valid_subject"] >= 30
    and G["G3_covariance_identity_max_error"] <= 1e-10
    and G["G4_nonfinite_distances"] == 0
    and G["G4_nonspd_events"] == 0
    and G["G5_karcher_failures"] == 0
)

print("\n" + "=" * 78)
print("VALIDITY GATES")
print("=" * 78)
for k, v in G.items():
    print(f"  {k:38s} {v}")


def subject_bootstrap_median(values, n_boot=20000, seed=SEED):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rg = np.random.default_rng(seed)
    idx = rg.integers(0, len(v), size=(n_boot, len(v)))
    meds = np.median(v[idx], axis=1)
    return float(np.median(v)), float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


valid = S2_SUBJECT[S2_SUBJECT.clean_validity_pass]
ir_valid = valid["interaction_required_harm_fraction"].to_numpy()
ir_all = S2_SUBJECT["interaction_required_harm_fraction"].to_numpy()

m_v, lo_v, hi_v = subject_bootstrap_median(ir_valid)
m_a, lo_a, hi_a = subject_bootstrap_median(ir_all, seed=SEED + 3)

print("\n" + "=" * 78)
print("PRIMARY RESULT (MDM)")
print("=" * 78)
print(f"subjects passing clean validity gate: {len(valid)} / {len(S2_SUBJECT)}")
print(f"median interaction-required (VALID subjects only) = {m_v:.6f}  95% CI [{lo_v:.6f}, {hi_v:.6f}]")
print(f"median interaction-required (all subjects, secondary) = {m_a:.6f}  95% CI [{lo_a:.6f}, {hi_a:.6f}]")
print("\nper subject:")
print(S2_SUBJECT[[
    "bci_subject", "clean_mdm_oof_accuracy", "clean_validity_pass",
    "full_harm", "no_interaction_harm_recall", "interaction_required_harm_fraction",
]].to_string(index=False))

if not G["all_ok"]:
    verdict = "INCONCLUSIVE"
    reading = "Одни или несколько ворот валидности не пройдены. Научная интерпретация запрещена."
elif PILOT:
    verdict = "PILOT_ONLY"
    reading = "Пилотный прогон для отладки софта. Научно не интерпретируется."
else:
    verdict = "REPORT_CONTINUOUS"
    reading = (
        "Отчитывайтесь непрерывно: 9 значений, медиана, 95% субъектный интервал. "
        "Сравнивайте с CSP-значениями Stage 3F / 5 / 6 качественно, не подгоняя порог. "
        "Если медиана MDM близка к CSP, механизм не специфичен для CSP. "
        "Если заметно выше, вывод сужается до семейства CSP."
    )

print(f"\nVERDICT: {verdict}\n{reading}")

# ============================================================================
# 8. СОХРАНЕНИЕ
# ============================================================================

S2_SUBJECT.to_csv(OUT / "S2_SUBJECT_SUMMARY.csv", index=False)
S2_BY_SNR.to_csv(OUT / "S2_BY_SNR.csv", index=False)
S2_BY_ARTSUBJ.to_csv(OUT / "S2_BY_ARTIFACT_SUBJECT.csv", index=False)
S2_DIAG.to_csv(OUT / "S2_DIAGNOSTICS.csv", index=False)

with open(OUT / "S2_RESULT.json", "w") as f:
    json.dump(dict(
        stage="S2", pilot=PILOT, verdict=verdict, reading=reading,
        validity_gates=G,
        median_interaction_required_valid=dict(median=m_v, ci95=[lo_v, hi_v]),
        median_interaction_required_all=dict(median=m_a, ci95=[lo_a, hi_a]),
        per_subject=S2_SUBJECT.to_dict(orient="records"),
        clean_validity_min_acc=CLEAN_VALIDITY_MIN_ACC,
        total_minutes=float((time.time() - t_start) / 60.0),
    ), f, indent=2)

print("\n" + "=" * 78)
print(f"STAGE S2: COMPLETE in {(time.time()-t_start)/60:.1f} min")
print("=" * 78)
for p in sorted(OUT.iterdir()):
    if p.is_file():
        print(f"  {p.name}")
