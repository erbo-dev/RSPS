# ============================================================================
# SPIKESHIELD STAGE S0 — CONTEXT BUILDER
#
# Пересобирает чистый декодер COMMON19/CAR19/8-32 CSP+LDA на 0train,
# сверяет его с замороженным ледгером (раздел 23.2 handoff),
# загружает development-банк CHEW-доноров и сохраняет всё на диск.
#
# НЕ использует: 1test, локированных артефактных субъектов, Stage 5/6 результаты.
#
# Kaggle: один cell. ~20-30 минут при первом запуске (скачивание BNCI2014_001).
# ============================================================================

import importlib.util
import subprocess
import sys

for pkg, spec in [("moabb", "moabb>=1.5,<1.7")]:
    if importlib.util.find_spec(pkg) is None:
        print(f"Installing {spec} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", spec])
        print(f"{pkg} installed. RESTART the Kaggle session once, then rerun this cell.")
        raise SystemExit(0)

import hashlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import mne
import moabb
from mne.decoding import CSP
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

mne.set_log_level("ERROR")
moabb.set_log_level("error")

# ============================================================================
# 1. FROZEN CONSTANTS
# ============================================================================

SEED = 20260813
np.random.seed(SEED)

SUBJECTS = list(range(1, 10))
CLASSES = ("left_hand", "right_hand")

FMIN, FMAX = 8.0, 32.0            # decoder band
SAFETY_FMIN, SAFETY_FMAX = 0.5, 100.0
TMIN, TMAX = 0.0, 4.0
SFREQ = 250.0
N_TIMES = 1001

COMMON19 = [
    "Fz",
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CP2", "CP4",
    "Pz", "POz",
]

SNR_GRID_DB = np.array([2.0, 0.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0], dtype=np.float64)

DEV_ARTIFACT_SUBJECTS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
    15, 16, 18, 20, 21, 22, 24, 26, 27, 28, 29, 30,
]
FORBIDDEN_ARTIFACT_SUBJECTS = [12, 14, 17, 19, 23, 25]

CSP_KW = dict(
    n_components=8,
    reg=None,
    log=True,
    norm_trace=False,
    transform_into="average_power",
)
LDA_KW = dict(solver="lsqr", shrinkage="auto")

# Frozen ledger, handoff section 23.2 (exact19 OOF).
LEDGER_ACC19 = {
    1: 0.840278, 2: 0.562500, 3: 0.951389, 4: 0.756944, 5: 0.548611,
    6: 0.715278, 7: 0.715278, 8: 0.986111, 9: 0.784722,
}
LEDGER_TOL = 0.02   # допуск на дрейф версий MNE/MOABB; при превышении печатается WARN

OUT = Path("/kaggle/working/spikeshield_stage_s")
CTX = OUT / "context"
CTX.mkdir(parents=True, exist_ok=True)

print("=" * 78)
print("STAGE S0 — CONTEXT BUILDER")
print("=" * 78)
print(f"mne {mne.__version__} | moabb {moabb.__version__} | numpy {np.__version__}")


# ============================================================================
# 2. LEAKAGE GUARD ON /kaggle/input
# ============================================================================

FORBIDDEN_TOKENS = ("locked", "stage5", "stage6", "1test", "test_e", "contam")


def scan_inputs():
    root = Path("/kaggle/input")
    if not root.exists():
        raise RuntimeError("/kaggle/input not found. Attach the donor-bank dataset.")
    bad = []
    files = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        files.append(p)
        low = p.name.lower()
        if any(tok in low for tok in FORBIDDEN_TOKENS):
            bad.append(str(p))
    if bad:
        raise RuntimeError(
            "Forbidden files present in /kaggle/input (locked / Stage5 / Stage6 / 1test):\n"
            + "\n".join(bad)
        )
    return files


input_files = scan_inputs()
print(f"\n[guard] /kaggle/input scanned: {len(input_files)} files, no forbidden tokens.")


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ============================================================================
# 3. LOAD BNCI2014_001 (0train only) IN TWO BANDS
# ============================================================================

def to_microvolts(X, label):
    """MOABB обычно отдаёт вольты. Приводим к µV и проверяем физиологичность."""
    X = np.asarray(X, dtype=np.float64)
    med = float(np.median(np.sqrt(np.mean(X ** 2, axis=(1, 2)))))
    if med < 1e-3:
        X = X * 1e6
        scaled = True
    else:
        scaled = False
    med2 = float(np.median(np.sqrt(np.mean(X ** 2, axis=(1, 2)))))
    if not (0.3 <= med2 <= 300.0):
        raise RuntimeError(
            f"{label}: implausible RMS after unit handling: {med2:.4f} µV"
        )
    return X, scaled, med2


dataset = BNCI2014_001()
assert dataset.__class__.__name__ == "BNCI2014_001"

paradigm_dec = LeftRightImagery(fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX)
paradigm_saf = LeftRightImagery(fmin=SAFETY_FMIN, fmax=SAFETY_FMAX, tmin=TMIN, tmax=TMAX)

data19 = {}
ch19_index = None

for s in SUBJECTS:
    print(f"\n--- Subject {s} ---")

    if ch19_index is None:
        ep = paradigm_dec.get_data(dataset, subjects=[s], return_epochs=True)[0]
        ch_names = list(ep.ch_names)
        missing = [c for c in COMMON19 if c not in ch_names]
        if missing:
            raise RuntimeError(f"COMMON19 channels missing from dataset: {missing}")
        ch19_index = np.array([ch_names.index(c) for c in COMMON19], dtype=int)
        print("  COMMON19 indices:", ch19_index.tolist())
        # порядок обязан совпадать буквально
        assert [ch_names[i] for i in ch19_index] == COMMON19
        del ep

    Xd, y, meta = paradigm_dec.get_data(dataset, subjects=[s])
    Xs, ys, metas = paradigm_saf.get_data(dataset, subjects=[s])

    meta = meta.reset_index(drop=True)
    metas = metas.reset_index(drop=True)

    # обе ветки обязаны содержать одни и те же трайлы в одном порядке
    assert np.array_equal(np.asarray(y), np.asarray(ys)), f"S{s}: label order mismatch"
    assert meta["session"].tolist() == metas["session"].tolist()
    assert meta["run"].tolist() == metas["run"].tolist()

    sess = meta["session"].astype(str)
    train_names = sorted({v for v in sess.unique() if "train" in v.lower() or v == "0"})
    if not train_names:
        train_names = [sorted(sess.unique())[0]]
    train_mask = sess.isin(train_names).to_numpy()
    print(f"  development session(s): {train_names}  trials={int(train_mask.sum())}")

    Xd = Xd[train_mask][:, ch19_index, :]
    Xs = Xs[train_mask][:, ch19_index, :]
    y = np.asarray(y)[train_mask].astype(str)
    runs = meta.loc[train_mask, "run"].astype(str).to_numpy()

    if Xd.shape[-1] != N_TIMES:
        raise RuntimeError(f"S{s}: expected {N_TIMES} samples, got {Xd.shape[-1]}")

    Xd, sc_d, rms_d = to_microvolts(Xd, f"S{s} decoder")
    Xs, sc_s, rms_s = to_microvolts(Xs, f"S{s} safety")

    # CAR19 после отбора каналов, как в замороженном пайплайне
    Xd = Xd - Xd.mean(axis=1, keepdims=True)
    Xs = Xs - Xs.mean(axis=1, keepdims=True)

    assert np.allclose(Xd.sum(axis=1), 0.0, atol=1e-6)
    assert set(np.unique(y)).issubset(set(CLASSES))

    data19[s] = dict(
        X_dec=Xd.astype(np.float64),
        X_saf=Xs.astype(np.float64),
        y=y,
        runs=runs,
    )
    print(f"  X_dec {Xd.shape}  median RMS {rms_d:.3f} µV (volts->µV: {sc_d})")
    print(f"  X_saf {Xs.shape}  median RMS {rms_s:.3f} µV")
    print(f"  runs: {sorted(set(runs))}")


# ============================================================================
# 4. LEAVE-ONE-RUN-OUT CSP+LDA + LEDGER REGRESSION GUARD
# ============================================================================

def make_decoder(seed=SEED):
    return make_pipeline(CSP(**CSP_KW), LDA(**LDA_KW))


def signed_margin(model, X, y):
    """margin > 0 <=> решение верное; ориентация счёта на right_hand."""
    lda = model.named_steps["lineardiscriminantanalysis"]
    Z = model.named_steps["csp"].transform(X)
    raw = np.asarray(lda.decision_function(Z)).reshape(-1)
    classes = np.asarray(lda.classes_)
    if classes[1] == "right_hand":
        score_right = raw
    elif classes[0] == "right_hand":
        score_right = -raw
    else:
        raise RuntimeError(f"unexpected LDA classes: {classes}")
    y_sign = np.where(np.asarray(y) == "right_hand", 1.0, -1.0)
    pred = np.where(score_right > 0, "right_hand", "left_hand")
    return score_right, y_sign * score_right, pred


models_oof = {}
oof_rows = []

print("\n" + "=" * 78)
print("CLEAN LEAVE-ONE-RUN-OUT DECODERS")
print("=" * 78)

for s in SUBJECTS:
    X = data19[s]["X_dec"]
    y = data19[s]["y"]
    runs = data19[s]["runs"]
    uruns = sorted(set(runs))
    if len(uruns) != 6:
        print(f"  WARN S{s}: {len(uruns)} runs (expected 6)")

    models_oof[s] = {}
    correct = 0
    total = 0
    for r in uruns:
        val = runs == r
        fit = ~val
        m = make_decoder()
        m.fit(X[fit], y[fit])
        models_oof[s][r] = m
        _, marg, pred = signed_margin(m, X[val], y[val])
        correct += int(np.sum(pred == y[val]))
        total += int(val.sum())

    acc = correct / total
    ref = LEDGER_ACC19[s]
    delta = acc - ref
    flag = "OK " if abs(delta) <= LEDGER_TOL else "WARN"
    print(f"  S{s}  OOF acc = {acc:.6f}   ledger = {ref:.6f}   delta = {delta:+.6f}  [{flag}]")
    oof_rows.append(dict(bci_subject=s, oof_acc=acc, ledger_acc=ref, delta=delta,
                         n_trials=total, n_correct=correct))

oof_df = pd.DataFrame(oof_rows)
mean_delta = float(np.mean(np.abs(oof_df["delta"])))
print(f"\nmean |delta| vs ledger 23.2 = {mean_delta:.6f}")
if (oof_df["delta"].abs() > LEDGER_TOL).any():
    print(
        "WARN: минимум один субъект вне допуска. Это НЕ повод менять ледгер.\n"
        "      Разберитесь в причине (версия MNE/MOABB, фильтр, порядок каналов)\n"
        "      прежде чем интерпретировать S1/S2 научно."
    )
else:
    print("LEDGER REGRESSION GUARD: PASS")


# ============================================================================
# 5. LOAD DEVELOPMENT CHEW DONOR BANK
# ============================================================================

print("\n" + "=" * 78)
print("DEVELOPMENT CHEW DONOR BANK")
print("=" * 78)


ALL_INPUT_FILES = [p for p in Path("/kaggle/input").rglob("*") if p.is_file()]


def _diagnose():
    lines = ["", "-" * 72, "СОДЕРЖИМОЕ /kaggle/input:", "-" * 72]
    if not ALL_INPUT_FILES:
        lines.append("  (пусто — ни один Input не приложен)")
    for p in sorted(ALL_INPUT_FILES)[:120]:
        lines.append(f"  {p.relative_to('/kaggle/input')}  ({p.stat().st_size/1e6:.1f} MB)")
    if len(ALL_INPUT_FILES) > 120:
        lines.append(f"  ... ещё {len(ALL_INPUT_FILES)-120} файлов")

    locked = [p for p in ALL_INPUT_FILES if "locked" in p.name.lower()]
    if locked:
        lines += [
            "",
            "!" * 72,
            "ОБНАРУЖЕН ЛОКИРОВАННЫЙ БАНК (65 доноров, субъекты 12/14/17/19/23/25).",
            "Его НЕЛЬЗЯ использовать как development-данные: эти шесть человек",
            "уже израсходованы на подтверждающий Stage 5 (правило 11 handoff).",
            "Нужен development-банк на 247 доноров от 24 других субъектов.",
            "!" * 72,
        ]
    lines += [
        "",
        "Требуются три файла:",
        "  chew_bank_decoder19_uV_250Hz.npy",
        "  chew_bank_safety19_uV_250Hz.npy",
        "  chew_bank_metadata_development.csv",
        "",
        "Вариант A: Add Input -> Notebook Output -> старый ноутбук со Stage 2F2.",
        "Вариант B: пересобрать через rebuild_dev_bank_cell.py.",
    ]
    return "\n".join(lines)


def find_one(primary, aliases=()):
    wanted = {primary} | set(aliases)
    hits = [p for p in ALL_INPUT_FILES if p.name in wanted]
    if len(hits) == 1:
        return hits[0]
    if len(hits) == 0:
        raise RuntimeError(f"Файл не найден: {primary}\n{_diagnose()}")
    raise RuntimeError(
        f"Найдено несколько копий {primary}: {[str(h) for h in hits]}\n"
        "Оставьте ровно один Input с банком, иначе непонятно, какой считать каноническим."
    )


p_dec = find_one("chew_bank_decoder19_uV_250Hz.npy")
p_saf = find_one("chew_bank_safety19_uV_250Hz.npy")
p_meta = find_one(
    "chew_bank_metadata_development.csv", aliases=("chew_bank_metadata_dev.csv",)
)

for p in (p_dec, p_saf, p_meta):
    print(f"  {p.name}  {p.stat().st_size} bytes  sha256={sha256(p)[:16]}...")

BANK_DEC_ALL = np.load(p_dec).astype(np.float64)
BANK_SAF_ALL = np.load(p_saf).astype(np.float64)
DEV_META = pd.read_csv(p_meta)

for col in ("donor_index", "artifact_subject", "safety_rms_uv_recomputed"):
    if col not in DEV_META.columns:
        raise RuntimeError(f"metadata missing column '{col}'. Columns: {list(DEV_META.columns)}")

if BANK_DEC_ALL.shape[1:] != (19, N_TIMES) or BANK_SAF_ALL.shape[1:] != (19, N_TIMES):
    raise RuntimeError(f"bank shapes {BANK_DEC_ALL.shape} / {BANK_SAF_ALL.shape}")
if BANK_DEC_ALL.shape[0] != BANK_SAF_ALL.shape[0]:
    raise RuntimeError("decoder/safety bank length mismatch")

donor_index = DEV_META["donor_index"].to_numpy(dtype=np.int64)
artifact_subject = DEV_META["artifact_subject"].to_numpy(dtype=np.int64)
donor_safety_rms = DEV_META["safety_rms_uv_recomputed"].to_numpy(dtype=np.float64)

# --- строгие проверки происхождения ---
bad_subj = sorted(set(artifact_subject) - set(DEV_ARTIFACT_SUBJECTS))
if bad_subj:
    raise RuntimeError(f"non-development artifact subjects in metadata: {bad_subj}")
leak = sorted(set(artifact_subject) & set(FORBIDDEN_ARTIFACT_SUBJECTS))
if leak:
    raise RuntimeError(f"LOCKED artifact subjects leaked into development bank: {leak}")
if len(DEV_META) != 247:
    print(f"WARN: development bank has {len(DEV_META)} donors, handoff says 247.")
if not np.all(donor_safety_rms > 0):
    raise RuntimeError("non-positive donor safety RMS")
if donor_index.max() >= BANK_DEC_ALL.shape[0]:
    raise RuntimeError("donor_index out of range for the full bank array")

BANK_DEC = BANK_DEC_ALL[donor_index]
BANK_SAF = BANK_SAF_ALL[donor_index]

# сверка RMS из метаданных с пересчитанным по массиву
rms_recomputed = np.sqrt(np.mean(BANK_SAF ** 2, axis=(1, 2)))
rms_err = float(np.max(np.abs(rms_recomputed - donor_safety_rms) / donor_safety_rms))
print(f"\n  donors        : {len(donor_index)}")
print(f"  artifact subj : {len(set(artifact_subject))}")
print(f"  safety RMS max relative mismatch vs metadata: {rms_err:.3e}")
if rms_err > 1e-3:
    raise RuntimeError(
        "safety RMS in metadata does not match the array. "
        "Вы почти наверняка перепутали полосы (safety=20-80) или порядок доноров."
    )

car_err_dec = float(np.max(np.abs(BANK_DEC.sum(axis=1))))
car_err_saf = float(np.max(np.abs(BANK_SAF.sum(axis=1))))
print(f"  CAR residual  : decoder {car_err_dec:.3e}, safety {car_err_saf:.3e} µV")
if max(car_err_dec, car_err_saf) > 1e-3:
    raise RuntimeError("donor bank is not CAR19-referenced")


# ============================================================================
# 6. FROZEN SNR SCALING (копия замороженной конвенции, не переписанная)
# ============================================================================

def artifact_scale_for_snr(clean_reference_rms_uv, artifact_rms_uv, snr_db):
    """SNR = 20*log10(clean_RMS / scaled_artifact_RMS). Один скаляр на донора."""
    return float(clean_reference_rms_uv) / (
        float(artifact_rms_uv) * 10.0 ** (float(snr_db) / 20.0)
    )


def epoch_rms(X):
    x = np.asarray(X, dtype=np.float64)
    return np.sqrt(np.mean(x * x, axis=(-2, -1)))


# численная проверка конвенции
_ref = float(np.median(epoch_rms(data19[1]["X_saf"])))
_sc = artifact_scale_for_snr(_ref, float(donor_safety_rms[0]), -3.0)
_realized = 20.0 * np.log10(_ref / (float(donor_safety_rms[0]) * _sc))
assert abs(_realized - (-3.0)) < 1e-10, _realized
print(f"\n  SNR convention check: target -3 dB, realized {_realized:.12f} dB  [PASS]")


# ============================================================================
# 7. FAST EXACT CSP ENGINE (перенос из Stage 3A, с гарантией эквивалентности)
# ============================================================================

def csp_geometry_terms(csp, clean_epochs, artifact_epochs):
    """
    clean_epochs   : (N, 19, T)
    artifact_epochs: (D, 19, T)
    returns p_clean (N,K), p_art (D,K), cross (N,D,K), rho (N,D,K)
    cross = E[z*u],  rho = cross / sqrt(p_clean*p_art)
    """
    clean = np.asarray(clean_epochs, dtype=np.float64)
    art = np.asarray(artifact_epochs, dtype=np.float64)
    if clean.ndim != 3 or clean.shape[1] != 19:
        raise ValueError(f"bad clean shape {clean.shape}")
    if art.ndim != 3 or art.shape[1] != 19:
        raise ValueError(f"bad artifact shape {art.shape}")

    k = int(csp.n_components)
    W = np.asarray(csp.filters_[:k], dtype=np.float64)

    z = np.einsum("kc,nct->nkt", W, clean, optimize=True)
    u = np.einsum("kc,dct->dkt", W, art, optimize=True)

    T = z.shape[-1]
    p_clean = np.mean(z ** 2, axis=-1)
    p_art = np.mean(u ** 2, axis=-1)
    cross = np.einsum("nkt,dkt->ndk", z, u, optimize=True) / T

    denom = np.sqrt(p_clean[:, None, :] * p_art[None, :, :])
    rho = cross / np.maximum(denom, 1e-300)
    return dict(p_clean=p_clean, p_art=p_art, cross=cross, rho=rho)


def decisions_from_power(lda, power):
    """power: (...,K) положительные мощности -> log -> LDA. Возвращает score_right."""
    power = np.maximum(np.asarray(power, dtype=np.float64), np.finfo(np.float64).tiny)
    feats = np.log(power)
    flat = feats.reshape(-1, feats.shape[-1])
    raw = np.asarray(lda.decision_function(flat)).reshape(-1)
    classes = np.asarray(lda.classes_)
    if classes[1] == "right_hand":
        score_right = raw
    elif classes[0] == "right_hand":
        score_right = -raw
    else:
        raise RuntimeError(f"unexpected LDA classes: {classes}")
    return score_right.reshape(feats.shape[:-1])


# --- гарантия эквивалентности быстрого движка настоящему MNE CSP ---
_s = 1
_r = sorted(set(data19[_s]["runs"]))[0]
_m = models_oof[_s][_r]
_csp, _lda = _m.named_steps["csp"], _m.named_steps["lineardiscriminantanalysis"]
_val = data19[_s]["runs"] == _r
_Xc = data19[_s]["X_dec"][_val][:6]
_A = BANK_DEC[:4]
_terms = csp_geometry_terms(_csp, _Xc, _A)
_lam = np.full((4, 1), 0.37)
_pw = (
    _terms["p_clean"][:, None, None, :]
    + 2.0 * _lam[None, :, :, None] * _terms["cross"][:, :, None, :]
    + (_lam[None, :, :, None] ** 2) * _terms["p_art"][None, :, None, :]
)
_fast = decisions_from_power(_lda, _pw)[:, :, 0]
_direct = np.empty_like(_fast)
for i in range(_Xc.shape[0]):
    mix = _Xc[i][None, :, :] + 0.37 * _A
    Zd = _csp.transform(mix)
    raw = np.asarray(_lda.decision_function(Zd)).reshape(-1)
    cl = np.asarray(_lda.classes_)
    _direct[i] = raw if cl[1] == "right_hand" else -raw
_engine_err = float(np.max(np.abs(_fast - _direct)))
print(f"  fast-CSP vs direct MNE CSP max score error: {_engine_err:.3e}")
if _engine_err > 1e-9:
    raise RuntimeError("fast CSP engine does not reproduce MNE CSP. STOP.")
print("  FAST ENGINE GUARD: PASS")


# ============================================================================
# 8. SAVE CONTEXT
# ============================================================================

np.savez_compressed(
    CTX / "clean_data19.npz",
    **{f"X_dec_{s}": data19[s]["X_dec"].astype(np.float64) for s in SUBJECTS},
    **{f"X_saf_{s}": data19[s]["X_saf"].astype(np.float32) for s in SUBJECTS},
    **{f"y_{s}": data19[s]["y"] for s in SUBJECTS},
    **{f"runs_{s}": data19[s]["runs"] for s in SUBJECTS},
)
np.save(CTX / "bank_decoder19_dev_uV.npy", BANK_DEC.astype(np.float32))
np.save(CTX / "bank_safety19_dev_uV.npy", BANK_SAF.astype(np.float32))
DEV_META.to_csv(CTX / "bank_metadata_dev.csv", index=False)
oof_df.to_csv(CTX / "clean_oof_ledger_check.csv", index=False)

import joblib
joblib.dump(models_oof, CTX / "models_oof.joblib")

context_manifest = dict(
    stage="S0",
    seed=SEED,
    decoder_band=[FMIN, FMAX],
    safety_band=[SAFETY_FMIN, SAFETY_FMAX],
    epoch_s=[TMIN, TMAX],
    sfreq=SFREQ,
    n_times=N_TIMES,
    common19=COMMON19,
    common19_indices=ch19_index.tolist(),
    snr_grid_db=SNR_GRID_DB.tolist(),
    csp=CSP_KW,
    lda=LDA_KW,
    n_donors=int(len(donor_index)),
    artifact_subjects=sorted(int(v) for v in set(artifact_subject)),
    forbidden_artifact_subjects=FORBIDDEN_ARTIFACT_SUBJECTS,
    ledger_mean_abs_delta=mean_delta,
    fast_engine_max_error=_engine_err,
    snr_convention_realized_minus3=_realized,
    bank_sha256=dict(
        decoder=sha256(p_dec), safety=sha256(p_saf), metadata=sha256(p_meta)
    ),
    versions=dict(mne=mne.__version__, moabb=moabb.__version__, numpy=np.__version__),
)
with open(CTX / "context_manifest.json", "w") as f:
    json.dump(context_manifest, f, indent=2)

print("\n" + "=" * 78)
print("STAGE S0: COMPLETE")
print("=" * 78)
print(f"saved to {CTX}")
for p in sorted(CTX.iterdir()):
    print(f"  {p.name}  {p.stat().st_size/1e6:.2f} MB")
