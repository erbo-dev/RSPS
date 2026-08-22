# ============================================================================
# SPIKESHIELD STAGE S-1 — DEVELOPMENT CHEW DONOR BANK REBUILD
#
# Самодостаточная ячейка. Ничего от старых ноутбуков не требует.
# Собрана из notebook3fc573e1d7 (Stage 2A / 2D / 2F0 / 2F1A / 2F1B / 2F2),
# логика извлечения донора воспроизведена без изменений.
#
# Что делает:
#   1  ставит moabb + boto3
#   2  берёт порядок каналов BCI2a из MOABB и строит COMMON19
#   3  строит замороженный фильтр декодера 8-32 (MOABB-совместимый)
#   4  индексирует 180 CSV аннотаций PhysioMotion, находит эпизоды CHEW
#   5  качает EDF по одному, извлекает доноров, чекпойнтит, удаляет EDF
#   6  агрегирует и сохраняет ровно те имена, которые ждёт stage_s0_context.py
#
# Только 24 development-субъекта. Локированные 12/14/17/19/23/25 исключены
# на уровне жёсткого assert (правило 11 handoff).
#
# ВРЕМЯ: 1.5-3 часа. Устойчиво к перезапуску: чекпойнт на каждый ран.
#        Если сессия Kaggle упала, просто запустите ячейку снова.
# ДИСК:  EDF удаляются сразу после обработки, пик около 1 ГБ.
# ============================================================================

import importlib.util
import subprocess
import sys

_need_restart = False
for pkg, spec in [("moabb", "moabb>=1.5,<1.7"), ("boto3", "boto3")]:
    if importlib.util.find_spec(pkg) is None:
        print(f"Installing {spec} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", spec])
        _need_restart = _need_restart or (pkg == "moabb")
if _need_restart:
    print("\nMOABB установлен. ПЕРЕЗАПУСТИТЕ сессию Kaggle ОДИН раз и запустите ячейку заново.")
    raise SystemExit(0)

import gc
import hashlib
import json
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import boto3
import mne
import moabb
from botocore import UNSIGNED
from botocore.config import Config
from scipy.signal import resample_poly
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery

try:
    from moabb.datasets.preprocessing import get_filter_pipeline
except Exception:
    from moabb.datasets.preprocessing import RawToEpochs  # noqa: F401
    raise RuntimeError(
        "get_filter_pipeline не найден в этой версии MOABB. "
        "Установите moabb>=1.5,<1.7."
    )

mne.set_log_level("ERROR")
moabb.set_log_level("error")

# ============================================================================
# 1. ЗАМОРОЖЕННЫЕ КОНСТАНТЫ
# ============================================================================

SEED = 20260813

UV_PER_V = 1e6
V_PER_UV = 1e-6

SOURCE_FS = 1000.0
TARGET_FS = 250.0

DECODER_FMIN, DECODER_FMAX = 8.0, 32.0
CHEW_FMIN, CHEW_FMAX = 20.0, 80.0
NOTCH_HZ = 50.0

CORE_S = 4.0
CORE_N_1000 = 4001
CORE_N_250 = 1001
MIN_CONTEXT_S = 1.0
MERGE_GAP_S = 0.20

BUCKET = "openneuro.org"
DATASET_PREFIX = "ds006386/"
ANNOTATION_WORKERS = 16

DEV_ARTIFACT_SUBJECTS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
    15, 16, 18, 20, 21, 22, 24, 26, 27, 28, 29, 30,
]
LOCKED_ARTIFACT_SUBJECTS = [12, 14, 17, 19, 23, 25]

assert len(DEV_ARTIFACT_SUBJECTS) == 24
assert not (set(DEV_ARTIFACT_SUBJECTS) & set(LOCKED_ARTIFACT_SUBJECTS))
assert sorted(DEV_ARTIFACT_SUBJECTS + LOCKED_ARTIFACT_SUBJECTS) == list(range(1, 31))

ROOT = Path("/kaggle/working/spikeshield_bank_rebuild")
PHYSIO_DIR = ROOT / "physiomotion"
BANK_ROOT = ROOT / "chew_bank"
RUN_CKPT_DIR = BANK_ROOT / "run_checkpoints"
TMP_RAW_DIR = BANK_ROOT / "_tmp_raw"
OUT_DIR = ROOT / "development_bank"
for d in (PHYSIO_DIR, BANK_ROOT, RUN_CKPT_DIR, TMP_RAW_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

print("=" * 78)
print("STAGE S-1 — DEVELOPMENT CHEW DONOR BANK REBUILD")
print("=" * 78)
print(f"mne {mne.__version__} | moabb {moabb.__version__} | numpy {np.__version__}")
print(f"development artifact subjects ({len(DEV_ARTIFACT_SUBJECTS)}): {DEV_ARTIFACT_SUBJECTS}")
print(f"EXCLUDED locked subjects: {LOCKED_ARTIFACT_SUBJECTS}")

# ============================================================================
# 2. ПОРЯДОК КАНАЛОВ BCI2a И COMMON19
# ============================================================================

print("\n[1/6] BCI2a channel order from MOABB ...")

_paradigm = LeftRightImagery(fmin=DECODER_FMIN, fmax=DECODER_FMAX, tmin=0.0, tmax=4.0)
_ep = _paradigm.get_data(dataset=BNCI2014_001(), subjects=[1], return_epochs=True)[0]
BCI2A_CHANNELS = list(_ep.ch_names)
del _ep
gc.collect()

assert len(BCI2A_CHANNELS) == 22, BCI2A_CHANNELS
EXCLUDED = {"CPz", "P1", "P2"}
COMMON19 = [ch for ch in BCI2A_CHANNELS if ch not in EXCLUDED]

assert len(COMMON19) == 19
assert set(BCI2A_CHANNELS) - set(COMMON19) == EXCLUDED
print(f"  BCI2a 22ch : {BCI2A_CHANNELS}")
print(f"  COMMON19   : {COMMON19}")
print(f"  excluded   : {sorted(EXCLUDED)} (отсутствуют в PhysioMotion)")

# ============================================================================
# 3. ЗАМОРОЖЕННЫЙ ФИЛЬТР ДЕКОДЕРА 8-32
# ============================================================================

print("\n[2/6] Building frozen MOABB-compatible 8-32 Hz filter ...")

MOABB_8_32 = get_filter_pipeline(DECODER_FMIN, DECODER_FMAX)


def decoder_filter_uv(X_uv, sfreq=TARGET_FS):
    """MOABB-совместимая фильтрация 8-32 Гц. Вход и выход в µV."""
    X = np.asarray(X_uv, dtype=np.float64)
    if X.ndim < 2:
        raise ValueError(f"Expected >=2 dimensions, got {X.shape}")
    original_shape = X.shape
    n_times = original_shape[-1]
    flat = X.reshape(-1, n_times)
    info = mne.create_info(
        ch_names=[f"EEG_{i:04d}" for i in range(flat.shape[0])],
        sfreq=float(sfreq),
        ch_types="eeg",
    )
    raw = mne.io.RawArray(flat * V_PER_UV, info, verbose="ERROR")
    filtered = MOABB_8_32.transform(raw).get_data(units="uV")
    return filtered.reshape(original_shape)


_rng = np.random.default_rng(SEED + 260)
_A = _rng.normal(0, 10, size=(3, 19, 1001))
_B = _rng.normal(0, 7, size=(3, 19, 1001))
_f = decoder_filter_uv(np.concatenate([_A, _B, _A + _B], axis=0))
_lin = float(
    np.max(np.abs(_f[6:9] - _f[0:3] - _f[3:6])) / max(np.max(np.abs(_f[6:9])), 1e-12)
)
print(f"  filter linearity relative error: {_lin:.3e}")
assert _lin < 1e-8, "фильтр декодера нелинеен, дальше идти нельзя"
print("  DECODER FILTER: PASS")
del _A, _B, _f
gc.collect()

# ============================================================================
# 4. ИНДЕКС ЭПИЗОДОВ CHEW (только метаданные, EDF не качаются)
# ============================================================================

print("\n[3/6] Indexing PhysioMotion CHEW episodes ...")

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

index_path = PHYSIO_DIR / "s3_keys.json"
if index_path.exists():
    all_keys = json.loads(index_path.read_text())
    print(f"  cached S3 index: {len(all_keys)} objects")
else:
    all_keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=DATASET_PREFIX):
        for obj in page.get("Contents", []):
            all_keys.append(obj["Key"])
    index_path.write_text(json.dumps(all_keys, indent=2))
    print(f"  indexed ds006386: {len(all_keys)} objects")

all_key_set = set(all_keys)


def parse_manual_id(key):
    m = re.fullmatch(r"sub-?0*(\d+)_run-?0*(\d+)\.csv", Path(key).name, flags=re.I)
    return (int(m.group(1)), int(m.group(2))) if m else None


manual_items = []
for key in sorted(all_keys):
    if "manual_annotations" in key.lower() and key.lower().endswith(".csv"):
        ident = parse_manual_id(key)
        if ident is not None:
            manual_items.append((ident[0], ident[1], key))
manual_items.sort()

print(f"  annotation files parsed: {len(manual_items)}")
assert len(manual_items) == 180, f"expected 180, got {len(manual_items)}"
assert {s for s, _, _ in manual_items} == set(range(1, 31))


def raw_key_for(subject, run):
    return (
        f"ds006386/sub-{subject}/eeg/"
        f"sub-{subject}_task-artifact_run-{run:02d}_eeg.edf"
    )


def read_annotation_from_s3(item):
    subject, run, key = item
    body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    ann = pd.read_csv(BytesIO(body))
    ann.columns = [str(c).strip().lower().replace(" ", "_") for c in ann.columns]
    expected = {"channel", "start_time", "stop_time", "label"}
    if not expected.issubset(ann.columns):
        raise RuntimeError(f"Bad annotation schema: {key}")

    label = ann["label"].astype(str).str.strip().str.lower()
    chew = ann.loc[label.eq("chew"), ["start_time", "stop_time"]].copy()
    if chew.empty:
        return subject, run, key, []

    chew["start_time"] = pd.to_numeric(chew["start_time"], errors="raise")
    chew["stop_time"] = pd.to_numeric(chew["stop_time"], errors="raise")
    chew = chew.sort_values(["start_time", "stop_time"])

    merged = []
    for start, stop in chew.to_numpy(dtype=float):
        start, stop = float(start), float(stop)
        if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
            continue
        if not merged or start > merged[-1][1] + MERGE_GAP_S:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return subject, run, key, merged


results = []
with ThreadPoolExecutor(max_workers=ANNOTATION_WORKERS) as pool:
    futures = {pool.submit(read_annotation_from_s3, it): it for it in manual_items}
    for fut in as_completed(futures):
        results.append(fut.result())

rows = []
for subject, run, annotation_key, episodes in sorted(results):
    rk = raw_key_for(subject, run)
    if rk not in all_key_set:
        raise RuntimeError(f"Missing raw EDF for P{subject:02d} R{run:02d}: {rk}")
    for episode_i, (start, stop) in enumerate(episodes, start=1):
        duration = stop - start
        available_context = max(0.0, (duration - CORE_S) / 2.0)
        rows.append({
            "artifact_subject": subject,
            "artifact_run": run,
            "episode_in_run": episode_i,
            "donor_id": f"P{subject:02d}_R{run:02d}_CHEW{episode_i:02d}",
            "start_s": start,
            "stop_s": stop,
            "duration_s": duration,
            "available_context_s": available_context,
            "usable": bool(available_context >= MIN_CONTEXT_S),
            "annotation_key": annotation_key,
            "raw_key": rk,
        })

CHEW_EPISODE_INDEX = (
    pd.DataFrame(rows)
    .sort_values(["artifact_subject", "artifact_run", "episode_in_run"])
    .reset_index(drop=True)
)
if CHEW_EPISODE_INDEX.empty:
    raise RuntimeError("No CHEW annotations found.")

usable_all = CHEW_EPISODE_INDEX[CHEW_EPISODE_INDEX["usable"]]
print(f"  all CHEW episodes    : {len(CHEW_EPISODE_INDEX)}")
print(f"  usable episodes      : {len(usable_all)}")
print(f"  subjects with usable : {usable_all['artifact_subject'].nunique()}")
print(f"  mean duration [s]    : {usable_all['duration_s'].mean():.3f}")

# сверка с замороженным ледгером (handoff E.7)
if len(CHEW_EPISODE_INDEX) != 327:
    print(f"  WARN: handoff фиксирует 327 эпизодов, здесь {len(CHEW_EPISODE_INDEX)}")
if len(usable_all) != 312:
    print(f"  WARN: handoff фиксирует 312 usable, здесь {len(usable_all)}")

CHEW_EPISODE_INDEX.to_csv(BANK_ROOT / "chew_episode_index_all.csv", index=False)
usable_all.to_csv(BANK_ROOT / "chew_episode_index_usable.csv", index=False)

# --- РАЗВИЛКА DEV / LOCKED ---
usable = usable_all[
    usable_all["artifact_subject"].isin(DEV_ARTIFACT_SUBJECTS)
].copy()
assert not usable["artifact_subject"].isin(LOCKED_ARTIFACT_SUBJECTS).any()

expected_runs = (
    usable[["artifact_subject", "artifact_run", "raw_key"]]
    .drop_duplicates()
    .sort_values(["artifact_subject", "artifact_run"])
    .reset_index(drop=True)
)
print(f"\n  DEVELOPMENT usable episodes : {len(usable)}")
print(f"  DEVELOPMENT runs to process : {len(expected_runs)}")

# ============================================================================
# 5. ПОТОКОВОЕ ИЗВЛЕЧЕНИЕ ДОНОРОВ
# ============================================================================

s3_stream = boto3.client(
    "s3",
    config=Config(
        signature_version=UNSIGNED,
        retries={"max_attempts": 5, "mode": "standard"},
    ),
)


def remote_size(key):
    return int(s3_stream.head_object(Bucket=BUCKET, Key=key)["ContentLength"])


def verified_download(key, subject, run):
    """Возвращает (path, delete_after). Проверяет полный размер файла."""
    relative = key[len(DATASET_PREFIX):] if key.startswith(DATASET_PREFIX) else key
    persistent = PHYSIO_DIR / relative
    expected = remote_size(key)

    if persistent.exists():
        if persistent.stat().st_size == expected:
            return persistent, False
        persistent.unlink(missing_ok=True)

    final_path = TMP_RAW_DIR / f"P{subject:02d}_R{run:02d}.edf"
    part_path = TMP_RAW_DIR / f"P{subject:02d}_R{run:02d}.edf.part"

    if final_path.exists() and final_path.stat().st_size != expected:
        final_path.unlink()
    if part_path.exists():
        part_path.unlink()

    if not final_path.exists():
        try:
            s3_stream.download_file(BUCKET, key, str(part_path))
            got = part_path.stat().st_size
            if got != expected:
                raise IOError(f"Truncated download {key}: got {got}, expected {expected}")
            os.replace(part_path, final_path)
        except Exception:
            part_path.unlink(missing_ok=True)
            final_path.unlink(missing_ok=True)
            raise
    return final_path, True


def hard_qc_common19(raw_core_uv):
    """Только явный отказ датчика. Большая амплитуда не является причиной отбраковки."""
    x = np.asarray(raw_core_uv, dtype=np.float64)
    bad_channels = []
    for i, ch in enumerate(COMMON19):
        sig = x[i]
        finite = bool(np.isfinite(sig).all())
        if finite:
            std_uv = float(np.std(sig))
            ptp_uv = float(np.ptp(sig))
            unchanged = float(np.mean(np.diff(sig) == 0)) if sig.size > 1 else 1.0
            bad = (std_uv < 1e-3) or (ptp_uv < 1e-3) or (unchanged > 0.999)
        else:
            bad = True
        if bad:
            bad_channels.append(ch)
    return bad_channels


def extract_episode(raw, episode):
    """Центральные 4 с эпизода CHEW. Возвращает (payload, reject_dict)."""
    start_s = float(episode.start_s)
    stop_s = float(episode.stop_s)

    center_s = (start_s + stop_s) / 2.0
    core_start_s = center_s - CORE_S / 2.0
    core_stop_s = core_start_s + CORE_S
    context_s = min(core_start_s - start_s, stop_s - core_stop_s)

    if context_s < MIN_CONTEXT_S:
        return None, {"reason": "insufficient_filter_context"}

    segment_start = int(round((core_start_s - context_s) * SOURCE_FS))
    segment_stop = int(round((core_stop_s + context_s) * SOURCE_FS)) + 1
    core_start = int(round(core_start_s * SOURCE_FS))
    core_stop = core_start + int(CORE_S * SOURCE_FS) + 1

    if core_start < segment_start or core_stop > segment_stop:
        return None, {"reason": "core_outside_segment"}

    segment_uv = raw.get_data(
        picks=COMMON19, start=segment_start, stop=segment_stop, units="uV"
    )

    core_slice = slice(core_start - segment_start, core_stop - segment_start)
    raw_core_uv = segment_uv[:, core_slice]
    if raw_core_uv.shape != (19, CORE_N_1000):
        return None, {"reason": f"bad_raw_core_shape_{raw_core_uv.shape}"}

    bad_channels = hard_qc_common19(raw_core_uv)
    if bad_channels:
        return None, {
            "reason": "bad_common19_channel",
            "bad_channels": ",".join(bad_channels),
        }

    # фильтруем весь аннотированный контекст, потом вырезаем ядро
    emg = mne.filter.notch_filter(
        segment_uv, Fs=SOURCE_FS, freqs=[NOTCH_HZ], verbose="ERROR"
    )
    emg = mne.filter.filter_data(
        emg, sfreq=SOURCE_FS, l_freq=CHEW_FMIN, h_freq=CHEW_FMAX, verbose="ERROR"
    )
    # точный CAR по точным COMMON19
    emg = emg - emg.mean(axis=0, keepdims=True)

    safety_context_uv = resample_poly(emg, up=1, down=4, axis=-1)

    core_start_250 = int(round((core_start - segment_start) / 4.0))
    core_stop_250 = core_start_250 + CORE_N_250
    if core_stop_250 > safety_context_uv.shape[1]:
        return None, {"reason": "resampled_core_outside_segment"}

    decoder_context_uv = decoder_filter_uv(safety_context_uv, sfreq=TARGET_FS)

    safety_core_uv = safety_context_uv[:, core_start_250:core_stop_250]
    decoder_core_uv = decoder_context_uv[:, core_start_250:core_stop_250]

    if safety_core_uv.shape != (19, CORE_N_250):
        return None, {"reason": f"bad_safety_shape_{safety_core_uv.shape}"}
    if decoder_core_uv.shape != (19, CORE_N_250):
        return None, {"reason": f"bad_decoder_shape_{decoder_core_uv.shape}"}
    if not np.isfinite(safety_core_uv).all() or not np.isfinite(decoder_core_uv).all():
        return None, {"reason": "nonfinite_processed_donor"}

    safety_rms = float(np.sqrt(np.mean(safety_core_uv ** 2)))
    decoder_rms = float(np.sqrt(np.mean(decoder_core_uv ** 2)))
    if safety_rms <= 0 or decoder_rms <= 0:
        return None, {"reason": "zero_rms"}

    return {
        "safety": safety_core_uv.astype(np.float32),
        "decoder": decoder_core_uv.astype(np.float32),
        "safety_rms_uv": safety_rms,
        "decoder_rms_uv": decoder_rms,
        "survival_ratio": decoder_rms / safety_rms,
        "context_s": context_s,
        "core_start_s": core_start / SOURCE_FS,
        "core_stop_s": (core_stop - 1) / SOURCE_FS,
    }, None


def checkpoint_paths(subject, run):
    stem = f"P{subject:02d}_R{run:02d}"
    return (
        RUN_CKPT_DIR / f"{stem}.npz",
        RUN_CKPT_DIR / f"{stem}_meta.csv",
        RUN_CKPT_DIR / f"{stem}_rejected.csv",
    )


def checkpoint_valid(subject, run):
    npz_path, meta_path, rej_path = checkpoint_paths(subject, run)
    if not (npz_path.exists() and meta_path.exists() and rej_path.exists()):
        return False
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            safety, decoder = z["safety"], z["decoder"]
            donor_ids = z["donor_ids"].astype(str)
            if safety.ndim != 3 or decoder.ndim != 3:
                return False
            if safety.shape[1:] != (19, CORE_N_250):
                return False
            if decoder.shape != safety.shape:
                return False
            if len(donor_ids) != safety.shape[0]:
                return False
        meta = pd.read_csv(meta_path)
        if len(meta) != len(donor_ids):
            return False
        if len(meta) and not np.array_equal(
            meta["donor_id"].astype(str).to_numpy(), donor_ids
        ):
            return False
        pd.read_csv(rej_path)
        return True
    except Exception:
        return False


META_COLUMNS = [
    "donor_id", "artifact_subject", "artifact_run", "episode_in_run",
    "annotation_start_s", "annotation_stop_s", "annotation_duration_s",
    "filter_context_s", "core_start_s", "core_stop_s",
    "safety_rms_uv", "decoder_rms_uv", "survival_ratio",
]
REJECT_COLUMNS = [
    "donor_id", "artifact_subject", "artifact_run", "episode_in_run",
    "reason", "bad_channels",
]

print("\n[4/6] Streaming donor extraction ...")
t_start = time.time()
n_done_now = 0

for run_i, run_row in enumerate(expected_runs.itertuples(index=False), start=1):
    subject = int(run_row.artifact_subject)
    run = int(run_row.artifact_run)
    rk = run_row.raw_key

    assert subject not in LOCKED_ARTIFACT_SUBJECTS, f"locked subject {subject} leaked"

    if checkpoint_valid(subject, run):
        print(f"  [{run_i:03d}/{len(expected_runs)}] P{subject:02d} R{run:02d} — checkpoint OK, skip")
        continue

    run_eps = usable[
        (usable["artifact_subject"] == subject) & (usable["artifact_run"] == run)
    ].sort_values("episode_in_run")

    local_path, delete_after, raw = None, False, None
    try:
        local_path, delete_after = verified_download(rk, subject, run)
        raw = mne.io.read_raw_edf(local_path, preload=False, verbose="ERROR")

        if not np.isclose(float(raw.info["sfreq"]), SOURCE_FS):
            raise RuntimeError(f"P{subject:02d} R{run:02d}: sfreq={raw.info['sfreq']}")
        missing_ch = [ch for ch in COMMON19 if ch not in raw.ch_names]
        if missing_ch:
            raise RuntimeError(f"P{subject:02d} R{run:02d}: missing COMMON19 {missing_ch}")

        # picks по списку имён обязан вернуть каналы ИМЕННО в порядке COMMON19.
        # Проверяем это один раз явно, а не полагаемся на поведение MNE.
        if not globals().get("_PICK_ORDER_VERIFIED", False):
            _probe = raw.get_data(picks=COMMON19, start=0, stop=4, units="uV")
            _idx = [raw.ch_names.index(ch) for ch in COMMON19]
            _ref = raw.get_data(picks=_idx, start=0, stop=4, units="uV")
            if not np.array_equal(_probe, _ref):
                raise RuntimeError("MNE picks-by-name не сохранил порядок COMMON19")
            globals()["_PICK_ORDER_VERIFIED"] = True
            print(f"      channel pick order verified against COMMON19")

        safety_list, decoder_list, donor_ids = [], [], []
        meta_rows, reject_rows = [], []

        for ep in run_eps.itertuples(index=False):
            payload, reject = extract_episode(raw, ep)
            if reject is not None:
                reject_rows.append({
                    "donor_id": ep.donor_id,
                    "artifact_subject": subject,
                    "artifact_run": run,
                    "episode_in_run": int(ep.episode_in_run),
                    **reject,
                })
                continue
            safety_list.append(payload["safety"])
            decoder_list.append(payload["decoder"])
            donor_ids.append(ep.donor_id)
            meta_rows.append({
                "donor_id": ep.donor_id,
                "artifact_subject": subject,
                "artifact_run": run,
                "episode_in_run": int(ep.episode_in_run),
                "annotation_start_s": float(ep.start_s),
                "annotation_stop_s": float(ep.stop_s),
                "annotation_duration_s": float(ep.duration_s),
                "filter_context_s": float(payload["context_s"]),
                "core_start_s": float(payload["core_start_s"]),
                "core_stop_s": float(payload["core_stop_s"]),
                "safety_rms_uv": float(payload["safety_rms_uv"]),
                "decoder_rms_uv": float(payload["decoder_rms_uv"]),
                "survival_ratio": float(payload["survival_ratio"]),
            })

        if safety_list:
            safety_arr = np.stack(safety_list).astype(np.float32)
            decoder_arr = np.stack(decoder_list).astype(np.float32)
        else:
            safety_arr = np.empty((0, 19, CORE_N_250), dtype=np.float32)
            decoder_arr = np.empty((0, 19, CORE_N_250), dtype=np.float32)

        npz_path, meta_path, rej_path = checkpoint_paths(subject, run)

        tmp = npz_path.with_name(npz_path.stem + ".tmp.npz")
        np.savez_compressed(
            tmp, safety=safety_arr, decoder=decoder_arr,
            donor_ids=np.asarray(donor_ids, dtype="U32"),
        )
        os.replace(tmp, npz_path)

        tmp = meta_path.with_name(meta_path.stem + ".tmp.csv")
        pd.DataFrame(meta_rows, columns=META_COLUMNS).to_csv(tmp, index=False)
        os.replace(tmp, meta_path)

        tmp = rej_path.with_name(rej_path.stem + ".tmp.csv")
        pd.DataFrame(reject_rows, columns=REJECT_COLUMNS).to_csv(tmp, index=False)
        os.replace(tmp, rej_path)

        n_done_now += 1
        elapsed = time.time() - t_start
        rate = elapsed / max(n_done_now, 1)
        remaining = len(expected_runs) - run_i
        print(
            f"  [{run_i:03d}/{len(expected_runs)}] P{subject:02d} R{run:02d} — "
            f"valid={len(safety_arr)}, rejected={len(reject_rows)}  "
            f"[{rate:.0f}s/run, ETA ~{rate*remaining/60:.0f} min]"
        )
    finally:
        if raw is not None:
            try:
                raw.close()
            except Exception:
                pass
        raw = None
        gc.collect()
        if delete_after and local_path is not None:
            Path(local_path).unlink(missing_ok=True)

leftover = list(TMP_RAW_DIR.glob("*.edf"))
for p in leftover:
    p.unlink(missing_ok=True)
print(f"  temporary EDFs cleaned: {len(leftover)}")

# ============================================================================
# 6. АГРЕГАЦИЯ + QC
# ============================================================================

print("\n[5/6] Aggregating development bank ...")

missing_ckpt = [
    (int(r.artifact_subject), int(r.artifact_run))
    for r in expected_runs.itertuples(index=False)
    if not checkpoint_valid(int(r.artifact_subject), int(r.artifact_run))
]
if missing_ckpt:
    raise RuntimeError(
        f"Missing/invalid checkpoints ({len(missing_ckpt)}): {missing_ckpt[:20]}\n"
        "Запустите ячейку ещё раз, она продолжит с места остановки."
    )

safety_parts, decoder_parts, meta_parts, reject_parts = [], [], [], []
for r in expected_runs.itertuples(index=False):
    subject, run = int(r.artifact_subject), int(r.artifact_run)
    npz_path, meta_path, rej_path = checkpoint_paths(subject, run)
    with np.load(npz_path, allow_pickle=False) as z:
        safety = z["safety"].astype(np.float32, copy=False)
        decoder = z["decoder"].astype(np.float32, copy=False)
        donor_ids = z["donor_ids"].astype(str)
    meta = pd.read_csv(meta_path)
    rej = pd.read_csv(rej_path)
    if len(meta) != len(safety):
        raise RuntimeError(f"P{subject:02d} R{run:02d}: metadata/array count mismatch")
    if len(meta):
        if not np.array_equal(meta["donor_id"].astype(str).to_numpy(), donor_ids):
            raise RuntimeError(f"P{subject:02d} R{run:02d}: donor order mismatch")
        safety_parts.append(safety)
        decoder_parts.append(decoder)
        meta_parts.append(meta)
    if len(rej):
        reject_parts.append(rej)

if not safety_parts:
    raise RuntimeError("No valid development CHEW donors after QC.")

BANK_SAFETY = np.concatenate(safety_parts, axis=0)
BANK_DECODER = np.concatenate(decoder_parts, axis=0)
BANK_META = pd.concat(meta_parts, ignore_index=True)
BANK_REJECTED = (
    pd.concat(reject_parts, ignore_index=True) if reject_parts else pd.DataFrame()
)

# donor_index указывает в ЭТОТ массив (0..N-1).
# Историческая нумерация в старом полном банке на 312 доноров была другой,
# стабильный ключ между версиями это donor_id, а не donor_index.
BANK_META.insert(0, "donor_index", np.arange(len(BANK_META), dtype=np.int32))

assert BANK_SAFETY.shape == BANK_DECODER.shape
assert BANK_SAFETY.shape[1:] == (19, CORE_N_250)
assert len(BANK_META) == len(BANK_SAFETY)
assert BANK_META["donor_id"].is_unique
assert np.isfinite(BANK_SAFETY).all() and np.isfinite(BANK_DECODER).all()

safety_rms = np.sqrt(np.mean(BANK_SAFETY.astype(np.float64) ** 2, axis=(1, 2)))
decoder_rms = np.sqrt(np.mean(BANK_DECODER.astype(np.float64) ** 2, axis=(1, 2)))
BANK_META["safety_rms_uv_recomputed"] = safety_rms
BANK_META["decoder_rms_uv_recomputed"] = decoder_rms
BANK_META["survival_ratio_recomputed"] = decoder_rms / safety_rms

car_err_safety = float(np.max(np.abs(BANK_SAFETY.mean(axis=1))))
car_err_decoder = float(np.max(np.abs(BANK_DECODER.mean(axis=1))))
subjects_built = sorted(BANK_META["artifact_subject"].astype(int).unique().tolist())

print(f"\n  artifact subjects : {len(subjects_built)}  {subjects_built}")
print(f"  valid donors      : {len(BANK_META)}")
print(f"  rejected episodes : {len(BANK_REJECTED)}")
print(f"  safety array      : {BANK_SAFETY.shape}")
print(f"  decoder array     : {BANK_DECODER.shape}")
print(f"  CAR err safety    : {car_err_safety:.3e} µV")
print(f"  CAR err decoder   : {car_err_decoder:.3e} µV")
print(f"  safety RMS median : {np.median(safety_rms):.4f} µV")
print(f"  safety RMS mean   : {np.mean(safety_rms):.4f} µV")
print(f"  safety RMS max    : {np.max(safety_rms):.4f} µV   <- handoff bank-wide max 97.945")
print(f"  8-32 survival med : {np.median(decoder_rms/safety_rms):.4f}")
print(
    "  ВНИМАНИЕ: числа 21.661 / 25.18 / 0.3670 / 0.3951 в разделе E.8 handoff\n"
    "            посчитаны BANK-WIDE по 312 донорам от 30 субъектов. Здесь собрано\n"
    "            подмножество 247 доноров от 24 субъектов, поэтому медиана и среднее\n"
    "            НЕ обязаны совпадать. Сравнивать напрямую нельзя.\n"
    "            Настоящие критерии воспроизведения ниже."
)

leak = sorted(set(subjects_built) & set(LOCKED_ARTIFACT_SUBJECTS))
if leak:
    raise RuntimeError(f"LOCKED artifact subjects leaked: {leak}")
assert set(subjects_built).issubset(set(DEV_ARTIFACT_SUBJECTS))
assert car_err_safety < 1e-3 and car_err_decoder < 1e-3

print("\n  --- сверка с замороженным ледгером (handoff E.8) ---")
ok_count = len(BANK_META) == 247
ok_subj = len(subjects_built) == 24
ok_rej = len(BANK_REJECTED) == 0
ok_max = abs(float(np.max(safety_rms)) - 97.945) < 0.05
print(f"  donors == 247               : {ok_count}  (получено {len(BANK_META)})")
print(f"  artifact subjects == 24     : {ok_subj}  (получено {len(subjects_built)})")
print(f"  rejected episodes == 0      : {ok_rej}  (получено {len(BANK_REJECTED)})")
print(f"  safety RMS max ~= 97.945    : {ok_max}  (получено {np.max(safety_rms):.4f})")
ok_count = ok_count and ok_rej and ok_max
if not (ok_count and ok_subj):
    print(
        "\n  WARN: банк не совпал с ледгером. НЕ подгоняйте число.\n"
        "        Вероятные причины: изменился снапшот ds006386, другая версия MNE,\n"
        "        другой набор отбракованных эпизодов. Разберитесь ДО запуска S1/S2,\n"
        "        иначе новые результаты нельзя сравнивать со Stage 3F/5/6."
    )
else:
    print("  BANK REPRODUCTION: PASS")

# ============================================================================
# 7. СОХРАНЕНИЕ ПОД ИМЕНАМИ, КОТОРЫЕ ЖДЁТ stage_s0_context.py
# ============================================================================

print("\n[6/6] Saving ...")

np.save(OUT_DIR / "chew_bank_safety19_uV_250Hz.npy", BANK_SAFETY)
np.save(OUT_DIR / "chew_bank_decoder19_uV_250Hz.npy", BANK_DECODER)
BANK_META.to_csv(OUT_DIR / "chew_bank_metadata_development.csv", index=False)
BANK_REJECTED.to_csv(OUT_DIR / "chew_bank_rejected_development.csv", index=False)

artifact_map = {
    "stage": "S-1 development bank rebuild",
    "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "dataset": "OpenNeuro ds006386 (PhysioMotion)",
    "development_artifact_subjects": DEV_ARTIFACT_SUBJECTS,
    "excluded_locked_artifact_subjects": LOCKED_ARTIFACT_SUBJECTS,
    "n_donors": int(len(BANK_META)),
    "n_artifact_subjects": len(subjects_built),
    "n_runs": int(len(expected_runs)),
    "array_shape": list(BANK_SAFETY.shape),
    "common19": COMMON19,
    "bands": {
        "safety_chew_hz": [CHEW_FMIN, CHEW_FMAX],
        "decoder_hz": [DECODER_FMIN, DECODER_FMAX],
        "notch_hz": NOTCH_HZ,
    },
    "sfreq_source": SOURCE_FS,
    "sfreq_target": TARGET_FS,
    "core_samples": CORE_N_250,
    "min_filter_context_s": MIN_CONTEXT_S,
    "merge_gap_s": MERGE_GAP_S,
    "reference": "CAR over exact COMMON19",
    "donor_index_note": (
        "donor_index нумерует строки ЭТОГО массива (0..N-1). "
        "В историческом полном банке на 312 доноров нумерация была другой. "
        "Стабильный ключ между версиями это donor_id."
    ),
    "ledger_check": {
        "donors_equal_247": bool(len(BANK_META) == 247),
        "subjects_equal_24": bool(ok_subj),
        "rejected_equal_0": bool(len(BANK_REJECTED) == 0),
        "safety_rms_max_matches_97_945": bool(ok_max),
        "all_hard_checks_pass": bool(ok_count and ok_subj),
        "safety_rms_median_dev_subset": float(np.median(safety_rms)),
        "safety_rms_mean_dev_subset": float(np.mean(safety_rms)),
        "safety_rms_max_dev_subset": float(np.max(safety_rms)),
        "survival_median_dev_subset": float(np.median(decoder_rms / safety_rms)),
        "note": (
            "handoff E.8 numbers (median 21.661, mean 25.18, survival median 0.3670) "
            "are BANK-WIDE over 312 donors / 30 subjects. This is the 247-donor / "
            "24-subject development subset, so median and mean are not expected to match."
        ),
    },
    "versions": {
        "mne": mne.__version__,
        "moabb": moabb.__version__,
        "numpy": np.__version__,
    },
    "files": {},
}
for p in sorted(OUT_DIR.iterdir()):
    if p.is_file() and p.name != "ARTIFACT_MAP.json":
        artifact_map["files"][p.name] = {
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
with open(OUT_DIR / "ARTIFACT_MAP.json", "w") as f:
    json.dump(artifact_map, f, indent=2)

print(f"\nsaved to {OUT_DIR}")
for name, info in artifact_map["files"].items():
    print(f"  {name:44s} {info['bytes']/1e6:8.2f} MB  sha256={info['sha256'][:16]}...")

print("\n" + "=" * 78)
print(f"STAGE S-1: COMPLETE in {(time.time()-t_start)/60:.1f} min")
print("=" * 78)
print(
    "\nЧТО ДАЛЬШЕ:\n"
    "  1. Save Version этого ноутбука (Save & Run All), дождитесь завершения.\n"
    "  2. В ноутбуке с S0: Add Input -> Notebook Output -> выберите этот ноутбук.\n"
    "  3. Запустите stage_s0_context.py.\n"
    "  4. ARTIFACT_MAP.json положите рядом с handoff: это начало карты артефактов,\n"
    "     которой у проекта пока нет."
)
