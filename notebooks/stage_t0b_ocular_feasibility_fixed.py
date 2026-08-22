# ============================================================================
# STAGE T0b — FEASIBILITY GATE, CORRECTED
#
# ПОЧЕМУ T0 ҚАЙТА ЖАСАЛЫП ЖАТЫР
#
# Алғашқы T0 нұсқасы MOABB арқылы алынған рандарды өлшеді. Бірақ MOABB-тың
# жүктеушісі EOG-блоктарын ҮНСІЗ тастап кетеді:
#
#     # some runs does not contains trials i.e baseline runs
#     if len(run.trial) > 0: ...
#     else: return None, None
#
# Сол себепті T0 көз артефактісін емес, кәдімгі MI-ранның ЭЭГ-сін өлшеді.
# Алынған 0,4999 саны — қалыпты ЭЭГ қуатының 8-32 Гц жолағындағы үлесі,
# көз артефактісіне қатысы жоқ. Ол ВЕРДИКТ ЖАРАМСЫЗ.
#
# Бұл нұсқа .mat файлын тікелей оқып, trial өрісі БОС рандарды алады.
# Дәл солар — сессия басындағы 5 минуттық EOG-блогы:
#   ран 1: 2 мин көз ашық
#   ран 2: 1 мин көз жұмық
#   ран 3: 1 мин КӨЗ ҚОЗҒАЛЫСЫ   <- бізге керегі осы
#
# Деректер бұрынғы прогоннан кэште қалған, қайта жүктеудің қажеті жоқ.
#
# ШЕШУШІ ЕРЕЖЕ (T0-дан өзгермеген, алдын ала бекітілген):
#   тірі қалу >= 0.15  -> GO
#   0.05 .. 0.15       -> BORDERLINE
#   < 0.05             -> STOP
#
# Kaggle: бір ұяшық. ~3-5 минут (жүктеу кэштен).
# ============================================================================

import importlib.util, subprocess, sys
if importlib.util.find_spec("moabb") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "moabb>=1.5,<1.7"])
    print("MOABB орнатылды. Сессияны ҚАЙТА ҚОСЫҢЫЗ да, ұяшықты қайта іске қосыңыз.")
    raise SystemExit(0)

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import mne
from scipy.io import loadmat
from moabb.datasets import BNCI2014_001

mne.set_log_level("ERROR")

# --- негізгі пайплайнмен бірдей қатырылған тұрақтылар ---
DECODER_BAND = (8.0, 32.0)
BROAD_BAND = (0.5, 100.0)
SFREQ = 250.0
SEG_S = 4.0
N_TIMES = 1001

BCI2A_CHANNELS = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1",
                  "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4",
                  "P1", "Pz", "P2", "POz"]
COMMON19 = [c for c in BCI2A_CHANNELS if c not in ("CPz", "P1", "P2")]
IDX19 = [BCI2A_CHANNELS.index(c) for c in COMMON19]
assert len(COMMON19) == 19

GO, BORDERLINE = 0.15, 0.05
CHEW_SURVIVAL = 0.3670
CHEW_RMS_MEDIAN = 21.5875     # мкВ, тіркелген донор банкінен

OUT = Path("/kaggle/working/spikeshield_T0b")
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 78)
print("STAGE T0b — КӨЗ БЛОГЫН ТІКЕЛЕЙ .mat ФАЙЛЫНАН ОҚУ")
print("=" * 78)
print(f"Шешуші ереже: >= {GO} GO, < {BORDERLINE} STOP")
print(f"Салыстыру: шайнау доноры  тірі қалу {CHEW_SURVIVAL}, RMS {CHEW_RMS_MEDIAN} мкВ")

ds = BNCI2014_001()

# ============================================================================
# 1. ФАЙЛДАРДЫ ТАБУ ЖӘНЕ ҚҰРЫЛЫМЫН АШУ
# ============================================================================

print("\n" + "=" * 78)
print(".mat ФАЙЛДАРЫНЫҢ ІШКІ ҚҰРЫЛЫМЫ")
print("=" * 78)

inv = []
eog_runs = {}       # (subject, session) -> список (индекс рана, массив)

for subj in range(1, 10):
    files = ds.data_path(subj)
    for fpath in files:
        fname = Path(fpath).name
        m = loadmat(fpath, struct_as_record=False, squeeze_me=True)
        arr = m["data"]
        if not isinstance(arr, np.ndarray):
            arr = np.array([arr])
        for ri, run in enumerate(arr):
            X = np.asarray(run.X)
            trial = np.atleast_1d(np.asarray(run.trial))
            n_trial = 0 if trial.size == 0 or trial.dtype == object else int(trial.size)
            is_eog = n_trial == 0
            dur = X.shape[0] / float(run.fs)
            inv.append(dict(subject=subj, file=fname, run_index=ri,
                            n_samples=X.shape[0], n_channels=X.shape[1],
                            duration_s=round(dur, 1), n_trials=n_trial,
                            is_eog_block=is_eog))
            if is_eog:
                eog_runs.setdefault((subj, fname), []).append((ri, X))
        del m, arr

INV = pd.DataFrame(inv)
INV.to_csv(OUT / "T0b_MAT_STRUCTURE.csv", index=False)
print(INV.head(20).to_string(index=False))
print(f"\nбарлық ран: {len(INV)}")
print(f"EOG блоктары (trial бос): {int(INV.is_eog_block.sum())}")
print(f"MI рандары: {int((~INV.is_eog_block).sum())}")

if INV.is_eog_block.sum() == 0:
    raise RuntimeError(
        "Бірде-бір EOG блогы табылмады. .mat құрылымы күтілгеннен өзгеше; "
        "T0b_MAT_STRUCTURE.csv файлын қараңыз."
    )

# ============================================================================
# 2. ӨЛШЕУ
# ============================================================================

print("\n" + "=" * 78)
print("КӨЗ БЛОКТАРЫН ӨЛШЕУ")
print("=" * 78)


def band_rms(x_uv, lo, hi):
    y = mne.filter.filter_data(np.asarray(x_uv, float), sfreq=SFREQ,
                               l_freq=lo, h_freq=hi, verbose="ERROR")
    return float(np.sqrt(np.mean(y ** 2)))


rows = []
n_win = int(SEG_S * SFREQ)

for (subj, fname), runs in sorted(eog_runs.items()):
    for ri, X in runs:
        # X: (n_samples, 25) = 22 ЭЭГ + 3 EOG. Бізге ЭЭГ арналары керек.
        if X.shape[1] < 22:
            print(f"  S{subj} {fname} run{ri}: {X.shape[1]} арна, өткізілді")
            continue
        eeg = np.asarray(X[:, :22], dtype=np.float64).T     # (22, n)
        if not np.isfinite(eeg).all():
            eeg = np.nan_to_num(eeg)
        sig = eeg[IDX19]
        sig = sig - sig.mean(axis=0, keepdims=True)          # CAR19

        for st in range(0, sig.shape[1] - n_win, n_win):
            seg = sig[:, st:st + n_win]
            broad = band_rms(seg, *BROAD_BAND)
            dec = band_rms(seg, *DECODER_BAND)
            if broad <= 0:
                continue
            rows.append(dict(subject=subj, file=fname, run_index=ri,
                             t_start_s=round(st / SFREQ, 1),
                             broad_rms_uv=broad, decoder_rms_uv=dec,
                             survival=dec / broad))
    print(f"  S{subj} {fname}: {len(runs)} EOG блок өлшенді")

SURV = pd.DataFrame(rows)
SURV.to_csv(OUT / "T0b_SURVIVAL.csv", index=False)
if len(SURV) == 0:
    raise RuntimeError("Терезе өлшенбеді.")

# ============================================================================
# 3. НӘТИЖЕ
# ============================================================================

print("\n" + "=" * 78)
print("БЛОК БОЙЫНША (0=көз ашық, 1=көз жұмық, 2=КӨЗ ҚОЗҒАЛЫСЫ)")
print("=" * 78)
byrun = SURV.groupby("run_index").agg(
    n=("survival", "size"),
    broad_rms=("broad_rms_uv", "median"),
    dec_rms=("decoder_rms_uv", "median"),
    survival=("survival", "median")).round(4)
print(byrun.to_string())

# көз қозғалысы блогы = ең үлкен амплитудалы блок
target_run = int(byrun.broad_rms.idxmax())
print(f"\nКөз қозғалысы блогы ретінде run_index={target_run} таңдалды "
      f"(ең жоғары амплитуда).")

EYE = SURV[SURV.run_index == target_run]
top = EYE.sort_values("broad_rms_uv", ascending=False).groupby("subject").head(8)

print("\n" + "=" * 78)
print("НӘТИЖЕ")
print("=" * 78)
print(f"\nКөз қозғалысы блогының барлық терезелері (n={len(EYE)}):")
print(f"  медианалық кең жолақ RMS : {EYE.broad_rms_uv.median():8.3f} мкВ")
print(f"  медианалық 8-32 RMS      : {EYE.decoder_rms_uv.median():8.3f} мкВ")
print(f"  медианалық тірі қалу     : {EYE.survival.median():8.4f}")

print(f"\nЕң күшті 8 терезе бойынша (n={len(top)}):")
print(f"  медианалық кең жолақ RMS : {top.broad_rms_uv.median():8.3f} мкВ")
print(f"  медианалық 8-32 RMS      : {top.decoder_rms_uv.median():8.3f} мкВ")
print(f"  медианалық тірі қалу     : {top.survival.median():8.4f}")

print(f"\n  Салыстыру, шайнау доноры : тірі қалу {CHEW_SURVIVAL:.4f}, "
      f"RMS {CHEW_RMS_MEDIAN:.3f} мкВ")

print("\nСынаушылар бойынша:")
per = top.groupby("subject").agg(
    broad_rms=("broad_rms_uv", "median"),
    dec_rms=("decoder_rms_uv", "median"),
    survival=("survival", "median")).round(4)
print(per.to_string())

surv = float(top.survival.median())
amp = float(top.decoder_rms_uv.median())

if surv >= GO:
    verdict = "GO"
    reading = ("Көз артефактісі декодер жолағында жеткілікті тірі қалады. "
               "«Өз донор» экспериментін жобалауға болады.")
elif surv >= BORDERLINE:
    verdict = "BORDERLINE"
    reading = ("Тірі қалу шекаралық аймақта. Эксперимент алдында күтілетін "
               "HARM оқиғаларының санын бағалау қажет.")
else:
    verdict = "STOP"
    reading = ("Көз артефактісі декодер жолағында жоғалады. Донор тым әлсіз. "
               "Мұны ЖАРИЯЛАҢЫЗ: бұл да өлшенген нәтиже, әрі ол 4-шектеудің "
               "неге ашық қалғанын түсіндіреді.")

print(f"\nВЕРДИКТ: {verdict}")
print(reading)

print("\nЕКІНШІ, ТӘУЕЛСІЗ ТЕКСЕРУ: амплитуда жеткілікті ме?")
print(f"  көз доноры 8-32 Гц RMS   : {amp:.3f} мкВ")
print(f"  шайнау доноры (медиана)  : 8-32 Гц RMS шамамен "
      f"{CHEW_RMS_MEDIAN*CHEW_SURVIVAL:.3f} мкВ")
print("  Тірі қалу пайызы жоғары бола тұра, абсолютті амплитуда төмен болса,")
print("  SNR-масштабтау кезінде донорды қатты күшейтуге тура келеді, ал бұл")
print("  физиологиялық шындықтан алшақтатады. Екеуін де қараңыз.")

with open(OUT / "T0b_RESULT.json", "w") as f:
    json.dump(dict(
        stage="T0b", verdict=verdict, reading=reading,
        eye_movement_run_index=target_run,
        survival_median_top=surv,
        survival_median_all_eye=float(EYE.survival.median()),
        decoder_rms_median_top=amp,
        broad_rms_median_top=float(top.broad_rms_uv.median()),
        chew_survival_reference=CHEW_SURVIVAL,
        chew_rms_reference=CHEW_RMS_MEDIAN,
        n_eog_blocks=int(INV.is_eog_block.sum()),
        n_windows=int(len(SURV)),
        per_subject=per.reset_index().to_dict(orient="records"),
        by_run=byrun.reset_index().to_dict(orient="records"),
        note=("Supersedes T0, which measured motor-imagery runs because MOABB "
              "silently drops baseline runs without trials."),
    ), f, indent=2, ensure_ascii=False)

print("\nСақталды:", OUT)
for p in sorted(OUT.iterdir()):
    print(" ", p.name)
print("\nБҰЛ ТЕК ОРЫНДАЛУ МҮМКІНДІГІНІҢ ТЕКСЕРІСІ. Ғылыми қорытынды жасамаңыз.")
