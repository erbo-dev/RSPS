# ============================================================================
# STAGE T0 — FEASIBILITY GATE FOR THE OCULAR-DONOR EXPERIMENT
#
# ЭТО НЕ ЭКСПЕРИМЕНТ. Это проверка выполнимости, одна цифра.
#
# Вопрос: выживает ли глазной артефакт из EOG-блока BNCI2014_001 в полосе
# декодера 8-32 Гц достаточно, чтобы служить донором?
#
# Логика. У жевательного донора медианный коэффициент выживаемости
# (RMS в 8-32) / (RMS в широкой полосе) = 0.3670. Если у глазного донора
# он окажется на порядок меньше, донор будет слишком слабым: P_a выйдет
# мизерным, событий HARM почти не будет, и весь эксперимент окажется
# недостаточно мощным.
#
# РЕШАЮЩЕЕ ПРАВИЛО, заморожено ДО запуска:
#   выживаемость >= 0.15  -> эксперимент имеет смысл, продолжаем
#   0.05 .. 0.15          -> пограничная зона, нужен расчёт мощности
#   < 0.05                -> донор слишком слаб, идею закрываем и пишем об этом
#
# Kaggle: один cell. ~15-20 минут (скачивание BNCI2014_001).
# ============================================================================

import importlib.util, subprocess, sys
if importlib.util.find_spec("moabb") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "moabb>=1.5,<1.7"])
    print("MOABB орнатылды. Сессияны ҚАЙТА ҚОСЫҢЫЗ да, ұяшықты қайта іске қосыңыз.")
    raise SystemExit(0)

import json, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import mne
from moabb.datasets import BNCI2014_001

mne.set_log_level("ERROR")

# --- замороженные константы, те же, что в основном пайплайне ---
DECODER_BAND = (8.0, 32.0)
BROAD_BAND = (0.5, 100.0)
SFREQ = 250.0
SEG_S = 4.0
N_TIMES = 1001
COMMON19 = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
            "C2", "C4", "C6", "CP3", "CP1", "CP2", "CP4", "Pz", "POz"]

GO_THRESHOLD = 0.15
BORDERLINE = 0.05
CHEW_SURVIVAL_MEDIAN = 0.3670   # тіркелген ледгерден

OUT = Path("/kaggle/working/spikeshield_T0")
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 78)
print("STAGE T0 — КӨЗ АРТЕФАКТІСІНІҢ ДЕКОДЕР ЖОЛАҒЫНДА ТІРІ ҚАЛУЫ")
print("=" * 78)
print(f"Шешуші ереже (алдын ала бекітілген): >= {GO_THRESHOLD} жалғастыру, "
      f"< {BORDERLINE} тоқтату")
print(f"Салыстыру нүктесі: шайнау доноры {CHEW_SURVIVAL_MEDIAN}")

ds = BNCI2014_001()
paths = ds.get_data(subjects=list(range(1, 10)))

rows = []
for subj in sorted(paths):
    for sess in sorted(paths[subj]):
        runs = paths[subj][sess]
        # EOG-блок жазбаның ЕҢ БАСЫНДА тұр, ол MI жүгірістерінен бөлек
        run_keys = sorted(runs)
        print(f"\nS{subj} / {sess}: {len(run_keys)} run  -> {run_keys}")

        for rk in run_keys:
            raw = runs[rk]
            try:
                raw.load_data()
            except Exception as e:
                print(f"  {rk}: жүктелмеді ({e})")
                continue

            ch = raw.ch_names
            missing = [c for c in COMMON19 if c not in ch]
            has19 = len(missing) == 0
            dur = raw.n_times / raw.info["sfreq"]
            n_ann = len(raw.annotations) if raw.annotations is not None else 0
            ann_types = sorted(set(raw.annotations.description)) if n_ann else []

            rows.append(dict(subject=subj, session=sess, run=rk,
                             duration_s=round(dur, 1), n_channels=len(ch),
                             has_common19=has19, n_annotations=n_ann,
                             annotation_types=",".join(ann_types[:6])))
            print(f"  {rk:14s} {dur:7.1f} с  {len(ch):2d} арна  "
                  f"COMMON19={'бар' if has19 else 'ЖОҚ ' + str(missing[:3])}  "
                  f"аннотация: {ann_types[:4]}")
            raw.close() if hasattr(raw, "close") else None

INV = pd.DataFrame(rows)
INV.to_csv(OUT / "T0_RAW_INVENTORY.csv", index=False)

print("\n" + "=" * 78)
print("ШИКІ ЖАЗБАЛАР ТІЗІМІ")
print("=" * 78)
print(INV.to_string(index=False))

# ============================================================================
# Көз блогын табу және тірі қалуды өлшеу
# ============================================================================

print("\n" + "=" * 78)
print("КӨЗ БЛОГЫН ІЗДЕУ ЖӘНЕ ӨЛШЕУ")
print("=" * 78)

def band_rms(x_uv, lo, hi):
    y = mne.filter.filter_data(np.asarray(x_uv, float), sfreq=SFREQ,
                               l_freq=lo, h_freq=hi, verbose="ERROR")
    return float(np.sqrt(np.mean(y ** 2)))

surv_rows = []
for subj in sorted(paths):
    sess = sorted(paths[subj])[0]
    runs = paths[subj][sess]
    rk = sorted(runs)[0]           # бірінші run: EOG блогы осында болуы тиіс
    raw = runs[rk]
    raw.load_data()

    if any(c not in raw.ch_names for c in COMMON19):
        print(f"S{subj}: COMMON19 толық емес, өткізілді")
        continue

    sf = float(raw.info["sfreq"])
    data = raw.get_data(picks=COMMON19, units="uV")
    data = data - data.mean(axis=0, keepdims=True)     # CAR19

    # алғашқы 5 минутты 4 секундтық терезелерге бөлеміз
    n_win = int(SEG_S * sf)
    max_t = min(int(300 * sf), data.shape[1])
    starts = list(range(0, max_t - n_win, n_win))
    if not starts:
        print(f"S{subj}: жазба тым қысқа")
        continue

    for st in starts:
        seg = data[:, st:st + n_win]
        if not np.isfinite(seg).all():
            continue
        broad = band_rms(seg, *BROAD_BAND)
        dec = band_rms(seg, *DECODER_BAND)
        if broad <= 0:
            continue
        surv_rows.append(dict(subject=subj, t_start_s=round(st / sf, 1),
                              broad_rms_uv=broad, decoder_rms_uv=dec,
                              survival=dec / broad))
    raw.close() if hasattr(raw, "close") else None
    print(f"S{subj}: {len(starts)} терезе өлшенді")

SURV = pd.DataFrame(surv_rows)
SURV.to_csv(OUT / "T0_SURVIVAL.csv", index=False)

if len(SURV) == 0:
    raise RuntimeError("Бірде-бір терезе өлшенбеді. Жазба құрылымын тексеріңіз.")

# ең күшті терезелер (көз қозғалысы блогы) бойынша бағалау
top = SURV.sort_values("broad_rms_uv", ascending=False).groupby("subject").head(8)

print("\n" + "=" * 78)
print("НӘТИЖЕ")
print("=" * 78)
print("\nБарлық терезелер бойынша:")
print(f"  медианалық кең жолақ RMS : {SURV.broad_rms_uv.median():.3f} мкВ")
print(f"  медианалық 8-32 RMS      : {SURV.decoder_rms_uv.median():.3f} мкВ")
print(f"  медианалық тірі қалу     : {SURV.survival.median():.4f}")

print("\nЕң жоғары амплитудалы 8 терезе бойынша (көз қозғалысы блогы):")
print(f"  медианалық кең жолақ RMS : {top.broad_rms_uv.median():.3f} мкВ")
print(f"  медианалық 8-32 RMS      : {top.decoder_rms_uv.median():.3f} мкВ")
print(f"  медианалық тірі қалу     : {top.survival.median():.4f}")
print(f"\n  Салыстыру: шайнау доноры  {CHEW_SURVIVAL_MEDIAN:.4f}")

print("\nСынаушылар бойынша (ең күшті терезелер):")
per = top.groupby("subject").agg(
    broad_rms=("broad_rms_uv", "median"),
    dec_rms=("decoder_rms_uv", "median"),
    survival=("survival", "median")).round(4)
print(per.to_string())

surv = float(top.survival.median())
if surv >= GO_THRESHOLD:
    verdict = "GO"
    reading = ("Көз артефактісі декодер жолағында жеткілікті түрде тірі қалады. "
               "«Өз донор» экспериментін жобалауға болады.")
elif surv >= BORDERLINE:
    verdict = "BORDERLINE"
    reading = ("Тірі қалу шекаралық аймақта. Экспериментті бастамас бұрын "
               "күтілетін HARM оқиғаларының санын бағалау қажет. Егер олар "
               "бірнеше мыңнан аз болса, статистика әлсіз болады.")
else:
    verdict = "STOP"
    reading = ("Көз артефактісі декодер жолағында дерлік жоғалады. Донор тым "
               "әлсіз, эксперимент мағынасыз. Мұны жасырмай, ЖАРИЯЛАҢЫЗ: "
               "бұл да нәтиже, әрі ол шектеу 4-тің неге ашық қалғанын түсіндіреді.")

print(f"\nВЕРДИКТ: {verdict}")
print(reading)

with open(OUT / "T0_RESULT.json", "w") as f:
    json.dump(dict(
        stage="T0", verdict=verdict, reading=reading,
        survival_median_top=surv,
        survival_median_all=float(SURV.survival.median()),
        chew_reference=CHEW_SURVIVAL_MEDIAN,
        go_threshold=GO_THRESHOLD, borderline=BORDERLINE,
        n_windows=int(len(SURV)),
        per_subject=per.reset_index().to_dict(orient="records"),
    ), f, indent=2, ensure_ascii=False)

print("\nСақталды:", OUT)
for p in sorted(OUT.iterdir()):
    print(" ", p.name)
print("\nЕСКЕРТУ: бұл тек орындалу мүмкіндігінің тексерісі. Ешқандай ғылыми")
print("қорытынды жасамаңыз. Вердикт GO болса ғана эксперимент жобаланады.")
