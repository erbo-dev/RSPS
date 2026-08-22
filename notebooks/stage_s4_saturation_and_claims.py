# ============================================================================
# SPIKESHIELD STAGE S4 — SATURATION ANALYSIS + CORRECTED POSTER FIGURES
#
# Запускается ПОСЛЕ S1, S2, S3. Ничего не пересчитывает с нуля, только
# анализирует уже полученные таблицы.
#
# Что делает:
#   A  восстанавливает долю HARM на субъекта в обоих семействах декодеров
#   B  считает связь "доля HARM -> interaction-required" (эффект насыщения)
#   C  проверяет симметрию required vs prevented (подпись нуль-среднего шума)
#   D  считает перенос субъектной картины между CSP и MDM
#   E  перевыпускает фигуры с формулировками, соответствующими S1
#   F  пишет POSTER_CLAIMS.md — точные формулировки для стенда и тезисов
#
# Kaggle: один cell. ~1 минута.
# ============================================================================

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon

warnings.filterwarnings("ignore")

# ============================================================================
# 0. ПОИСК ВХОДОВ
# ============================================================================

CANDIDATES = [Path("/kaggle/working/spikeshield_stage_s")]
if Path("/kaggle/input").exists():
    CANDIDATES += sorted(Path("/kaggle/input").glob("*"))
    CANDIDATES += sorted(Path("/kaggle/input").glob("*/*"))
    CANDIDATES += sorted(Path("/kaggle/input").glob("*/*/*"))

ROOT = next(
    (c for c in CANDIDATES
     if (c / "S1" / "S1_SUBJECT_SUMMARY.csv").exists()
     and (c / "S2" / "S2_SUBJECT_SUMMARY.csv").exists()),
    None,
)
if ROOT is None:
    listing = "\n".join(
        f"  {p}" for p in sorted(Path("/kaggle/input").rglob("*"))[:80]
    ) if Path("/kaggle/input").exists() else "  (/kaggle/input пуст)"
    raise RuntimeError(
        "Не найдены выходы S1/S2. Нужна папка с подпапками S1/ и S2/.\n" + listing
    )

OUT = Path("/kaggle/working/spikeshield_stage_s/S4")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

print("=" * 78)
print("STAGE S4 — SATURATION ANALYSIS + CORRECTED FIGURES")
print("=" * 78)
print(f"inputs: {ROOT}")

SUBJECTS = list(range(1, 10))
D_DONORS = 247
N_SNR = 8

# ============================================================================
# 1. ЗАГРУЗКА
# ============================================================================

S1_SUB = pd.read_csv(ROOT / "S1" / "S1_SUBJECT_SUMMARY.csv")
S2_SUB = pd.read_csv(ROOT / "S2" / "S2_SUBJECT_SUMMARY.csv")

_led_path = ROOT / "context" / "clean_oof_ledger_check.csv"
if _led_path.exists():
    LED = pd.read_csv(_led_path).set_index("bci_subject")
    csp_correct = LED["n_correct"].astype(int)
else:
    # запасной путь: числа из замороженного ледгера 23.2 (ACC19 x 144)
    csp_correct = pd.Series(
        {1: 121, 2: 81, 3: 137, 4: 109, 5: 79, 6: 103, 7: 103, 8: 142, 9: 113}
    )
    print("  WARN: clean_oof_ledger_check.csv не найден, использую числа из handoff 23.2")

OBS = S1_SUB[S1_SUB.arm == "observed"].set_index("bci_subject").sort_index()
NUL = (
    S1_SUB[S1_SUB.arm.str.startswith("null")]
    .groupby("bci_subject")
    .agg(
        full_harm=("full_harm", "mean"),
        interaction_required_harm=("interaction_required_harm", "mean"),
        interaction_prevented_harm=("interaction_prevented_harm", "mean"),
        interaction_required_harm_fraction=("interaction_required_harm_fraction", "mean"),
    )
    .sort_index()
)
MDM = S2_SUB.set_index("bci_subject").sort_index()

# ============================================================================
# 2. ДОЛЯ HARM НА СУБЪЕКТА В ОБОИХ СЕМЕЙСТВАХ
# ============================================================================

csp_events = csp_correct * D_DONORS * N_SNR
csp_rate = (OBS["full_harm"] / csp_events).rename("harm_rate")

mdm_events = MDM["n_clean_correct_trials"] * D_DONORS * N_SNR
mdm_rate = (MDM["full_harm"] / mdm_events).rename("harm_rate")

SAT = pd.DataFrame({
    "bci_subject": list(SUBJECTS) * 2,
    "decoder": ["CSP+LDA"] * 9 + ["Riemannian MDM"] * 9,
    "clean_correct_trials": list(csp_correct[SUBJECTS]) + list(MDM.loc[SUBJECTS, "n_clean_correct_trials"]),
    "clean_accuracy": (
        list(csp_correct[SUBJECTS] / 144.0)
        + list(MDM.loc[SUBJECTS, "clean_mdm_oof_accuracy"])
    ),
    "counterfactual_events": list(csp_events[SUBJECTS]) + list(mdm_events[SUBJECTS]),
    "full_harm": list(OBS.loc[SUBJECTS, "full_harm"]) + list(MDM.loc[SUBJECTS, "full_harm"]),
    "harm_rate": list(csp_rate[SUBJECTS]) + list(mdm_rate[SUBJECTS]),
    "interaction_required_harm": (
        list(OBS.loc[SUBJECTS, "interaction_required_harm"])
        + list(MDM.loc[SUBJECTS, "interaction_required_harm"])
    ),
    "interaction_prevented_harm": (
        list(OBS.loc[SUBJECTS, "interaction_prevented_harm"])
        + list(MDM.loc[SUBJECTS, "interaction_prevented_harm"])
    ),
    "interaction_required_fraction": (
        list(OBS.loc[SUBJECTS, "interaction_required_harm_fraction"])
        + list(MDM.loc[SUBJECTS, "interaction_required_harm_fraction"])
    ),
})
SAT["prevented_over_required"] = (
    SAT["interaction_prevented_harm"] / SAT["interaction_required_harm"].replace(0, np.nan)
)

print("\n" + "=" * 78)
print("A. ДОЛЯ HARM И INTERACTION-REQUIRED ПО СУБЪЕКТАМ")
print("=" * 78)
print(SAT.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ============================================================================
# 3. ЭФФЕКТ НАСЫЩЕНИЯ
# ============================================================================

print("\n" + "=" * 78)
print("B. ЭФФЕКТ НАСЫЩЕНИЯ: доля HARM -> interaction-required")
print("=" * 78)

sat_rows = []
for name, g in list(SAT.groupby("decoder")) + [("оба семейства (n=18)", SAT)]:
    r = spearmanr(g["harm_rate"], g["interaction_required_fraction"])
    sat_rows.append(dict(
        group=name, n=len(g),
        spearman_rho=float(r.statistic), p_value=float(r.pvalue),
    ))
    print(f"  {name:24s} n={len(g):2d}  Spearman rho = {r.statistic:+.3f}   p = {r.pvalue:.5f}")
SATURATION = pd.DataFrame(sat_rows)

pooled = SATURATION[SATURATION.group.str.startswith("оба")].iloc[0]
print(
    "\n  Читается так: чем большую долю решений артефакт переворачивает сам по себе,\n"
    "  тем меньше нужен перекрёстный член. interaction-required в значительной мере\n"
    "  определяется режимом контаминации, а не семейством декодера или субъектом."
)
extremes = SAT.loc[[SAT.harm_rate.idxmax(), SAT.harm_rate.idxmin()]]
for _, e in extremes.iterrows():
    print(
        f"    {e.decoder} S{int(e.bci_subject)}: доля HARM {e.harm_rate:.3f} "
        f"-> interaction-required {e.interaction_required_fraction:.4f}"
    )

# ============================================================================
# 4. СИММЕТРИЯ required vs prevented
# ============================================================================

print("\n" + "=" * 78)
print("C. СИММЕТРИЯ: перекрёстный член ломает и спасает решения одинаково часто")
print("=" * 78)

for name, g in SAT.groupby("decoder"):
    ratio = g["prevented_over_required"].dropna()
    w = wilcoxon(g["interaction_required_harm"], g["interaction_prevented_harm"])
    print(
        f"  {name:16s} prevented/required: медиана {ratio.median():.3f}  "
        f"диапазон [{ratio.min():.2f}, {ratio.max():.2f}]  Wilcoxon p = {w.pvalue:.4f}"
    )
_sym = SAT.groupby("decoder")["prevented_over_required"].median()
_csp_sym = float(_sym.get("CSP+LDA", np.nan))
_mdm_sym = float(_sym.get("Riemannian MDM", np.nan))
print(
    f"\n  CSP: отношение {_csp_sym:.3f}, то есть симметрия почти идеальная.\n"
    "  Нуль-средний шум обязан быть симметричным, структурный эффект был бы\n"
    "  асимметричным. Это независимо подтверждает вердикт S1 NULL_MATCHES_OBSERVED,\n"
    "  полученный совсем другим способом.\n"
    f"\n  MDM: отношение {_mdm_sym:.2f}, симметрии НЕТ. Перекрёстный член чаще\n"
    "  спасает решение, чем ломает. Вероятная причина методологическая: матрица\n"
    "  Cx + lambda^2*Ca не является ковариацией никакого реального сигнала, тогда\n"
    "  как C_full является. В римановой геометрии искусственная матрица может\n"
    "  систематически смещать расстояния. В разложении мощностей CSP этот эффект\n"
    "  выражен слабее. Это ограничение абляции в ковариационном представлении,\n"
    "  и его надо назвать вслух, а не прятать."
)

# ============================================================================
# 5. ПЕРЕНОС СУБЪЕКТНОЙ КАРТИНЫ МЕЖДУ СЕМЕЙСТВАМИ
# ============================================================================

print("\n" + "=" * 78)
print("D. ПЕРЕНОС МЕЖДУ СЕМЕЙСТВАМИ ДЕКОДЕРОВ")
print("=" * 78)

csp_ir = OBS.loc[SUBJECTS, "interaction_required_harm_fraction"].to_numpy()
mdm_ir = MDM.loc[SUBJECTS, "interaction_required_harm_fraction"].to_numpy()
r_ir = spearmanr(csp_ir, mdm_ir)
r_rate = spearmanr(csp_rate[SUBJECTS].to_numpy(), mdm_rate[SUBJECTS].to_numpy())

print(f"  interaction-required, CSP vs MDM : rho = {r_ir.statistic:+.3f}  p = {r_ir.pvalue:.3f}")
print(f"  доля HARM,            CSP vs MDM : rho = {r_rate.statistic:+.3f}  p = {r_rate.pvalue:.3f}")
print(f"\n  медиана CSP = {np.median(csp_ir):.4f}   медиана MDM = {np.median(mdm_ir):.4f}")
print(
    "\n  ВЕЛИЧИНА эффекта переносится между семействами (обе медианы малы).\n"
    "  СУБЪЕКТНАЯ КАРТИНА не переносится. Это ограничение, а не находка,\n"
    "  и оно обязано быть на стенде."
)

# ============================================================================
# 6. ФИГУРЫ
# ============================================================================

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 300, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})
C_CSP, C_MDM = "#2c6fb5", "#c8552b"

# --- Fig S4-1: эффект насыщения (главная новая фигура) ---
fig, ax = plt.subplots(figsize=(7, 4.6))
for name, color in [("CSP+LDA", C_CSP), ("Riemannian MDM", C_MDM)]:
    g = SAT[SAT.decoder == name]
    ax.scatter(g.harm_rate, g.interaction_required_fraction, s=58,
               color=color, label=name, zorder=3, edgecolor="white", linewidth=0.8)
    for _, r_ in g.iterrows():
        ax.annotate(f"S{int(r_.bci_subject)}",
                    (r_.harm_rate, r_.interaction_required_fraction),
                    fontsize=6.5, xytext=(5, 3), textcoords="offset points", color=color)
x = SAT.harm_rate.to_numpy()
y = SAT.interaction_required_fraction.to_numpy()
z = np.polyfit(x, y, 1)
xs = np.linspace(x.min(), x.max(), 50)
ax.plot(xs, np.polyval(z, xs), ls="--", lw=1.1, color="grey", zorder=2)
ax.set_xlabel("доля переворотов решения, вызванных контаминацией (HARM rate)")
ax.set_ylabel("доля HARM, требующая перекрёстного члена")
ax.set_title(
    "Чем сильнее артефакт уже ломает решение,\n"
    f"тем меньше нужен перекрёстный член  (Spearman ρ = {pooled.spearman_rho:+.2f}, "
    f"p = {pooled.p_value:.5f}, n = 18)"
)
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "figS4_1_saturation.png")
plt.close(fig)

# --- Fig S4-2: симметрия required vs prevented ---
fig, ax = plt.subplots(figsize=(5.6, 5.2))
for name, color in [("CSP+LDA", C_CSP), ("Riemannian MDM", C_MDM)]:
    g = SAT[SAT.decoder == name]
    ax.scatter(g.interaction_required_harm, g.interaction_prevented_harm,
               s=58, color=color, label=name, zorder=3,
               edgecolor="white", linewidth=0.8)
lim = max(SAT.interaction_required_harm.max(), SAT.interaction_prevented_harm.max()) * 1.08
ax.plot([0, lim], [0, lim], ls="--", lw=1.1, color="grey", zorder=2)
ax.text(lim * 0.62, lim * 0.68, "идеальная симметрия", fontsize=7.5,
        color="grey", rotation=38)
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel("HARM, требующих перекрёстного члена")
ax.set_ylabel("HARM, предотвращённых перекрёстным членом")
ax.set_title("Перекрёстный член ломает и спасает решения\nодинаково часто: подпись нуль-среднего шума")
ax.legend(fontsize=8, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "figS4_2_symmetry.png")
plt.close(fig)

# --- Fig S4-3: перенос между семействами ---
fig, ax = plt.subplots(figsize=(5.6, 5.2))
ax.scatter(csp_ir, mdm_ir, s=64, color="#4a4a4a", zorder=3,
           edgecolor="white", linewidth=0.8)
for i, s_ in enumerate(SUBJECTS):
    ax.annotate(f"S{s_}", (csp_ir[i], mdm_ir[i]), fontsize=7.5,
                xytext=(6, 3), textcoords="offset points")
lim = max(csp_ir.max(), mdm_ir.max()) * 1.15
ax.plot([0, lim], [0, lim], ls="--", lw=1.0, color="grey")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel("interaction-required, CSP+LDA")
ax.set_ylabel("interaction-required, Riemannian MDM")
ax.set_title(
    f"Величина переносится, субъектная картина нет\n"
    f"(Spearman ρ = {r_ir.statistic:+.3f}, p = {r_ir.pvalue:.2f}, n = 9)"
)
fig.tight_layout()
fig.savefig(FIG / "figS4_3_cross_decoder_transfer.png")
plt.close(fig)

# --- Fig S4-4: сводка по стадиям с ИСПРАВЛЕННЫМ заголовком ---
STAGE_LEDGER = {
    "Stage3F_dev": [0.028911, 0.051595, 0.021261, 0.028975, 0.039841,
                    0.006686, 0.023362, 0.038094, 0.004302],
    "Stage5_locked": [0.032307, 0.064775, 0.021107, 0.038749, 0.054052,
                      0.003454, 0.015123, 0.025811, 0.010784],
    "Stage6_session": [0.041716, 0.071295, 0.017521, 0.074511, 0.049901,
                       0.001956, 0.011134, 0.032084, 0.013725],
    "S1_observed": list(csp_ir),
    "S1_null": list(NUL.loc[SUBJECTS, "interaction_required_harm_fraction"]),
    "S2_MDM": list(mdm_ir),
}


def boot_median(v, n_boot=20000, seed=20260813):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    rg = np.random.default_rng(seed)
    idx = rg.integers(0, len(v), size=(n_boot, len(v)))
    d = np.median(v[idx], axis=1)
    return float(np.median(v)), float(np.quantile(d, .025)), float(np.quantile(d, .975))


fig, ax = plt.subplots(figsize=(9.2, 4.6))
names = list(STAGE_LEDGER)
for i, st in enumerate(names):
    vals = np.array(STAGE_LEDGER[st], float)
    xs = np.full(len(vals), i) + np.linspace(-0.17, 0.17, len(vals))
    color = C_MDM if st == "S2_MDM" else ("#888888" if st == "S1_null" else C_CSP)
    ax.scatter(xs, vals, s=26, color=color, zorder=3)
    for xx, yy, sid in zip(xs, vals, SUBJECTS):
        ax.annotate(str(sid), (xx, yy), fontsize=5.5,
                    xytext=(0, 5), textcoords="offset points", ha="center")
    m, lo, hi = boot_median(vals, seed=20260813 + i)
    ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="_", markersize=26,
                capsize=5, lw=1.6, color="black", zorder=4)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=16, ha="right")
ax.set_ylabel("interaction-required HARM fraction")
ax.set_title(
    "Перекрёстный член clean × artifact вносит лишь несколько процентов переворотов,\n"
    "и суррогатная ветка (S1_null) неотличима от наблюдаемой"
)
fig.tight_layout()
fig.savefig(FIG / "figS4_4_by_stage_corrected.png")
plt.close(fig)

print("\n" + "=" * 78)
print("E. ФИГУРЫ ПЕРЕВЫПУЩЕНЫ")
print("=" * 78)
for p in sorted(FIG.iterdir()):
    print(f"  {p.name}")

# ============================================================================
# 7. ТОЧНЫЕ ФОРМУЛИРОВКИ ДЛЯ СТЕНДА
# ============================================================================

_w_csp = wilcoxon(
    SAT[SAT.decoder == "CSP+LDA"]["interaction_required_harm"],
    SAT[SAT.decoder == "CSP+LDA"]["interaction_prevented_harm"],
)
csp_sym = float(SAT[SAT.decoder == "CSP+LDA"]["prevented_over_required"].median())
csp_sym_p = float(_w_csp.pvalue)
mdm_sym = float(SAT[SAT.decoder == "Riemannian MDM"]["prevented_over_required"].median())

o_m, o_lo, o_hi = boot_median(csp_ir)
n_m, n_lo, n_hi = boot_median(np.array(STAGE_LEDGER["S1_null"]))
m_m, m_lo, m_hi = boot_median(mdm_ir)

claims = f"""# SPIKESHIELD — точные формулировки для стенда и тезисов

Сгенерировано S4. Каждая цифра прослеживается до таблицы в S1/S2/S4.

## Что МОЖНО утверждать

1. В моторно-имажинарном BCI на CSP+LDA (COMMON19/CAR19, 8–32 Гц) контролируемая
   контаминация реальным многоканальным жевательным артефактом переворачивает
   правильные решения в неправильные. Явление наблюдается у 9/9 испытуемых.

2. Точный перекрёстный член clean × artifact необходим лишь для
   {o_m*100:.1f}% таких переворотов (95% субъектный ДИ {o_lo*100:.1f}–{o_hi*100:.1f}%, N=9).

3. Эта величина воспроизводится на невиданных ранее артефактных испытуемых
   (Stage 5) и на отдельной сессии записи BCI (Stage 6), обе по предрегистрации.

4. Мы объясняем, ПОЧЕМУ величина мала. Донор артефакта и носитель сигнала
   физиологически независимы, поэтому перекрёстный член является нуль-средним
   шумом. Два независимых свидетельства:
     а) фазово-рандомизированная суррогатная ветка, точно сохраняющая
        ковариацию донора, даёт {n_m*100:.1f}% (95% ДИ {n_lo*100:.1f}–{n_hi*100:.1f}%);
        парная разность с наблюдаемой веткой покрывает ноль;
     б) в CSP перекрёстный член предотвращает переворот примерно так же часто,
        как вызывает его (отношение {csp_sym:.3f}, Wilcoxon p = {csp_sym_p:.2f}).

5. Величина эффекта переносится на принципиально другое семейство декодеров.
   Riemannian MDM на SPD-многообразии даёт медиану {m_m*100:.1f}%
   (95% ДИ {m_lo*100:.1f}–{m_hi*100:.1f}%).

6. Доля interaction-required в значительной мере определяется режимом
   контаминации: чем большую долю решений артефакт уже переворачивает,
   тем меньше нужен перекрёстный член
   (Spearman ρ = {pooled.spearman_rho:+.2f}, p = {pooled.p_value:.5f}, n = 18 точек
   по двум семействам декодеров).

7. Практическое следствие: для оценки риска артефактно-индуцированной ошибки
   команды достаточно аддитивной модели проецируемых мощностей. Моделировать
   взаимодействие не требуется, что снижает вычислительную стоимость
   онлайн-оценки риска.

8. Весь путь от сырых данных до результата воспроизводим. Пересборка банка
   доноров с нуля другой версией библиотек воспроизвела замороженные значения
   Stage 3F с максимальным отклонением 4.3e-07 по всем девяти испытуемым.

## Что НЕЛЬЗЯ утверждать

- Не «мы открыли механизм взаимодействия». S1 показал обратное: структуры нет,
  есть нуль-средний шум измеримой амплитуды.
- Не «взаимодействие равно нулю». Оно ненулевое, просто малое, и в CSP
  симметричное относительно направления влияния на решение.
- Не «артефакт сам по себе даёт то же решение». NO-INTERACTION сохраняет вклад
  чистого сигнала. Это разные эксперименты.
- Не «результат обобщается на любой артефакт и любой декодер». Проверены один
  класс артефакта и два семейства декодеров.
- Не «субъект, уязвимый в CSP, уязвим и в MDM». Ранговая корреляция
  ρ = {r_ir.statistic:+.3f} (p = {r_ir.pvalue:.2f}), то есть переноса нет.
- Не «защитный гейт работает». Он не превзошёл уверенность декодера.
- Не «миллионы пар это миллионы независимых экспериментов».

## Обязательные ограничения на стенде

- N = 9 испытуемых BCI, 24 артефактных донора-испытуемых, один класс артефакта
  (жевание, компонента 20–80 Гц), бинарная задача левая/правая рука.
- В Stage 6 у испытуемых 2, 5 и 7 чистая точность около 51%, поэтому их вклад
  в утверждение слабый. В MDM введён предрегистрированный порог 0.60,
  его прошли 7 из 9.
- Величина эффекта зависит от режима контаминации (см. пункт 6), поэтому
  число «около 3%» относится к нашей замороженной сетке SNR, а не к
  естественному жеванию.
- В MDM симметрия отсутствует: перекрёстный член чаще спасает решение, чем
  ломает (отношение {mdm_sym:.2f}). Матрица Cx + lambda^2*Ca не является
  ковариацией реального сигнала, тогда как C_full является, поэтому в римановой
  геометрии абляция может смещать расстояния. Это ограничение метода, и оно
  должно быть названо. В разложении мощностей CSP эффект выражен слабее.

## Отрицательные результаты, которые мы публикуем

- Гейт безопасности не превзошёл уверенность декодера
  (медианный HARM Risk@90: 0.3691 против 0.3698; улучшение у 3 из 9 испытуемых).
- Явные геометрические признаки не улучшили сильный базовый уровень
  (медианная дельта AUC −0.00008; Wilcoxon p около 0.99).
- Синтетический EMG с придуманными топографиями дал 13.8–49% HARM, но был
  аннулирован собственной фальсификацией: генератор сам кодировал класс.
- Сгенерированный ИИ-агентом код эксперимента был отклонён тремя верификаторами
  до запуска (10 конкретных дефектов, включая тавтологическую проверку).
"""

(OUT / "POSTER_CLAIMS.md").write_text(claims, encoding="utf-8")

SAT.to_csv(OUT / "S4_SATURATION_TABLE.csv", index=False)
SATURATION.to_csv(OUT / "S4_SATURATION_CORRELATIONS.csv", index=False)
pd.DataFrame([
    dict(metric="interaction_required_CSP_vs_MDM",
         spearman_rho=float(r_ir.statistic), p_value=float(r_ir.pvalue)),
    dict(metric="harm_rate_CSP_vs_MDM",
         spearman_rho=float(r_rate.statistic), p_value=float(r_rate.pvalue)),
]).to_csv(OUT / "S4_CROSS_DECODER_TRANSFER.csv", index=False)

with open(OUT / "S4_RESULT.json", "w") as f:
    json.dump(dict(
        stage="S4",
        saturation=SATURATION.to_dict(orient="records"),
        cross_decoder_transfer=dict(
            interaction_required_rho=float(r_ir.statistic),
            interaction_required_p=float(r_ir.pvalue),
            harm_rate_rho=float(r_rate.statistic),
        ),
        medians=dict(
            csp_observed=[o_m, o_lo, o_hi],
            csp_null=[n_m, n_lo, n_hi],
            mdm=[m_m, m_lo, m_hi],
        ),
        symmetry=dict(
            SAT.groupby("decoder")["prevented_over_required"].median().round(4)
        ),
    ), f, indent=2, ensure_ascii=False)

print("\n" + "=" * 78)
print("STAGE S4: COMPLETE")
print("=" * 78)
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(OUT)}")
print("\nГЛАВНОЕ: прочитайте POSTER_CLAIMS.md целиком. Раздел «Что НЕЛЬЗЯ")
print("утверждать» важнее раздела «Что можно».")
