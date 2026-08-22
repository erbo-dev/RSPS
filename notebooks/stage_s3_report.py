# ============================================================================
# SPIKESHIELD STAGE S3 — FINAL TABLES, BOOTSTRAP, FIGURES
#
# Сводит замороженный ледгер (Stage 3F / 5 / 6) с новыми S1 и S2,
# считает субъектный и многоуровневый бутстрап и рисует стендовые фигуры.
#
# Работает и без S1/S2: то, чего нет, просто не рисуется.
# Kaggle: один cell. ~5 минут.
# ============================================================================

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path("/kaggle/working/spikeshield_stage_s")
OUT = ROOT / "S3"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SEED = 20260813
SUBJECTS = list(range(1, 10))

# ============================================================================
# 1. ЗАМОРОЖЕННЫЙ ЛЕДГЕР (handoff 23.3 / 23.4 / 23.5) — НЕ ПЕРЕСЧИТЫВАЕТСЯ
# ============================================================================

LEDGER = pd.DataFrame([
    # subject, stage, interaction_required, no_int_recall, full_harm, clean_acc
    (1, "Stage3F_dev",  0.028911, 0.971089, None,  None),
    (2, "Stage3F_dev",  0.051595, 0.948405, None,  None),
    (3, "Stage3F_dev",  0.021261, 0.978739, None,  None),
    (4, "Stage3F_dev",  0.028975, 0.971025, None,  None),
    (5, "Stage3F_dev",  0.039841, 0.960159, None,  None),
    (6, "Stage3F_dev",  0.006686, 0.993314, None,  None),
    (7, "Stage3F_dev",  0.023362, 0.976638, None,  None),
    (8, "Stage3F_dev",  0.038094, 0.961906, None,  None),
    (9, "Stage3F_dev",  0.004302, 0.995698, None,  None),

    (1, "Stage5_locked", 0.032307, 0.967693, 21203, None),
    (2, "Stage5_locked", 0.064775, 0.935225, 15191, None),
    (3, "Stage5_locked", 0.021107, 0.978893, 22173, None),
    (4, "Stage5_locked", 0.038749, 0.961251, 22246, None),
    (5, "Stage5_locked", 0.054052, 0.945948, 16854, None),
    (6, "Stage5_locked", 0.003454, 0.996546, 26639, None),
    (7, "Stage5_locked", 0.015123, 0.984877, 24665, None),
    (8, "Stage5_locked", 0.025811, 0.974189, 18752, None),
    (9, "Stage5_locked", 0.010784, 0.989216, 26707, None),

    (1, "Stage6_session", 0.041716, 0.958284, 18482, 0.840278),
    (2, "Stage6_session", 0.071295, 0.928705, 15527, 0.513889),
    (3, "Stage6_session", 0.017521, 0.982479, 19919, 0.840278),
    (4, "Stage6_session", 0.074511, 0.925489, 15528, 0.680556),
    (5, "Stage6_session", 0.049901, 0.950099, 19739, 0.513889),
    (6, "Stage6_session", 0.001956, 0.998044, 23517, 0.638889),
    (7, "Stage6_session", 0.011134, 0.988866,   988, 0.506944),
    (8, "Stage6_session", 0.032084, 0.967916, 17080, 0.958333),
    (9, "Stage6_session", 0.013725, 0.986275, 29580, 0.854167),
], columns=["bci_subject", "stage", "interaction_required", "no_int_recall",
            "full_harm", "clean_acc"])

# ============================================================================
# 2. ПОДТЯГИВАЕМ S1 И S2, ЕСЛИ ЕСТЬ
# ============================================================================

def load_csv(p):
    p = Path(p)
    return pd.read_csv(p) if p.exists() else None


S1_SUB = load_csv(ROOT / "S1" / "S1_SUBJECT_SUMMARY.csv")
S1_SNR = load_csv(ROOT / "S1" / "S1_BY_SNR.csv")
S1_ART = load_csv(ROOT / "S1" / "S1_BY_ARTIFACT_SUBJECT.csv")
S1_PC = load_csv(ROOT / "S1" / "S1_POSITIVE_CONTROL.csv")
S2_SUB = load_csv(ROOT / "S2" / "S2_SUBJECT_SUMMARY.csv")
S2_SNR = load_csv(ROOT / "S2" / "S2_BY_SNR.csv")
S2_ART = load_csv(ROOT / "S2" / "S2_BY_ARTIFACT_SUBJECT.csv")

print("loaded:",
      "S1" if S1_SUB is not None else "-",
      "S2" if S2_SUB is not None else "-")

rows = [LEDGER]

if S1_SUB is not None:
    obs = S1_SUB[S1_SUB.arm == "observed"].copy()
    rows.append(pd.DataFrame(dict(
        bci_subject=obs.bci_subject,
        stage="S1_observed",
        interaction_required=obs.interaction_required_harm_fraction,
        no_int_recall=obs.no_interaction_harm_recall,
        full_harm=obs.full_harm,
        clean_acc=np.nan,
    )))
    nul = (S1_SUB[S1_SUB.arm.str.startswith("null")]
           .groupby("bci_subject")
           .agg(interaction_required=("interaction_required_harm_fraction", "mean"),
                no_int_recall=("no_interaction_harm_recall", "mean"),
                full_harm=("full_harm", "mean"))
           .reset_index())
    nul["stage"] = "S1_null"
    nul["clean_acc"] = np.nan
    rows.append(nul[["bci_subject", "stage", "interaction_required",
                     "no_int_recall", "full_harm", "clean_acc"]])

if S2_SUB is not None:
    rows.append(pd.DataFrame(dict(
        bci_subject=S2_SUB.bci_subject,
        stage="S2_MDM",
        interaction_required=S2_SUB.interaction_required_harm_fraction,
        no_int_recall=S2_SUB.no_interaction_harm_recall,
        full_harm=S2_SUB.full_harm,
        clean_acc=S2_SUB.clean_mdm_oof_accuracy,
    )))

MASTER = pd.concat(rows, ignore_index=True)
MASTER.to_csv(OUT / "MASTER_SUBJECT_TABLE.csv", index=False)

# ============================================================================
# 3. БУТСТРАП
# ============================================================================

def subject_bootstrap(values, n_boot=20000, seed=SEED, stat=np.median):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(n=0, point=np.nan, lo=np.nan, hi=np.nan)
    rg = np.random.default_rng(seed)
    idx = rg.integers(0, len(v), size=(n_boot, len(v)))
    draws = stat(v[idx], axis=1)
    return dict(n=int(len(v)), point=float(stat(v)),
                lo=float(np.quantile(draws, 0.025)),
                hi=float(np.quantile(draws, 0.975)))


boot_rows = []
for stage, g in MASTER.groupby("stage"):
    b = subject_bootstrap(g["interaction_required"].to_numpy(),
                          seed=SEED + abs(hash(stage)) % 10000)
    boot_rows.append(dict(stage=stage, metric="interaction_required", **b))
    b2 = subject_bootstrap(g["no_int_recall"].to_numpy(),
                           seed=SEED + 1 + abs(hash(stage)) % 10000)
    boot_rows.append(dict(stage=stage, metric="no_interaction_recall", **b2))

BOOT = pd.DataFrame(boot_rows)
BOOT.to_csv(OUT / "SUBJECT_BOOTSTRAP.csv", index=False)

print("\n" + "=" * 78)
print("SUBJECT-LEVEL BOOTSTRAP (median, 95% CI)")
print("=" * 78)
print(BOOT.to_string(index=False))


def multiway_bootstrap(df, n_boot=5000, seed=SEED):
    """
    df: колонки bci_subject, artifact_subject, full_harm, interaction_required_harm.
    Ресемплим независимо BCI-субъектов и артефактных субъектов,
    пересобираем медиану по субъектам из счётчиков.
    """
    bci = np.sort(df.bci_subject.unique())
    art = np.sort(df.artifact_subject.unique())
    piv_f = df.pivot_table(index="bci_subject", columns="artifact_subject",
                           values="full_harm", aggfunc="sum", fill_value=0).reindex(
                               index=bci, columns=art, fill_value=0).to_numpy(float)
    piv_r = df.pivot_table(index="bci_subject", columns="artifact_subject",
                           values="interaction_required_harm", aggfunc="sum",
                           fill_value=0).reindex(
                               index=bci, columns=art, fill_value=0).to_numpy(float)
    rg = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        bi = rg.integers(0, len(bci), len(bci))
        ai = rg.integers(0, len(art), len(art))
        f = piv_f[np.ix_(bi, ai)].sum(axis=1)
        r = piv_r[np.ix_(bi, ai)].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(f > 0, r / f, np.nan)
        out[b] = np.nanmedian(frac)
    point = float(np.nanmedian(
        np.where(piv_f.sum(1) > 0, piv_r.sum(1) / np.maximum(piv_f.sum(1), 1), np.nan)))
    return dict(point=point, lo=float(np.nanquantile(out, 0.025)),
                hi=float(np.nanquantile(out, 0.975)), draws=out)


mw_rows = []
for name, d in [("S1_observed", S1_ART[S1_ART.arm == "observed"] if S1_ART is not None else None),
                ("S2_MDM", S2_ART)]:
    if d is None or len(d) == 0:
        continue
    res = multiway_bootstrap(d[["bci_subject", "artifact_subject",
                                "full_harm", "interaction_required_harm"]])
    mw_rows.append(dict(source=name, point=res["point"], lo=res["lo"], hi=res["hi"]))
    np.save(OUT / f"multiway_draws_{name}.npy", res["draws"])

if mw_rows:
    MW = pd.DataFrame(mw_rows)
    MW.to_csv(OUT / "MULTIWAY_BOOTSTRAP.csv", index=False)
    print("\n" + "=" * 78)
    print("MULTIWAY BOOTSTRAP (BCI subjects x artifact subjects)")
    print("=" * 78)
    print(MW.to_string(index=False))

# ============================================================================
# 4. ФИГУРЫ
# ============================================================================

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 300, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})

STAGE_ORDER = ["Stage3F_dev", "Stage5_locked", "Stage6_session",
               "S1_observed", "S1_null", "S2_MDM"]
STAGES = [s for s in STAGE_ORDER if s in set(MASTER.stage)]


# --- Fig 1: interaction-required по субъектам и стадиям ---
fig, ax = plt.subplots(figsize=(9, 4.5))
for i, st in enumerate(STAGES):
    g = MASTER[MASTER.stage == st].sort_values("bci_subject")
    xs = np.full(len(g), i) + np.linspace(-0.16, 0.16, len(g))
    ax.scatter(xs, g["interaction_required"], s=26, zorder=3)
    for x, yv, sid in zip(xs, g["interaction_required"], g["bci_subject"]):
        if np.isfinite(yv):
            ax.annotate(str(int(sid)), (x, yv), fontsize=5.5,
                        xytext=(0, 5), textcoords="offset points", ha="center")
    b = BOOT[(BOOT.stage == st) & (BOOT.metric == "interaction_required")].iloc[0]
    ax.errorbar(i, b["point"], yerr=[[b["point"] - b["lo"]], [b["hi"] - b["point"]]],
                fmt="_", markersize=26, capsize=5, lw=1.6, color="black", zorder=4)
ax.axhline(0.05, ls="--", lw=0.9, color="grey")
ax.text(len(STAGES) - 0.45, 0.0515, "5%", fontsize=7, color="grey")
ax.set_xticks(range(len(STAGES)))
ax.set_xticklabels(STAGES, rotation=18, ha="right")
ax.set_ylabel("interaction-required HARM fraction")
ax.set_title("Перекрёстный член вторичен: все 9 субъектов, медиана и 95% субъектный интервал")
fig.tight_layout()
fig.savefig(FIG / "fig1_interaction_required_by_stage.png")
plt.close(fig)

# --- Fig 2: тренд по SNR ---
snr_sets = []
if S1_SNR is not None:
    for arm in ["observed", "null0"]:
        d = S1_SNR[S1_SNR.arm == arm]
        if len(d):
            snr_sets.append((f"S1 {arm}", d.groupby("snr_db")[
                "interaction_required_harm_fraction"].median()))
if S2_SNR is not None and len(S2_SNR):
    snr_sets.append(("S2 MDM", S2_SNR.groupby("snr_db")[
        "interaction_required_harm_fraction"].median()))
# замороженные значения Stage 3F и Stage 6 из handoff
snr_sets.append(("Stage3F dev (frozen)", pd.Series(
    {2.0: 0.05254, 0.0: 0.041785, -2.0: 0.03277, -3.0: 0.029986,
     -4.0: 0.026096, -5.0: 0.021727, -6.0: 0.017235, -7.0: 0.015023})))
snr_sets.append(("Stage6 (frozen)", pd.Series(
    {2.0: 0.039421, 0.0: 0.033801, -2.0: 0.034061, -3.0: 0.033593,
     -4.0: 0.033898, -5.0: 0.029967, -6.0: 0.029270, -7.0: 0.025684})))

fig, ax = plt.subplots(figsize=(7, 4))
for label, ser in snr_sets:
    ser = ser.sort_index(ascending=False)
    ax.plot(ser.index, ser.values, marker="o", ms=4, lw=1.4, label=label)
ax.invert_xaxis()
ax.set_xlabel("SNR донора, дБ (правее = сильнее контаминация)")
ax.set_ylabel("median interaction-required")
ax.set_title("Доля interaction-required падает с ростом мощности артефакта")
ax.legend(fontsize=7.5)
fig.tight_layout()
fig.savefig(FIG / "fig2_snr_trend.png")
plt.close(fig)

# --- Fig 3: S1 observed vs null, парно ---
if S1_SUB is not None:
    obs = (S1_SUB[S1_SUB.arm == "observed"].set_index("bci_subject")
           ["interaction_required_harm_fraction"])
    nul = (S1_SUB[S1_SUB.arm.str.startswith("null")]
           .groupby("bci_subject")["interaction_required_harm_fraction"].mean())
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for sid in obs.index:
        axes[0].plot([0, 1], [obs[sid], nul[sid]], marker="o", ms=4, lw=1.1)
        axes[0].annotate(str(int(sid)), (0, obs[sid]), fontsize=6,
                         xytext=(-11, -2), textcoords="offset points")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["наблюдаемые\nдоноры", "фазово-рандом.\nсуррогаты"])
    axes[0].set_ylabel("interaction-required")
    axes[0].set_title("Парное сравнение по субъектам")

    delta = (obs - nul).to_numpy()
    b = subject_bootstrap(delta, seed=SEED + 21)
    axes[1].axvline(0, color="grey", lw=1, ls="--")
    axes[1].scatter(delta, np.arange(1, len(delta) + 1), s=30, zorder=3)
    axes[1].errorbar(b["point"], 0.2,
                     xerr=[[b["point"] - b["lo"]], [b["hi"] - b["point"]]],
                     fmt="D", ms=6, capsize=4, color="black")
    axes[1].set_yticks(list(range(1, len(delta) + 1)) + [0.2])
    axes[1].set_yticklabels([f"S{i}" for i in obs.index] + ["медиана"])
    axes[1].set_xlabel("observed − null")
    axes[1].set_title("Разность и 95% субъектный интервал")
    fig.suptitle("Калибровка перекрёстного члена относительно независимой контаминации",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_s1_observed_vs_null.png")
    plt.close(fig)

# --- Fig 4: позитивный контроль ---
if S1_PC is not None and len(S1_PC):
    g = S1_PC.groupby("beta").agg(
        ir=("interaction_required_harm_fraction", "median"),
        rho=("abs_rho_median", "median")).reset_index()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(g.beta, g.ir, marker="o", lw=1.6, label="interaction-required")
    ax2 = ax.twinx()
    ax2.plot(g.beta, g.rho, marker="s", lw=1.2, ls="--", color="darkorange",
             label="median |rho|")
    ax.set_xlabel("beta (сила принудительного выравнивания a' = a + beta·x)")
    ax.set_ylabel("interaction-required")
    ax2.set_ylabel("median |rho|")
    ax.set_title("Позитивный контроль: измеритель реагирует на реальное взаимодействие")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_positive_control.png")
    plt.close(fig)

# --- Fig 5: слабое место Stage 6 (чистая точность против объёма HARM) ---
s6 = LEDGER[LEDGER.stage == "Stage6_session"]
fig, ax = plt.subplots(figsize=(6.2, 4.2))
sc = ax.scatter(s6.clean_acc, s6.interaction_required,
                s=np.sqrt(s6.full_harm.astype(float)) * 1.6, alpha=0.75, zorder=3)
for _, r in s6.iterrows():
    ax.annotate(f"S{int(r.bci_subject)}", (r.clean_acc, r.interaction_required),
                fontsize=7, xytext=(6, 3), textcoords="offset points")
ax.axvline(0.60, ls="--", color="crimson", lw=1.1)
ax.text(0.605, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
        "порог валидности 0.60", fontsize=7, color="crimson")
ax.axhline(0.05, ls=":", color="grey", lw=1)
ax.set_xlabel("чистая cross-session accuracy (Stage 6)")
ax.set_ylabel("interaction-required")
ax.set_title("Ограничение Stage 6: часть субъектов около случайного уровня\n"
             "(размер точки ~ число событий HARM)")
fig.tight_layout()
fig.savefig(FIG / "fig5_stage6_clean_accuracy_limitation.png")
plt.close(fig)

# --- Fig 6: многоуровневый бутстрап ---
draw_files = sorted(OUT.glob("multiway_draws_*.npy"))
if draw_files:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for p in draw_files:
        d = np.load(p)
        d = d[np.isfinite(d)]
        ax.hist(d, bins=60, alpha=0.6, label=p.stem.replace("multiway_draws_", ""))
    ax.set_xlabel("медиана interaction-required по субъектам")
    ax.set_ylabel("частота (bootstrap)")
    ax.set_title("Многоуровневый бутстрап: BCI-субъекты × артефактные субъекты")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_multiway_bootstrap.png")
    plt.close(fig)

# ============================================================================
# 5. ТЕКСТОВАЯ СВОДКА ДЛЯ ТЕЗИСОВ
# ============================================================================

def fmt(stage, metric="interaction_required"):
    r = BOOT[(BOOT.stage == stage) & (BOOT.metric == metric)]
    if len(r) == 0:
        return "н/д"
    r = r.iloc[0]
    return f"{r['point']:.4f} (95% CI {r['lo']:.4f}–{r['hi']:.4f}, n={int(r['n'])})"


lines = [
    "SPIKESHIELD — сводка для тезисов",
    "=" * 60,
    "",
    "Первичная величина: доля событий HARM, для которых точный перекрёстный",
    "член clean x artifact был НЕОБХОДИМ, чтобы решение перевернулось.",
    "Статистическая единица: BCI-субъект (N=9). Медиана и 95% интервал",
    "получены субъектным бутстрапом, а не по числу сгенерированных пар.",
    "",
]
for st in STAGES:
    lines.append(f"  {st:16s} interaction-required = {fmt(st)}")
lines.append("")
for st in STAGES:
    lines.append(f"  {st:16s} no-interaction recall = {fmt(st, 'no_interaction_recall')}")
lines += [
    "",
    "Ограничения, которые обязаны быть на стенде:",
    "  - N=9 субъектов, один класс артефакта (жевание, компонента 20-80 Гц),",
    "    бинарная задача левая/правая рука, полоса декодера 8-32 Гц.",
    "  - миллионы контрфактических пар это вычислительные реализации,",
    "    вложенные в малое число людей, а не независимые эксперименты.",
    "  - NO-INTERACTION не равно ARTIFACT-ONLY: убирается только",
    "    перекрёстный член, вклад чистого сигнала и артефакта остаётся.",
    "  - в Stage 6 у S2, S5 и S7 чистая точность около 51%, поэтому",
    "    их вклад в утверждение слабый; см. рис. 5.",
    "  - защитный гейт не превзошёл уверенность декодера. Это опубликованный",
    "    отрицательный результат, а не скрытая неудача.",
]

if (ROOT / "S1" / "S1_RESULT.json").exists():
    with open(ROOT / "S1" / "S1_RESULT.json") as f:
        r1 = json.load(f)
    lines += ["", f"S1 вердикт: {r1['verdict']}", f"  {r1['reading']}"]
if (ROOT / "S2" / "S2_RESULT.json").exists():
    with open(ROOT / "S2" / "S2_RESULT.json") as f:
        r2 = json.load(f)
    lines += ["", f"S2 вердикт: {r2['verdict']}", f"  {r2['reading']}"]

summary = "\n".join(lines)
(OUT / "ABSTRACT_NUMBERS.txt").write_text(summary, encoding="utf-8")
print("\n" + summary)

print("\n" + "=" * 78)
print("STAGE S3: COMPLETE")
print("=" * 78)
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(OUT)}")
