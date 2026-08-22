# ============================================================================
# SPIKESHIELD STAGE S5 — ARTIFACT-ONLY CONTROL + INDEPENDENCE MODEL
#
# Запускается после S0/S1. Новых данных не читает, только пересчитывает
# уже сохранённый контекст. Ветка CSP+LDA (первичная).
#
# ЧАСТЬ A — контроль "только артефакт".
#   В handoff трижды написано NO-INTERACTION != ARTIFACT-ONLY, но сам
#   эксперимент никогда не проводился на уровне отдельных событий HARM.
#   Здесь декодеру подаётся ТОЛЬКО масштабированный артефакт, без ЭЭГ:
#   мощность = lambda^2 * p_art. Закрывает главное возражение читателя:
#   "раз перекрёстный член не нужен, значит решает сам артефакт?".
#
# ЧАСТЬ B — модель независимости (POST-HOC, разведочная).
#   m_full = m_noint + I, где I точный эффект перекрёстного члена на счёт.
#   Если I не зависит от m_noint, то доля interaction-required полностью
#   определяется двумя маргинальными распределениями.
#   Первичная проверка: перемешивание I между событиями. Если перемешанный
#   результат совпадает с наблюдаемым, независимость подтверждена.
#   Вторичная: замкнутая формула f_m(0)*E[(-I)+] / P(m<0), ноль подгоночных
#   параметров.
#
# ВАЖНО: часть B придумана ПОСЛЕ просмотра результата S1, поэтому она
# помечается как post-hoc и не является предрегистрированной. Первичная
# величина проекта остаётся замороженной.
#
# Kaggle: один cell, ~3-5 минут.
# ============================================================================

RUN_PART_B = True          # выключите, если нужна только часть A

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

warnings.filterwarnings("ignore")
import joblib

# ============================================================================
# 0. КОНТЕКСТ
# ============================================================================

CAND = [Path("/kaggle/working/spikeshield_stage_s")]
if Path("/kaggle/input").exists():
    CAND += sorted(Path("/kaggle/input").glob("*"))
    CAND += sorted(Path("/kaggle/input").glob("*/*"))
    CAND += sorted(Path("/kaggle/input").glob("*/*/*"))

ROOT = next((c for c in CAND if (c / "context" / "context_manifest.json").exists()), None)
if ROOT is None:
    raise RuntimeError("Контекст S0 не найден (ищется папка context/).")

CTX = ROOT / "context"
OUT = Path("/kaggle/working/spikeshield_stage_s/S5")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MANIFEST = json.loads((CTX / "context_manifest.json").read_text())
SEED = int(MANIFEST["seed"])
SNR_GRID_DB = np.array(MANIFEST["snr_grid_db"], dtype=np.float64)
SUBJECTS = list(range(1, 10))
S = len(SNR_GRID_DB)

print("=" * 78)
print("STAGE S5 — ARTIFACT-ONLY CONTROL + INDEPENDENCE MODEL")
print("=" * 78)
print(f"context: {CTX}")

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
META = pd.read_csv(CTX / "bank_metadata_dev.csv")
models_oof = joblib.load(CTX / "models_oof.joblib")

donor_safety_rms = META["safety_rms_uv_recomputed"].to_numpy(dtype=np.float64)
artifact_subject = META["artifact_subject"].to_numpy(dtype=np.int64)
D = BANK_DEC.shape[0]
print(f"donors D={D}, SNR levels S={S}")

# замороженные значения Stage 3F для регрессионной сверки
LEDGER_3F = {
    1: 0.028911, 2: 0.051595, 3: 0.021261, 4: 0.028975, 5: 0.039841,
    6: 0.006686, 7: 0.023362, 8: 0.038094, 9: 0.004302,
}

# ============================================================================
# 1. ЯДРО
# ============================================================================


def csp_terms(csp, clean, art):
    k = int(csp.n_components)
    W = np.asarray(csp.filters_[:k], dtype=np.float64)
    z = np.einsum("kc,nct->nkt", W, clean, optimize=True)
    u = np.einsum("kc,dct->dkt", W, art, optimize=True)
    T = z.shape[-1]
    p_clean = np.mean(z ** 2, axis=-1)
    p_art = np.mean(u ** 2, axis=-1)
    cross = np.einsum("nkt,dkt->ndk", z, u, optimize=True) / T
    return p_clean, p_art, cross


def score_right(lda, power):
    power = np.maximum(power, np.finfo(np.float64).tiny)
    f = np.log(power)
    raw = np.asarray(lda.decision_function(f.reshape(-1, f.shape[-1]))).reshape(-1)
    cl = np.asarray(lda.classes_)
    sr = raw if cl[1] == "right_hand" else -raw
    return sr.reshape(f.shape[:-1])


# ============================================================================
# 2. ГЛАВНЫЙ ЦИКЛ
# ============================================================================

t0 = time.time()
rows_a, rows_b, rows_snr_a = [], [], []
diag = []

for s in SUBJECTS:
    X_dec, X_saf = data19[s]["X_dec"], data19[s]["X_saf"]
    y, runs = data19[s]["y"], data19[s]["runs"]

    # --- A: счётчики ---
    n_harm = 0
    n_harm_art_same = 0        # артефакт один даёт ТУ ЖЕ неверную метку
    n_events_elig = 0
    n_art_correct = 0          # артефакт один угадывает истинную метку
    n_art_right = 0            # артефакт один говорит "right"
    n_harm_true_right = 0
    n_harm_true_left = 0
    a_snr = {j: dict(harm=0, same=0) for j in range(S)}

    # --- B: накопители ---
    m_parts, I_parts = [], []

    # --- регрессионная сверка ---
    c_full = c_req = c_prev = 0

    for r in sorted(set(runs)):
        val, fit = runs == r, runs != r
        model = models_oof[s][r]
        csp = model.named_steps["csp"]
        lda = model.named_steps["lineardiscriminantanalysis"]

        Xc, yv = X_dec[val], y[val]
        y_sign = np.where(yv == "right_hand", 1.0, -1.0)

        p_clean, p_art, cross = csp_terms(csp, Xc, BANK_DEC)
        sr_clean = score_right(lda, p_clean)
        clean_margin = y_sign * sr_clean
        clean_correct = clean_margin > 0.0

        ref_rms = float(np.median(np.sqrt(np.mean(X_saf[fit] ** 2, axis=(1, 2)))))
        lam = ref_rms / (donor_safety_rms[:, None] * (10.0 ** (SNR_GRID_DB[None, :] / 20.0)))

        pc = p_clean[:, None, None, :]
        pa = p_art[None, :, None, :]
        cr = cross[:, :, None, :]
        lm = lam[None, :, :, None]

        sr_full = score_right(lda, pc + 2.0 * lm * cr + lm ** 2 * pa)
        sr_noint = score_right(lda, pc + lm ** 2 * pa)

        # --- ТОЛЬКО АРТЕФАКТ: чистый сигнал полностью удалён ---
        # мощность = lambda^2 * p_art, не зависит от трайла -> форма (D,S)
        sr_art = score_right(lda, (lam[:, :, None] ** 2) * p_art[:, None, :])
        pred_art_right = sr_art > 0.0                       # (D,S)

        m_full = y_sign[:, None, None] * sr_full
        m_noint = y_sign[:, None, None] * sr_noint
        elig = clean_correct[:, None, None]

        full_harm = elig & (m_full < 0.0)
        noint_harm = elig & (m_noint < 0.0)
        c_full += int(full_harm.sum())
        c_req += int((full_harm & ~noint_harm).sum())
        c_prev += int(((~full_harm) & noint_harm & elig).sum())

        # ---- ЧАСТЬ A ----
        # решение полной контаминации на событиях HARM всегда неверно,
        # значит "та же метка" = артефакт один тоже указывает не на истину
        true_right = (yv == "right_hand")[:, None, None]
        art_same_as_full = (pred_art_right[None, :, :] != true_right)
        n_harm += int(full_harm.sum())
        n_harm_art_same += int((full_harm & art_same_as_full).sum())
        n_events_elig += int(np.broadcast_to(elig, full_harm.shape).sum())
        n_art_correct += int(
            (np.broadcast_to(elig, full_harm.shape) & ~art_same_as_full).sum()
        )
        n_art_right += int(
            (np.broadcast_to(elig, full_harm.shape)
             & np.broadcast_to(pred_art_right[None, :, :], full_harm.shape)).sum()
        )
        n_harm_true_right += int((full_harm & true_right).sum())
        n_harm_true_left += int((full_harm & ~true_right).sum())
        for j in range(S):
            a_snr[j]["harm"] += int(full_harm[:, :, j].sum())
            a_snr[j]["same"] += int((full_harm & art_same_as_full)[:, :, j].sum())

        # ---- ЧАСТЬ B ----
        if RUN_PART_B:
            mask = np.broadcast_to(elig, m_noint.shape)
            m_parts.append(m_noint[mask].astype(np.float32))
            I_parts.append((m_full - m_noint)[mask].astype(np.float32))

        del p_clean, p_art, cross, sr_full, sr_noint, m_full, m_noint

    ir_frac = c_req / c_full if c_full else np.nan
    delta_led = ir_frac - LEDGER_3F[s]

    rows_a.append(dict(
        bci_subject=s,
        eligible_events=n_events_elig,
        full_harm=n_harm,
        harm_artifact_only_same_label=n_harm_art_same,
        harm_artifact_only_agreement=n_harm_art_same / n_harm if n_harm else np.nan,
        artifact_only_accuracy=n_art_correct / n_events_elig if n_events_elig else np.nan,
        artifact_only_right_fraction=n_art_right / n_events_elig if n_events_elig else np.nan,
        harm_true_right=n_harm_true_right,
        harm_true_left=n_harm_true_left,
        harm_frac_true_right=(
            n_harm_true_right / (n_harm_true_right + n_harm_true_left)
            if (n_harm_true_right + n_harm_true_left) else np.nan
        ),
        interaction_required_fraction=ir_frac,
        ledger_3f=LEDGER_3F[s],
        ledger_delta=delta_led,
    ))
    for j in range(S):
        rows_snr_a.append(dict(
            bci_subject=s, snr_db=float(SNR_GRID_DB[j]),
            full_harm=a_snr[j]["harm"],
            artifact_only_agreement=(
                a_snr[j]["same"] / a_snr[j]["harm"] if a_snr[j]["harm"] else np.nan
            ),
        ))

    if RUN_PART_B:
        m = np.concatenate(m_parts).astype(np.float64)
        I = np.concatenate(I_parts).astype(np.float64)
        del m_parts, I_parts

        req_obs = int(np.sum((m > 0) & (m + I < 0)))
        prev_obs = int(np.sum((m < 0) & (m + I > 0)))
        full_obs = int(np.sum(m + I < 0))

        # --- первичная проверка: перемешивание I между событиями ---
        rg = np.random.default_rng(SEED + s)
        req_sh, frac_sh = [], []
        for _ in range(20):
            Ish = rg.permutation(I)
            rq = int(np.sum((m > 0) & (m + Ish < 0)))
            fl = int(np.sum(m + Ish < 0))
            req_sh.append(rq)
            frac_sh.append(rq / fl if fl else np.nan)
        frac_sh = np.array(frac_sh)

        # --- вторичная: замкнутая формула, ноль подгоночных параметров ---
        p_neg = float(np.mean(m < 0))
        E_I_neg = float(np.mean(np.maximum(-I, 0.0)))
        preds = {}
        for q in (0.01, 0.02, 0.05):
            h = float(np.quantile(np.abs(m), q))
            f0 = float(np.mean(np.abs(m) < h) / (2.0 * h))
            preds[q] = f0 * E_I_neg / p_neg if p_neg > 0 else np.nan

        rho_indep = spearmanr(np.abs(m), np.abs(I))
        rows_b.append(dict(
            bci_subject=s,
            n_events=len(m),
            observed_fraction=req_obs / full_obs if full_obs else np.nan,
            shuffled_fraction_mean=float(np.nanmean(frac_sh)),
            shuffled_fraction_sd=float(np.nanstd(frac_sh)),
            closed_form_q01=preds[0.01],
            closed_form_q02=preds[0.02],
            closed_form_q05=preds[0.05],
            P_m_negative=p_neg,
            E_I_negative_part=E_I_neg,
            I_mean=float(np.mean(I)),
            I_sd=float(np.std(I)),
            abs_corr_m_I_rho=float(rho_indep.statistic),
            abs_corr_m_I_p=float(rho_indep.pvalue),
            observed_prevented=prev_obs,
            observed_required=req_obs,
        ))
        del m, I

    diag.append(dict(bci_subject=s, seconds=time.time() - t0))
    a = rows_a[-1]
    msg = (f"  S{s}: HARM={a['full_harm']:>7}  "
           f"artifact-only agreement={a['harm_artifact_only_agreement']:.4f}  "
           f"IR={ir_frac:.6f} (ledger delta {delta_led:+.1e})")
    if RUN_PART_B:
        b = rows_b[-1]
        msg += f"  | shuffled={b['shuffled_fraction_mean']:.6f}"
    print(msg)

A = pd.DataFrame(rows_a)
A_SNR = pd.DataFrame(rows_snr_a)
B = pd.DataFrame(rows_b) if RUN_PART_B else pd.DataFrame()

# ============================================================================
# 3. РЕГРЕССИОННАЯ СВЕРКА
# ============================================================================

max_delta = float(A["ledger_delta"].abs().max())
print("\n" + "=" * 78)
print("РЕГРЕССИОННАЯ СВЕРКА С ЗАМОРОЖЕННЫМ Stage 3F")
print("=" * 78)
print(f"  max |delta| = {max_delta:.2e}   -> "
      f"{'PASS' if max_delta < 1e-5 else 'РАСХОЖДЕНИЕ, СТОП'}")
if max_delta >= 1e-5:
    raise RuntimeError("S5 не воспроизвёл Stage 3F. Не интерпретируйте результат.")

# ============================================================================
# 4. ЧАСТЬ A — ИНТЕРПРЕТАЦИЯ
# ============================================================================

print("\n" + "=" * 78)
print("ЧАСТЬ A — КОНТРОЛЬ 'ТОЛЬКО АРТЕФАКТ'")
print("=" * 78)
print(A[[
    "bci_subject", "full_harm", "harm_frac_true_right",
    "artifact_only_right_fraction", "artifact_only_accuracy",
    "harm_artifact_only_agreement",
]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

agree_med = float(A["harm_artifact_only_agreement"].median())
acc_med = float(A["artifact_only_accuracy"].median())
right_med = float(A["artifact_only_right_fraction"].median())
acc_dev = float((A["artifact_only_accuracy"] - 0.5).abs().max())
bias_lo = float(A["artifact_only_right_fraction"].min())
bias_hi = float(A["artifact_only_right_fraction"].max())

# ключевая проверка на циркулярность метрики согласия:
# если артефакт толкает декодер в фиксированном направлении, то HARM обязан
# концентрироваться в противоположном истинном классе, и высокое "согласие"
# получается механически, а не как самостоятельное доказательство
rho_push = spearmanr(A["harm_frac_true_right"], A["artifact_only_right_fraction"])

print(f"\n  точность артефакта одного, медиана : {acc_med:.4f}  "
      f"(случайный уровень 0.5, макс отклонение {acc_dev:.4f})")
print(f"  доля ответов 'right', диапазон     : [{bias_lo:.4f}, {bias_hi:.4f}]  "
      f"(нейтральный уровень 0.5)")
print(f"  согласие на событиях HARM, медиана : {agree_med:.4f}")
print(f"  Spearman(доля HARM с истиной right, смещение артефакта) = "
      f"{rho_push.statistic:+.3f}  p = {rho_push.pvalue:.4f}")

if acc_dev <= 0.05:
    verdict_a = "ARTIFACT_CLASS_UNINFORMATIVE_BUT_DIRECTIONAL"
    read_a = (
        "ГЛАВНОЕ: реальный жевательный артефакт, поданный один, распознаётся "
        f"на уровне случайности ({acc_med:.3f}), то есть НЕ несёт информации о "
        "воображаемом классе. Это тот самый контроль, который провалила "
        "аннулированная ветка синтетического EMG, где придуманные топографии "
        "давали класс с экстремальной уверенностью.\n"
        "  ОДНАКО артефакт навязывает декодеру субъект-специфичное фиксированное "
        f"направление сдвига (доля ответов right от {bias_lo:.3f} до {bias_hi:.3f}), "
        "и события HARM концентрируются в противоположном истинном классе "
        f"(Spearman {rho_push.statistic:+.3f}).\n"
        f"  ПОЭТОМУ цифру согласия {agree_med:.3f} НЕЛЬЗЯ читать как "
        "'артефакт один решает исход'. Она в значительной мере является "
        "механическим следствием общего направления сдвига, а не независимым "
        "доказательством. Это ограничение метрики, и его надо назвать вслух."
    )
else:
    verdict_a = "ARTIFACT_CARRIES_CLASS_INFORMATION"
    read_a = (
        f"ТРЕВОГА: артефакт один даёт точность {acc_med:.3f}, отклонение от "
        f"случайности {acc_dev:.3f}. Это признак того же дефекта, который "
        "погубил ветку синтетического EMG. Разберитесь до интерпретации."
    )
print(f"\n  ВЕРДИКТ A: {verdict_a}\n  {read_a}")

# ============================================================================
# 5. ЧАСТЬ B — ИНТЕРПРЕТАЦИЯ
# ============================================================================

verdict_b, read_b = "NOT_RUN", ""
if RUN_PART_B:
    print("\n" + "=" * 78)
    print("ЧАСТЬ B — МОДЕЛЬ НЕЗАВИСИМОСТИ (POST-HOC)")
    print("=" * 78)
    print(B[[
        "bci_subject", "observed_fraction", "shuffled_fraction_mean",
        "shuffled_fraction_sd", "closed_form_q02", "abs_corr_m_I_rho",
    ]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    rel = (B.shuffled_fraction_mean - B.observed_fraction).abs() / B.observed_fraction
    w = wilcoxon(B.observed_fraction, B.shuffled_fraction_mean)
    print(f"\n  перемешивание vs наблюдение: медианное отн. расхождение "
          f"{rel.median()*100:.2f}%, макс {rel.max()*100:.2f}%, Wilcoxon p={w.pvalue:.4f}")
    rel_cf = (B.closed_form_q02 - B.observed_fraction).abs() / B.observed_fraction
    print(f"  замкнутая формула vs наблюдение: медианное отн. расхождение "
          f"{rel_cf.median()*100:.1f}%, макс {rel_cf.max()*100:.1f}%")
    print(f"  |m| vs |I|, Spearman rho: медиана {B.abs_corr_m_I_rho.median():+.3f}")

    if rel.median() < 0.05:
        verdict_b = "INDEPENDENCE_HOLDS"
        read_b = ("Перемешивание эффекта перекрёстного члена между событиями "
                  "воспроизводит наблюдаемую долю. Значит доля interaction-required "
                  "полностью определяется двумя маргинальными распределениями и не "
                  "содержит специфичной для события связи. Это независимое "
                  "подтверждение вердикта S1, полученное третьим способом.")
    elif rel.median() < 0.20:
        verdict_b = "INDEPENDENCE_APPROXIMATE"
        read_b = ("Независимость выполняется приближённо. Докладывайте расхождение "
                  "честно, не подгоняйте.")
    else:
        verdict_b = "INDEPENDENCE_FAILS"
        read_b = ("Независимость НЕ выполняется: есть связь между margin и эффектом "
                  "перекрёстного члена. Это интересно и требует отдельного описания.")
    print(f"\n  ВЕРДИКТ B: {verdict_b}\n  {read_b}")

# ============================================================================
# 6. ФИГУРЫ
# ============================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 300, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))

ax = axes[0]
ax.bar(A.bci_subject, A.artifact_only_accuracy, width=0.6, color="#2c6fb5")
ax.axhline(0.5, ls="--", lw=1.2, color="crimson")
ax.text(9.4, 0.505, "случайный\nуровень", fontsize=7.5, color="crimson")
ax.set_xticks(A.bci_subject)
ax.set_ylim(0, 1.0)
ax.set_xlabel("испытуемый BCI")
ax.set_ylabel("точность декодера на одном артефакте")
ax.set_title("Реальное жевание НЕ несёт информации о классе\n"
             "(контроль, который провалил синтетический EMG)")

ax = axes[1]
ax.scatter(A.artifact_only_right_fraction, A.harm_frac_true_right,
           s=64, color="#c8552b", zorder=3, edgecolor="white", linewidth=0.8)
for _, r_ in A.iterrows():
    ax.annotate(f"S{int(r_.bci_subject)}",
                (r_.artifact_only_right_fraction, r_.harm_frac_true_right),
                fontsize=7.5, xytext=(6, 3), textcoords="offset points")
ax.axhline(0.5, ls=":", lw=0.9, color="grey")
ax.axvline(0.5, ls=":", lw=0.9, color="grey")
ax.set_xlabel("артефакт один: доля ответов «right»")
ax.set_ylabel("HARM: доля событий с истинным «right»")
ax.set_title("Но навязывает фиксированное направление сдвига,\n"
             f"и HARM копится в противоположном классе (ρ = {rho_push.statistic:+.2f})")
fig.tight_layout()
fig.savefig(FIG / "figS5_1_artifact_only.png")
plt.close(fig)

if RUN_PART_B:
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.scatter(B.observed_fraction, B.shuffled_fraction_mean, s=64,
               color="#2c6fb5", zorder=3, edgecolor="white", linewidth=0.8)
    for _, r_ in B.iterrows():
        ax.annotate(f"S{int(r_.bci_subject)}",
                    (r_.observed_fraction, r_.shuffled_fraction_mean),
                    fontsize=7.5, xytext=(6, 3), textcoords="offset points")
    lim = max(B.observed_fraction.max(), B.shuffled_fraction_mean.max()) * 1.15
    ax.plot([0, lim], [0, lim], ls="--", lw=1.0, color="grey")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("наблюдаемая доля interaction-required")
    ax.set_ylabel("доля после перемешивания эффекта")
    ax.set_title("Перемешивание эффекта перекрёстного члена\nвоспроизводит наблюдаемую долю")
    fig.tight_layout()
    fig.savefig(FIG / "figS5_2_shuffle_model.png")
    plt.close(fig)

# ============================================================================
# 7. СОХРАНЕНИЕ
# ============================================================================

A.to_csv(OUT / "S5_ARTIFACT_ONLY_SUBJECT.csv", index=False)
A_SNR.to_csv(OUT / "S5_ARTIFACT_ONLY_BY_SNR.csv", index=False)
if RUN_PART_B:
    B.to_csv(OUT / "S5_INDEPENDENCE_MODEL.csv", index=False)

with open(OUT / "S5_RESULT.json", "w") as f:
    json.dump(dict(
        stage="S5",
        note="Часть B post-hoc, придумана после S1, не предрегистрирована.",
        ledger_regression_max_delta=max_delta,
        part_a=dict(
            verdict=verdict_a, reading=read_a,
            median_harm_agreement=agree_med,
            median_artifact_only_accuracy=acc_med,
            median_right_fraction=right_med,
            max_accuracy_deviation_from_chance=acc_dev,
            right_fraction_range=[bias_lo, bias_hi],
            push_direction_spearman=float(rho_push.statistic),
            push_direction_p=float(rho_push.pvalue),
            per_subject=A.to_dict(orient="records"),
        ),
        part_b=(dict(
            verdict=verdict_b, reading=read_b,
            per_subject=B.to_dict(orient="records"),
        ) if RUN_PART_B else None),
        minutes=float((time.time() - t0) / 60.0),
    ), f, indent=2, ensure_ascii=False)

print("\n" + "=" * 78)
print(f"STAGE S5: COMPLETE in {(time.time()-t0)/60:.1f} min")
print("=" * 78)
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(OUT)}")
