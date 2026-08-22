"""Acceptance test: does harmkit reproduce the frozen study values?

A benchmark package that cannot reproduce the numbers it was extracted from
is worthless. This runs the real pipeline through the public API and checks
every per-subject value against the frozen ledger.

Run:
    python examples/reproduce_frozen_result.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harmkit import (ArtifactBank, HarmBenchmark, PowerDecoder,
                     artifact_only_control, phase_randomised_bank,
                     predict_interaction_required, subject_bootstrap)

# Frozen ledger, protocol/LEDGER.md
LEDGER = {1: 0.028911, 2: 0.051595, 3: 0.021261, 4: 0.028975, 5: 0.039841,
          6: 0.006686, 7: 0.023362, 8: 0.038094, 9: 0.004302}
TOL = 1e-5

CTX = Path(sys.argv[1] if len(sys.argv) > 1 else "context")
if not (CTX / "context_manifest.json").exists():
    raise SystemExit(f"context not found at {CTX}; pass its path as the first argument")

M = json.loads((CTX / "context_manifest.json").read_text())
SNR = np.array(M["snr_grid_db"], dtype=float)
Z = np.load(CTX / "clean_data19.npz", allow_pickle=True)
BANK_DEC = np.load(CTX / "bank_decoder19_dev_uV.npy").astype(np.float64)
BANK_SAF = np.load(CTX / "bank_safety19_dev_uV.npy").astype(np.float64)
import pandas as pd
META = pd.read_csv(CTX / "bank_metadata_dev.csv")
models = joblib.load(CTX / "models_oof.joblib")

bank = ArtifactBank(BANK_DEC, BANK_SAF,
                    META["donor_id"].to_numpy(dtype=object),
                    META["artifact_subject"].to_numpy())
print("bank checks:", bank.validate())

print("\n" + "=" * 66)
print("REPRODUCING THE FROZEN PER-SUBJECT VALUES")
print("=" * 66)

worst = 0.0
all_frac, margins_m, margins_i = [], [], []

for s in range(1, 10):
    Xd = Z[f"X_dec_{s}"].astype(np.float64)
    Xs = Z[f"X_saf_{s}"].astype(np.float64)
    y = Z[f"y_{s}"].astype(str)
    runs = Z[f"runs_{s}"].astype(str)

    tot_full = tot_req = 0
    for r in sorted(set(runs)):
        val, fit = runs == r, runs != r
        pipe = models[s][r]
        csp = pipe.named_steps["csp"]
        lda = pipe.named_steps["lineardiscriminantanalysis"]
        flip = 1.0 if np.asarray(lda.classes_)[1] == "right_hand" else -1.0

        def score_fn(logpow, _lda=lda, _flip=flip):
            flat = logpow.reshape(-1, logpow.shape[-1])
            raw = np.asarray(_lda.decision_function(flat)).reshape(-1)
            return (_flip * raw).reshape(logpow.shape[:-1])

        y_bin = (y[val] == "right_hand").astype(int)
        dec = PowerDecoder(csp.filters_[:csp.n_components], score_fn, y_bin)

        ref = float(np.median(np.sqrt(np.mean(Xs[fit] ** 2, axis=(1, 2)))))
        bench = HarmBenchmark(dec, bank, snr_db=SNR)
        res = bench.run(Xd[val], clean_reference_rms=ref,
                        subject_ids=np.full(int(val.sum()), s), keep_margins=True)
        tot_full += res.full_harm
        tot_req += res.interaction_required
        margins_m.append(res.margins["no_interaction"])
        margins_i.append(res.margins["interaction"])

    frac = tot_req / tot_full
    all_frac.append(frac)
    delta = frac - LEDGER[s]
    worst = max(worst, abs(delta))
    print(f"  S{s}  harmkit {frac:.6f}   ledger {LEDGER[s]:.6f}   delta {delta:+.2e}")

print(f"\nmax |delta| = {worst:.2e}  ->  {'PASS' if worst < TOL else 'FAIL'}")
if worst >= TOL:
    raise SystemExit("harmkit does not reproduce the frozen values")

ci = subject_bootstrap(all_frac, seed=20260813)
print(f"\nsubject-level median {ci['point']:.6f}  95% CI [{ci['lo']:.6f}, {ci['hi']:.6f}]  N={ci['n']}")

# ------------------------------------------------------------------
print("\n" + "=" * 66)
print("CLOSED-FORM LAW, no fitted parameters")
print("=" * 66)
m = np.concatenate(margins_m).astype(np.float64)
I = np.concatenate(margins_i).astype(np.float64)
pred, diag = predict_interaction_required(m, I)
obs = float(np.sum((m > 0) & (m + I < 0)) / np.sum(m + I < 0))
print(f"  predicted {pred:.6f}   observed {obs:.6f}   "
      f"relative error {abs(pred-obs)/obs*100:.2f}%")
print(f"  f_m(0)={diag['f_m_0']:.4f}  E[(-I)+]={diag['E_neg_I']:.5f}  "
      f"P(m<0)={diag['P_m_neg']:.4f}")

# ------------------------------------------------------------------
print("\n" + "=" * 66)
print("CONTROL: does the artifact alone reveal the class?")
print("=" * 66)
pipe = models[1]["0"] if "0" in models[1] else models[1][sorted(models[1])[0]]
csp = pipe.named_steps["csp"]; lda = pipe.named_steps["lineardiscriminantanalysis"]
flip = 1.0 if np.asarray(lda.classes_)[1] == "right_hand" else -1.0
score_fn = lambda lp: (flip * np.asarray(
    lda.decision_function(lp.reshape(-1, lp.shape[-1]))).reshape(-1)).reshape(lp.shape[:-1])
runs1 = Z["runs_1"].astype(str); val1 = runs1 == sorted(set(runs1))[0]
y1 = (Z["y_1"].astype(str)[val1] == "right_hand").astype(int)
dec1 = PowerDecoder(csp.filters_[:csp.n_components], score_fn, y1)
Xs1 = Z["X_saf_1"].astype(np.float64)
ref1 = float(np.median(np.sqrt(np.mean(Xs1[~val1] ** 2, axis=(1, 2)))))
lam1 = bank.scale_for_snr(ref1, SNR)
ctrl = artifact_only_control(dec1, bank, lam1, y1)
for k, v in ctrl.items():
    print(f"  {k}: {v}")

print("\n" + "=" * 66)
print("ALL ACCEPTANCE CHECKS PASSED")
print("=" * 66)
