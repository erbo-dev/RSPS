# REPO STATUS: CURRENT. Natural SNR in the safety-band convention, and the cohort recount.
# Run order, retractions and full context: notebooks_cho/RUN_ORDER.md

"""
STAGE T3d + T3e  ---  natural SNR in the right convention, and the cohort recount.

Three jobs, no download, minutes.

T3d  Natural SNR, recomputed in harmkit's convention.
     SNR_nat = 20 log10( clean safety-band RMS / donor safety-band RMS )
     Both in 20-80 Hz, both over the frozen 0.5-3.0 s window, both from the
     same person in the same session on the same amplifier, so the ratio is
     unit free and directly interpretable.
     The previously reported median of -9.52 dB is RETRACTED: it divided a
     clean RMS taken in 8-32 Hz by a donor RMS in 20-80 Hz, which is not the
     convention harmkit implements. That is the second retraction of this
     number; the first was the zero-padded resample in bank v1.

T3e  Cohort recount under bank v3, subject by subject, against v2.
     The purity DEFINITION did not change. Two things about its ESTIMATE did:
     the band-pass is now applied to the whole block instead of a 750-sample
     window, and purity is measured on the 625-sample segment that is actually
     injected rather than on a 750-sample window that is not. Both changes are
     towards correctness, and the second is the substantive one.
     The v2-versus-v3 check in T2d used six subjects and reported a worst
     drift of 0.021. That ledger was too small: subject 31 moved by 0.111.
     A six-subject spot check was the wrong instrument and this recount is the
     right one.

     RULE, FROZEN HERE, BEFORE ANY COUNT IS PRINTED:
       the primary cohort is the v3 count, because v3 measures purity on the
       signal that is injected;
       the v2 count is published beside it as a sensitivity analysis;
       the 0.90 threshold does not move, whatever either count turns out to be.

CAR  The absolute check in harmkit.validate uses car_tol = 1e-3, calibrated
     for microvolts: the PhysioMotion bank peaked at 97.9 uV. Raw Cho units
     are about ten thousand times larger, so an absolute tolerance is
     meaningless here. The residual scales with amplitude, which is the
     signature of float32 rounding and not of a broken average reference:
     CAR is exact by construction, since subtracting the channel mean gives a
     sum of exactly zero and every later operation is linear and identical
     across channels. Only the float32 cast breaks it. This checks the
     RELATIVE residual instead, which is the quantity that has meaning.

Run as one Kaggle CPU cell with cho_bank_v3 attached.
"""

import glob
import json
import os

import numpy as np

BANK, OUT = "/kaggle/working/cho_bank_v3", "/kaggle/working"
if not glob.glob(f"{BANK}/sub-*.npz"):
    for a in glob.glob("/kaggle/input/**/", recursive=True):
        if glob.glob(f"{a}/sub-*.npz"):
            BANK = a.rstrip("/")
            break
FILES = sorted(glob.glob(f"{BANK}/sub-*.npz"))
if not FILES:
    raise SystemExit("STOP. cho_bank_v3 not found. Attach the Stage T2d output.")

PURITY, ACC_MIN = 0.90, 0.60
FROZEN_GRID = [2, 0, -2, -3, -4, -5, -6, -7]
RETRACTED_MEDIAN = -9.52

# Recorded during the Stage T3 run on bank v2. Column 'jaw>=pur' and 'clean acc'.
V2_NPURE = {6: 2, 7: 1, 9: 3, 10: 3, 11: 0, 12: 1, 13: 0, 14: 3, 18: 3, 19: 0,
            20: 1, 21: 0, 22: 3, 23: 2, 25: 1, 26: 1, 27: 2, 29: 0, 31: 0,
            32: 1, 33: 3, 34: 0, 35: 3, 37: 3, 38: 3, 39: 0, 40: 3, 41: 1,
            44: 0, 47: 3, 49: 2, 50: 3}
V2_ACC = {6: .6450, 7: .4728, 9: .7292, 10: .8300, 11: .6200, 12: .7600,
          13: .5600, 14: .9800, 18: .6800, 19: .5500, 20: .7250, 21: .7337,
          22: .7136, 23: .8550, 25: .6935, 26: .6000, 27: .6382, 29: .4850,
          31: .7337, 32: .5200, 33: .6382, 34: .4300, 35: .7550, 37: .7000,
          38: .4600, 39: .7588, 40: .5500, 41: .9196, 44: .6850, 47: .6450,
          49: .6850, 50: .5250}

ACC = dict(V2_ACC)
p = next((q for q in [f"{OUT}/T3_RESULT.json"] + glob.glob("/kaggle/input/**/T3_RESULT.json",
                                                           recursive=True) if os.path.exists(q)), None)
if p:
    ACC = {t["subject"]: t["clean_acc"] for t in json.loads(open(p).read())["subjects"]}
    print(f"clean accuracies read from {p}")
else:
    print("clean accuracies taken from the embedded T3 ledger")

print("=" * 80)
print("STAGE T3d + T3e")
print(f"bank: {BANK}   purity threshold {PURITY} (frozen, not moved)")
print("PRIMARY COHORT IS THE v3 COUNT. v2 is the sensitivity analysis.")
print("=" * 80)

rows = []
for f in FILES:
    d = np.load(f, allow_pickle=True)
    s, grp = int(d["subject"]), str(d["group"])
    kept = np.concatenate([np.asarray(d["clean_saf_rms_L"])[d["keep_L"]],
                           np.asarray(d["clean_saf_rms_R"])[d["keep_R"]]])
    ref = float(np.median(kept))
    jr = np.asarray(d["jaw_saf_rms"], float)
    snr = 20.0 * np.log10(ref / np.maximum(jr, 1e-300))
    pur = np.asarray(d["jaw_purity"], float)

    jd = np.asarray(d["jaw_dec"], float)
    car_rel = float(np.max(np.abs(jd.sum(1))) / max(np.max(np.abs(jd)), 1e-300))

    rows.append(dict(subject=s, group=grp, clean_saf_ref=ref,
                     snr_nat_med=float(np.median(snr)), snr_nat_min=float(snr.min()),
                     snr_all=[float(x) for x in snr],
                     n_pure_v3=int((pur >= PURITY).sum()),
                     n_pure_v2=V2_NPURE.get(s), purity_med=float(np.median(pur)),
                     car_rel=car_rel, acc=ACC.get(s)))

conf = [r for r in rows if r["group"] != "pilot"]
dev = [r for r in rows if r["group"] == "dev"]

# ---------------------------------------------------------------- CAR
cr = np.array([r["car_rel"] for r in rows])
print(f"\nCAR residual, RELATIVE: median {np.median(cr):.2e}, max {cr.max():.2e}")
print("  float32 storage alone accounts for about 1e-7. The absolute tolerance")
print("  of 1e-3 in harmkit.validate is calibrated for microvolts and does not")
print("  apply to raw Cho units; use this relative figure instead.")
if cr.max() > 1e-5:
    print("  WARNING: larger than float32 rounding explains. Investigate.")

# ---------------------------------------------------------------- T3d
print("\n" + "-" * 80)
print("T3d  NATURAL SNR, safety-band convention, confirmatory subjects")
print("-" * 80)
v = np.array([x for r in conf for x in r["snr_all"]])
vm = np.array([r["snr_nat_med"] for r in conf])
for nm, a in (("all donor windows", v), ("per-subject median", vm)):
    q = np.percentile(a, [5, 25, 50, 75, 95])
    print(f"  {nm:>20}: n={a.size:4d}  median {np.median(a):7.2f} dB   "
          f"p5 {q[0]:7.2f}  p25 {q[1]:7.2f}  p75 {q[3]:7.2f}  p95 {q[4]:7.2f}")
lo, hi = min(FROZEN_GRID), max(FROZEN_GRID)
print(f"\n  frozen grid {lo:+d} to {hi:+d} dB")
print(f"    inside  {100*np.mean((v>=lo)&(v<=hi)):5.1f}%    "
      f"harsher {100*np.mean(v<lo):5.1f}%    milder {100*np.mean(v>hi):5.1f}%")
print(f"\n  RETRACTED value under the wrong convention: {RETRACTED_MEDIAN:+.2f} dB")
print(f"  corrected median                           : {np.median(vm):+.2f} dB")
print(f"  shift                                      : "
      f"{np.median(vm) - RETRACTED_MEDIAN:+.2f} dB")
print("""
  Read this before writing. Clean EEG carries far less power at 20-80 Hz than
  at 8-32 Hz, so the corrected reference is smaller and the corrected SNR is
  more negative. Whichever way the grid comparison lands, state it plainly:
  a grid that sits milder than nature makes the headline number an upper
  bound; a grid that sits harsher means it does not transfer, and section 7
  already showed the quantity depends on the contamination regime.
""")

# ---------------------------------------------------------------- T3e
print("-" * 80)
print("T3e  COHORT RECOUNT, development subjects")
print("-" * 80)
print(f"  {'subj':>5} {'acc':>7} {'pur med':>8} {'n v2':>5} {'n v3':>5} {'change':>22}")
moved_in, moved_out, same = [], [], []
for r in sorted(dev, key=lambda x: x["subject"]):
    a, n2, n3 = r["acc"], r["n_pure_v2"], r["n_pure_v3"]
    e2 = a is not None and a >= ACC_MIN and (n2 or 0) >= 1
    e3 = a is not None and a >= ACC_MIN and n3 >= 1
    tag = ""
    if e3 and not e2:
        tag = "ENTERS the cohort"; moved_in.append(r["subject"])
    elif e2 and not e3:
        tag = "LEAVES the cohort"; moved_out.append(r["subject"])
    elif e3:
        same.append(r["subject"])
    print(f"  {r['subject']:5d} {a if a is not None else float('nan'):7.4f} "
          f"{r['purity_med']:8.3f} {n2 if n2 is not None else -1:5d} {n3:5d} {tag:>22}")

n2_tot = len([r for r in dev if r["acc"] and r["acc"] >= ACC_MIN and (r["n_pure_v2"] or 0) >= 1])
n3_tot = len(same) + len(moved_in)
print(f"""
  eligible under v2 (sensitivity) : {n2_tot}
  eligible under v3 (PRIMARY)     : {n3_tot}
  unchanged {len(same)}   entered {moved_in}   left {moved_out}
""")
for p_ in (0.75, 0.90, 0.95):
    n = len([r for r in dev if r["acc"] and r["acc"] >= ACC_MIN
             and (np.asarray(np.load(f"{BANK}/sub-{r['subject']:02d}.npz")["jaw_purity"], float) >= p_).sum() >= 1])
    print(f"    sensitivity, purity >= {p_:.2f}: {n} subjects")

print(f"""
  What must go in the text, whichever way the count fell:
    the v3 count as primary, the v2 count beside it, and the reason v3 is
    primary, namely that purity is now measured on the injected segment;
    the list of subjects that entered and left, by number;
    that the 0.90 threshold was not moved.

  If the two counts differ, the decoders do NOT need refitting. Trials were
  filtered identically in both banks, so T3_MODELS.npz and every clean
  accuracy remain valid. Only the donor-side membership changed.
""")

with open(f"{OUT}/T3de_RESULT.json", "w") as f:
    json.dump(dict(stage="T3d+T3e", bank=BANK, purity=PURITY,
                   retracted_median_db=RETRACTED_MEDIAN,
                   corrected_median_db=float(np.median(vm)),
                   grid_inside=float(np.mean((v >= lo) & (v <= hi))),
                   grid_harsher=float(np.mean(v < lo)),
                   grid_milder=float(np.mean(v > hi)),
                   car_relative_max=float(cr.max()),
                   n_eligible_v2=n2_tot, n_eligible_v3=n3_tot,
                   entered=moved_in, left=moved_out, unchanged=same,
                   subjects=rows), f, indent=2, default=float)
print(f"written: {OUT}/T3de_RESULT.json")
