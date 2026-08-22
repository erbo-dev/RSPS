# REPO STATUS: RETRACTED. Ran on the void bank v1. See ledger R3.
# Run order, retractions and full context: notebooks_cho/RUN_ORDER.md

"""
STAGE T2b  ---  natural SNR and donor purity, from the bank already on disk.

No download.  Reads the npz files written by Stage T2 and answers two
questions that the bank makes answerable and nothing else in the project does.

Question 1, natural SNR.
------------------------
Limitation 8 of the current write-up says the headline number applies to the
frozen SNR grid and not to natural chewing, and the grid +2 to -7 dB is
justified only by a citation to an EMG-removal paper.  In Cho2017 the jaw
block and the imagery block come from the same person, the same session, the
same electrodes, the same impedances and the same amplifier gain, so their
amplitudes are directly comparable and the SNR that person would have
inflicted on themselves is a measurable quantity:

    SNR_nat = 20 log10( RMS(clean imagery trial) / RMS(jaw donor window) )

using the same band convention as the frozen protocol: the clean trial in its
decoder band 8-32 Hz, the donor in its donor band 20-80 Hz.  Both signals are
in the same arbitrary amplifier units, so the ratio is unit free.

That turns limitation 8 from a caveat into a measurement.

Question 2, donor purity.
-------------------------
The design rescales every donor to a target SNR, so a donor's absolute power
means nothing.  What matters is whether the thing being injected is muscle or
the subject's own background.  Injecting a person's own background EEG back
into their own trial and calling it an artifact is the confound that killed
the ocular branch, and MI_d >= 3.0 does not rule it out: MI_d is measured in
40-100 Hz, while what gets injected is 20-80 Hz.

    purity = 1 - P_20_80(subject's resting window) / P_20_80(donor window)

This is a LOWER BOUND on the muscle fraction, not an estimate of it.  It is a
bound because alpha and beta are higher at rest than during any task, so the
numerator of the subtracted term is overstated.  Report it as a bound.

No harm statistic here either.  Stage T3 is still unfrozen.

Run as one Kaggle CPU cell with the Stage T2 output attached.
"""

import glob
import json
import os

import numpy as np

BANK = "/kaggle/working/cho_bank"
if not glob.glob(f"{BANK}/sub-*.npz"):
    for alt in glob.glob("/kaggle/input/*/cho_bank") + glob.glob("/kaggle/input/*"):
        if glob.glob(f"{alt}/sub-*.npz"):
            BANK = alt
            break

FILES = sorted(glob.glob(f"{BANK}/sub-*.npz"))
if not FILES:
    raise SystemExit("STOP. No sub-XX.npz found. Attach the Stage T2 output.")

PILOT = {1, 2, 3, 4, 5}
LOCKED = {8, 15, 16, 17, 24, 28, 30, 36, 42, 43, 45, 46, 48, 51, 52}
FROZEN_GRID_DB = [2, 0, -2, -3, -4, -5, -6, -7]

print("=" * 78)
print(f"STAGE T2b  natural SNR and donor purity     bank: {BANK}")
print(f"{len(FILES)} subject files")
print("=" * 78)


def rms(x):
    """Root mean square over channels and time, one number per epoch."""
    x = np.asarray(x, dtype=np.float64)
    return np.sqrt((x ** 2).mean(axis=(-2, -1)))


rows, per_donor = [], []
for f in FILES:
    d = np.load(f, allow_pickle=True)
    s = int(d["subject"])
    group = "pilot" if s in PILOT else ("locked" if s in LOCKED else "dev")

    clean = rms(d["mi_X"])                       # 8-32 Hz, per trial
    jaw = rms(d["jaw_D"])                        # 20-80 Hz, per donor window
    rest = rms(d["rest_D"])                      # 20-80 Hz, per rest window
    mi_d = np.asarray(d["jaw_mi"], dtype=np.float64)

    c_med = float(np.median(clean))
    r_med2 = float(np.median(rest) ** 2)         # background power in donor band

    snr = 20.0 * np.log10(c_med / np.maximum(jaw, 1e-300))
    purity = np.clip(1.0 - r_med2 / np.maximum(jaw ** 2, 1e-300), 0.0, 1.0)

    for k in range(jaw.size):
        per_donor.append(dict(subject=s, group=group, window=k,
                              mi_d=float(mi_d[k]) if k < mi_d.size else np.nan,
                              snr_nat_db=float(snr[k]), purity=float(purity[k]),
                              donor_rms=float(jaw[k])))

    rows.append(dict(subject=s, group=group, n_trials=int(d["mi_y"].size),
                     clean_rms_median=c_med, rest_rms_median=float(np.median(rest)),
                     mi_d_max=float(mi_d.max()) if mi_d.size else np.nan,
                     snr_nat_min=float(snr.min()), snr_nat_median=float(np.median(snr)),
                     purity_max=float(purity.max()), purity_median=float(np.median(purity))))

conf = [r for r in rows if r["group"] != "pilot"]
D_conf = [d for d in per_donor if d["group"] != "pilot"]
print(f"confirmatory subjects {len(conf)}, confirmatory donor windows {len(D_conf)}")

# ----------------------------------------------------------------------
print("\n" + "-" * 78)
print("NATURAL SNR, confirmatory subjects only")
print("-" * 78)
snr_all = np.array([d["snr_nat_db"] for d in D_conf])
snr_min = np.array([r["snr_nat_min"] for r in conf])   # strongest clench per person

for name, v in [("all donor windows", snr_all), ("worst window per subject", snr_min)]:
    q = np.percentile(v, [5, 25, 50, 75, 95])
    print(f"  {name:>26}: n={v.size:4d}  median {np.median(v):7.2f} dB   "
          f"p5 {q[0]:7.2f}   p25 {q[1]:7.2f}   p75 {q[3]:7.2f}   p95 {q[4]:7.2f}")

lo, hi = min(FROZEN_GRID_DB), max(FROZEN_GRID_DB)
inside = float(np.mean((snr_all >= lo) & (snr_all <= hi)))
below = float(np.mean(snr_all < lo))
above = float(np.mean(snr_all > hi))
print(f"\n  frozen grid spans {lo:+d} to {hi:+d} dB")
print(f"    natural windows inside the grid : {100 * inside:5.1f}%")
print(f"    harsher than the grid (< {lo:+d} dB) : {100 * below:5.1f}%")
print(f"    milder  than the grid (> {hi:+d} dB) : {100 * above:5.1f}%")
print("""
  Read this carefully before writing anything. If most natural windows sit
  MILDER than the grid, your grid is conservative and the headline number is
  an upper bound on natural harm, which is a defensible thing to say. If most
  sit HARSHER, the grid does not cover the regime people actually generate,
  and section 7 already showed the quantity depends on the contamination
  regime, so the headline number would not transfer. Say whichever it is.
""")

# ----------------------------------------------------------------------
print("-" * 78)
print("DONOR PURITY, lower bound on the muscle fraction in the injected band")
print("-" * 78)
pur = np.array([d["purity"] for d in D_conf])
mid = np.array([d["mi_d"] for d in D_conf])
q = np.percentile(pur, [5, 25, 50, 75, 95])
print(f"  n={pur.size}  median {np.median(pur):.3f}  "
      f"p5 {q[0]:.3f}  p25 {q[1]:.3f}  p75 {q[3]:.3f}  p95 {q[4]:.3f}")

ok = np.isfinite(mid) & np.isfinite(pur)
if ok.sum() > 2:
    from scipy.stats import spearmanr
    rho, p = spearmanr(mid[ok], pur[ok])
    print(f"  Spearman(MI_d, purity) = {rho:+.3f}, p = {p:.2g}, n = {int(ok.sum())}")
    print("  A high correlation means MI_d is already a good proxy and a separate")
    print("  purity threshold adds little. A low one means MI_d passes donors whose")
    print("  injected content is mostly the subject's own background.")

# ----------------------------------------------------------------------
print("\n" + "-" * 78)
print("COHORT SIZE UNDER CANDIDATE THRESHOLDS  (subjects with >= 1 surviving window)")
print("-" * 78)
print(f"  {'MI_d >=':>8} {'purity >=':>10} {'dev':>5} {'locked':>7} {'total':>6}")
grid = []
for t_mi in [3, 10, 20, 50]:
    for t_pu in [0.0, 0.5, 0.75, 0.9]:
        keep = {}
        for d in D_conf:
            if d["mi_d"] >= t_mi and d["purity"] >= t_pu:
                keep.setdefault(d["group"], set()).add(d["subject"])
        nd, nl = len(keep.get("dev", ())), len(keep.get("locked", ()))
        grid.append(dict(mi_d=t_mi, purity=t_pu, dev=nd, locked=nl, total=nd + nl))
        print(f"  {t_mi:>8} {t_pu:>10.2f} {nd:5d} {nl:7d} {nd + nl:6d}")
print("""
  Pick ONE cell and freeze it in the T3 preregistration, with the reason
  written down, before any decoder accuracy exists. Picking it after seeing
  accuracies is the difference between a threshold and a knob.
""")

# ----------------------------------------------------------------------
print("-" * 78)
print("ARM C MATCHING")
print("-" * 78)
n_jaw = int(np.median([len([d for d in per_donor if d["subject"] == r["subject"]])
                       for r in conf]))
print(f"  jaw donors per host  : {n_jaw}")
print(f"  rest donors per host : 22")
print(f"  arm C must be subsampled to {n_jaw} per host, redrawn and averaged,")
print(f"  on the same rule already written for arm B. Otherwise the comparison")
print(f"  is a comparison of counting noise, not of donor type.")

# ----------------------------------------------------------------------
print("\n" + "-" * 78)
print("PER SUBJECT, confirmatory")
print("-" * 78)
print(f"  {'subj':>5} {'grp':>7} {'trials':>7} {'MI_d max':>10} "
      f"{'SNR min':>9} {'SNR med':>9} {'purity max':>11}")
for r in sorted(conf, key=lambda x: x["subject"]):
    print(f"  {r['subject']:5d} {r['group']:>7} {r['n_trials']:7d} "
          f"{r['mi_d_max']:10.2f} {r['snr_nat_min']:9.2f} "
          f"{r['snr_nat_median']:9.2f} {r['purity_max']:11.3f}")

with open("/kaggle/working/T2b_NATURAL_SNR.json", "w") as f:
    json.dump(dict(stage="T2b", n_subjects=len(rows), subjects=rows,
                   donors=per_donor, frozen_grid_db=FROZEN_GRID_DB,
                   fraction_inside_grid=inside, fraction_below=below,
                   fraction_above=above, threshold_grid=grid),
              f, indent=2, default=float)
print("\nwritten: /kaggle/working/T2b_NATURAL_SNR.json")
print("""
Two decisions come out of this run and both belong in T3, frozen, before any
accuracy is computed:
  1. epoch window, 0-3 s as built or 0.5-3.5 s which costs a re-extraction
  2. one cell from the threshold table above
""")
