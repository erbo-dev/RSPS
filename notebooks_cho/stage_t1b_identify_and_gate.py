# REPO STATUS: SUPERSEDED by t1c. Kept: its physically impossible output is the evidence in ledger R1.
# Run order, retractions and full context: notebooks_cho/RUN_ORDER.md

"""
STAGE T1b  ---  identify the jaw-clench slot by physics, then run the gate.

Why this exists
---------------
The PHASE 1 dump of s01.mat showed:

    eeg.noise : cell, shape (5, 1), each block (68, 5120) float32
    eeg.rest  : (68, 34048) float32
    eeg.srate : uint16

So there are five non-task blocks plus a separate resting block, and each
non-task block is exactly 10 seconds at 512 Hz.  The publication lists the six
non-task recordings in prose, but prose ordering is not a guarantee of cell
ordering, and building a donor bank on the wrong slot would silently poison
every downstream number.

This script therefore does NOT trust the stated order.  It identifies the
jaw-clench block from its physics and requires the answer to agree across all
probe subjects before it will proceed.

Discriminator
-------------
Jaw clenching is surface EMG: broadband, roughly flat spectrum, maximal over
temporal sites.  Ocular and head-movement blocks live below about 4 Hz and are
maximal over frontal sites.  Two independent signatures are therefore used:

    1. HF/LF power ratio, mean power 40-100 Hz over mean power 0.5-4 Hz.
       Jaw clenching must be the argmax among the five blocks.
    2. Topography, the EEG channel carrying the most 40-100 Hz power must be
       a temporal or fronto-temporal site, not a frontopolar one.

Both must agree, in every probe subject, on the same slot index.  Anything
less and the script stops.

Then, and only then, the preregistered feasibility gate from Stage T1 runs:
does the jaw block add ABSOLUTE 8-32 Hz power relative to the same subject's
resting block?  The gate is on absolute power, never on a ratio.  The ratio
criterion is the documented failure of the ocular probe and is not repeated.

This script still computes no harm statistic of any kind.  Do not add one
until the Stage T3 preregistration is frozen and committed.

Run as a single Kaggle CPU cell.  Internet must be enabled.  The five .mat
files are already cached from Stage T1, so this costs no download.
"""

import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Protocol constants for the Cho2017 branch.
# These are NOT the PhysioMotion constants.  Record them as a separate block
# in FROZEN_CONSTANTS.json.  Do not overwrite the existing entry.
# ----------------------------------------------------------------------
SEED = 20260813
SFREQ_TARGET = 250.0
DECODER_BAND = (8.0, 32.0)
BROAD_BAND = (0.5, 100.0)
HF_BAND = (40.0, 100.0)          # muscle
LF_BAND = (0.5, 4.0)             # ocular and head movement
NOTCH_HZ = 60.0                  # Korea, not 50
WINDOW_S = 4.0                   # for THIS stage only, see note below
PROBE_SUBJECTS = [1, 2, 3, 4, 5]

# Note on WINDOW_S.  Stage T1b measures block power, so 4 s is fine here.
# Stage T2 must switch to 3 s, because Cho2017 gives about 3 s of imagery per
# trial and the frozen 4 s epoch of the BNCI branch does not exist in this
# dataset.  Freeze that as a separate constant before building the bank.

ROOT_KEY = "eeg"
FIELD_SRATE = "srate"
FIELD_NOISE = "noise"
FIELD_REST = "rest"

N_EEG = 64                       # channels 65-68 are arm EMG, never used here

COMMON19 = ["Fz",
            "FC3", "FC1", "FCz", "FC2", "FC4",
            "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP3", "CP1", "CP2", "CP4",
            "Pz", "POz"]

# Sites where surface EMG from the masseter and temporalis is expected to be
# largest.  Used only as a sanity check on the identified slot, never to pick
# channels for any measurement.
TEMPORAL_SITES = {"T7", "T8", "T9", "T10", "FT7", "FT8", "TP7", "TP8",
                  "F7", "F8", "FT9", "FT10", "TP9", "TP10"}
FRONTOPOLAR_SITES = {"Fp1", "Fp2", "Fpz", "AF7", "AF8", "AF3", "AF4", "AFz"}

# Preregistered gate, frozen before any number below was seen.
GATE_MIN_SUBJECTS = 4
GATE_MIN_ABSOLUTE_RATIO = 1.50
GATE_REQUIRE_ABSOLUTE_INCREASE = True

OUT_DIR = "/kaggle/working"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 74)
print("STAGE T1b  identify the jaw slot, then gate")
print("=" * 74)

try:
    import mne
    import moabb
    from moabb.datasets import Cho2017
    from scipy.io import loadmat
    from scipy.signal import resample_poly
except ImportError:
    os.system(f"{sys.executable} -m pip install -q mne moabb")
    import mne
    import moabb
    from moabb.datasets import Cho2017
    from scipy.io import loadmat
    from scipy.signal import resample_poly

mne.set_log_level("ERROR")
print(f"mne {mne.__version__}   moabb {moabb.__version__}   numpy {np.__version__}")

ds = Cho2017()

# ----------------------------------------------------------------------
# Channel names.  MOABB hardcodes the montage; the .mat carries the same row
# order but no names.  Borrow the names, then assert the counts agree.  A
# silent channel misalignment here would invalidate everything downstream.
# ----------------------------------------------------------------------
raw_mi = None
for by_session in ds.get_data(subjects=[PROBE_SUBJECTS[0]]).values():
    for by_run in by_session.values():
        for run in by_run.values():
            raw_mi = run
            break
        break
    break
if raw_mi is None:
    raise RuntimeError("MOABB returned no runs")

ch_all = [c for c in raw_mi.ch_names if c.upper() not in ("STIM", "STI 014")]
print(f"\nMOABB reports {len(ch_all)} non-stim channels")
print(f"  first 5 : {ch_all[:5]}")
print(f"  last 6  : {ch_all[-6:]}   <- expect 4 EMG at the tail")

missing = [c for c in COMMON19 if c not in ch_all]
if missing:
    raise RuntimeError(f"COMMON19 absent from Cho2017 montage: {missing}")
idx19 = np.array([ch_all.index(c) for c in COMMON19])
eeg_names = ch_all[:N_EEG]
print(f"  all 19 COMMON19 present, max index {idx19.max()} (must be < {N_EEG})")
if idx19.max() >= N_EEG:
    raise RuntimeError("a COMMON19 channel landed in the EMG tail, order is wrong")


# ----------------------------------------------------------------------
def unwrap(struct, field):
    v = struct[field]
    while isinstance(v, np.ndarray) and v.dtype == object and v.size == 1:
        v = v.ravel()[0]
    return v


def prep(block, sfreq_in, band, ch_idx, car=True):
    """CAR over the given channels, notch, band-pass, resample to 250 Hz."""
    x = np.asarray(block, dtype=np.float64)[ch_idx, :]
    if car:
        x = x - x.mean(axis=0, keepdims=True)
    g = int(round(sfreq_in / np.gcd(int(sfreq_in), int(SFREQ_TARGET))))
    up = int(round(SFREQ_TARGET * g / sfreq_in))
    x = resample_poly(x, up, g, axis=-1)
    x = mne.filter.notch_filter(x, SFREQ_TARGET, [NOTCH_HZ], verbose=False)
    return mne.filter.filter_data(x, SFREQ_TARGET, band[0], band[1], verbose=False)


def window_rms(x):
    n = int(WINDOW_S * SFREQ_TARGET)
    k = x.shape[-1] // n
    if k == 0:
        return np.array([])
    w = x[:, : k * n].reshape(x.shape[0], k, n)
    return np.sqrt((w ** 2).mean(axis=(0, 2)))


# ======================================================================
# PART A  ---  identify the jaw slot
# ======================================================================
print("\n" + "-" * 74)
print("PART A  slot identification, five blocks per subject")
print("-" * 74)

eeg_idx = np.arange(N_EEG)
identified, detail = {}, []

for s in PROBE_SUBJECTS:
    p = ds.data_path(s)
    p = p[0] if isinstance(p, (list, tuple)) else p
    m = loadmat(p, squeeze_me=False, struct_as_record=True)
    root = m[ROOT_KEY].ravel()[0]
    sf = float(np.asarray(unwrap(root, FIELD_SRATE)).ravel()[0])
    noise = unwrap(root, FIELD_NOISE)
    blocks = list(noise.ravel()) if noise.dtype == object else list(np.asarray(noise))

    print(f"\nsubject {s:2d}   srate {sf:g} Hz   n_blocks {len(blocks)}")
    print(f"  {'slot':>4} {'HF/LF':>9} {'broad RMS':>10} {'8-32 RMS':>9}  argmax-HF channel")
    ratios = []
    for j, b in enumerate(blocks):
        b = np.asarray(b, dtype=np.float64)
        if b.shape[0] != len(ch_all):
            b = b.T
        hf = prep(b, sf, HF_BAND, eeg_idx)
        lf = prep(b, sf, LF_BAND, eeg_idx)
        bb = prep(b, sf, BROAD_BAND, idx19)
        db = prep(b, sf, DECODER_BAND, idx19)
        r = float((hf ** 2).mean() / max((lf ** 2).mean(), 1e-300))
        top = eeg_names[int(np.argmax((hf ** 2).mean(axis=-1)))]
        ratios.append(r)
        detail.append(dict(subject=s, slot=j, hf_lf_ratio=r, top_hf_channel=top,
                           broad_rms=float(np.median(window_rms(bb))),
                           decoder_rms=float(np.median(window_rms(db)))))
        print(f"  {j:>4} {r:9.3f} {np.median(window_rms(bb)):10.3f} "
              f"{np.median(window_rms(db)):9.3f}  {top}")

    j = int(np.argmax(ratios))
    top = [d for d in detail if d["subject"] == s and d["slot"] == j][0]["top_hf_channel"]
    ok_topo = top in TEMPORAL_SITES
    identified[s] = dict(slot=j, top=top, topography_ok=ok_topo,
                         margin=float(ratios[j] / max(sorted(ratios)[-2], 1e-300)))
    flag = "OK" if ok_topo else ("FRONTOPOLAR, suspicious" if top in FRONTOPOLAR_SITES
                                 else "not temporal, check by hand")
    print(f"  -> argmax HF/LF is slot {j}, top channel {top}  [{flag}], "
          f"{identified[s]['margin']:.2f}x over runner-up")

slots = {v["slot"] for v in identified.values()}
topo_ok = sum(v["topography_ok"] for v in identified.values())
consistent = len(slots) == 1

print("\n" + "-" * 74)
print(f"slot agreement across subjects : {sorted(slots)}  "
      f"{'CONSISTENT' if consistent else 'INCONSISTENT'}")
print(f"temporal topography            : {topo_ok}/{len(identified)}")

if not consistent:
    raise SystemExit(
        "\nSTOP. The five subjects do not agree on which slot is jaw clenching.\n"
        "Cell ordering cannot be assumed stable across files. Inspect the\n"
        "waveforms by hand, or abandon this dataset. Do NOT pick a majority\n"
        "vote and continue.")
if topo_ok < len(identified) - 1:
    raise SystemExit(
        "\nSTOP. The high-frequency maximum is not over temporal sites.\n"
        "The identified block is probably not jaw clenching. Plot the five\n"
        "blocks before going further.")

JAW_SLOT = slots.pop()
print(f"JAW_SLOT identified from physics = {JAW_SLOT}")

# ======================================================================
# PART B  ---  the preregistered feasibility gate
# ======================================================================
print("\n" + "-" * 74)
print("PART B  does jaw clenching add ABSOLUTE 8-32 Hz power vs the same")
print("        subject's resting block?")
print("-" * 74)
print(f"{'subj':>5} {'n_jaw':>6} {'n_rest':>7} {'jaw P(8-32)':>12} "
      f"{'rest P(8-32)':>13} {'ratio':>7} {'abs.up':>7}")

rows = []
for s in PROBE_SUBJECTS:
    p = ds.data_path(s)
    p = p[0] if isinstance(p, (list, tuple)) else p
    m = loadmat(p, squeeze_me=False, struct_as_record=True)
    root = m[ROOT_KEY].ravel()[0]
    sf = float(np.asarray(unwrap(root, FIELD_SRATE)).ravel()[0])
    noise = unwrap(root, FIELD_NOISE)
    blocks = list(noise.ravel()) if noise.dtype == object else list(np.asarray(noise))

    jaw = np.asarray(blocks[JAW_SLOT], dtype=np.float64)
    rest = np.asarray(unwrap(root, FIELD_REST), dtype=np.float64)
    if jaw.shape[0] != len(ch_all):
        jaw = jaw.T
    if rest.shape[0] != len(ch_all):
        rest = rest.T

    jd = window_rms(prep(jaw, sf, DECODER_BAND, idx19))
    rd = window_rms(prep(rest, sf, DECODER_BAND, idx19))
    jb = window_rms(prep(jaw, sf, BROAD_BAND, idx19))
    rb = window_rms(prep(rest, sf, BROAD_BAND, idx19))

    pj, pr = float(np.median(jd) ** 2), float(np.median(rd) ** 2)
    ratio = pj / pr if pr > 0 else np.inf
    up = pj > pr
    rows.append(dict(subject=s, sfreq=sf, jaw_slot=JAW_SLOT,
                     n_jaw_windows=int(jd.size), n_rest_windows=int(rd.size),
                     jaw_power_decoder=pj, rest_power_decoder=pr, ratio=ratio,
                     jaw_rms_broad=float(np.median(jb)),
                     rest_rms_broad=float(np.median(rb)),
                     absolute_increase=bool(up)))
    print(f"{s:5d} {jd.size:6d} {rd.size:7d} {pj:12.4f} {pr:13.4f} "
          f"{ratio:7.2f} {'yes' if up else 'NO':>7}")

n_abs = sum(r["absolute_increase"] for r in rows)
n_ratio = sum(r["ratio"] >= GATE_MIN_ABSOLUTE_RATIO for r in rows)
passed = n_abs >= GATE_MIN_SUBJECTS and n_ratio >= GATE_MIN_SUBJECTS

print("\n" + "=" * 74)
print("PREREGISTERED GATE")
print(f"  absolute 8-32 Hz power increase : {n_abs}/{len(rows)} "
      f"(need >= {GATE_MIN_SUBJECTS})")
print(f"  power ratio >= {GATE_MIN_ABSOLUTE_RATIO}               : {n_ratio}/{len(rows)} "
      f"(need >= {GATE_MIN_SUBJECTS})")
print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
print("=" * 74)

if passed:
    print("""
Jaw clenching is a usable within-subject donor for an 8-32 Hz decoder.

Rewrite section 9.4. The impossibility claim goes; the defensible claim that
survives is narrower: OCULAR artifacts cannot serve as donors for a mu-beta
decoder, measured 0/9. That is still a real result and it is still yours.

Before Stage T2, freeze three things that the dump forced on you:
  1. epoch 3 s for this branch, because Cho2017 has no 4 s of imagery
  2. notch 60 Hz, not 50
  3. donor window selection rule, e.g. max broadband RMS inside the block
Write them as a NEW entry in FROZEN_CONSTANTS.json. Do not overwrite the
PhysioMotion entry, and never mix the two banks in one number.

Then Stage T3. Freeze the preregistration BEFORE any harm statistic exists.
""")
else:
    print("""
Gate failed. Do not loosen the threshold to rescue it.

Report it the way the ocular probe was reported: a stated criterion, a
measured outcome, published either way. Section 9.4 still has to lose the
impossibility claim, because that claim was wrong independently of this gate.
""")

with open(f"{OUT_DIR}/T1b_SLOT_AND_GATE.json", "w") as f:
    json.dump(dict(stage="T1b", seed=SEED, subjects=PROBE_SUBJECTS,
                   jaw_slot=int(JAW_SLOT), slot_identification=identified,
                   block_detail=detail,
                   constants=dict(sfreq_target=SFREQ_TARGET,
                                  decoder_band=DECODER_BAND, hf_band=HF_BAND,
                                  lf_band=LF_BAND, notch_hz=NOTCH_HZ,
                                  window_s=WINDOW_S),
                   gate=dict(min_subjects=GATE_MIN_SUBJECTS,
                             min_ratio=GATE_MIN_ABSOLUTE_RATIO,
                             require_absolute=GATE_REQUIRE_ABSOLUTE_INCREASE),
                   n_absolute_increase=n_abs, n_ratio_pass=n_ratio,
                   verdict="PASS" if passed else "FAIL", rows=rows), f, indent=2)
print(f"written: {OUT_DIR}/T1b_SLOT_AND_GATE.json")
