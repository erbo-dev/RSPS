# REPO STATUS: SUPERSEDED by t1b and t1c. Kept because ledger R1 refers to it.
# Run order, retractions and full context: notebooks_cho/RUN_ORDER.md

"""
STAGE T1  ---  Cho2017 jaw-clench feasibility probe.

Purpose
-------
Section 9.4 of the current write-up concludes that closing the dependence axis
with open data is physically impossible, on the evidence that ocular artifacts
carry no energy in the 8-32 Hz decoder band.  That conclusion is too strong.
Cho et al. 2017 (GigaScience 6(7):gix034, GigaDB 100295) recorded, for each of
52 subjects and inside the same session, both left/right motor imagery and six
non-task blocks, one of which is jaw clenching.  Same person, same montage,
same impedances.

This stage answers exactly one question, and refuses to answer any other:

    Does the jaw-clench block add ABSOLUTE power in the 8-32 Hz decoder band,
    relative to the resting block, for the same subject?

The gate is on absolute power, not on a ratio.  The ratio criterion is the
documented mistake of Stage T0; repeating it would be inexcusable.

What this script does NOT do
----------------------------
It does not train a decoder, does not build a donor bank, does not contaminate
anything, and does not compute any harm statistic.  If it did, feasibility and
result would be measured on the same pass and the preregistration of Stage T3
would already be compromised.

Structure discovery
-------------------
The exact field names inside the GigaDB .mat files were NOT verified by the
author of this script.  The script therefore DISCOVERS the structure, prints
it, and fails loudly rather than guessing.  Run PHASE 1 first, read the dump,
then set the four constants in PHASE 2 to whatever the dump actually shows.

Run as a single Kaggle CPU cell.  Internet must be enabled.
"""

import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Protocol constants, copied from the frozen constants of the main study.
# Do not change these to make a result appear.
# ----------------------------------------------------------------------
SEED = 20260813
SFREQ_TARGET = 250.0          # BNCI2014_001 rate, kept for comparability
DECODER_BAND = (8.0, 32.0)    # the band the decoder actually uses
BROAD_BAND = (0.5, 100.0)     # for reporting only
NOTCH_HZ = 60.0               # Cho2017 was recorded in Korea, mains is 60 Hz
WINDOW_S = 4.0
PROBE_SUBJECTS = [1, 2, 3, 4, 5]

COMMON19 = ["Fz",
            "FC3", "FC1", "FCz", "FC2", "FC4",
            "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP3", "CP1", "CP2", "CP4",
            "Pz", "POz"]

# Preregistered gate for this stage.  Frozen before the numbers were seen.
GATE_MIN_SUBJECTS = 4                 # out of len(PROBE_SUBJECTS)
GATE_MIN_ABSOLUTE_RATIO = 1.50        # jaw band power / rest band power
GATE_REQUIRE_ABSOLUTE_INCREASE = True # the T0 lesson, in code

OUT_DIR = "/kaggle/working"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 72)
print("STAGE T1  Cho2017 jaw-clench feasibility")
print("=" * 72)

# ----------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------
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

# ======================================================================
# PHASE 1  ---  locate the raw .mat and dump its structure
# ======================================================================
print("\n[PHASE 1] locating raw .mat files via the MOABB downloader")

ds = Cho2017()
raw_paths = {}
for s in PROBE_SUBJECTS:
    try:
        p = ds.data_path(s)
        raw_paths[s] = p[0] if isinstance(p, (list, tuple)) else p
        print(f"  subject {s:2d}: {raw_paths[s]}")
    except Exception as e:
        print(f"  subject {s:2d}: FAILED  {type(e).__name__}: {e}")

if not raw_paths:
    raise RuntimeError(
        "No .mat could be downloaded. Enable internet in the notebook settings. "
        "If the GigaDB mirror is down, check moabb/datasets/gigadb.py for the "
        "current GIGA_URL and fetch manually."
    )

first = raw_paths[min(raw_paths)]


def describe(obj, name="root", depth=0, max_depth=3):
    """Print the shape of whatever MATLAB handed us, without assuming names."""
    pad = "  " * depth
    if depth > max_depth:
        print(f"{pad}{name}: ...")
        return
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        print(f"{pad}{name}: struct, fields = {list(obj.dtype.names)}")
        flat = obj.ravel()
        if flat.size:
            for f in obj.dtype.names:
                describe(flat[0][f], f, depth + 1, max_depth)
    elif isinstance(obj, np.ndarray) and obj.dtype == object:
        print(f"{pad}{name}: cell, shape {obj.shape}")
        flat = obj.ravel()
        if flat.size:
            describe(flat[0], f"{name}[0]", depth + 1, max_depth)
    elif isinstance(obj, np.ndarray):
        print(f"{pad}{name}: array, shape {obj.shape}, dtype {obj.dtype}")
    else:
        print(f"{pad}{name}: {type(obj).__name__} = {str(obj)[:80]}")


print(f"\n[PHASE 1] dumping structure of {os.path.basename(first)}")
mat = loadmat(first, squeeze_me=False, struct_as_record=True)
top = [k for k in mat.keys() if not k.startswith("__")]
print(f"  top-level keys: {top}")
for k in top:
    describe(mat[k], k)

# ======================================================================
# PHASE 2  ---  field mapping.  EDIT THESE AFTER READING THE PHASE 1 DUMP.
# ======================================================================
# Set to None to make the script stop and ask you to look.  Filling these in
# from a guess instead of from the dump is how silent bugs get published.
ROOT_KEY = "eeg"          # top-level struct that holds everything
FIELD_SRATE = "srate"     # scalar sampling rate
FIELD_NOISE = "noise"     # container of the six non-task blocks
FIELD_REST = "rest"       # resting block, used as the honest baseline

# Which slot inside FIELD_NOISE is jaw clenching.  The paper lists six blocks
# (eye blinking, eyeball up/down, eyeball left/right, head movement, jaw
# clenching, resting).  The ORDER IS NOT VERIFIED.  Read the dump, then set it.
JAW_SLOT = None           # e.g. 4  (zero-based)

for nm, val in [("ROOT_KEY", ROOT_KEY), ("FIELD_SRATE", FIELD_SRATE),
                ("FIELD_NOISE", FIELD_NOISE), ("FIELD_REST", FIELD_REST),
                ("JAW_SLOT", JAW_SLOT)]:
    if val is None:
        raise SystemExit(
            f"\nSTOP. {nm} is unset.\n"
            "Read the PHASE 1 dump above, set the four constants and JAW_SLOT to\n"
            "what the file actually contains, then rerun. Do not guess.\n"
            "Cross-check the block order against the GigaDB readme for dataset\n"
            "100295 before trusting JAW_SLOT."
        )

# Channel names come from MOABB, which hardcodes the Cho2017 montage.  The raw
# .mat carries the same channel order but no names, so we borrow them and then
# assert the counts agree.  If this assertion fires, stop: a silent channel
# misalignment would invalidate every downstream number.
sess = ds.get_data(subjects=[min(raw_paths)])
raw_mi = None
for by_session in sess.values():
    for by_run in by_session.values():
        for run in by_run.values():
            raw_mi = run
            break
        break
    break
if raw_mi is None:
    raise RuntimeError("MOABB returned no runs for this subject")
mi_ch = [c for c in raw_mi.ch_names if c.upper() not in ("STIM", "STI 014")]
print(f"\n[PHASE 2] MOABB reports {len(mi_ch)} non-stim channels")

missing = [c for c in COMMON19 if c not in mi_ch]
if missing:
    raise RuntimeError(f"COMMON19 channels absent from Cho2017 montage: {missing}")
idx19 = np.array([mi_ch.index(c) for c in COMMON19])
print(f"  all 19 COMMON19 channels present, indices {idx19.tolist()}")

# ======================================================================
# PHASE 3  ---  measurement
# ======================================================================


def unwrap(struct, field):
    v = struct[field]
    while isinstance(v, np.ndarray) and v.dtype == object and v.size == 1:
        v = v.ravel()[0]
    return v


def as_channels_by_time(a, n_expected):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"expected 2-D block, got shape {a.shape}")
    if a.shape[0] == n_expected:
        return a
    if a.shape[1] == n_expected:
        return a.T
    raise ValueError(f"neither axis of {a.shape} matches {n_expected} channels")


def band_rms(block, sfreq_in, band, ch_idx):
    """CAR19 over COMMON19, notch, band-pass, then RMS per 4 s window."""
    x = block[ch_idx, :]
    x = x - x.mean(axis=0, keepdims=True)          # CAR over exactly 19
    g = int(round(sfreq_in / np.gcd(int(sfreq_in), int(SFREQ_TARGET))))
    up = int(SFREQ_TARGET * g / sfreq_in)
    x = resample_poly(x, up, g, axis=-1)
    x = mne.filter.notch_filter(x, SFREQ_TARGET, [NOTCH_HZ], verbose=False)
    x = mne.filter.filter_data(x, SFREQ_TARGET, band[0], band[1], verbose=False)
    n = int(WINDOW_S * SFREQ_TARGET)
    k = x.shape[-1] // n
    if k == 0:
        return np.array([])
    w = x[:, : k * n].reshape(x.shape[0], k, n)
    return np.sqrt((w ** 2).mean(axis=(0, 2)))     # one RMS per window


rows = []
print("\n[PHASE 3] measuring")
print(f"{'subj':>5} {'n_jaw':>6} {'n_rest':>7} "
      f"{'jaw 8-32':>10} {'rest 8-32':>10} {'ratio':>7} "
      f"{'jaw broad':>10} {'rest broad':>11}")

for s, path in sorted(raw_paths.items()):
    try:
        m = loadmat(path, squeeze_me=False, struct_as_record=True)
        root = m[ROOT_KEY].ravel()[0]
        sf = float(np.asarray(unwrap(root, FIELD_SRATE)).ravel()[0])

        noise = unwrap(root, FIELD_NOISE)
        noise_flat = noise.ravel() if noise.dtype == object else None
        jaw = (noise_flat[JAW_SLOT] if noise_flat is not None
               else np.asarray(noise)[JAW_SLOT])
        rest = unwrap(root, FIELD_REST)

        jaw = as_channels_by_time(jaw, len(mi_ch))
        rest = as_channels_by_time(rest, len(mi_ch))

        jd = band_rms(jaw, sf, DECODER_BAND, idx19)
        rd = band_rms(rest, sf, DECODER_BAND, idx19)
        jb = band_rms(jaw, sf, BROAD_BAND, idx19)
        rb = band_rms(rest, sf, BROAD_BAND, idx19)
        if jd.size == 0 or rd.size == 0:
            raise ValueError("block shorter than one 4 s window")

        # Power, not amplitude.  Median over windows, robust to a stray clench.
        pj, pr = float(np.median(jd) ** 2), float(np.median(rd) ** 2)
        ratio = pj / pr if pr > 0 else np.inf
        rows.append(dict(subject=s, sfreq=sf, n_jaw=int(jd.size), n_rest=int(rd.size),
                         jaw_power_decoder=pj, rest_power_decoder=pr, ratio=ratio,
                         jaw_rms_broad=float(np.median(jb)),
                         rest_rms_broad=float(np.median(rb)),
                         absolute_increase=bool(pj > pr)))
        print(f"{s:5d} {jd.size:6d} {rd.size:7d} {pj:10.4f} {pr:10.4f} "
              f"{ratio:7.2f} {np.median(jb):10.4f} {np.median(rb):11.4f}")
    except Exception as e:
        print(f"{s:5d}  FAILED  {type(e).__name__}: {e}")

if not rows:
    raise RuntimeError("no subject measured, nothing to gate on")

# ======================================================================
# PHASE 4  ---  the preregistered gate
# ======================================================================
n_abs = sum(r["absolute_increase"] for r in rows)
n_ratio = sum(r["ratio"] >= GATE_MIN_ABSOLUTE_RATIO for r in rows)
passed = (n_abs >= GATE_MIN_SUBJECTS) and (n_ratio >= GATE_MIN_SUBJECTS)

print("\n" + "=" * 72)
print("PREREGISTERED GATE")
print(f"  absolute 8-32 Hz power increase        : {n_abs}/{len(rows)} "
      f"(need >= {GATE_MIN_SUBJECTS})")
print(f"  power ratio >= {GATE_MIN_ABSOLUTE_RATIO}                  : {n_ratio}/{len(rows)} "
      f"(need >= {GATE_MIN_SUBJECTS})")
print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
print("=" * 72)

if passed:
    print("\nJaw clenching is a usable donor for an 8-32 Hz decoder, within subject.")
    print("Section 9.4 must be rewritten: the impossibility claim is false, and")
    print("the honest statement is that OCULAR artifacts cannot serve as donors,")
    print("which is a narrower and defensible result.")
    print("Next: Stage T2 builds the bank.  Do NOT look at any harm statistic")
    print("before Stage T3 freezes the preregistration.")
else:
    print("\nGate failed.  Do not rescue it by loosening the threshold.")
    print("Report it as a measured negative, exactly as the ocular probe was")
    print("reported, and keep the impossibility claim out of the text either way.")

with open(f"{OUT_DIR}/T1_FEASIBILITY.json", "w") as f:
    json.dump(dict(stage="T1", seed=SEED, subjects=PROBE_SUBJECTS,
                   decoder_band=DECODER_BAND, window_s=WINDOW_S,
                   notch_hz=NOTCH_HZ, sfreq_target=SFREQ_TARGET,
                   gate=dict(min_subjects=GATE_MIN_SUBJECTS,
                             min_ratio=GATE_MIN_ABSOLUTE_RATIO,
                             require_absolute=GATE_REQUIRE_ABSOLUTE_INCREASE),
                   n_absolute_increase=n_abs, n_ratio_pass=n_ratio,
                   verdict="PASS" if passed else "FAIL",
                   rows=rows), f, indent=2)
print(f"\nwritten: {OUT_DIR}/T1_FEASIBILITY.json")
