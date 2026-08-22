# REPO STATUS: CURRENT. Identifies the jaw-clench slot from physics, 5/5 subjects.
# Run order, retractions and full context: notebooks_cho/RUN_ORDER.md

"""
STAGE T1c  ---  diagnose T1b, then identify all five non-task blocks.

Why this replaces T1b
---------------------
T1b returned a physically impossible result: five blocks that are supposed to
be eye blinks, two kinds of eye movement, head movement and jaw clenching came
out within 3 per cent of each other on broadband RMS, with HF/LF pinned at
0.007 everywhere, and the high-frequency argmax landing on a different random
central-parietal channel in each subject.

Leading hypothesis: the fault is in the T1b processing, not the data.  T1b
band-passed a 10 s block (2500 samples after resampling to 250 Hz) from 0.5 Hz
upward.  An FIR at 0.5 Hz needs on the order of 1650 samples of transient, so
the transient covers most of the block.  BioSemi ActiveTwo raw data carries
large per-channel DC offsets, and ringing off that step swamps the physiology.

That is a hypothesis, not an established fact.  This script tests it.

Design decisions
----------------
1. PHASE 0 touches nothing.  No filter, no resample.  If the five blocks
   already differ several-fold in raw amplitude, the T1b filter was the fault
   and everything is recoverable.  If they are identical raw, the file or the
   loader is handing back the same array and the Cho branch dies here.
2. Block hashes.  Three lines, and they settle whether the five cells are
   genuinely distinct arrays.
3. No time-domain band-pass anywhere.  Band power comes from a Welch PSD at
   the native 512 Hz, so there is no filter to design and no edge to ring.
   Mains is handled by dropping bins near 60, 120 and 180 Hz from the
   integration, not by notching.
4. Per-channel demean and linear detrend before the PSD, which is mandatory
   for raw BioSemi and was missing in T1b.
5. Identification is attempted for all five blocks, not just jaw clenching.
   If the physics also recovers a frontopolar low-frequency blink and an
   antisymmetric left/right eye movement, the whole cell ordering becomes
   verifiable rather than assumed.

Expected signatures, stated before the numbers are seen:
    jaw clenching      highest 40-100 Hz power, flattest spectrum,
                       HF maximum over temporal sites (T7/T8/FT7/FT8/F7/F8)
    eye blink          highest 1-4 Hz power, frontopolar, spiky, high kurtosis
    eyeball up/down    frontopolar, low frequency, lower kurtosis than blink
    eyeball left/right low frequency, NEGATIVE correlation between AF7 and AF8
    head movement      low frequency, large, spread over all channels

Still no harm statistic of any kind.  Stage T3 has not been frozen yet.

Run as one Kaggle CPU cell.  The .mat files are cached from T1, no download.
"""

import hashlib
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

SEED = 20260813
SFREQ_NATIVE = 512.0            # no resampling in this stage, on purpose
DECODER_BAND = (8.0, 32.0)
LF_BAND = (1.0, 4.0)
HF_BAND = (40.0, 100.0)
MAINS_HZ = [60.0, 120.0, 180.0]
MAINS_HALFWIDTH = 2.0
NPERSEG = 512                   # 1 s segments -> 1 Hz resolution
PROBE_SUBJECTS = [1, 2, 3, 4, 5]
N_EEG = 64

COMMON19 = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
            "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP3", "CP1", "CP2", "CP4", "Pz", "POz"]

TEMPORAL = ["T7", "T8", "FT7", "FT8", "TP7", "TP8", "F7", "F8"]
FRONTOPOLAR = ["Fp1", "Fp2", "Fpz", "AF7", "AF8", "AF3", "AF4", "AFz"]
LATERAL_PAIR = ("AF7", "AF8")   # antisymmetry probe for horizontal eye movement

ROOT_KEY, FIELD_SRATE, FIELD_NOISE, FIELD_REST = "eeg", "srate", "noise", "rest"

GATE_MIN_SUBJECTS = 4
GATE_MIN_ABSOLUTE_RATIO = 1.50

OUT_DIR = "/kaggle/working"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 78)
print("STAGE T1c  diagnose, then identify all five blocks")
print("=" * 78)

try:
    import mne
    import moabb
    from moabb.datasets import Cho2017
    from scipy.io import loadmat
    from scipy.signal import detrend, welch
    from scipy.stats import kurtosis
except ImportError:
    os.system(f"{sys.executable} -m pip install -q mne moabb")
    import mne
    import moabb
    from moabb.datasets import Cho2017
    from scipy.io import loadmat
    from scipy.signal import detrend, welch
    from scipy.stats import kurtosis

mne.set_log_level("ERROR")
print(f"mne {mne.__version__}  moabb {moabb.__version__}  numpy {np.__version__}")

ds = Cho2017()

# ----------------------------------------------------------------------
# Channel names.  The T1b run already proved the montage is the standard
# BioSemi 64 order followed by four arm EMG channels, so this is a re-assert.
# ----------------------------------------------------------------------
raw_mi = None
for a in ds.get_data(subjects=[PROBE_SUBJECTS[0]]).values():
    for b in a.values():
        for r in b.values():
            raw_mi = r
            break
        break
    break
ch_all = [c for c in raw_mi.ch_names if c.upper() not in ("STIM", "STI 014")]
eeg_names = ch_all[:N_EEG]
idx19 = np.array([ch_all.index(c) for c in COMMON19])
i_temporal = [eeg_names.index(c) for c in TEMPORAL if c in eeg_names]
i_frontopolar = [eeg_names.index(c) for c in FRONTOPOLAR if c in eeg_names]
i_left, i_right = (eeg_names.index(LATERAL_PAIR[0]) if LATERAL_PAIR[0] in eeg_names else None,
                   eeg_names.index(LATERAL_PAIR[1]) if LATERAL_PAIR[1] in eeg_names else None)
print(f"channels {len(ch_all)}, EEG {N_EEG}, COMMON19 max index {idx19.max()}")
print(f"temporal probes {len(i_temporal)}, frontopolar probes {len(i_frontopolar)}")


def unwrap(struct, field):
    v = struct[field]
    while isinstance(v, np.ndarray) and v.dtype == object and v.size == 1:
        v = v.ravel()[0]
    return v


def load_blocks(subject):
    p = ds.data_path(subject)
    p = p[0] if isinstance(p, (list, tuple)) else p
    m = loadmat(p, squeeze_me=False, struct_as_record=True)
    root = m[ROOT_KEY].ravel()[0]
    sf = float(np.asarray(unwrap(root, FIELD_SRATE)).ravel()[0])
    noise = unwrap(root, FIELD_NOISE)
    blocks = list(noise.ravel()) if noise.dtype == object else list(np.asarray(noise))
    out = []
    for b in blocks:
        b = np.asarray(b, dtype=np.float64)
        out.append(b if b.shape[0] == len(ch_all) else b.T)
    rest = np.asarray(unwrap(root, FIELD_REST), dtype=np.float64)
    rest = rest if rest.shape[0] == len(ch_all) else rest.T
    return sf, out, rest


def band_power(x, sf, band):
    """Welch band power per channel. No filter is designed, so no edge effects."""
    x = detrend(x, axis=-1, type="linear")          # kills DC and slow drift
    nps = int(min(NPERSEG, x.shape[-1]))
    f, P = welch(x, fs=sf, nperseg=nps, noverlap=nps // 2, axis=-1)
    mask = (f >= band[0]) & (f <= band[1])
    for m0 in MAINS_HZ:                              # drop mains, never notch
        mask &= ~((f > m0 - MAINS_HALFWIDTH) & (f < m0 + MAINS_HALFWIDTH))
    if not mask.any():
        return np.zeros(x.shape[0])
    return np.trapezoid(P[..., mask], f[mask], axis=-1)


def spectral_flatness(x, sf, lo=5.0, hi=100.0):
    """Geometric over arithmetic mean of the PSD. EMG is flat, near 1."""
    x = detrend(x, axis=-1, type="linear")
    nps = int(min(NPERSEG, x.shape[-1]))
    f, P = welch(x, fs=sf, nperseg=nps, noverlap=nps // 2, axis=-1)
    m = (f >= lo) & (f <= hi)
    for m0 in MAINS_HZ:
        m &= ~((f > m0 - MAINS_HALFWIDTH) & (f < m0 + MAINS_HALFWIDTH))
    Q = np.maximum(P[..., m], 1e-300)
    return float(np.median(np.exp(np.log(Q).mean(-1)) / Q.mean(-1)))


# ======================================================================
# PHASE 0  ---  raw, nothing applied
# ======================================================================
print("\n" + "-" * 78)
print("PHASE 0  raw amplitudes, zero processing")
print("-" * 78)

phase0 = []
for s in PROBE_SUBJECTS:
    sf, blocks, rest = load_blocks(s)
    print(f"\nsubject {s:2d}  srate {sf:g}  blocks {len(blocks)}  "
          f"lengths {[b.shape[1] for b in blocks]}")
    print(f"  {'slot':>4} {'sha256[:12]':>14} {'|DC| med':>10} "
          f"{'std med':>10} {'ptp max':>11} {'kurt med':>9}")
    hashes = []
    for j, b in enumerate(blocks):
        e = b[:N_EEG]
        h = hashlib.sha256(np.ascontiguousarray(e).tobytes()).hexdigest()[:12]
        hashes.append(h)
        dc = float(np.median(np.abs(e.mean(-1))))
        sd = float(np.median(e.std(-1)))
        pp = float(np.max(e.max(-1) - e.min(-1)))
        ku = float(np.median(kurtosis(e - e.mean(-1, keepdims=True), axis=-1)))
        phase0.append(dict(subject=s, slot=j, sha=h, dc_median=dc,
                           std_median=sd, ptp_max=pp, kurtosis_median=ku))
        print(f"  {j:>4} {h:>14} {dc:10.2f} {sd:10.2f} {pp:11.2f} {ku:9.2f}")
    if len(set(hashes)) < len(hashes):
        print("  !! DUPLICATE BLOCKS: the five cells are not five distinct arrays")

stds = {}
for r in phase0:
    stds.setdefault(r["subject"], []).append(r["std_median"])
spread = {s: max(v) / max(min(v), 1e-12) for s, v in stds.items()}
print("\nraw std spread within subject (max/min across the five blocks):")
for s, v in spread.items():
    print(f"  subject {s:2d}: {v:6.2f}x")

if max(spread.values()) < 1.5:
    print("""
VERDICT PHASE 0: the five blocks are near-identical in RAW amplitude.
That is not consistent with eye blinks versus jaw clenching, so the T1b
filter was NOT the only problem. Before writing anything, plot slot 0 and
slot 3 for subject 1 side by side and look at them. If they look the same,
the Cho branch is dead and section 9.4 still loses its impossibility claim.
""")
else:
    print(f"""
VERDICT PHASE 0: blocks differ up to {max(spread.values()):.1f}x in raw amplitude.
The T1b result was a processing artefact, as suspected. Continue.
""")

# ======================================================================
# PHASE 1 and 2  ---  Welch band powers and identification
# ======================================================================
print("-" * 78)
print("PHASE 1/2  Welch band power and per-block identification")
print("-" * 78)

rows, votes = [], {}
for s in PROBE_SUBJECTS:
    sf, blocks, rest = load_blocks(s)
    print(f"\nsubject {s:2d}")
    print(f"  {'slot':>4} {'P(1-4)':>11} {'P(8-32)':>11} {'P(40-100)':>11} "
          f"{'HF/LF':>8} {'flat':>6} {'HF top':>8} {'LF top':>8} {'r(AF7,AF8)':>11}")
    per = []
    for j, b in enumerate(blocks):
        e = b[:N_EEG]
        lf = band_power(e, sf, LF_BAND)
        db = band_power(e, sf, DECODER_BAND)
        hf = band_power(e, sf, HF_BAND)
        flat = spectral_flatness(e, sf)
        hf_top = eeg_names[int(np.argmax(hf))]
        lf_top = eeg_names[int(np.argmax(lf))]
        if i_left is not None and i_right is not None:
            d = detrend(e[[i_left, i_right]], axis=-1, type="linear")
            r_lr = float(np.corrcoef(d[0], d[1])[0, 1])
        else:
            r_lr = np.nan
        rec = dict(subject=s, slot=j,
                   p_lf=float(lf.mean()), p_dec=float(db.mean()), p_hf=float(hf.mean()),
                   hf_lf=float(hf.mean() / max(lf.mean(), 1e-300)),
                   flatness=flat, hf_top=hf_top, lf_top=lf_top,
                   hf_temporal=float(hf[i_temporal].mean()),
                   hf_frontopolar=float(hf[i_frontopolar].mean()),
                   lf_frontopolar=float(lf[i_frontopolar].mean()),
                   corr_lateral=r_lr)
        per.append(rec)
        rows.append(rec)
        print(f"  {j:>4} {lf.mean():11.3e} {db.mean():11.3e} {hf.mean():11.3e} "
              f"{rec['hf_lf']:8.4f} {flat:6.3f} {hf_top:>8} {lf_top:>8} {r_lr:11.3f}")

    jaw = max(per, key=lambda r: r["hf_lf"])
    blink = max(per, key=lambda r: r["lf_frontopolar"])
    horiz = min(per, key=lambda r: r["corr_lateral"])
    votes.setdefault("jaw", []).append(jaw["slot"])
    votes.setdefault("blink", []).append(blink["slot"])
    votes.setdefault("horizontal", []).append(horiz["slot"])
    print(f"   -> jaw candidate  slot {jaw['slot']}  HF/LF {jaw['hf_lf']:.4f}, "
          f"temporal/frontopolar HF "
          f"{jaw['hf_temporal'] / max(jaw['hf_frontopolar'], 1e-300):.2f}, "
          f"flatness {jaw['flatness']:.3f}")
    print(f"   -> blink candidate slot {blink['slot']}, "
          f"horizontal-eye candidate slot {horiz['slot']} "
          f"(r = {horiz['corr_lateral']:.3f})")

print("\n" + "-" * 78)
for k, v in votes.items():
    print(f"  {k:>11} candidate per subject: {v}  "
          f"{'CONSISTENT' if len(set(v)) == 1 else 'INCONSISTENT'}")

jaw_slots = set(votes["jaw"])
identified = len(jaw_slots) == 1
conflict = identified and (set(votes["blink"]) == jaw_slots or
                           set(votes["horizontal"]) == jaw_slots)

if not identified:
    print("""
STOP. Subjects disagree on the jaw slot. Cell order is not stable across
files. Do not take a majority vote. Plot the blocks by hand.
""")
    raise SystemExit(0)
if conflict:
    print("""
STOP. The same slot won both the muscle test and an ocular test. The
discriminators are not separating anything, which usually means the blocks
are not what the publication describes. Plot before proceeding.
""")
    raise SystemExit(0)

JAW_SLOT = jaw_slots.pop()
print(f"\nJAW_SLOT = {JAW_SLOT}, and the ocular candidates land on other slots,")
print("which is the corroboration the T1b run never produced.")

# ======================================================================
# PHASE 3  ---  the preregistered feasibility gate
# ======================================================================
print("\n" + "-" * 78)
print("PHASE 3  absolute 8-32 Hz power, jaw block versus the same subject's rest")
print("-" * 78)
print(f"  {'subj':>5} {'jaw P(8-32)':>13} {'rest P(8-32)':>14} {'ratio':>8} {'abs.up':>7}")

gate_rows = []
for s in PROBE_SUBJECTS:
    sf, blocks, rest = load_blocks(s)
    pj = float(band_power(blocks[JAW_SLOT][:N_EEG][idx19], sf, DECODER_BAND).mean())
    pr = float(band_power(rest[:N_EEG][idx19], sf, DECODER_BAND).mean())
    ratio = pj / pr if pr > 0 else np.inf
    gate_rows.append(dict(subject=s, jaw_power=pj, rest_power=pr,
                          ratio=ratio, absolute_increase=bool(pj > pr)))
    print(f"  {s:5d} {pj:13.4e} {pr:14.4e} {ratio:8.2f} "
          f"{'yes' if pj > pr else 'NO':>7}")

n_abs = sum(r["absolute_increase"] for r in gate_rows)
n_ratio = sum(r["ratio"] >= GATE_MIN_ABSOLUTE_RATIO for r in gate_rows)
passed = n_abs >= GATE_MIN_SUBJECTS and n_ratio >= GATE_MIN_SUBJECTS

print("\n" + "=" * 78)
print(f"  absolute increase : {n_abs}/{len(gate_rows)} (need >= {GATE_MIN_SUBJECTS})")
print(f"  ratio >= {GATE_MIN_ABSOLUTE_RATIO}      : {n_ratio}/{len(gate_rows)} "
      f"(need >= {GATE_MIN_SUBJECTS})")
print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
print("=" * 78)
print("""
Whatever the verdict, three things are now fixed for the Cho branch and must
be committed as a NEW entry in FROZEN_CONSTANTS.json, not an overwrite:
  epoch 3 s, because Cho2017 has no 4 s of imagery
  notch 60 Hz, because the recordings are Korean mains
  band power by Welch PSD, never by a time-domain filter on a short block
And read the imagery arrays straight from the .mat. The MOABB loader stacks
7 s trials through zero buffers, so filtering its Raw object and then epoching
puts an edge artefact on every trial boundary.
""")

with open(f"{OUT_DIR}/T1c_DIAGNOSTIC.json", "w") as f:
    json.dump(dict(stage="T1c", seed=SEED, subjects=PROBE_SUBJECTS,
                   raw_std_spread=spread, phase0=phase0, blocks=rows,
                   votes=votes, jaw_slot=int(JAW_SLOT), gate=gate_rows,
                   n_absolute_increase=n_abs, n_ratio_pass=n_ratio,
                   verdict="PASS" if passed else "FAIL"), f, indent=2, default=float)
print(f"written: {OUT_DIR}/T1c_DIAGNOSTIC.json")
