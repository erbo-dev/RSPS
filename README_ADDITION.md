<!-- Append this to the existing README.md, after the "Headline numbers" table. -->

---

## Within-subject branch: is the independence an artifact of the design?

The strongest objection to this study is that donor and host come from
different people in unsynchronised recordings, so `E[zu] = 0` follows
algebraically and the result is arithmetic rather than physiology. Limitation
10.3.4 said so plainly and added that the value of `|rho|` for one person
recorded at one sitting was unknown.

It is no longer unknown. It was measured.

**Data.** Cho et al. 2017, GigaScience 6(7):gix034, GigaDB 100295. 52
participants, 64 channels in the 10-10 system, 512 Hz, BioSemi ActiveTwo. Each
participant performed left and right hand motor imagery **and** six non-task
blocks, one of them jaw clenching, **inside the same session**. Same person,
same montage, same impedances, same amplifier.

The jaw block was identified by physics rather than by the publication's prose
ordering, which would have pointed at slot 4. Slot 4 holds head movement. All
five probe subjects agreed on slot 3 on three independent signatures.

### Headline numbers, within-subject branch

| Result | Value |
|---|---|
| Feasibility gate, MI_d >= 3.0 | 45 of 47 (threshold 20) |
| Eligible development subjects | 19 of 32 |
| Eligible sealed subjects, opened once | 7 of 15 |
| \|rho\|, own donor vs stranger's donor | 0.0231 vs 0.0230 |
| Paired difference | +0.0001, p = 0.94, n = 23 (+0.0003, p = 0.71, n = 18) |
| Excess over a phase-randomised floor, arm A | -0.0001, 95% CI [-0.0004, +0.0002] |
| Surrogate covariance preservation | 5.0e-16 |
| Interaction-required, development | 0.0484 [0.0336, 0.0534] |
| Interaction-required, sealed group | 0.0336 [0.0205, 0.0583] |
| Primary endpoint, A minus B symmetry | +0.0942 [-0.1392, +0.1844], p = 0.47 |
| Natural SNR, 47 subjects | median -10.79 dB; 75.9% harsher than the frozen grid, 0% milder |
| Saturation correlation, new cohort | Spearman -0.514, p = 0.024, n = 19 (was -0.50, p = 0.17, n = 9) |
| Preregistered prediction, monotone -7 to -30 dB | held, on subjects never opened |
| Preregistered prediction, level below 0.015 at -15 dB | refuted, 0.0192 |
| Preregistered criteria on the sealed group | 3 of 5 passed |

**Conclusion.** Recording the donor from the same person in the same session
introduces no measurable linear dependence. The independence is not an artifact
of the design. What this does **not** close is simultaneity with the task: jaw
and imagery are still separate blocks, and no open dataset we know of provides
the simultaneous case.

**What is measured, precisely.** The linear correlation of CSP projections over
a 2.5 s window. Higher-order dependence is not excluded by this. The measure is
the right one nonetheless, because the cross-term `2*E[zu]` is that linear
correlation.

### Nine retractions

Building this branch forced us to withdraw nine of our own results, including a
donor bank declared void in full and a validity gate that turned out to be
algebraically incapable of failing. Every one is documented with its mechanism,
the check that caught it, and what survived, in
[`protocol/LEDGER_CHO.md`](protocol/LEDGER_CHO.md). The scripts that produced
the void results are still in the repository, marked in their headers. A ledger
whose evidence has been deleted is not a ledger.

### Layout

```
notebooks_cho/   thirteen stages plus the harm engine; RUN_ORDER.md has status per file
protocol/        preregistrations, both amendments, frozen constants, the ledger
results_cho/     result JSONs from every current stage
docs_kz/         Kazakh drafts of the affected document sections
```

### Instrument verification

The harm engine self tests on import: SNR convention to 3.55e-15, the algebraic
power decomposition against the actual contaminated waveform to 2.60e-15, the
vectorised path against an independent triple loop, zero harm at lambda = 0,
and a non-vacuity check that fails the test outright if no harm event was
produced, since the earlier checks would then have compared zero with zero.

It was cross-checked against this repository's own `harmkit` on identical
synthetic inputs: all counters identical across four SNR grids and five seeds,
scale factors identical to `harmkit.scale_for_snr` with a difference of exactly
0.0.
