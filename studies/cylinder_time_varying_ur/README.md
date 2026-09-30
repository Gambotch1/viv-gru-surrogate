# Cylinder time-varying-Ur continuation study

An operating-condition continuation test: one continuous coupled
simulation in which the reduced velocity Ur changes sequentially through
the cylinder sweep, as opposed to independently-initialised fixed-Ur runs.

**Status: complete.** The production runs (one seed each; schedules
ascending, ascending_cosine and triangular_cosine, run with
`--single_seed_production`) are reported in thesis Sec. 5.7 and
Appendix F. Model: `gru_cylinder200_nd_context_noacc` (h/D, hdot/U,
Ur context; no acceleration).

## Directory layout

- `scripts/time_varying_coupled.py` -- `run_coupled_viv_time_varying_ur`,
  the generalization of `viv_analysis.coupled_inference.run_coupled_viv`
  to a time-varying `Ur_schedule` array. Imports `Newmark_beta` directly
  from production, unmodified; the structural solver itself is reused
  verbatim. Only the per-step orchestration (force scaling / ND-
  normalization now driven by a schedule instead of one fixed scalar) is
  new. Proven to collapse to `run_coupled_viv`'s own output within
  floating-point tolerance for a constant schedule (float32 throughout,
  matching production exactly -- see Tests).
- `scripts/schedules.py` -- Ur(t) schedule builders: `ascending`
  (instantaneous transitions), `ascending_cosine` (one-Tn smooth ramp
  between plateaus), `triangular` (2->12->2, no reset at the reversal;
  `transition="cosine"` is the approved production variant --
  `transition="instantaneous"` remains available for diagnostics only).
  Every returned schedule dict records `transition_duration_convention`
  explicitly (`"added_to_dwell"` for the two cosine variants -- the
  transition time is ADDED ON TOP of the full requested `dwell_s` at each
  plateau, never carved out of it; `"instantaneous_zero_added_time"` for
  the instantaneous variant).
- `scripts/run_time_varying_sweep.py` -- driver. `--smoke` bypasses the
  production gate and uses the development checkpoint directly (for ad
  hoc checks; the canonical smoke test is `run_smoke_test.py`, below, and
  doesn't go through this flag). Without `--smoke`, it calls
  `assert_stage2_selection_frozen()` (see "Production gate") and then
  runs the requested schedule once per each of the 3 selected seeds
  independently, saving each seed's result/receipt separately. Performs
  the CFD warm-up **exactly once** at Ur=2.00 (reusing `warmup_history`
  verbatim) per seed run. Never touches the repo's existing fixed-Ur
  `results/` directories.
- `scripts/run_smoke_test.py` -- the required smoke test (Ur=[2.0, 2.5],
  dwell=2Tn=10s), run for both an instantaneous and a one-Tn cosine
  transition. Reports **adjacent-step changes** in h/h_dot near the
  transition (not "discontinuities" -- a continuous state still changes
  over one timestep; only U/force actually jump for the instantaneous
  schedule). See "Smoke test results" below.
- `scripts/plot_time_varying_ur.py` -- thesis figures:
  - `plot_time_varying_panels`: aligned panels (Ur(t), h/D, hdot/(fn*D),
    CL_hat, envelope A/D) for a full sweep. hdot is normalized by the
    CONSTANT fn*D, not the time-varying U(t), to avoid an artificial
    normalization jump at every transition. The envelope panel calls
    `envelope_or_unavailable`, which reports "unavailable" instead of
    plotting a 1-2-point line whenever the window doesn't span enough
    cycles (< `MIN_ENVELOPE_WINDOWS`=5) for the estimate to mean anything.
  - `plateau_summary`: compares the final n_cycles of each dwell plateau
    against independent fixed-Ur CFD/GRU references; a plateau is only
    labelled `stabilised` if its envelope is both low-variance AND backed
    by enough cycles (`sufficient_data_for_n_cycles`) -- never called a
    steady-state amplitude otherwise.
  - `plot_transition_window` + `LOCKIN_REGION_TRANSITIONS`: the 3 main
    transition figures -- `onset` (3.50->4.00), `lockin` (5.25->5.50),
    `departure_from_lockin` (6.00->6.25, the 4.5x amplitude collapse that
    terminates lock-in) -- plus `POST_LOCKIN_APPENDIX_TRANSITION`
    (7.00->8.00, the settled decaying tail) kept as a separate appendix
    figure. All windowed to 5 Tn before / 15 Tn after the named
    transition, with a `fixed_xlim` parameter so all figures in the main
    set share identical axis limits/window duration. Transition Ur pairs
    are confirmed against the real fixed-Ur CFD amplitude curve (see the
    module for the full table and reasoning), not chosen arbitrarily.
    Structurally verified against synthetic multi-condition data (locates
    the right transition, windows and labels correctly); the real figures
    await the actual full sweep.
  - `plot_multiseed_panels` + `aggregate_envelope_across_seeds` +
    `aggregate_scalar_metrics_across_seeds`: multi-seed aggregation.
    **Never** takes a pointwise median of raw h(t)/hdot(t)/CL(t) across
    seeds -- small phase differences between seeds would attenuate the
    median waveform (demonstrated in
    `test_envelope_aggregation_not_attenuated_by_seed_phase_differences`:
    a naive pointwise median of 3 same-amplitude, different-phase
    synthetic seeds understates the true amplitude, while the median of
    their per-seed envelopes does not). Aggregates via the median
    envelope with an IQR band, direct median/IQR of scalar metrics
    (RMS amplitude, frequency, energy), and exactly one **predeclared**
    representative seed (`REPRESENTATIVE_SEED = 123`, fixed in the module
    before any results exist) for the raw oscillatory time histories,
    with the other seeds overlaid faintly for context.
- `tests/test_time_varying_ur.py` -- 18 tests: the 9 required regression
  tests, 5 for the production gate, and 4 for multi-seed aggregation. All
  use a small synthetic model/scalers except where the claim under test
  specifically requires exercising the real CFD-loading path (with
  `merge_dataframes`/`compute_kinematics` monkeypatched to tiny synthetic
  data, so the real multi-minute cylinder200 load never runs inside the
  test suite).

## Production gate

Full (non-`--smoke`) sweeps require:

    studies/gru_architecture_history_sensitivity/manifests/selection_manifest_stage2.json
    with "frozen": true AND "selection_complete": true

`selection_complete` is set by that OTHER study's `select_configuration.py
--freeze` only when `--stage 2` (its final stage) -- a frozen Stage 1
manifest unblocks that study's own Stage 2 job generation but does not
mean the architecture/history-length choice is final, so it does not
satisfy this gate. Without this, a full 21-condition sweep would run
against a configuration that could still be superseded, exposing held-out
validation conditions across the whole Ur range before model selection is
actually finished. `assert_stage2_selection_frozen()` also requires
exactly 3 `selected_run_dirs` (one per seed 123/456/789) and raises
`SystemExit` otherwise.

## Input-history handling (the critical invariant)

The rolling model-input `history` buffer is a FIFO exactly as in
`run_coupled_viv` (`np.roll` + overwrite the last row). A row, once
written using the Ur/U active at that step, is **never** revisited or
renormalized when Ur changes later -- it only leaves the window naturally
as the roll evicts it. This is not an added safeguard; it is the same
mechanism `run_coupled_viv` already uses for its (constant) Ur, simply
carried through unchanged while U/Ur now vary per step. Verified directly
by `test_history_rows_retain_original_U_and_Ur`.

## Tests (18/18 pass)

1. Constant Ur=5.5 schedule reproduces `run_coupled_viv`'s own fixed-Ur
   trajectory within floating-point tolerance.
2. h, h_dot are continuous across a Ur transition (no explicit
   jump/reset applied to structural state).
3. U, dynamic pressure, and force scaling change correctly with Ur.
4. History rows retain their original U_j/Ur_j after a later transition.
5. Acceleration is absent from the GRU input (hard-rejects "acc" in
   `input_cols`).
6. CFD warm-up occurs exactly once per sweep, regardless of schedule
   length.
7. The production Newmark implementation is reused (imported, not
   redefined; behaviorally cross-checked).
8. Schedule durations and transition locations are exact.
9. Existing fixed-Ur result files are untouched by a sweep run.
10-14. The production gate: fails without a manifest, fails if not
   frozen, fails if `selection_complete` is not true, fails without
   exactly 3 `selected_run_dirs`, succeeds when all three hold.
15-18. Multi-seed aggregation: envelope-median is not attenuated by seed
   phase differences (a naive pointwise median of the raw waveform is,
   demonstrated directly); scalar-metric median/IQR; the representative
   seed is a fixed module constant, not data-dependent; rejects a
   representative seed not present in the results.

Run with:

    cd studies/cylinder_time_varying_ur/tests
    PYTHONPATH=../../../src:../scripts python3 -m pytest -q

## Smoke test results (item 7 -- complete, no bugs found)

Real checkpoint, real CFD warm-up, batch job (64s runtime). Force-scale
ratio matches (2.5/2.0)^2 = 1.5625 exactly for the instantaneous
transition and ~1.0 for the cosine transition's smooth start; zero
regression error against a constant-Ur=2.0 baseline for the pre-transition
segment of both schedules; no NaNs/OOD warnings. Full report:
`results/smoke_report.json`, figures: `figures/smoke_panels_*.{pdf,png}`.

## Next steps (once the Stage 2 selection is frozen and complete)

1. `run_time_varying_sweep.py --schedule ascending` (instantaneous)
2. `run_time_varying_sweep.py --schedule ascending_cosine`
3. `run_time_varying_sweep.py --schedule triangular_cosine`

Each runs all 3 selected seeds automatically. Recommended review order:
review ascending and ascending-cosine receipts/plateau classifications
first, then run the triangular sweep. `plot_multiseed_panels`/
`aggregate_envelope_across_seeds`/`aggregate_scalar_metrics_across_seeds`
are ready for the resulting 3-seed data; the real
`plot_transition_window` figures for the 3 main transitions (plus the
post-lock-in appendix) await the actual full sweep.
