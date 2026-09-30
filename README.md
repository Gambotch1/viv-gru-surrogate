# GRU surrogate for vortex-induced vibration in a coupled fluid–structure loop

Code for the master's thesis *"Machine Learning Surrogates for VIV Prediction"* (Omar Ghariani, Bauhaus-Universität Weimar, Chair of Modelling and Simulation of Structures, August 2026).

A recurrent neural network (GRU) learns the lift coefficient C_L from the recent motion of a structure. It then replaces the CFD solver inside a two-way coupled simulation: the GRU predicts the lift, a Newmark-β integrator moves the structure, and the new motion is fed back to the GRU. The method is tested on two cases:

- a **circular cylinder** at Re = 200 (Leontini et al. benchmark), where the coupled surrogate reproduces the lock-in response (thesis Ch. 5);
- a section of the **Rio–Niterói bridge deck**, where the open-loop predictions are accurate but the coupled response decays (thesis Ch. 6).

The main lesson of the thesis is that open-loop accuracy does not guarantee a correct coupled response. Every model in this repository should therefore be judged in closed loop, with `evaluate_all.py`.

---

## Contents

1. [How the pipeline works](#1-how-the-pipeline-works)
2. [Repository layout](#2-repository-layout)
3. [Installation](#3-installation)
4. [Data](#4-data)
5. [Quick start](#5-quick-start)
6. [Step-by-step workflow](#6-step-by-step-workflow)
7. [Which script makes which thesis figure](#7-which-script-makes-which-thesis-figure)
8. [Conventions to know before changing the code](#8-conventions-to-know-before-changing-the-code)
9. [Tests](#9-tests)
10. [Results and reproducibility](#10-results-and-reproducibility)
11. [Known limitations and open problems](#11-known-limitations-and-open-problems)

---

## 1. How the pipeline works

```
Fluent .out monitor files          data/cylinder_Re_200/…, data/Bridge/…
        │
        ▼
preprocess.py        merge disp / C_L / C_D, drop the start, downsample (bridge),
                     compute velocity and acceleration; bridge result cached as parquet
        │
        ▼
train_gru.py         one-step (teacher-forced) training: window of past motion → C_L
                     → results/<model>/  (weights, scalers, split, settings)
        │
        ▼
coupled_inference.py one closed-loop run at one Ur:
                     GRU → force → Newmark-β → new state → GRU → …
                     → coupled_*.npz + .receipt.json
        │
        ▼
evaluate_all.py      runs coupled_inference for every Ur of a model and scores each run
                     (closed_loop_metrics.py, reference_quality.py) → sweep_results.csv
        │
        ▼
plotting/*.py        thesis figures from the saved results (no retraining)
```

Two extra pieces are only used for the bridge:

- **Rollout-informed refinement** (Sec. 6.6.1): `train_rollout.py`, using the differentiable loop in `rollout_training.py`.
- **Residual forcing test** (Sec. 6.6.2): `build_pooled_tf_residual.py` → `coupled_inference.py --residual_npz` → `plotting/summarize_stochastic_closure.py` → `plotting/plot_stochastic_closure_figures.py`.

Two self-contained studies live in `studies/`:

- `gru_architecture_history_sensitivity/`: network size and input-history length (Sec. 4.4.3, 5.2.1, Appendix C);
- `cylinder_time_varying_ur/`: one continuous run with Ur changing in time (Sec. 5.7, Appendix F).

---

## 2. Repository layout

```
src/viv_analysis/
    config.py                   all fixed parameters (SI units)
    utils.py                    project root, Ur case labels, time-gap splitting
    preprocess.py               read Fluent files, compute kinematics, bridge cache
    models/gru.py               GRU model, sequence dataset, scalers
    train_gru.py                one-step training
    evaluate.py                 open-loop metrics (MAE, RMSE, R²)
    coupled_inference.py        closed loop for one Ur; Newmark-β integrator
    evaluate_all.py             closed-loop sweep over all Ur of a model
    closed_loop_metrics.py      amplitude, frequency, energy, phase, stability label
    reference_quality.py        which bridge CFD cases can serve as a steady reference
    rollout_training.py         differentiable closed loop (torch)
    train_rollout.py            rollout-informed refinement of a bridge model
    residual_forcing.py         random lift forcing for the residual forcing test
    build_pooled_tf_residual.py residual data for the residual forcing test
    plotting/                   thesis figures and tables (see section 7)
studies/                        the two separate studies, each with its own README and tests
tests/                          tests of the main package
data/                           CFD data (not in the repository, see section 4)
results/                        everything the scripts write (not in the repository)
```

Every script runs as a module from the repository root:

```bash
PYTHONPATH=src python -m viv_analysis.<module> --help
```

The first lines of each file explain what it does, what it reads and writes, and give an example command.

---

## 3. Installation

Python 3.10 or newer. A GPU is recommended for training and long closed-loop runs, but everything also runs on the CPU.

```bash
git clone https://github.com/Gambotch1/viv-gru-surrogate.git
cd viv-gru-surrogate
python -m venv .venv && source .venv/bin/activate
pip install -e .              # installs the package and its dependencies
pip install pytest            # for the tests
```

`pip install -e .` makes `viv_analysis` importable. If you don't install the package, prefix the commands with `PYTHONPATH=src` as shown in this README. `requirements.txt` is a full snapshot of the environment on the HPC cluster; the list in `pyproject.toml` is all the code needs.

---

## 4. Data

The CFD results are not stored in the repository. Put the Fluent monitor files (`.out`, one file per case and signal) here:

```
data/cylinder_Re_200/disp/   cl/   cd/   vel/   force/
data/Bridge/disp/            cl/   cm/   vel/   force/
```

- **Cylinder:** 21 cases, Ur = 2 … 12, Re = 200, m* = 10, ζ = 0.01, D = 0.2 m, fn = 0.2 Hz, Δt = 0.005 s. File names contain the case, e.g. `…Ur5.5….out`.
- **Bridge:** wind speeds 11 … 19 m/s, D = 7.42 m, B = 25.9 m, fn = 0.32 Hz, Δt = 0.0001 s. File names contain the speed, e.g. `disp-16.out`; the code converts it to Ur = U / (fn·D). The 19.5 m/s case is excluded (`BRIDGE_EXCLUDED_RAW_SPEEDS` in `preprocess.py`).

`vel/` and `force/` are needed to compute the acceleration. Without them, velocity and acceleration are obtained by differentiating the displacement.

The bridge preprocessing takes about an hour. Its result is cached in `data/cache/bridge_ds20_trim100_v2.parquet` and reused by all scripts. Delete the file (or call `load_bridge_df_cached(force_rebuild=True)`) after changing the raw data.

---

## 5. Quick start

Train a cylinder model, run one coupled simulation, then run the full closed-loop sweep:

```bash
# 1. Train (disp and vel as inputs, nondimensional, with Ur context)
PYTHONPATH=src python -m viv_analysis.train_gru --cfd_dataset cylinder200 \
    --input_cols disp vel --nd_inputs --exp_subdir my_cylinder_model

# 2. One closed-loop run at Ur = 5, until t = 300 s
PYTHONPATH=src python -m viv_analysis.coupled_inference --cfd_dataset cylinder200 \
    --model_subdir my_cylinder_model --Ur 5 --total_time 300 --output_dir my_first_run

# 3. Closed-loop sweep over all 21 cases, with scoring
PYTHONPATH=src python -m viv_analysis.evaluate_all --dataset cylinder200 \
    --model_subdir my_cylinder_model --output_dir my_cylinder_model_coupled_eval
```

Look at `results/my_cylinder_model_coupled_eval/sweep_results.csv` for amplitude, stability label and pass/fail per Ur.

---

## 6. Step-by-step workflow

### 6.1 Train a model — `train_gru.py`

Main options:

| Option | Meaning |
|---|---|
| `--cfd_dataset cylinder200 \| bridge` | which dataset |
| `--input_cols disp vel` | input signals; the thesis models leave out `acc` (Sec. 5.4) |
| `--nd_inputs` | use h/D, ḣ/U, ḧ·D/U² instead of physical units |
| `--use_ur_context` / `--no_ur_context` | add Ur as an extra input (default: on) |
| `--noise_std 0.05` | Gaussian noise on the scaled inputs during training |
| `--exp_subdir <name>` | name of the output folder under `results/` |
| `--holdout_ur`, `--exclude_ur`, `--force_train_ur` | change the train/val/test split |
| `--preflight_only` | only load, split and scale; print the setup and stop |

The cylinder split is fixed (test: Ur 3.5, 5.5, 7, 11; validation: Ur 4.25, 6.25, 9, 10). The bridge split spreads about 20 % test and 20 % validation cases evenly over Ur.

Output folder `results/<exp_subdir>/`: `gru_best.pt`, `x_scaler.pkl`, `y_scaler.pkl`, `ur_stats.pkl`, `run_config.json`, `metrics_gru.json`, plots. All later steps read this folder.

### 6.2 Closed-loop run — `coupled_inference.py`

The run starts from real CFD data. The first `seq_len` samples after the release fill the GRU input window, then the surrogate takes over at

```
handoff index = release index + seq_len + handoff_offset      (default offset: 2000 steps)
```

From there on, the GRU and Newmark-β run alone until `--total_time` (absolute time in seconds, not duration). The output `.npz` contains the surrogate trajectory (`t, h, cl, h_dot, h_ddot`), the CFD reference over the same time (`h_cfd, cl_cfd`) and the forcing `e`. The `.receipt.json` records all settings and the git commit.

Useful options: `--handoff_offset` (start later in the CFD record), `--run_replay_diag` (Newmark driven by the CFD lift, to check the structural part, Fig. 6.8).

### 6.3 Sweep and scoring — `evaluate_all.py`, `closed_loop_metrics.py`, `reference_quality.py`

`evaluate_all.py` runs step 6.2 for every Ur of the model's split and scores each run on the last half of the trajectory:

- **Stability label** (`classify_stability`): `stationary_lco`, `divergence` or `decay_to_rest`, from the growth of the displacement envelope (±10 % counts as stationary).
- **Amplitude A\* = A/D and frequency**, averaged over complete cycles.
- **Energy per cycle** E_f = ∮ C_L ḣ dt and the lift–velocity phase.
- **Pass** = stationary limit cycle and amplitude error ≤ 20 % against CFD.

For the bridge, most CFD cases never reach a steady limit cycle. Run `PYTHONPATH=src python -m viv_analysis.reference_quality` once to classify the CFD cases (`results/bridge_reference_status.csv`); the bridge figures then compare amplitudes only where the CFD reference is a settled limit cycle (Sec. 4.7).

### 6.4 Bridge only: rollout refinement — `train_rollout.py`

Fine-tunes a trained bridge model with L = L_TF + λ_roll·L_roll (λ_roll = 0.08064), where L_roll is the displacement/velocity error after a 0.5 s closed-loop rollout. The output folder has the same format as `train_gru.py`, so it can go straight into `evaluate_all.py`.

### 6.5 Bridge only: residual forcing test

```bash
PYTHONPATH=src python -m viv_analysis.build_pooled_tf_residual \
    --model_subdir gru_bridge_nd_context_noacc --target_ur 6.7385
PYTHONPATH=src python -m viv_analysis.coupled_inference --cfd_dataset bridge --model_dataset bridge \
    --model_subdir gru_bridge_nd_context_noacc --Ur 6.7385 --total_time 500 \
    --residual_npz results/gru_bridge_nd_context_noacc/pooled_tf_residual_train_cases.npz \
    --noise_mode surrogate --noise_seed 0 --output_dir <folder>
```

Repeat for `--noise_mode white` and several seeds, then run `plotting/summarize_stochastic_closure.py` and `plotting/plot_stochastic_closure_figures.py`.

---

## 7. Which script makes which thesis figure

All plotting scripts read saved results; none of them trains a model. They write into a `thesis_figures/` folder next to the results they read.

| Thesis | Script |
|---|---|
| Fig. 5.5, 6.4 (learning curves) | `plotting/regenerate_learning_curves.py` |
| Fig. 5.4, 5.7, App. D (cylinder open loop) | `plotting/regenerate_teacher_forcing_plots.py` |
| Fig. 5.3, 5.8, 5.9, App. E (cylinder closed loop) | `plotting/regenerate_cylinder_plots.py` |
| Fig. 5.10 (dimensional vs nondimensional) | `plotting/regenerate_ablation_comparison_plot.py` |
| Fig. 5.11–5.13, App. F (time-varying Ur) | `studies/cylinder_time_varying_ur/scripts/plot_time_varying_ur.py` |
| Tables 5.2, C.1 (architecture study) | `studies/gru_architecture_history_sensitivity/scripts/generate_appendix_tables.py` |
| Fig. 6.5, App. G (bridge open loop) | `plotting/regenerate_bridge_open_loop_plots.py` |
| Fig. 6.6, 6.7, App. H (bridge closed loop) | `plotting/regenerate_bridge_plots.py` |
| Fig. 6.8 (Newmark replay) | `coupled_inference.py --run_replay_diag` |
| Fig. 6.9 (aerodynamic work) | `plotting/plot_bridge_aerodynamic_work.py` |
| Fig. 6.11, 6.12 (rollout refinement) | `plotting/plot_rollout_refinement_figures.py` |
| Fig. 6.13 (residual forcing) | `plotting/plot_stochastic_closure_figures.py` |

The CFD reference figures of Ch. 3 and Appendices A–B were not produced with this repository.

---

## 8. Conventions to know before changing the code

- **Window and target.** A window covers rows i − seq_len … i − 1 and the target is C_L at row i. Training (`VIVSequenceDataset`), open-loop prediction and the closed loop all use this alignment. If you change it in one place, change it everywhere.
- **History length.** seq_len = 1000 (cylinder, 5 s) and 2500 (bridge, 5 s after downsampling to 0.002 s).
- **Nondimensional inputs.** disp/D, vel/U, acc/(U²/D) with U = Ur·fn·D of each case. The mode a model was trained with is stored in `run_config.json`; `coupled_inference` reads it and refuses a contradicting command-line choice.
- **Scaling.** Scalers and Ur statistics are fitted on the training cases only.
- **Force.** F = ½ ρ U² B C_L with B = 25.9 m for the bridge and B = D = 0.2 m for the cylinder.
- **Structure.** m, c, k come from `config.py` and are the same values the Fluent UDF used. Newmark-β uses β = 1/4, γ = 1/2.
- **Case names.** Always `Ur<value>` with trailing zeros removed (`utils.format_ur_label`), e.g. `Ur5`, `Ur6.7385`.
- **Acceleration as input.** With the default `acc_source="force_residual"`, C_L is an exact linear function of (disp, vel, acc). A model given `acc` learns that identity instead of the aerodynamics (thesis Sec. 5.4). Keep `acc` out of the inputs.

---

## 9. Tests

```bash
PYTHONPATH=src python -m pytest tests -q                                         # main package
PYTHONPATH=src python -m pytest studies/gru_architecture_history_sensitivity/tests -q
PYTHONPATH=src python -m pytest studies/cylinder_time_varying_ur/tests -q
```

The main tests check the Newmark integrator, the stability classification and metrics, the bridge reference classification, the nondimensional transform in the closed loop, the dataset options, and that gradients flow through the differentiable rollout.

---

## 10. Results and reproducibility

`results/` and `data/` are not part of the repository. Every closed-loop run writes a `.receipt.json` next to its `.npz` with the settings, the git commit and a flag plus hash for uncommitted changes. Commit your code before production runs, so the recorded commit is enough to reproduce them.

This repository starts from the final, cleaned state of the code. The full development history is kept in the archived repository [Gambotch1/Cylinder](https://github.com/Gambotch1/Cylinder), including the code the thesis runs were made with. Receipts of those runs refer to commit hashes in that archive. The relevant commits are tagged there:

| Tag | Commit | Content |
|---|---|---|
| `thesis-runs-2026-08-23` | `a34e975` | code of the August runs; receipts stamped `914e686` were made with the uncommitted code first committed here |
| `thesis-code-2026-08-30` | `458c52a` | last complete state before the cleanup |
| `curriculum-prototype` | `73d6031` | period-based curriculum driver (not in the thesis) |

Several thesis runs were made from a working copy with uncommitted changes (`git_dirty: true` in their receipts), so these commits are the closest committed states, not exact copies.

---

## 11. Known limitations and open problems

From the thesis (Sec. 7.4–7.5):

- **The bridge coupled response decays** although the open-loop prediction is accurate. Small errors in the lift–velocity phase reverse the aerodynamic work (Sec. 6.5). The rollout refinement and the residual forcing did not fix this.
- **Aerodynamic state.** The GRU only sees the structural motion. A low-dimensional wake state (e.g. from POD of the flow field) could let it follow the flow memory.
- **Curriculum learning.** The refinement used a 0.5 s rollout; the coupled assessment runs for about 90 structural periods. Training should move gradually from one-step to multi-cycle rollouts. A prototype period-based curriculum driver exists in the archived repository (tag `curriculum-prototype`, file `src/viv_analysis/train_rollout.py`).
- **Time-varying Ur has no CFD counterpart.** The history dependence seen in Sec. 5.7 cannot be confirmed as physical hysteresis without CFD runs with the same velocity schedule.
- **Two degrees of freedom.** Extending to heave and pitch needs a C_M prediction and coupled structural modes.

In the code:

- The closed-loop state arrays in `run_coupled_viv` are float32. A test with a forced linear oscillator showed about 1 % amplitude difference against float64 over 80 000 steps. This has not yet been checked on a real coupled run.
- `studies/cylinder_time_varying_ur/scripts/time_varying_coupled.py` contains its own copy of the coupled loop (tested to match `run_coupled_viv` for constant Ur). A change to one of the two loops has to be made in both.
