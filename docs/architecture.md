# Architecture

veropt is a Bayesian optimisation library for expensive black-box problems (~100 evaluations),
originally built to tune the [VEROS](https://veros.readthedocs.io/) ocean simulator. It is split
into three layers with a deliberately narrow seam between them:

```mermaid
flowchart LR
    subgraph graphical["veropt/graphical"]
        VIS["visualisation.py<br/>(plotly figures, Dash app)"]
    end

    subgraph optimiser["veropt/optimiser"]
        BO["BayesianOptimiser"]
        PRED["BotorchPredictor"]
        MODEL["GPyTorchFullModel<br/>(one GP per objective)"]
        ACQ["Acquisition function<br/>(UCB / qLogEHVI)"]
        ACQOPT["Acquisition optimiser<br/>(dual annealing + proximity punishment)"]
        OBJ["Objective<br/>(CallableObjective | InterfaceObjective)"]
        BO --> PRED
        PRED --> MODEL
        PRED --> ACQ
        PRED --> ACQOPT
        BO --> OBJ
    end

    subgraph interfaces["veropt/interfaces"]
        EXP["Experiment"]
        BM["BatchManager<br/>(local | local_slurm | remote_slurm)"]
        SR["SimulationRunner"]
        RP["ResultProcessor"]
        EXP --> BM
        BM --> SR
        EXP --> RP
    end

    VIS -- "reads optimiser + predictor state" --> BO
    EXP -- "drives" --> BO
    EXP -- "JSON files via<br/>ExperimentObjective" --> OBJ
```

- **`veropt/optimiser`** is the mathematical core (torch/gpytorch/botorch). It knows nothing about
  simulations, slurm, or plotting. See [optimiser.md](optimiser.md).
- **`veropt/interfaces`** runs expensive simulations as objectives, locally or on a cluster, with
  resumable on-disk state. It talks to the core *only* through the two-method
  `InterfaceObjective` contract. See [interfaces.md](interfaces.md).
- **`veropt/graphical`** renders plotly figures by reading the optimiser's public properties and
  the predictor's prediction methods. See [graphical.md](graphical.md).

Entry points users actually call:

- `veropt.bayesian_optimiser(...)` (`optimiser/constructors.py`) — build an optimiser.
- `veropt.interfaces.constructors.experiment(...)` — build a resumable simulation experiment.
- `veropt.graphical.visualisation.*` — plot it.
- `veropt.save_to_json` / `load_optimiser_from_state` / `load_optimiser_from_settings` — persistence.

## The two objective flavours

Everything pivots on which kind of objective the optimiser holds (`optimiser/objective.py`):

| | `CallableObjective` | `InterfaceObjective` |
|---|---|---|
| What it is | A Python function (`_run()`) | A two-method contract: `save_candidates()` / `load_evaluated_points()` |
| Step style | Synchronous | Asynchronous (an external process evaluates the points between optimiser steps) |
| Used by | Practice objectives, user functions | `veropt/interfaces` (`ExperimentObjective` writes/reads JSON exchange files) |

## One optimisation step

`BayesianOptimiser.run_optimisation_step()` (`optimiser/optimiser.py`). Note the inverted order in
the interface case — results of the *previous* batch are loaded first, then new candidates are
suggested and handed off:

```mermaid
sequenceDiagram
    participant U as caller
    participant O as BayesianOptimiser
    participant P as BotorchPredictor
    participant F as objective

    rect rgb(235, 244, 255)
    note over U,F: CallableObjective (synchronous)
    U->>O: run_optimisation_step()
    O->>O: suggest_candidates()
    note right of O: initial phase: slice of pre-generated points<br/>bayesian phase: predictor.suggest_points()
    O->>F: __call__(suggested points, real units)
    F-->>O: objective values
    O->>O: _add_new_points()
    O->>P: _train_and_normalise_if_needed()
    end

    rect rgb(255, 244, 230)
    note over U,F: InterfaceObjective (asynchronous)
    U->>O: run_optimisation_step()
    O->>F: load_evaluated_points()
    F-->>O: previous batch results
    O->>P: _train_and_normalise_if_needed()
    O->>O: suggest_candidates()
    O->>F: save_candidates(new points, real units)
    end
```

The optimiser is in **initial mode** while `n_points_evaluated < n_initial_points` (serving slices
of pre-generated random points) and switches to **bayesian mode** afterwards (the
`optimisation_mode` property). The GP is first trained once
`n_points_evaluated >= settings.n_points_before_fitting`; before that, suggested points carry no
prediction.

## One experiment step (submitted/slurm mode)

`Experiment.run_experiment_step_submitted()` (`interfaces/experiment.py`) interleaves the
optimiser with slurm so that exactly one batch is in flight at a time:

```mermaid
sequenceDiagram
    participant E as Experiment
    participant B as SubmitBatchManager
    participant S as slurm
    participant R as ResultProcessor
    participant O as BayesianOptimiser

    note over E: step N (N > 0)
    E->>B: wait_for_jobs(state)
    B->>S: poll scontrol until batch N-1 done
    E->>R: process(results of batch N-1)
    R-->>E: objective values (failed runs → NaN → imputed)
    E->>E: save objectives to state + exchange JSON
    E->>O: run_optimisation_step()
    note right of O: loads batch N-1 results,<br/>trains GP, suggests batch N
    E->>E: save optimiser state JSON
    E->>B: submit_batch(batch N)
    B->>S: sbatch each point, record job ids
```

Direct mode (`run_experiment_step_direct()`, `experiment_mode == "local"`) is the simple
synchronous version: suggest → run each simulation blocking → process → feed back, all in one call.

Every step persists the full experiment state (`ExperimentalState` JSON + optimiser state JSON),
which is what makes `experiment(..., continue_if_possible=True)` able to resume after a crash,
requeue, or login-node logout.

## Cross-cutting mechanisms

- **Dual-unit data.** The optimiser stores variables/objectives in real units and serves
  normalised views once normalisers are fitted (default: zero mean, unit variance per dimension).
  Tensors cross internal API boundaries wrapped in `TensorWithNormalisationFlag`; attributes use a
  `_real_units` suffix. Multi-objective runs renormalise every step by default, single-objective
  runs don't.
- **Persistence.** Every savable component subclasses `SavableClass`
  (`optimiser/saver_loader_utility.py`): it serialises via `gather_dicts_to_save()` and is
  reconstructed by `rehydrate_object()`, which dispatches on a unique `name` class attribute.
  See [extending.md](extending.md) — forgetting the `name` is the classic mistake.
- **NaN handling.** The core cannot ingest NaNs yet, so `interfaces/experiment.py:_mask_nans()`
  imputes failed simulations with `nanmin − 2·nanstd` of the objective so far (a pessimistic but
  finite value).
