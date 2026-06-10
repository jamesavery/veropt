# veropt documentation

Developer documentation for veropt. For a user-facing introduction (install, quick start), see the
top-level [README.md](../README.md).

## Contents

| Document | What it covers |
|---|---|
| [architecture.md](architecture.md) | The big picture: the three subpackages, how an optimisation step flows, diagrams |
| [optimiser.md](optimiser.md) | Deep dive into `veropt/optimiser`: the `BayesianOptimiser` lifecycle, the prediction stack (model → acquisition → acquisition optimiser), normalisation, saving/loading |
| [interfaces.md](interfaces.md) | Deep dive into `veropt/interfaces`: the `Experiment` loop, batch managers (local/slurm), simulation runners, result processing, on-disk layout, resuming |
| [graphical.md](graphical.md) | The public plotting API in `veropt/graphical` and how it reads data from the optimiser |
| [extending.md](extending.md) | How to add new kernels, acquisition functions, objectives, runners, etc., and the save/load contract every new component must honour |

## Other documentation in this repository

- [README.md](../README.md) — user-facing introduction: installation, a quick-start example,
  the visualisation tools, and a sketch of the interfaces subpackage.
- [CHANGELOG.md](../CHANGELOG.md) — release notes per version.
- [CLAUDE.md](../CLAUDE.md) — instructions for AI coding agents (commands, conventions);
  the conventions listed there (import ordering for `torch.set_default_dtype`, mypy strictness,
  British spelling) apply to human contributors too.
- `examples/` — runnable, typed example scripts; `examples/interfaces/template_experiment.py`
  is the skeleton to copy when wiring up your own simulation.

## Conventions that bite

- veropt always **maximises**. Negate your objective to minimise.
- `torch.set_default_dtype(torch.float64)` is set at import time in `optimiser/optimiser.py` and
  `interfaces/experiment.py` *before* the other imports. This ordering is load-bearing (it is why
  flake8 E402 is disabled repo-wide). Preserve it.
- Everything is typed, including tests and examples (`mypy veropt tests examples` must pass with
  `disallow_untyped_defs`).
- British spelling throughout the API: optimiser, normaliser, visualisation.
- Tensor shape convention: `[n_points, n_variables]` and `[n_points, n_objectives]`
  (see `DataShape` in `optimiser/utility.py`).
