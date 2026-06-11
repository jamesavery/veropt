# `veropt/optimiser` — the Bayesian optimisation core

## Construction

`bayesian_optimiser(...)` (`constructors.py`) is the user entry point (re-exported as
`veropt.bayesian_optimiser`). Required arguments: `n_initial_points`, `n_bayesian_points`,
`n_evaluations_per_step`, `objective`. Each component can be customised either with a **choice
TypedDict** (string options + settings dicts, validated against `Literal` types) or by passing an
**already-built instance**:

| Parameter | Choice dict | Options (defaults from `default_settings.json` first) |
|---|---|---|
| `model` | `GPytorchModelChoice` | kernels: `matern`, `double_matern`, `rational_quadratic`, `rational_quadratic_and_matern`, `SMK`, `spectral_delta`; one kernel name for all objectives or a list (one per objective); `kernel_optimiser`: `adam`; plus `training_settings` (e.g. `max_iter`, learning rate); optionally `proxy_prior` + `proxy_prior_settings` (see "Proxy-informed priors" below) |
| `acquisition_function` | `AcquisitionChoice` | `qlogehvi` (multi-objective default), `ucb` (single-objective default, setting `beta`) |
| `acquisition_optimiser` | `AcquisitionOptimiserChoice` | `dual_annealing` (setting `max_iter`); `allow_proximity_punishment` (default on when `n_evaluations_per_step > 1`) with `proximity_punish_settings` (`alpha`, `omega`, `refresh_setting`) |
| `normaliser` | `NormaliserChoice` | `zero_mean_unit_variance` |
| `**kwargs` | `OptimiserSettingsInputDict` | `normalise` (True), `verbose` (True), `renormalise_each_step` (None → True iff multi-objective), `n_points_before_fitting`, `objective_weights`, `initial_points_generator` (`random`) |

Internally it builds a `BotorchPredictor` via `botorch_predictor()` and calls
`BayesianOptimiser.from_the_beginning()`.

## `BayesianOptimiser` (`optimiser.py`)

State held (all in real units, with normalised views computed on demand):

- `objective` and its bounds (`[2, n_variables]` tensor: lower row, upper row)
- `_initial_points_real_units` — all initial points, pre-generated up front
  (`initial_points.py: generate_initial_points`, currently uniform random in bounds)
- `evaluated_variables/objectives_real_units` — the growing data set, shape
  `[n_points, n_variables|n_objectives]`
- `suggested_points_real_units` + `suggested_points_history` — `SuggestedPoints` records
  (variable values, predicted objective values, step, mode)
- normalisers for variables and objectives (None until fitted)
- `settings: OptimiserSettings`

Key properties: `optimisation_mode` (initial while `n_points_evaluated < n_initial_points`, then
bayesian), `model_has_been_trained`, `return_normalised_data` (True only when `settings.normalise`
and the normalisers exist), `current_step`.

### Step lifecycle

`run_optimisation_step()` branches on objective kind (see the sequence diagram in
[architecture.md](architecture.md)):

- **callable**: `suggest_candidates()` → `_evaluate_points()` (unnormalises, calls the objective)
  → `_add_new_points()` → `_train_and_normalise_if_needed()`.
- **interface**: `_load_latest_points()` (via `objective.load_evaluated_points()`, names matched
  against `variable_names`/`objective_names`) → `_train_and_normalise_if_needed()` →
  `suggest_candidates()` → `_save_candidates()` (real units out).

`_train_and_normalise_if_needed()` logic:

- model already trained → optionally refit normalisers (`renormalise_each_step`) → retrain on all
  data (`_update_predictor()`).
- model not yet trained and `n_points_evaluated >= n_points_before_fitting` → fit normalisers →
  first training.
- otherwise nothing (early initial phase).

`suggest_candidates()` takes the next slice of `initial_points` (initial mode) or calls
`predictor.suggest_points()` (bayesian mode), attaches model predictions when available, and
stores the result in real units.

## The prediction stack (`prediction.py`)

`BotorchPredictor` owns the three pluggable pieces and the normaliser callables it needs to move
between unit systems:

- `suggest_points()` → runs the acquisition optimiser over the acquisition function, returns
  candidates.
- `update_with_new_data(variables, objectives, train=True)` → `model.train_model()` +
  `acquisition_function.refresh()`.
- `predict_values()`, `get_samples_from_model()`, `get_acquisition_values()` — used by the
  optimiser and by `veropt/graphical`.

### Model (`model.py`, `kernels.py`)

`GPyTorchFullModel` is a **list of independent single-objective GPs** (`GPyTorchSingleModel`, one
per objective) — there is no inter-objective covariance. Training
(`train_model()`): build a fresh `GPyTorchDataModel` (gpytorch `ExactGP` + botorch
`GPyTorchModel`) per objective from the current data, form a `SumMarginalLogLikelihood`, and run an
Adam loop (`TorchModelOptimiser`) for `max_iter` iterations. `train_noise` controls whether
likelihood noise is optimised alongside mean/kernel hyperparameters. Lengthscale/noise constraints
are applied per kernel class in `_set_up_model_constraints()`.

Concrete kernels live in `kernels.py`; each is a `GPyTorchSingleModel` subclass with a unique
`name` and a TypedDict of constraint/settings options.

### Acquisition (`acquisition.py`)

`BotorchAcquisitionFunction` wraps a botorch acquisition function and rebuilds it on `refresh()`
(new model + data after each training):

- `UpperConfidenceBound` (`ucb`) — single-objective, wraps the botorch analytic UCB; setting
  `beta` (default 3.0).
- `QLogExpectedHyperVolumeImprovement` (`qlogehvi`) — multi-objective, wraps botorch
  qLogEHVI with `FastNondominatedPartitioning`; the reference (nadir) point comes from
  `optimiser_utility.get_nadir_point()`.

### Acquisition optimiser (`acquisition_optimiser.py`)

- `DualAnnealingOptimiser` (`dual_annealing`) — wraps `scipy.optimize.dual_annealing`;
  `TorchNumpyWrapper` adapts between torch and numpy, and the objective is negated in the
  `dual_annealing` call (scipy minimises, veropt maximises). Produces **one** candidate per call.
- `ProximityPunishmentSequentialOptimiser` — turns a single-candidate optimiser into a batch
  optimiser. For each of the `n_evaluations_per_step` candidates it optimises a
  `ProximityPunishAcquisitionFunction`, which *subtracts* a Gaussian bump around every
  already-chosen candidate:

  `punished(x) = acq(x) − Σₖ ω·s·exp(−‖x − xₖ‖² / α²)`

  `α` (width, default 0.7 in normalised units) and `ω` (relative strength, default 1.0) are user
  settings; the scale `s` is re-estimated each step from 1000 random acquisition samples — either
  simply (`refresh_setting='simple'`: sample std) or via a Gaussian-mixture fit that picks the
  dominant high-value cluster (`'advanced'`, the default).

  Without this (or with `allow_proximity_punishment=False`), a batch of q candidates from a
  deterministic optimiser would collapse onto the same maximum.

## Proxy-informed priors (`proxy_prior.py`)

When a fast approximation of the objective exists with a known pointwise deviation bound (e.g.
"the true objective is within 1% of the proxy everywhere"), the GP prior can be built from it
instead of the uninformative default. Subclass `ProxyMeanFunction` (real units in, real units out,
unique `name`) and pass it through the model choice:

```python
optimiser = bayesian_optimiser(
    ...,
    model={
        'kernels': 'matern',
        'proxy_prior': my_proxy,                      # or a list, one entry (or None) per objective
        'proxy_prior_settings': {'bound_value': 0.01}
    },
    n_points_before_fitting=n_evaluations_per_step,   # informative prior pays off from the first batch
)
```

The encoding: prior mean `m(x) = proxy(x)` (a `ProxyMean` gpytorch mean that maps the model's
normalised inputs to real units, evaluates the proxy, and maps back), and prior covariance
`k(x,x') = c²·σ(x)σ(x')·k_base(x,x')` (`ProxyScaledKernel` wrapping any of the existing kernels,
which are all correlation kernels). With the default relative bound,
`σ(x) = bound_value·|proxy(x)| / bound_in_n_sigmas` — i.e. "1%" is read as a 2σ band by default.
The scalar amplitude factor `c` is trained by the marginal likelihood but hard-capped at 1 via a
gpytorch `Interval` constraint, so the band can tighten with data but never exceed the promised
bound; set `train_amplitude_factor: False` to pin it.

Settings (`ProxyPriorSettingsInputDict`): `bound_value` (required), `bound_type`
(`relative`/`absolute`), `bound_in_n_sigmas` (2.0), `train_amplitude_factor` (True),
`amplitude_factor_lower_bound` (0.1), `amplitude_floor` (0.0, real units — keeps σ away from zero
where a relative-bound proxy crosses zero, which otherwise pins the GP and can upset Cholesky).

Caveats: the bound is encoded softly (a 2σ Gaussian band, not a hard constraint); the proxy is
called thousands of times per suggestion step inside dual annealing, so it should cost well under
a second; the proxy is evaluated without gradients (fine for the current derivative-free
acquisition optimisers); normalisation round-trips can hand the proxy points a float-epsilon
outside the variable bounds, so proxies with strictly validated domains should clamp their input. Saving/loading works like user-defined objectives: the
`ProxyMeanFunction` subclass must be importable when the state file is loaded. See
`examples/example_proxy_informed_prior.py`.

## Normalisation (`normalisation.py`)

`Normaliser` ABC with `transform` / `inverse_transform`; the only implementation is
`NormaliserZeroMeanUnitVariance` (per-dimension mean/variance over the data points). Variables
*and* objectives are normalised before training; bounds, initial points and suggested points get
normalised views. The `TensorWithNormalisationFlag` wrapper (`utility.py`) keeps callers honest
about which unit system a tensor is in.

## Saving and loading

Public API in `optimiser_saver_loader.py`:

- `save_to_json(obj, path)` — `gather_dicts_to_save()` + `TensorsAsListsEncoder`
  (tensors → lists, ±inf → strings).
- `load_optimiser_from_state(path)` — full state round-trip; rehydrates the whole object tree and
  retrains the model if it had been trained.
- `load_optimiser_from_settings(path, objective)` — settings-only JSON (validated against the
  `bayesian_optimiser` signature) + a fresh objective; for starting a *new* run from a config file.

The mechanism (`saver_loader_utility.py`) is described in [extending.md](extending.md).

## Practice objectives (`practice_objectives.py`)

Wrappers around `botorch.test_functions` for experimentation and tests: `Hartmann` (single
objective, 3/4/6 variables), `VehicleSafety` (3 objectives, 5 variables), `DTLZ1` (configurable,
default 5 objectives / 10 variables). All are `CallableObjective`s and savable.
