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
| `**kwargs` | `OptimiserSettingsInputDict` | `normalise` (True), `verbose` (True), `renormalise_each_step` (None → True iff multi-objective), `n_points_before_fitting`, `objective_weights`, `initial_points_generator` (`random`), `variable_normalisation` (`data`; `bounds` normalises the variables by the bounds' moments, so lengthscale bounds keep their real-unit meaning as points cluster) |

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
`k(x,x') = c²·σ(x)σ(x')·ρ(x,x')` (`ProxyScaledKernel` wrapping any of the existing kernels;
the wrapped kernel is normalised by its own diagonal, so even base kernels with `k(x,x) ≠ 1` —
kernel sums, spectral mixtures — become correlation kernels and the band stays calibrated).
With the default relative bound,
`σ(x) = bound_value·|proxy(x)| / bound_in_n_sigmas` — i.e. "1%" is read as a 2σ band by default.
The scalar amplitude factor `c` is trained by the marginal likelihood but, by default, hard-capped
at 1 via a gpytorch `Interval` constraint, so the band can tighten with data but never exceed the
promised bound; set `train_amplitude_factor: False` to pin it, or raise
`amplitude_factor_upper_bound` when the bound is an assumption rather than a guarantee (below).

Settings (`ProxyPriorSettingsInputDict`): `bound_value` (required), `bound_type`
(`relative`/`absolute`), `bound_in_n_sigmas` (2.0), `train_amplitude_factor` (True),
`amplitude_factor_lower_bound` (0.1), `amplitude_factor_upper_bound` (1.0), `amplitude_floor`
(0.0, real units — keeps σ away from zero where a relative-bound proxy crosses zero, which otherwise
pins the GP and can upset Cholesky), and the three optional settings of the next subsection.

Caveats: the bound is encoded softly (a 2σ Gaussian band, not a hard constraint); the proxy is
called thousands of times per suggestion step inside dual annealing, so it should cost well under
a second; the proxy is evaluated without gradients (fine for the current derivative-free
acquisition optimisers); normalisation round-trips can hand the proxy points a float-epsilon
outside the variable bounds, so proxies with strictly validated domains should clamp their input. Saving/loading works like user-defined objectives: the
`ProxyMeanFunction` subclass must be importable when the state file is loaded. See
`examples/example_proxy_informed_prior.py`.

### When the bound is an assumption, and the deviation has known structure near a point

The settings above encode a *certified* bound. Three optional settings cover the other common
case: the proxy is a cheaper model of the same physics, the search is local around a reference
point `x0` (say the proxy's own optimum), and what is known about the deviation `d = f − proxy` is
its low-order structure there rather than a guaranteed width. All default to the behaviour
described above, and states saved before they existed load unchanged.

- `amplitude_factor_upper_bound` (1.0): the cap on the trained factor `c`. Above 1, training starts
  from the stated band and the data may widen it as well as tighten it.
- `reference_point` (real units) with `anchor_at_reference: True`: for objectives defined relative
  to their own value at `x0`, so that `d(x0) = 0` exactly. The band is conditioned on that,
  `k(x,x') − k(x,x0)·k(x0,x') / k(x0,x0)`, which is what feeding the model the exact datum
  `(x0, f(x0))` does. Use one or the other, not both.
- `local_expansion` (needs `reference_point`): a prior on the deviation's Taylor coefficients at
  `x0`, added to the band. With `u = x − x0`, `H` the magnitude of the proxy's curvature there
  (`curvature_metric`: a matrix, or a number `h` for `h·I`; veropt maximises, so a proxy's Hessian
  at its optimum is negative semi-definite — either sign is accepted, an indefinite matrix is not)
  and `e(u) = ½·uᵀHu`,

  `k(x,x') = uᵀC u' + β_s²·e(u)e(u') + ¼β_a²·(uᵀH u')²`

  `C` is the covariance of the deviation's gradient at `x0` — `gradient_standard_deviation` (one
  number, or one per variable; zero where a symmetry forbids a deviation) or a full
  `gradient_covariance`. `β_s` (`shared_curvature_standard_deviation`) and `β_a`
  (`general_curvature_standard_deviation`) are dimensionless: the standard deviations of a common
  rescaling of `H` and of an unstructured symmetric perturbation of it; along any one direction
  they add in quadrature. The first term is what lets a handful of evaluations locate a *displaced*
  optimum: it correlates the deviation across the whole domain, where the band alone has to relearn
  a slope within every lengthscale. These scales are declared assumptions and stay fixed unless
  `train_amplitude_factor` is set inside `local_expansion`, which fits one common factor within
  `amplitude_factor_lower_bound`/`_upper_bound` (0.1/10, which must strictly enclose 1). Matrices
  are plain lists (`.tolist()` an array), so that the settings save as JSON.

With an expansion the band is what remains *after* its low-order terms, so `bound_value` should
shrink accordingly; it should not vanish, because the expansion has finite rank
(`n + n(n+1)/2` coefficients) and would otherwise claim certainty everywhere once that many points
are in. Note that the band contributes gradient (and curvature) uncertainty at `x0` too — for a
Matérn-5/2 base kernel and a band of constant width, `(5σ²/3)·diag(ℓ⁻²)`; a relative bound whose
width varies adds `∇σ∇σᵀ` — so the prior's statement about the gradient is the *total*, which
`ProxyScaledKernel.gradient_covariance_at_reference_real_units()` reports in real units (reach the
kernel as `optimiser.predictor.model[0].kernel`). The quantity exists only for a base kernel that is
twice differentiable at zero distance: Matérn ν ≥ 1.5 (ν = 1.5 converges slowly); ν = 0.5, as in
`rational_quadratic_and_matern`, is refused. The estimate is checked against one at half the step
and warns if the two disagree. The lengthscale bounds live in normalised variables, and with
`renormalise_each_step` the variable normaliser follows the evaluated points, so their meaning in
real units drifts as the points cluster; the diagnostic accounts for the current normalisation.

### Starting from known points

Points evaluated outside the optimiser go in with
`optimiser.add_evaluated_points_real_units(variable_values, objective_values)` (real units, any
number of points). They count as initial points, so `n_initial_points=0` is allowed, and the model
is fitted as soon as `n_points_before_fitting` points are in. With a proxy prior on every
objective that can be a *single* point: fewer than two points have no spread to normalise from,
so the variables are then normalised from the bounds and the objectives from the scale the prior
declares at the evaluated points (σ = bound/κ), and the model is the prior conditioned on the
point. Without a proxy prior, at least two points are needed and the optimiser says so.

### Observation noise in real units

The kernels' own `noise` setting is a variance in *normalised* objective units, so its physical
meaning changes whenever the objective normaliser is refitted. A measured noise level is better
given as `model={'observation_noise_standard_deviation': 0.05, ...}` (the objective's own units; a
number, or a list with one entry or `None` per objective). The model converts it at every fit,
after the kernel's own `noise` and `noise_lower_bound`, which it overrides: if the converted value
lies below the kernel's lower bound the bound is lowered to admit it, never the other way round.
It is saved and restored with the state.

- With the kernel's `train_noise` off (the default) the noise is *held* at the measurement.
- With `train_noise: True` the measurement is a *floor*: it becomes the lower bound of the noise
  constraint, the fit starts a little above it and may only raise it. A noise fitted freely to tens
  of points can absorb exactly the shallow curvature a search is after and report a flat surface
  with wide error bars, so the measurement is not negotiable downwards. On every fit and on reload
  the value in force is `max(stored or trained value, current floor)`, so a re-measured, larger
  floor wins over a stale trained value.

Read back what is actually in force with `optimiser.predictor.model[i].observation_noise_real_units()`,
which returns the effective standard deviation in real units (from the likelihood, not from the
setting), the floor, and the regime: `'fixed'`, `'at_floor'`, `'trained'`, or `'kernel_setting'`
when no measurement was given. Note that `predict_values()` passes the model through the
likelihood, so its band is the predictive one (latent plus noise), whereas the acquisition
functions use botorch's latent posterior.

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
