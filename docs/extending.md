# Extending veropt

The library is extended by subclassing one of the abstract bases. The pattern is uniform across
the codebase.

## The save/load contract (read this first)

Almost every component subclasses `SavableClass` (`optimiser/saver_loader_utility.py`):

- `gather_dicts_to_save() -> dict` returns `{'name': self.name, 'state': {...}}` with
  JSON-serialisable state (tensors are handled by `TensorsAsListsEncoder`).
- `from_saved_state(saved_state) -> Self` reconstructs the instance.
- Loading dispatches through `rehydrate_object(superclass, name, saved_state)`, which scans
  `get_all_subclasses(superclass)` for a matching `name` **class attribute**.

Therefore every concrete subclass in a savable hierarchy **must set a unique `name`**, and the
subclass must be imported (so it exists in the subclass registry) before loading a state file that
references it. A dataclass variant, `SavableDataClass`, derives both methods automatically.

## Extension points

| To add a… | Subclass | Module | Also touch |
|---|---|---|---|
| Objective (Python function) | `CallableObjective` — implement `_run(parameter_values) -> tensor [n_points, n_objectives]` | `optimiser/objective.py` | — |
| Objective (external process) | `InterfaceObjective` — implement `save_candidates()` / `load_evaluated_points()` | `optimiser/objective.py` | — |
| Proxy prior mean (fast approximate objective) | `ProxyMeanFunction` — implement `_run(variable_values) -> tensor [n_points]`, real units | `optimiser/proxy_prior.py` | pass via `model={'proxy_prior': ..., 'proxy_prior_settings': {...}}` |
| Kernel / single-objective GP | `GPyTorchSingleModel` | `optimiser/kernels.py` | add the option string to `SingleKernelOptions` and the constructor dispatch in `optimiser/constructors.py` |
| Acquisition function | `BotorchAcquisitionFunction` — implement `refresh()` to (re)build the botorch object | `optimiser/acquisition.py` | `AcquisitionOptions` literal + dispatch in `constructors.py`; expand `AcquisitionSettings` |
| Acquisition optimiser | `AcquisitionOptimiser` — implement `optimise()`; mind `maximum_evaluations_per_step` | `optimiser/acquisition_optimiser.py` | `AcquisitionOptimiserOptions` literal + dispatch |
| Normaliser | `Normaliser` — `transform` / `inverse_transform` | `optimiser/normalisation.py` | `NormaliserChoice` literal |
| Initial-points generator | extend `generate_initial_points()` | `optimiser/initial_points.py` | `InitialPointsChoice` literal |
| Simulation runner | `SimulationRunner` — implement `set_up_and_run() -> SimulationResult` (+ a pydantic config) | `interfaces/local_simulation.py` / `slurm_simulation.py` | — |
| Result processor | `ResultProcessor` — implement `calculate_objectives()` and `open_output_file()` | `interfaces/result_processing.py` | — |
| Batch manager | `DirectBatchManager` (implement `run_batch()`) or `SubmitBatchManager` (implement `submit_batch()` / `wait_for_jobs()`) | `interfaces/batch_manager.py` | `ExperimentMode` + `_get_batch_manager_class()`, or pass `batch_manager_class` to `experiment()` |
| Visualisation | plain function | `graphical/visualisation.py` (public wrapper) + an underscore module (implementation) | — |

The string-option pattern: user-facing choices are `Literal` types validated at construction, and
defaults come from `optimiser/default_settings.json`. When adding an option, update the `Literal`,
the dispatch in `constructors.py`, and — if it has settings — the corresponding
`…SettingsInputDict` TypedDict, so mypy keeps users honest.

`examples/interfaces/template_experiment.py` is a runnable skeleton for the
runner + processor pair.

## House rules

- Run the CI trio before pushing: `flake8 .`, `mypy veropt tests examples`, `pytest`
  (also via `python local_workflows/run_all.py`). Type annotations are mandatory everywhere,
  including tests and examples.
- `torch.set_default_dtype(torch.float64)` must stay before the other imports in
  `optimiser/optimiser.py` and `interfaces/experiment.py`.
- veropt maximises; keep that convention in new objectives and acquisition functions
  (e.g. `DualAnnealingOptimiser` negates the acquisition function because scipy minimises).
- Tests mirror the module layout (`tests/test_<module>.py`, `tests/interfaces/…`); use
  `MockSimulationRunner` / `MockResultProcessor` to test experiment plumbing without slurm.
- British spelling in all public API.
