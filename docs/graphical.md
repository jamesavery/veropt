# `veropt/graphical` — visualisation

All public functions live in `visualisation.py`; the underscore-prefixed modules
(`_model_visualisation.py`, `_overview.py`, `_pareto_front.py`, `_table.py`,
`_visualisation_utility.py`) are internal. Figures are plotly (interactive, saveable as HTML);
`run_prediction_grid_app` additionally spins up a Dash app.

Every function takes the `optimiser` as its first argument and a `normalised` flag (default
False — plots are in real units by default since v1.2).

## Public API

| Function | Plots |
|---|---|
| `plot_prediction_grid(optimiser, evaluated_point=…, plot_acquisition=…)` | The flagship plot: a grid of 1D cross-sections (one per variable × objective) through a chosen point, showing GP mean ± uncertainty, model samples, acquisition function, evaluated points and suggested points with predicted error bars |
| `plot_prediction_surface(optimiser, variable_x, variable_y, objective)` | 3D GP surface over two variables (others fixed at a chosen point) |
| `plot_prediction_surface_grid(optimiser, objective)` | Grid of 3D surfaces for all pairwise variable combinations |
| `run_prediction_grid_app(optimiser)` | Interactive Dash app for browsing prediction grids across evaluated points |
| `plot_point_overview(optimiser, points='all'\|'pareto-optimal'\|'best'\|'suggested'\|'bayes')` | Parallel-coordinates view of points across all variables and objectives |
| `plot_progression(optimiser)` | Objective values over evaluation order |
| `plot_pareto_front(optimiser, plotted_objective_indices)` | 2D/3D Pareto front for chosen objectives (can include the reference point) |
| `plot_pareto_front_grid(optimiser)` | All pairwise 2D Pareto projections |
| `build_table` / `plot_table` / `save_table_to_csv` | Tabular comparison of chosen points (values + bounds), as dict, plotly table, or CSV |

## How the plots get their data

The graphical layer is read-only and uses only public surface:

- optimiser properties: `bounds`, `evaluated_variable_values` / `evaluated_objective_values`
  (flagged tensors), `suggested_points`, `n_objectives`, `reference_point`,
  `get_best_points()`, `get_pareto_optimal_points()`, `objective.variable_names` /
  `objective_names`;
- predictor methods: `predict_values()` (mean/lower/upper), `get_samples_from_model()`,
  `get_acquisition_values()`.

`_visualisation_utility.ModelPredictionContainer` caches grids of predictions so the Dash app can
flip between points without recomputation.

If you add a public plotting function, put the user-facing wrapper in `visualisation.py` and the
implementation in an underscore module, following the existing split.
