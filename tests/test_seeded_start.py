from pathlib import Path
from typing import Any, Literal

import pytest
import torch

from veropt import bayesian_optimiser, load_optimiser_from_state, save_to_json
from veropt.optimiser.optimiser import BayesianOptimiser
from veropt.optimiser.practice_objectives import Hartmann
from veropt.optimiser.proxy_prior import ProxyMeanFunction


class HarmonicProxy(ProxyMeanFunction):

    # g(x) = -2 |x - 0.5|^2, with its optimum at the box centre

    name = 'harmonic_proxy_for_seeded_start_tests'

    def _run(
            self,
            variable_values: torch.Tensor
    ) -> torch.Tensor:
        return -2.0 * ((variable_values - 0.5) ** 2).sum(dim=1)


N_VARIABLES: Literal[3] = 3
BOUND_VALUE = 0.4
BOUND_IN_N_SIGMAS = 2.0


def make_optimiser(
        n_initial_points: int = 0,
        proxy_prior: bool = True,
        **settings: Any
) -> BayesianOptimiser:

    model: dict[str, Any] = {'kernels': 'matern', 'training_settings': {'max_iter': 20}}

    if proxy_prior:
        model['proxy_prior'] = HarmonicProxy(n_variables=N_VARIABLES)
        model['proxy_prior_settings'] = {
            'bound_value': BOUND_VALUE, 'bound_type': 'absolute', 'bound_in_n_sigmas': BOUND_IN_N_SIGMAS
        }

    return bayesian_optimiser(
        n_initial_points=n_initial_points,
        n_bayesian_points=2,
        n_evaluations_per_step=1,
        objective=Hartmann(n_variables=N_VARIABLES),
        verbose=False,
        model=model,  # type: ignore[arg-type]
        acquisition_optimiser={'optimiser': 'dual_annealing', 'optimiser_settings': {'max_iter': 50}},
        n_points_before_fitting=1,
        **settings
    )


def seed(optimiser: BayesianOptimiser, n_points: int = 1) -> tuple[torch.Tensor, torch.Tensor]:

    variable_values = torch.full([n_points, N_VARIABLES], 0.5) + 0.01 * torch.arange(n_points).unsqueeze(-1)
    objective_values = Hartmann(n_variables=N_VARIABLES)(variable_values)

    optimiser.add_evaluated_points_real_units(variable_values=variable_values, objective_values=objective_values)

    return variable_values, objective_values


def test_a_proxy_prior_run_starts_from_a_single_seeded_point(tmp_path: Path) -> None:

    torch.manual_seed(0)

    optimiser = make_optimiser()
    variable_values, objective_values = seed(optimiser)

    # One point in, model fitted from it, in Bayesian mode: no random initial point was needed
    assert optimiser.n_points_evaluated == 1
    assert optimiser.model_has_been_trained
    assert optimiser.optimisation_mode.name == 'bayesian'

    # The variables are normalised from the bounds ([0, 1]^3 for Hartmann), the objective by the
    # prior's declared scale sigma = bound / kappa around the seeded value
    variables_normaliser: Any = optimiser._normaliser_variables
    objectives_normaliser: Any = optimiser._normaliser_objectives

    assert torch.allclose(variables_normaliser.means, torch.full([N_VARIABLES], 0.5))
    assert torch.allclose(variables_normaliser.variances, torch.full([N_VARIABLES], 1.0 / 12.0))
    assert torch.allclose(objectives_normaliser.means, objective_values[0])
    assert torch.allclose(objectives_normaliser.variances, torch.tensor([(BOUND_VALUE / BOUND_IN_N_SIGMAS) ** 2]))

    # The model is the prior conditioned on the seed: it reproduces the seed, and elsewhere its mean
    # lies between the proxy and the proxy shifted by the seed's residual, by the correlation
    at_seed = optimiser.predictor.predict_values(variable_values=variable_values, normalised=False)
    assert torch.allclose(at_seed['mean'], objective_values, atol=1e-4)

    proxy = HarmonicProxy(n_variables=N_VARIABLES)
    far = torch.tensor([[0.1, 0.9, 0.2]])
    prediction = float(optimiser.predictor.predict_values(variable_values=far, normalised=False)['mean'].detach())
    residual_at_seed = float(objective_values[0] - proxy(variable_values)[0])
    proxy_far = float(proxy(far)[0])

    assert residual_at_seed > 0.0
    assert proxy_far <= prediction <= proxy_far + residual_at_seed

    # A Bayesian step runs from there, and the whole thing survives a checkpoint
    optimiser.run_optimisation_step()
    assert optimiser.n_points_evaluated == 2

    file_path = str(tmp_path / 'optimiser_state.json')
    save_to_json(object_to_save=optimiser, file_path=file_path)
    loaded = load_optimiser_from_state(file_name=file_path)

    assert loaded.n_points_evaluated == 2
    assert loaded.settings.variable_normalisation == 'data'
    loaded.run_optimisation_step()
    assert loaded.n_points_evaluated == 3


def test_without_a_proxy_prior_a_single_point_is_refused_with_advice() -> None:

    optimiser = make_optimiser(proxy_prior=False)

    with pytest.raises(AssertionError, match="needs a proxy prior on every objective"):
        seed(optimiser)


def test_a_relative_bound_that_vanishes_at_the_seed_is_refused() -> None:

    optimiser = bayesian_optimiser(
        n_initial_points=0, n_bayesian_points=1, n_evaluations_per_step=1,
        objective=Hartmann(n_variables=N_VARIABLES), verbose=False,
        model={
            'kernels': 'matern',
            'proxy_prior': HarmonicProxy(n_variables=N_VARIABLES),
            'proxy_prior_settings': {'bound_value': 0.1, 'bound_type': 'relative'}  # zero at the box centre
        },
        n_points_before_fitting=1
    )

    with pytest.raises(AssertionError, match="deviation band is zero"):
        seed(optimiser)


def test_two_seeded_points_normalise_from_data_as_before() -> None:

    torch.manual_seed(1)

    optimiser = make_optimiser(proxy_prior=False)
    optimiser.settings.n_points_before_fitting = 2
    variable_values, objective_values = seed(optimiser, n_points=2)

    variables_normaliser: Any = optimiser._normaliser_variables
    objectives_normaliser: Any = optimiser._normaliser_objectives

    assert optimiser.model_has_been_trained
    assert torch.allclose(objectives_normaliser.means, objective_values.mean(dim=0))
    assert torch.allclose(variables_normaliser.means, variable_values.mean(dim=0))


def test_a_step_with_no_points_and_no_model_says_what_to_do() -> None:

    optimiser = make_optimiser()

    with pytest.raises(AssertionError, match="add_evaluated_points_real_units"):
        optimiser.run_optimisation_step()


def test_bounds_normalisation_does_not_move_with_the_points(tmp_path: Path) -> None:

    torch.manual_seed(2)

    optimiser = make_optimiser(n_initial_points=2, proxy_prior=False, variable_normalisation='bounds')
    optimiser.settings.n_points_before_fitting = 2
    optimiser.settings.renormalise_each_step = True

    expected_means, expected_variances = torch.full([N_VARIABLES], 0.5), torch.full([N_VARIABLES], 1.0 / 12.0)

    optimiser.run_optimisation_step()  # the first point: nothing fitted yet

    for _ in range(3):

        optimiser.run_optimisation_step()

        variables_normaliser: Any = optimiser._normaliser_variables

        assert torch.allclose(variables_normaliser.means, expected_means)
        assert torch.allclose(variables_normaliser.variances, expected_variances)

    # The objectives are still normalised from the data
    objectives_normaliser: Any = optimiser._normaliser_objectives

    assert torch.allclose(objectives_normaliser.means, optimiser.evaluated_objectives_real_units.mean(dim=0))

    file_path = str(tmp_path / 'optimiser_state.json')
    save_to_json(object_to_save=optimiser, file_path=file_path)
    loaded = load_optimiser_from_state(file_name=file_path)

    assert loaded.settings.variable_normalisation == 'bounds'


def test_an_unknown_variable_normalisation_is_refused() -> None:

    with pytest.raises(AssertionError, match="variable_normalisation"):
        make_optimiser(variable_normalisation='points')  # type: ignore[arg-type]
