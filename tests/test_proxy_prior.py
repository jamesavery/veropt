from typing import Any, Literal

import botorch
import gpytorch
import pytest
import torch

from veropt import bayesian_optimiser, load_optimiser_from_state, save_to_json
from veropt.optimiser.constructors import gpytorch_model, gpytorch_single_model
from veropt.optimiser.kernels import SingleKernelOptions
from veropt.optimiser.normalisation import NormaliserZeroMeanUnitVariance
from veropt.optimiser.practice_objectives import Hartmann
from veropt.optimiser.proxy_prior import (
    ProxyMean, ProxyMeanFunction, ProxyPrior, ProxyPriorSettings, ProxyScaledKernel,
    make_single_column_objective_normaliser
)


class QuadraticProxy(ProxyMeanFunction):

    name = 'quadratic_proxy'

    def _run(
            self,
            variable_values: torch.Tensor
    ) -> torch.Tensor:
        return (variable_values ** 2).sum(dim=1)


class WrongShapeProxy(ProxyMeanFunction):

    name = 'wrong_shape_proxy'

    def _run(
            self,
            variable_values: torch.Tensor
    ) -> torch.Tensor:
        return torch.ones(variable_values.shape[0], 2)


class NanProxy(ProxyMeanFunction):

    name = 'nan_proxy'

    def _run(
            self,
            variable_values: torch.Tensor
    ) -> torch.Tensor:
        return torch.full([variable_values.shape[0]], float('nan'))


class DistortedHartmannProxy(ProxyMeanFunction):

    name = 'distorted_hartmann_proxy'

    def __init__(
            self,
            n_variables: int
    ) -> None:

        super().__init__(
            n_variables=n_variables
        )

        self.function = botorch.test_functions.Hartmann(
            dim=n_variables,
            negate=True
        )

    def _run(
            self,
            variable_values: torch.Tensor
    ) -> torch.Tensor:

        true_values = self.function(variable_values)

        return true_values * (1.0 + 0.01 * torch.sin(6.0 * torch.pi * variable_values[:, 0]))


def identity_function(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


def make_proxy_mean_with_identity_normalisation(
        mean_function: ProxyMeanFunction
) -> ProxyMean:

    proxy_mean = ProxyMean(
        mean_function=mean_function
    )

    proxy_mean.update_normalisation_functions(
        unnormaliser_variables=identity_function,
        normaliser_objectives=identity_function
    )

    return proxy_mean


def test_proxy_mean_function_validates_input_shape() -> None:

    proxy = QuadraticProxy(n_variables=3)

    with pytest.raises(ValueError):
        proxy(torch.rand(5, 2))

    with pytest.raises(ValueError):
        proxy(torch.rand(5))


def test_proxy_mean_function_validates_output_shape() -> None:

    proxy = WrongShapeProxy(n_variables=3)

    with pytest.raises(ValueError):
        proxy(torch.rand(5, 3))


def test_proxy_mean_function_rejects_non_finite_values() -> None:

    proxy = NanProxy(n_variables=3)

    with pytest.raises(ValueError):
        proxy(torch.rand(5, 3))


def test_proxy_mean_forward_with_identity_normalisation() -> None:

    proxy = QuadraticProxy(n_variables=3)
    proxy_mean = make_proxy_mean_with_identity_normalisation(mean_function=proxy)

    variable_values = torch.rand(7, 3)

    assert torch.allclose(proxy_mean(variable_values), proxy(variable_values))


def test_proxy_mean_forward_with_fitted_normalisers() -> None:

    n_variables = 3
    n_points = 50

    proxy = QuadraticProxy(n_variables=n_variables)

    variable_values_real_units = 2.0 * torch.rand(n_points, n_variables) + 1.0
    objective_values_real_units = proxy(variable_values_real_units).unsqueeze(1)

    normaliser_variables = NormaliserZeroMeanUnitVariance.from_tensor(tensor=variable_values_real_units)
    normaliser_objectives = NormaliserZeroMeanUnitVariance.from_tensor(tensor=objective_values_real_units)

    proxy_mean = ProxyMean(
        mean_function=proxy
    )

    proxy_mean.update_normalisation_functions(
        unnormaliser_variables=normaliser_variables.inverse_transform,
        normaliser_objectives=make_single_column_objective_normaliser(
            normaliser_objectives=normaliser_objectives.transform,
            objective_index=0,
            n_objectives=1
        )
    )

    variable_values_normalised = normaliser_variables.transform(variable_values_real_units)

    expected_values = normaliser_objectives.transform(objective_values_real_units)[:, 0]

    assert torch.allclose(proxy_mean(variable_values_normalised), expected_values)


def test_proxy_mean_forward_handles_batch_dimensions() -> None:

    proxy = QuadraticProxy(n_variables=3)
    proxy_mean = make_proxy_mean_with_identity_normalisation(mean_function=proxy)

    batched_variable_values = torch.rand(4, 6, 3)

    output = proxy_mean(batched_variable_values)

    assert list(output.shape) == [4, 6]

    assert torch.allclose(output[2], proxy(batched_variable_values[2]))


def make_proxy_scaled_matern_kernel(
        n_variables: int,
        settings: ProxyPriorSettings
) -> ProxyScaledKernel:

    proxy_mean = make_proxy_mean_with_identity_normalisation(
        mean_function=QuadraticProxy(n_variables=n_variables)
    )

    base_kernel = gpytorch.kernels.MaternKernel(
        ard_num_dims=n_variables
    )

    return ProxyScaledKernel(
        base_kernel=base_kernel,
        proxy_mean=proxy_mean,
        settings=settings
    )


def test_proxy_scaled_kernel_diagonal_matches_amplitude() -> None:

    n_variables = 2
    bound_value = 0.05
    bound_in_n_sigmas = 2.0

    kernel = make_proxy_scaled_matern_kernel(
        n_variables=n_variables,
        settings=ProxyPriorSettings(
            bound_value=bound_value,
            bound_in_n_sigmas=bound_in_n_sigmas,
            train_amplitude_factor=False
        )
    )

    variable_values = torch.rand(10, n_variables)

    proxy_values = QuadraticProxy(n_variables=n_variables)(variable_values)

    expected_diagonal = (bound_value * proxy_values.abs() / bound_in_n_sigmas) ** 2

    diagonal = kernel(variable_values, variable_values, diag=True)

    assert torch.allclose(diagonal, expected_diagonal)


def test_proxy_scaled_kernel_absolute_bound() -> None:

    n_variables = 2
    bound_value = 0.5
    bound_in_n_sigmas = 2.0

    kernel = make_proxy_scaled_matern_kernel(
        n_variables=n_variables,
        settings=ProxyPriorSettings(
            bound_value=bound_value,
            bound_type='absolute',
            bound_in_n_sigmas=bound_in_n_sigmas,
            train_amplitude_factor=False
        )
    )

    variable_values = torch.rand(10, n_variables)

    expected_diagonal = torch.full([10], (bound_value / bound_in_n_sigmas) ** 2)

    diagonal = kernel(variable_values, variable_values, diag=True)

    assert torch.allclose(diagonal, expected_diagonal)


def test_proxy_scaled_kernel_amplitude_floor() -> None:

    n_variables = 2
    amplitude_floor = 0.1
    bound_in_n_sigmas = 2.0

    kernel = make_proxy_scaled_matern_kernel(
        n_variables=n_variables,
        settings=ProxyPriorSettings(
            bound_value=0.01,
            bound_in_n_sigmas=bound_in_n_sigmas,
            train_amplitude_factor=False,
            amplitude_floor=amplitude_floor
        )
    )

    # The quadratic proxy is ~0 near the origin, so the floor should kick in
    variable_values = torch.zeros(3, n_variables)

    expected_diagonal = torch.full([3], (amplitude_floor / bound_in_n_sigmas) ** 2)

    diagonal = kernel(variable_values, variable_values, diag=True)

    assert torch.allclose(diagonal, expected_diagonal)


def test_proxy_scaled_kernel_amplitude_factor_constraint() -> None:

    kernel = make_proxy_scaled_matern_kernel(
        n_variables=2,
        settings=ProxyPriorSettings(
            bound_value=0.05,
            amplitude_factor_lower_bound=0.1
        )
    )

    assert 0.1 < float(kernel.amplitude_factor.detach()) <= 1.0

    # Even an absurd raw value cannot push the factor past the bound
    kernel.raw_amplitude_factor.data = torch.tensor(100.0)

    assert 0.1 < float(kernel.amplitude_factor.detach()) <= 1.0


def test_proxy_scaled_kernel_fixed_amplitude_has_no_parameters() -> None:

    kernel = make_proxy_scaled_matern_kernel(
        n_variables=2,
        settings=ProxyPriorSettings(
            bound_value=0.05,
            train_amplitude_factor=False
        )
    )

    assert len(list(kernel.parameters(recurse=False))) == 0

    assert float(kernel.amplitude_factor) == 1.0


@pytest.mark.parametrize(
    'kernel_name',
    ['matern', 'double_matern', 'rational_quadratic', 'rational_quadratic_and_matern', 'SMK', 'spectral_delta']
)
def test_set_proxy_prior_with_all_kernels(kernel_name: SingleKernelOptions) -> None:

    torch.manual_seed(42)

    n_variables = 2
    n_points = 12

    single_model = gpytorch_single_model(
        n_variables=n_variables,
        kernel=kernel_name
    )

    single_model.set_proxy_prior(
        proxy_prior=ProxyPrior.from_mean_function_and_settings(
            mean_function=QuadraticProxy(n_variables=n_variables),
            settings={'bound_value': 0.05, 'amplitude_floor': 0.01}
        )
    )

    single_model.update_normalisation_functions(
        unnormaliser_variables=identity_function,
        normaliser_objectives=identity_function
    )

    variable_values = torch.rand(n_points, n_variables)
    objective_values = QuadraticProxy(n_variables=n_variables)(variable_values)

    single_model.initialise_model_with_data(
        train_inputs=variable_values,
        train_targets=objective_values
    )

    assert isinstance(single_model.model_with_data.covar_module, ProxyScaledKernel)  # type: ignore[union-attr]

    # The optimiser must not raise on empty parameter groups (the proxy mean has no parameters)
    for parameter_group in single_model.trained_parameters:
        assert len(parameter_group['params']) > 0


def test_proxy_prior_integration_tracks_proxy_far_from_data() -> None:

    torch.manual_seed(42)

    n_variables: Literal[3] = 3
    n_evaluations_per_step = 4

    objective = Hartmann(
        n_variables=n_variables
    )

    proxy = DistortedHartmannProxy(
        n_variables=n_variables
    )

    optimiser = bayesian_optimiser(
        n_initial_points=4,
        n_bayesian_points=8,
        n_evaluations_per_step=n_evaluations_per_step,
        objective=objective,
        verbose=False,
        model={
            'kernels': 'matern',
            'proxy_prior': proxy,
            'proxy_prior_settings': {
                'bound_value': 0.01,
                'amplitude_floor': 0.01
            },
            'training_settings': {
                'max_iter': 50
            }
        },
        acquisition_optimiser={
            'optimiser': 'dual_annealing',
            'optimiser_settings': {
                'max_iter': 20
            }
        },
        renormalise_each_step=True,
        n_points_before_fitting=n_evaluations_per_step
    )

    for i in range(3):
        optimiser.run_optimisation_step()

    # Far away from the few evaluated points, the posterior mean should track the proxy
    # to within (roughly) the deviation band
    test_points = torch.rand(200, n_variables)

    prediction = optimiser.predictor.predict_values(
        variable_values=test_points,
        normalised=False
    )

    proxy_values = proxy(test_points)

    largest_deviation = (prediction['mean'].detach().flatten() - proxy_values).abs().max()

    assert float(largest_deviation) < 0.1


def test_proxy_prior_save_load_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]

    torch.manual_seed(42)

    n_variables: Literal[3] = 3

    objective = Hartmann(
        n_variables=n_variables
    )

    proxy = DistortedHartmannProxy(
        n_variables=n_variables
    )

    optimiser = bayesian_optimiser(
        n_initial_points=4,
        n_bayesian_points=8,
        n_evaluations_per_step=4,
        objective=objective,
        verbose=False,
        model={
            'kernels': 'matern',
            'proxy_prior': proxy,
            'proxy_prior_settings': {
                'bound_value': 0.01,
                'bound_in_n_sigmas': 3.0,
                'amplitude_floor': 0.01
            },
            'training_settings': {
                'max_iter': 20
            }
        },
        n_points_before_fitting=4
    )

    optimiser.run_optimisation_step()

    file_path = str(tmp_path / 'optimiser_state.json')

    save_to_json(
        object_to_save=optimiser,
        file_path=file_path
    )

    loaded_optimiser = load_optimiser_from_state(
        file_name=file_path
    )

    loaded_model = loaded_optimiser.predictor.model[0]  # type: ignore[attr-defined]

    assert loaded_model.proxy_prior is not None
    assert loaded_model.proxy_prior.settings.bound_value == 0.01
    assert loaded_model.proxy_prior.settings.bound_in_n_sigmas == 3.0
    assert isinstance(loaded_model.proxy_prior.mean_function, DistortedHartmannProxy)

    test_points = torch.rand(20, n_variables)

    prediction = optimiser.predictor.predict_values(
        variable_values=test_points,
        normalised=False
    )

    loaded_prediction = loaded_optimiser.predictor.predict_values(
        variable_values=test_points,
        normalised=False
    )

    assert torch.allclose(prediction['mean'], loaded_prediction['mean'], atol=1e-10)
    assert torch.allclose(prediction['upper'], loaded_prediction['upper'], atol=1e-10)


def test_multi_objective_with_partial_proxy_priors() -> None:

    n_variables = 2
    n_objectives = 2

    full_model = gpytorch_model(
        n_variables=n_variables,
        n_objectives=n_objectives,
        kernels='matern',
        proxy_prior=[QuadraticProxy(n_variables=n_variables), None],
        proxy_prior_settings=[{'bound_value': 0.05}, None]
    )

    assert full_model[0].proxy_prior is not None
    assert isinstance(full_model[0].kernel, ProxyScaledKernel)
    assert isinstance(full_model[0].mean_module, ProxyMean)

    assert full_model[1].proxy_prior is None
    assert not isinstance(full_model[1].kernel, ProxyScaledKernel)


def test_constructor_validation() -> None:

    n_variables = 2

    with pytest.raises(AssertionError):
        # Settings without a proxy prior
        gpytorch_model(
            n_variables=n_variables,
            n_objectives=1,
            proxy_prior_settings={'bound_value': 0.05}
        )

    with pytest.raises(AssertionError):
        # Wrong list length
        gpytorch_model(
            n_variables=n_variables,
            n_objectives=2,
            proxy_prior=[QuadraticProxy(n_variables=n_variables)]
        )

    with pytest.raises(AssertionError):
        # Missing the (required) bound value
        gpytorch_model(
            n_variables=n_variables,
            n_objectives=1,
            proxy_prior=QuadraticProxy(n_variables=n_variables)
        )

    settings_with_unknown_key: Any = {'bound_value': 0.05, 'not_a_real_setting': 1.0}

    with pytest.raises(AssertionError):
        # Unknown settings key
        gpytorch_model(
            n_variables=n_variables,
            n_objectives=1,
            proxy_prior=QuadraticProxy(n_variables=n_variables),
            proxy_prior_settings=settings_with_unknown_key
        )

    with pytest.raises(AssertionError):
        # Proxy expecting the wrong number of variables
        gpytorch_model(
            n_variables=n_variables,
            n_objectives=1,
            proxy_prior=QuadraticProxy(n_variables=5),
            proxy_prior_settings={'bound_value': 0.05}
        )


def test_proxy_prior_without_normalisation() -> None:

    torch.manual_seed(42)

    n_variables: Literal[3] = 3

    objective = Hartmann(
        n_variables=n_variables
    )

    proxy = DistortedHartmannProxy(
        n_variables=n_variables
    )

    optimiser = bayesian_optimiser(
        n_initial_points=4,
        n_bayesian_points=8,
        n_evaluations_per_step=4,
        objective=objective,
        verbose=False,
        model={
            'kernels': 'matern',
            'proxy_prior': proxy,
            'proxy_prior_settings': {
                'bound_value': 0.01,
                'amplitude_floor': 0.01
            },
            'training_settings': {
                'max_iter': 20
            }
        },
        normalise=False,
        n_points_before_fitting=4
    )

    optimiser.run_optimisation_step()

    # Without normalisation the model lives in real units, so the (normalised-space)
    # prediction should match the proxy in real units far from the data
    test_points = torch.rand(100, n_variables)

    prediction = optimiser.predictor.predict_values(
        variable_values=test_points,
        normalised=True
    )

    proxy_values = proxy(test_points)

    largest_deviation = (prediction['mean'].detach().flatten() - proxy_values).abs().max()

    assert float(largest_deviation) < 0.1
