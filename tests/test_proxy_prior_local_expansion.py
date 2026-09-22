import json
from pathlib import Path
from typing import Any, Literal, Optional

import gpytorch
import pytest
import torch

from veropt import bayesian_optimiser, load_optimiser_from_state, save_to_json
from veropt.optimiser.constructors import gpytorch_single_model
from veropt.optimiser.kernels import SingleKernelOptions
from veropt.optimiser.normalisation import NormaliserZeroMeanUnitVariance
from veropt.optimiser.optimiser import BayesianOptimiser
from veropt.optimiser.practice_objectives import Hartmann, VehicleSafety
from veropt.optimiser.proxy_prior import (
    LocalExpansionSettings, ProxyMean, ProxyMeanFunction, ProxyPrior, ProxyPriorSettings, ProxyScaledKernel,
    make_single_column_objective_normaliser
)


class HarmonicProxy(ProxyMeanFunction):

    # g(x) = -(1/2) metric |x - x0|^2: an optimum at x0 = (0.5, ..., 0.5), zero there

    name = 'harmonic_proxy_for_local_expansion_tests'

    metric = 4.0
    reference_value = 0.5

    def _run(
            self,
            variable_values: torch.Tensor
    ) -> torch.Tensor:
        return -0.5 * self.metric * ((variable_values - self.reference_value) ** 2).sum(dim=1)


class ShiftedHarmonicProxy(HarmonicProxy):

    # The same with its optimum at the origin, so that it is non-zero at x0 = 0.5

    name = 'shifted_harmonic_proxy_for_local_expansion_tests'

    reference_value = 0.0


def identity_function(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


def reference_point(n_variables: int) -> list[float]:
    return [0.5] * n_variables


def make_kernel(
        n_variables: int,
        settings: ProxyPriorSettings,
        normaliser_variables: Optional[NormaliserZeroMeanUnitVariance] = None,
        normaliser_objectives: Optional[NormaliserZeroMeanUnitVariance] = None,
        lengthscale: float = 0.7,
        nu: float = 2.5,
        proxy_class: type[HarmonicProxy] = HarmonicProxy
) -> ProxyScaledKernel:

    # A proxy-scaled Matern kernel with identity normalisation unless normalisers are given

    proxy_mean = ProxyMean(mean_function=proxy_class(n_variables=n_variables))

    if normaliser_variables is None or normaliser_objectives is None:
        proxy_mean.update_normalisation_functions(identity_function, identity_function, identity_function)

    else:
        proxy_mean.update_normalisation_functions(
            unnormaliser_variables=normaliser_variables.inverse_transform,
            normaliser_objectives=make_single_column_objective_normaliser(
                normaliser_objectives=normaliser_objectives.transform, objective_index=0, n_objectives=1
            ),
            normaliser_variables=normaliser_variables.transform
        )

    base_kernel = gpytorch.kernels.MaternKernel(nu=nu, ard_num_dims=n_variables)
    base_kernel.lengthscale = lengthscale

    # Validates the settings the way the constructor path does
    ProxyPrior(mean_function=proxy_mean.mean_function, settings=settings)

    return ProxyScaledKernel(base_kernel=base_kernel, proxy_mean=proxy_mean, settings=settings)


def band_settings(
        n_variables: int,
        bound_value: float = 0.6,
        bound_type: Literal['relative', 'absolute'] = 'absolute',
        **settings: Any
) -> ProxyPriorSettings:

    return ProxyPriorSettings(
        bound_value=bound_value, bound_type=bound_type, train_amplitude_factor=False,
        reference_point=reference_point(n_variables), **settings
    )


def expansion_settings(
        n_variables: int,
        anchor_at_reference: bool = False,
        bound_value: float = 1e-9,
        **expansion: Any
) -> ProxyPriorSettings:

    # The band cannot be switched off, so it is negligible where a test is about the expansion

    return band_settings(
        n_variables, bound_value=bound_value, anchor_at_reference=anchor_at_reference,
        local_expansion=LocalExpansionSettings(**expansion)
    )


def make_optimiser(
        model: dict[str, Any],
        n_evaluations_per_step: int = 2,
        renormalise_each_step: bool = False
) -> BayesianOptimiser:

    # One initial step, then one Bayesian step, with a short acquisition search: the integration
    # tests are about the model passing through the optimiser, not about where the search ends up

    model_choice: Any = {'kernels': 'matern', 'training_settings': {'max_iter': 20}, **model}

    return bayesian_optimiser(
        n_initial_points=n_evaluations_per_step,
        n_bayesian_points=n_evaluations_per_step,
        n_evaluations_per_step=n_evaluations_per_step,
        objective=Hartmann(n_variables=3),
        verbose=False,
        model=model_choice,
        acquisition_optimiser={'optimiser': 'dual_annealing', 'optimiser_settings': {'max_iter': 50}},
        n_points_before_fitting=n_evaluations_per_step,
        renormalise_each_step=renormalise_each_step
    )


def single_model(optimiser: BayesianOptimiser, objective_no: int = 0) -> Any:
    return optimiser.predictor.model[objective_no]  # type: ignore[attr-defined]


# --- the band's amplitude cap ---------------------------------------------------------------------

def test_amplitude_factor_upper_bound_lets_the_data_widen_the_band() -> None:

    settings = ProxyPriorSettings(bound_value=0.5, bound_type='absolute', amplitude_factor_upper_bound=3.0)
    kernel = make_kernel(2, settings)

    # Training starts from the stated bound itself and can reach, but not pass, the cap
    assert float(kernel.amplitude_factor.detach()) == pytest.approx(1.0)

    kernel.raw_amplitude_factor.data = torch.tensor(100.0)

    assert float(kernel.amplitude_factor.detach()) == pytest.approx(3.0)


# --- the local expansion ---------------------------------------------------------------------------

def test_local_expansion_variance_in_real_units_under_normalisation() -> None:

    # k(x, x) = u^T C u + (beta_s^2 + beta_a^2) e(u)^2 in real units, whatever the normalisation

    torch.manual_seed(0)

    n_variables = 3
    gradient_standard_deviation = [0.3, 0.0, 1.1]
    shared, general = 0.2, 0.1

    variable_values = 3.0 * torch.rand(40, n_variables) - 1.0
    normaliser_variables = NormaliserZeroMeanUnitVariance.from_tensor(tensor=variable_values)
    normaliser_objectives = NormaliserZeroMeanUnitVariance.from_tensor(tensor=5.0 * torch.randn(40, 1) + 2.0)

    kernel = make_kernel(
        n_variables,
        expansion_settings(
            n_variables, gradient_standard_deviation=gradient_standard_deviation, curvature_metric=HarmonicProxy.metric,
            shared_curvature_standard_deviation=shared, general_curvature_standard_deviation=general
        ),
        normaliser_variables, normaliser_objectives
    )

    displacement = variable_values - 0.5
    harmonic = 0.5 * HarmonicProxy.metric * (displacement ** 2).sum(dim=1)
    expected = (
        (displacement ** 2 * torch.tensor(gradient_standard_deviation) ** 2).sum(dim=1) +
        (shared ** 2 + general ** 2) * harmonic ** 2
    ) / normaliser_objectives.variances[0]

    normalised = normaliser_variables.transform(variable_values)

    assert torch.allclose(kernel(normalised, normalised, diag=True), expected, rtol=1e-6, atol=1e-12)


def test_local_expansion_is_the_covariance_of_the_random_taylor_model() -> None:

    # d(u) = a.u + (1/2) w^T B w with w = H^(1/2) u, a ~ N(0, C), B = beta_s xi I + beta_a (Z + Z^T) / 2

    torch.manual_seed(1)

    n_variables, n_draws = 3, 400_000
    metric = torch.tensor([[3.0, 0.5, 0.0], [0.5, 2.0, 0.2], [0.0, 0.2, 1.0]])
    gradient_covariance = torch.tensor([[0.5, 0.1, 0.0], [0.1, 0.3, 0.0], [0.0, 0.0, 0.0]])
    shared, general = 0.3, 0.25

    kernel = make_kernel(
        n_variables,
        expansion_settings(
            n_variables, gradient_covariance=gradient_covariance.tolist(), curvature_metric=metric.tolist(),
            shared_curvature_standard_deviation=shared, general_curvature_standard_deviation=general
        )
    )

    points = torch.rand(5, n_variables)
    displacement = points - 0.5

    eigenvalues, eigenvectors = torch.linalg.eigh(metric)
    whitened = displacement @ (eigenvectors * eigenvalues.sqrt()) @ eigenvectors.T

    gradients = torch.randn(n_draws, n_variables) @ torch.linalg.cholesky(gradient_covariance + 1e-12 * torch.eye(3)).T
    unstructured = torch.randn(n_draws, n_variables, n_variables)
    curvature = (
        shared * torch.randn(n_draws, 1, 1) * torch.eye(n_variables) +
        general * 0.5 * (unstructured + unstructured.transpose(1, 2))
    )
    deviations = gradients @ displacement.T + 0.5 * torch.einsum('pi,nij,pj->np', whitened, curvature, whitened)

    assert torch.allclose(
        kernel(points, points).to_dense().detach(), deviations.T @ deviations / n_draws, rtol=0.03, atol=2e-5
    )


def test_local_expansion_vanishes_at_the_reference_and_agrees_across_evaluation_paths() -> None:

    torch.manual_seed(2)

    n_variables = 3
    kernel = make_kernel(
        n_variables,
        expansion_settings(
            n_variables, anchor_at_reference=True, bound_value=0.4, gradient_standard_deviation=0.4,
            curvature_metric=2.0, shared_curvature_standard_deviation=0.3, general_curvature_standard_deviation=0.1
        )
    )

    points = torch.cat([torch.full([1, n_variables], 0.5), torch.rand(30, n_variables)])
    values = kernel(points, points).to_dense().detach()

    assert torch.allclose(values[0], torch.zeros(31), atol=1e-14)  # anchored band plus vanishing expansion
    assert torch.allclose(values, values.T)
    assert float(torch.linalg.eigvalsh(values).min()) > -1e-10 * float(values.abs().max())

    # Batched inputs, the diagonal path, and the diagonal path with two different inputs
    batched, other = torch.rand(2, 7, n_variables), torch.rand(2, 7, n_variables)
    full, cross = kernel(batched, batched).to_dense().detach(), kernel(batched, other).to_dense().detach()

    assert list(full.shape) == [2, 7, 7]
    assert torch.allclose(full.diagonal(dim1=-2, dim2=-1), kernel(batched, batched, diag=True).detach(), atol=1e-12)
    assert torch.allclose(full[1], kernel(batched[1], batched[1]).to_dense().detach(), atol=1e-12)
    assert torch.allclose(cross.diagonal(dim1=-2, dim2=-1), kernel(batched, other, diag=True).detach(), atol=1e-12)
    assert torch.allclose(cross, kernel(other, batched).to_dense().detach().transpose(-1, -2), atol=1e-12)


def test_a_forbidden_gradient_direction_has_no_gradient_variance() -> None:

    # A projector-type covariance tau^2 P: no deviation of the gradient along (1, -1) / sqrt(2)

    allowed, forbidden = torch.tensor([1.0, 1.0]) / 2 ** 0.5, torch.tensor([1.0, -1.0]) / 2 ** 0.5
    kernel = make_kernel(2, expansion_settings(2, gradient_covariance=(0.49 * torch.outer(allowed, allowed)).tolist()))

    covariance = kernel.gradient_covariance_at_reference_real_units()

    assert float(forbidden @ covariance @ forbidden) == pytest.approx(0.0, abs=1e-8)
    assert float(allowed @ covariance @ allowed) == pytest.approx(0.49, rel=1e-6)


def test_a_negative_semidefinite_curvature_metric_is_read_as_its_magnitude() -> None:

    # veropt maximises, so a proxy's Hessian at its optimum is negative semi-definite; the kernel
    # is even in the metric, so it is the same prior either way

    torch.manual_seed(3)

    metric = torch.tensor([[3.0, 0.5], [0.5, 2.0]])

    def make(sign: float) -> ProxyScaledKernel:
        return make_kernel(2, expansion_settings(
            2, curvature_metric=(sign * metric).tolist(), shared_curvature_standard_deviation=0.2,
            general_curvature_standard_deviation=0.3
        ))

    points = torch.rand(9, 2)

    assert torch.allclose(make(-1.0)(points, points).to_dense(), make(1.0)(points, points).to_dense())

    scalar = make_kernel(2, expansion_settings(2, curvature_metric=-4.0, shared_curvature_standard_deviation=0.2))

    assert scalar.local_expansion is not None
    assert torch.allclose(scalar.local_expansion._curvature_metric, 4.0 * torch.eye(2))


def test_the_expansion_amplitude_factor_is_fixed_unless_asked_for_and_then_capped() -> None:

    fixed = make_kernel(2, expansion_settings(2, gradient_standard_deviation=1.0))

    assert len(list(fixed.local_expansion.parameters())) == 0  # type: ignore[union-attr]

    trained = make_kernel(2, expansion_settings(
        2, gradient_standard_deviation=1.0, train_amplitude_factor=True, amplitude_factor_upper_bound=5.0
    ))
    assert trained.local_expansion is not None
    assert float(trained.local_expansion.amplitude_factor.detach()) == pytest.approx(1.0)

    points = torch.rand(6, 2)
    before = trained(points, points, diag=True).detach()

    trained.local_expansion.raw_amplitude_factor.data = torch.tensor(100.0)

    assert float(trained.local_expansion.amplitude_factor.detach()) == pytest.approx(5.0)
    assert torch.allclose(trained(points, points, diag=True).detach(), 25.0 * before)


@pytest.mark.parametrize(
    'kernel_name',
    ['matern', 'double_matern', 'rational_quadratic', 'rational_quadratic_and_matern', 'SMK', 'spectral_delta']
)
def test_the_expansion_amplitude_factor_is_trained_with_every_kernel(kernel_name: SingleKernelOptions) -> None:

    # Kernel classes that pick their trained parameters by hand must include the expansion's factor

    torch.manual_seed(4)

    model = gpytorch_single_model(n_variables=2, kernel=kernel_name)
    model.set_proxy_prior(ProxyPrior.from_mean_function_and_settings(
        mean_function=HarmonicProxy(n_variables=2),
        settings={
            'bound_value': 0.3, 'bound_type': 'absolute', 'reference_point': reference_point(2),
            'local_expansion': {'gradient_standard_deviation': 1.0, 'train_amplitude_factor': True}
        }
    ))
    model.update_normalisation_functions(identity_function, identity_function, identity_function)

    variable_values = torch.rand(12, 2)
    model.initialise_model_with_data(
        train_inputs=variable_values, train_targets=HarmonicProxy(n_variables=2)(variable_values)
    )

    covar_module = model.model_with_data.covar_module  # type: ignore[union-attr]
    trained = {id(parameter) for group in model.trained_parameters for parameter in group['params']}

    assert id(covar_module.local_expansion.raw_amplitude_factor) in trained
    assert id(covar_module.raw_amplitude_factor) in trained


# --- anchoring -------------------------------------------------------------------------------------

def test_anchoring_is_conditioning_on_the_reference_value() -> None:

    torch.manual_seed(5)

    n_variables = 3
    points = torch.cat([torch.full([1, n_variables], 0.5), torch.rand(12, n_variables)])

    def values(settings: ProxyPriorSettings) -> torch.Tensor:
        return make_kernel(n_variables, settings)(points, points).to_dense().detach()

    unanchored = values(band_settings(n_variables))
    anchored = values(band_settings(n_variables, anchor_at_reference=True))

    expected = unanchored - torch.outer(unanchored[:, 0], unanchored[0, :]) / unanchored[0, 0]

    assert torch.allclose(anchored, expected, atol=1e-12)
    assert torch.allclose(anchored[0], torch.zeros(13), atol=1e-12)
    assert float(torch.linalg.eigvalsh(anchored).min()) > -1e-10

    # A relative bound around a proxy that is zero at the reference point is anchored already
    already = values(band_settings(n_variables, bound_value=0.3, bound_type='relative', anchor_at_reference=True))

    assert bool(torch.isfinite(already).all())
    assert torch.allclose(already, values(band_settings(n_variables, bound_value=0.3, bound_type='relative')))


def test_anchoring_equals_giving_the_model_the_exact_reference_datum() -> None:

    # The two ways of stating d(x0) = 0 must give the same posterior, so that exactly one is used

    torch.manual_seed(6)

    n_variables = 2

    def posterior(anchor: bool, inputs: torch.Tensor, targets: torch.Tensor, test_inputs: torch.Tensor) -> Any:

        kernel = make_kernel(n_variables, band_settings(n_variables, anchor_at_reference=anchor))
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-12))
        likelihood.noise = 1e-10

        class Model(gpytorch.models.ExactGP):  # type: ignore[misc]

            def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
                return gpytorch.distributions.MultivariateNormal(kernel.proxy_mean(x), kernel(x))

        with torch.no_grad():
            return Model(inputs, targets, likelihood).eval()(test_inputs)

    proxy = HarmonicProxy(n_variables=n_variables)
    inputs, test_inputs = torch.rand(6, n_variables), torch.rand(9, n_variables)
    targets = proxy(inputs) + 0.2 * torch.randn(6)
    x0 = torch.full([1, n_variables], 0.5)

    with_datum = posterior(False, torch.cat([x0, inputs]), torch.cat([proxy(x0), targets]), test_inputs)
    with_anchoring = posterior(True, inputs, targets, test_inputs)

    assert torch.allclose(with_datum.mean, with_anchoring.mean, atol=1e-5)
    assert torch.allclose(with_datum.variance, with_anchoring.variance, atol=1e-5)


# --- the gradient covariance diagnostic ------------------------------------------------------------

def test_total_gradient_covariance_includes_the_band() -> None:

    # Cov(grad d(x0)) = C + (5 sigma^2 / 3) diag(l^-2) for a Matern-5/2 band of constant sigma, in
    # real units; conditioning on the reference value leaves a stationary band's gradient alone

    torch.manual_seed(7)

    n_variables, bound_value, lengthscale = 2, 0.8, 0.7
    gradient_standard_deviation = [0.5, 0.2]

    normaliser_variables = NormaliserZeroMeanUnitVariance.from_tensor(tensor=2.0 * torch.rand(30, n_variables))
    normaliser_objectives = NormaliserZeroMeanUnitVariance.from_tensor(tensor=3.0 * torch.randn(30, 1))

    # The lengthscale lives in normalised variables; in real units it is stretched per variable
    lengthscales_real_units = lengthscale * torch.sqrt(normaliser_variables.variances)
    expected = (
        torch.diag(torch.tensor(gradient_standard_deviation) ** 2) +
        (5.0 / 3.0) * (bound_value / 2.0) ** 2 * torch.diag(lengthscales_real_units ** -2.0)
    )

    for anchor in [False, True]:

        kernel = make_kernel(
            n_variables,
            expansion_settings(
                n_variables, anchor_at_reference=anchor, bound_value=bound_value,
                gradient_standard_deviation=gradient_standard_deviation
            ),
            normaliser_variables, normaliser_objectives, lengthscale=lengthscale
        )

        assert torch.allclose(kernel.gradient_covariance_at_reference_real_units(), expected, rtol=5e-3, atol=1e-6)


def test_the_gradient_covariance_of_a_varying_band_and_the_diagnostic_limits() -> None:

    n_variables = 2

    # A relative bound around a proxy that is non-zero at x0 has grad sigma(x0) != 0, which adds a
    # rank-one term; anchoring removes it. sigma(x) = 0.3 |g(x)| / 2 with g = -2 |x|^2, so at
    # x0 = (0.5, 0.5): sigma = 0.15, grad sigma = (0.3, 0.3)
    stationary = (5.0 / 3.0) * 0.15 ** 2 / 0.7 ** 2 * torch.eye(n_variables)
    rank_one = torch.full([n_variables, n_variables], 0.3 ** 2)

    def varying(anchor: bool) -> torch.Tensor:
        return make_kernel(
            n_variables, band_settings(n_variables, bound_value=0.3, bound_type='relative', anchor_at_reference=anchor),
            proxy_class=ShiftedHarmonicProxy
        ).gradient_covariance_at_reference_real_units()

    assert torch.allclose(varying(False), stationary + rank_one, rtol=1e-4)
    assert torch.allclose(varying(True), stationary, rtol=1e-4, atol=1e-8)

    # A band that is not differentiable is refused; an unconverged estimate is reported
    with pytest.raises(ValueError, match='not differentiable'):
        make_kernel(n_variables, band_settings(n_variables), nu=0.5).gradient_covariance_at_reference_real_units()

    with pytest.warns(UserWarning, match='not converged'):
        kernel = make_kernel(n_variables, band_settings(n_variables))
        kernel.gradient_covariance_at_reference_real_units(relative_step=0.5)


# --- settings ------------------------------------------------------------------------------------

@pytest.mark.parametrize('settings, message', [
    ({'amplitude_factor_upper_bound': 0.5}, "amplitude_factor_upper_bound"),
    ({'local_expansion': {'gradient_standard_deviation': 1.0}}, "reference_point"),
    ({'anchor_at_reference': True}, "reference_point"),
    ({'reference_point': [0.5]}, "one number per variable"),
    ({'reference_point': torch.tensor([0.5, 0.5])}, "one number per variable"),
    ({'reference_point': [0.5, 0.5], 'local_expansion': {
        'gradient_standard_deviation': 1.0, 'gradient_covariance': [[1.0, 0.0], [0.0, 1.0]]}}, "not both"),
    ({'reference_point': [0.5, 0.5], 'local_expansion': {
        'gradient_covariance': [[1.0, 2.0], [2.0, 1.0]]}}, "positive semi-definite"),
    ({'reference_point': [0.5, 0.5], 'local_expansion': {
        'gradient_covariance': [[1.0, 0.5], [0.0, 1.0]]}}, "symmetric"),
    ({'reference_point': [0.5, 0.5], 'local_expansion': {'gradient_covariance': torch.eye(2)}}, "use .tolist()"),
    ({'reference_point': [0.5, 0.5], 'local_expansion': {'gradient_standard_deviation': [1.0, 1.0, 1.0]}},
     "per variable"),
    ({'reference_point': [0.5, 0.5], 'local_expansion': {'gradient_standard_deviation': 'large'}}, "number"),
    ({'reference_point': [0.5, 0.5], 'local_expansion': {'gradient_standard_deviation': -1.0}}, "non-negative"),
    ({'reference_point': [0.5, 0.5], 'local_expansion': {'shared_curvature_standard_deviation': 0.1}},
     "curvature_metric"),
    ({'reference_point': [0.5, 0.5], 'local_expansion': {
        'general_curvature_standard_deviation': -0.1, 'curvature_metric': 1.0}}, "non-negative"),
    ({'reference_point': [0.5, 0.5], 'local_expansion': {
        'curvature_metric': [[1.0, 0.0], [0.0, -1.0]]}}, "positive or negative semi-definite"),
    ({'reference_point': [0.5, 0.5], 'local_expansion': {
        'gradient_standard_deviation': 1.0, 'train_amplitude_factor': True, 'amplitude_factor_upper_bound': 1.0}},
     "strictly enclose 1"),
    ({'reference_point': [0.5, 0.5], 'local_expansion': {'gradient_sd': 1.0}}, "not recognised"),
])
def test_settings_validation(settings: dict, message: str) -> None:

    with pytest.raises(AssertionError, match=message):
        ProxyPrior.from_mean_function_and_settings(
            mean_function=HarmonicProxy(n_variables=2),
            settings={'bound_value': 0.5, 'bound_type': 'absolute', **settings}
        )


# --- through the optimiser -----------------------------------------------------------------------

LOCAL_EXPANSION_MODEL = {
    'proxy_prior': HarmonicProxy(n_variables=3),
    'proxy_prior_settings': {
        'bound_value': 0.5, 'bound_type': 'absolute', 'amplitude_factor_upper_bound': 4.0,
        'reference_point': reference_point(3), 'anchor_at_reference': True,
        'local_expansion': {
            'gradient_standard_deviation': [1.0, 0.0, 2.0], 'curvature_metric': 4.0,
            'shared_curvature_standard_deviation': 0.2, 'general_curvature_standard_deviation': 0.1,
            'train_amplitude_factor': True
        }
    }
}


def test_local_expansion_runs_in_the_optimiser_and_survives_save_and_load(tmp_path: Path) -> None:

    torch.manual_seed(8)

    optimiser = make_optimiser(LOCAL_EXPANSION_MODEL)
    optimiser.run_optimisation_step()
    optimiser.run_optimisation_step()

    file_path = str(tmp_path / 'optimiser_state.json')
    save_to_json(object_to_save=optimiser, file_path=file_path)
    loaded = load_optimiser_from_state(file_name=file_path)

    settings = single_model(loaded).proxy_prior.settings

    assert settings.anchor_at_reference is True
    assert settings.amplitude_factor_upper_bound == 4.0
    assert isinstance(settings.local_expansion, LocalExpansionSettings)
    assert settings.local_expansion.gradient_standard_deviation == [1.0, 0.0, 2.0]

    test_points = torch.rand(20, 3)
    prediction = optimiser.predictor.predict_values(variable_values=test_points, normalised=False)
    loaded_prediction = loaded.predictor.predict_values(variable_values=test_points, normalised=False)

    assert torch.allclose(prediction['mean'], loaded_prediction['mean'], atol=1e-10)
    assert torch.allclose(prediction['upper'], loaded_prediction['upper'], atol=1e-10)

    # The anchored prior knows the objective at the reference point without having evaluated it:
    # zero latent variance, so the predictive band there is the observation noise alone
    at_reference = optimiser.predictor.predict_values(variable_values=torch.full([1, 3], 0.5), normalised=False)
    noise = single_model(optimiser).observation_noise_real_units()['standard_deviation']

    assert torch.allclose(at_reference['mean'], torch.zeros(1, 1), atol=1e-6)
    assert float((at_reference['upper'] - at_reference['mean']).detach()) == pytest.approx(2.0 * noise, rel=1e-3)


def test_states_saved_before_these_settings_existed_still_load(tmp_path: Path) -> None:

    torch.manual_seed(9)

    optimiser = make_optimiser({
        'proxy_prior': HarmonicProxy(n_variables=3),
        'proxy_prior_settings': {'bound_value': 0.5, 'bound_type': 'absolute'}
    })
    optimiser.run_optimisation_step()

    file_path = str(tmp_path / 'optimiser_state.json')
    save_to_json(object_to_save=optimiser, file_path=file_path)

    # Strip every setting this feature added: the four proxy prior settings (from the proxy
    # settings dictionary only; the optimiser has an unrelated 'reference_point' of its own) and
    # the model's noise setting, wherever it sits in the state
    proxy_keys = {'amplitude_factor_upper_bound', 'reference_point', 'anchor_at_reference', 'local_expansion'}

    def strip(node: Any) -> Any:
        if isinstance(node, dict):
            skipped = (proxy_keys if 'bound_value' in node else set()) | {'observation_noise_standard_deviation'}
            return {key: strip(value) for key, value in node.items() if key not in skipped}
        if isinstance(node, list):
            return [strip(value) for value in node]
        return node

    with open(file_path) as file:
        state_text = json.dumps(strip(json.load(file)))

    assert '"observation_noise_standard_deviation"' not in state_text and '"local_expansion"' not in state_text

    with open(file_path, 'w') as file:
        file.write(state_text)

    loaded = load_optimiser_from_state(file_name=file_path)
    test_points = torch.rand(10, 3)

    assert torch.allclose(
        optimiser.predictor.predict_values(variable_values=test_points, normalised=False)['mean'],
        loaded.predictor.predict_values(variable_values=test_points, normalised=False)['mean'],
        atol=1e-10
    )


# --- observation noise in real units ---------------------------------------------------------------

@pytest.mark.parametrize('standard_deviation', [0.05, 1e-7])
def test_a_fixed_observation_noise_follows_the_normalisation_and_is_never_clamped(
        standard_deviation: float,
        tmp_path: Path
) -> None:

    # 1e-7 is far below the kernel's default lower bound (1e-8 in normalised variance, i.e. a
    # standard deviation of 1e-4 times the objective's spread): the bound yields, the value does not

    torch.manual_seed(10)

    optimiser = make_optimiser(
        {'observation_noise_standard_deviation': standard_deviation},
        n_evaluations_per_step=4, renormalise_each_step=True
    )

    objective_variances = []

    for _ in range(2):

        optimiser.run_optimisation_step()

        objective_variances.append(float(optimiser._normaliser_objectives.variances[0]))  # type: ignore[union-attr]
        information = single_model(optimiser).observation_noise_real_units()

        assert information['regime'] == 'fixed'
        assert information['floor'] == standard_deviation
        assert information['standard_deviation'] == pytest.approx(standard_deviation, rel=1e-9)
        assert bool(torch.isfinite(single_model(optimiser).model_with_data.likelihood.noise_covar.raw_noise))

    # The normalisation did change between the steps, so a fixed normalised level would have drifted
    assert objective_variances[0] != pytest.approx(objective_variances[1], rel=1e-3)

    file_path = str(tmp_path / 'optimiser_state.json')
    save_to_json(object_to_save=optimiser, file_path=file_path)
    reloaded = single_model(load_optimiser_from_state(file_name=file_path))

    assert reloaded.observation_noise_standard_deviation == standard_deviation
    assert reloaded.observation_noise_real_units()['standard_deviation'] == pytest.approx(standard_deviation, rel=1e-9)


def test_a_measured_noise_is_a_floor_when_the_noise_is_trained(tmp_path: Path) -> None:

    torch.manual_seed(11)

    optimiser = make_optimiser(
        {'kernel_settings': {'train_noise': True}, 'observation_noise_standard_deviation': 0.05},
        n_evaluations_per_step=4
    )
    optimiser.run_optimisation_step()

    model = single_model(optimiser)
    raw_noise = model.model_with_data.likelihood.noise_covar.raw_noise

    # The noise is in the trained set, off its bound (a value on the bound has raw = -inf and no
    # gradient), and cannot have gone below the floor
    assert id(raw_noise) in {id(parameter) for group in model.trained_parameters for parameter in group['params']}
    assert bool(torch.isfinite(raw_noise))

    information = model.observation_noise_real_units()

    assert information['regime'] in ('at_floor', 'trained')
    assert information['standard_deviation'] >= 0.05 * (1.0 - 1e-9)

    # A re-measured, larger floor wins over the trained value; a smaller one leaves it in force
    model.set_observation_noise_real_units(standard_deviation=0.4)
    raised = model.observation_noise_real_units()

    assert raised['regime'] == 'at_floor'
    start_above_floor = (1.0 + model.trained_noise_start_margin) ** 0.5

    assert raised['standard_deviation'] == pytest.approx(0.4 * start_above_floor, rel=1e-6)

    model.set_observation_noise_real_units(standard_deviation=0.01)
    lowered = model.observation_noise_real_units()

    assert lowered['regime'] == 'trained'
    assert lowered['standard_deviation'] == pytest.approx(raised['standard_deviation'], rel=1e-9)

    # Reload keeps the value in force, since it is above the floor
    file_path = str(tmp_path / 'optimiser_state.json')
    save_to_json(object_to_save=optimiser, file_path=file_path)
    reloaded = single_model(load_optimiser_from_state(file_name=file_path)).observation_noise_real_units()

    assert reloaded['regime'] == 'trained'
    assert reloaded['standard_deviation'] == pytest.approx(lowered['standard_deviation'], rel=1e-6)


def test_observation_noise_and_proxy_settings_per_objective() -> None:

    torch.manual_seed(12)

    n_variables = 5

    optimiser = bayesian_optimiser(
        n_initial_points=4, n_bayesian_points=0, n_evaluations_per_step=4, objective=VehicleSafety(), verbose=False,
        model={
            'kernels': 'matern',
            'observation_noise_standard_deviation': [0.05, None, 0.2],
            'proxy_prior': [None, HarmonicProxy(n_variables=n_variables), HarmonicProxy(n_variables=n_variables)],
            'proxy_prior_settings': [
                None,
                {'bound_value': 0.5, 'bound_type': 'absolute', 'reference_point': reference_point(n_variables),
                 'anchor_at_reference': True},
                {'bound_value': 0.5, 'bound_type': 'absolute', 'reference_point': [0.2] * n_variables,
                 'local_expansion': {'gradient_standard_deviation': 1.0}}
            ],
            'training_settings': {'max_iter': 20}
        },
        n_points_before_fitting=4
    )
    optimiser.run_optimisation_step()

    models = [single_model(optimiser, objective_no) for objective_no in range(3)]

    assert [model.observation_noise_real_units()['regime'] for model in models] == ['fixed', 'kernel_setting', 'fixed']
    assert models[0].observation_noise_real_units()['standard_deviation'] == pytest.approx(0.05, rel=1e-6)
    assert models[2].observation_noise_real_units()['standard_deviation'] == pytest.approx(0.2, rel=1e-6)
    assert models[1].kernel.settings.anchor_at_reference is True
    assert torch.allclose(models[2].kernel.local_expansion._reference_point, torch.full([n_variables], 0.2))

    prediction = optimiser.predictor.predict_values(variable_values=torch.rand(4, n_variables), normalised=False)

    assert bool(torch.isfinite(prediction['mean']).all()) and bool(torch.isfinite(prediction['upper']).all())


@pytest.mark.parametrize(
    'standard_deviation, error', [('0.05', ValueError), (-0.05, AssertionError), ([0.1, 0.2], AssertionError)]
)
def test_wrong_observation_noise_settings_are_refused(standard_deviation: Any, error: type[Exception]) -> None:

    with pytest.raises(error, match='observation_noise_standard_deviation|for each objective'):
        make_optimiser({'observation_noise_standard_deviation': standard_deviation})
