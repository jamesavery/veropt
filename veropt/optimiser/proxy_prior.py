import abc
import warnings
from dataclasses import dataclass, fields
from typing import Any, Callable, Literal, Mapping, Optional, Self, TypedDict, Union

import gpytorch
import torch
from gpytorch.constraints import Interval

from veropt.optimiser.saver_loader_utility import SavableClass, SavableDataClass, rehydrate_object
from veropt.optimiser.utility import _validate_typed_dict


NormalisationFunction = Callable[[torch.Tensor], torch.Tensor]


class ProxyMeanFunction(SavableClass, metaclass=abc.ABCMeta):

    name: str = 'meta'

    def __init__(
            self,
            n_variables: int
    ) -> None:

        self.n_variables = n_variables

        assert 'name' in self.__class__.__dict__, (
            f"Must give subclass '{self.__class__.__name__}' the static class variable 'name'."
        )

    def __call__(
            self,
            variable_values: torch.Tensor
    ) -> torch.Tensor:

        error_message_input = (
            f"Tensor 'variable_values' should have shape [n_points, n_variables = {self.n_variables}] "
            f"but received shape {list(variable_values.shape)} "
            f"(in '{self.__class__.__name__}')."
        )

        if not len(variable_values.shape) == 2:
            raise ValueError(error_message_input)

        if not variable_values.shape[1] == self.n_variables:
            raise ValueError(error_message_input)

        proxy_values = self._run(
            variable_values=variable_values
        )

        n_points = variable_values.shape[0]

        if len(proxy_values.shape) == 2 and proxy_values.shape[1] == 1:
            proxy_values = proxy_values.flatten()

        if not list(proxy_values.shape) == [n_points]:
            raise ValueError(
                f"Proxy mean function '{self.__class__.__name__}' should return shape [n_points = {n_points}] "
                f"but returned shape {list(proxy_values.shape)}."
            )

        if not bool(torch.isfinite(proxy_values).all()):
            raise ValueError(
                f"Proxy mean function '{self.__class__.__name__}' returned non-finite values."
            )

        return proxy_values

    @abc.abstractmethod
    def _run(
            self,
            variable_values: torch.Tensor
    ) -> torch.Tensor:
        pass

    def gather_dicts_to_save(self) -> dict:
        return {
            'name': self.name,
            'state': {
                'n_variables': self.n_variables
            }
        }

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:
        return cls(
            n_variables=saved_state['n_variables']
        )


class LocalExpansionSettingsInputDict(TypedDict, total=False):
    gradient_standard_deviation: Union[float, list[float]]
    gradient_covariance: list[list[float]]
    curvature_metric: Union[float, list[list[float]]]
    shared_curvature_standard_deviation: float
    general_curvature_standard_deviation: float
    train_amplitude_factor: bool
    amplitude_factor_lower_bound: float
    amplitude_factor_upper_bound: float


@dataclass
class LocalExpansionSettings(SavableDataClass):

    # A prior on the low-order Taylor coefficients of the deviation d = f - g about the reference
    # point x0, with u = x - x0 in real units:
    #
    #   d(x) = a.u + (1/2) u^T H^(1/2) B H^(1/2) u + (higher order, left to the scaled kernel)
    #
    #   - a, the deviation's gradient at x0, has covariance diag(gradient_standard_deviation^2)
    #     (a number, or one per variable; zero where a symmetry forbids a deviation) or the full
    #     'gradient_covariance'
    #   - H, 'curvature_metric', is the magnitude of the proxy's curvature at x0 (a number h stands
    #     for h * identity), so that (1/2) u^T H u is the proxy's harmonic rise. veropt maximises,
    #     so a proxy's Hessian at its optimum is negative semi-definite; either sign is accepted
    #   - B is the relative curvature deviation: a common rescaling of H with standard deviation
    #     'shared_curvature_standard_deviation' plus an unstructured symmetric perturbation with
    #     'general_curvature_standard_deviation', both dimensionless
    #
    # The scales are declared assumptions and stay fixed unless 'train_amplitude_factor' is set, in
    # which case one factor multiplying all of them is fitted within its bounds.

    gradient_standard_deviation: Union[float, list[float], None] = None
    gradient_covariance: Optional[list[list[float]]] = None
    curvature_metric: Union[float, list[list[float]], None] = None
    shared_curvature_standard_deviation: float = 0.0
    general_curvature_standard_deviation: float = 0.0
    train_amplitude_factor: bool = False
    amplitude_factor_lower_bound: float = 0.1
    amplitude_factor_upper_bound: float = 10.0


class ProxyPriorSettingsInputDict(TypedDict, total=False):
    bound_value: float
    bound_type: Literal['relative', 'absolute']
    bound_in_n_sigmas: float
    train_amplitude_factor: bool
    amplitude_factor_lower_bound: float
    amplitude_factor_upper_bound: float
    amplitude_floor: float
    reference_point: list[float]
    anchor_at_reference: bool
    local_expansion: LocalExpansionSettingsInputDict


@dataclass
class ProxyPriorSettings(SavableDataClass):
    bound_value: float
    bound_type: Literal['relative', 'absolute'] = 'relative'
    bound_in_n_sigmas: float = 2.0
    train_amplitude_factor: bool = True
    amplitude_factor_lower_bound: float = 0.1
    # 1.0 makes the bound a hard cap: data may tighten the band but never widen it, which is right
    # for a certified bound. For a bound that is itself an assumption, a larger value lets the
    # data revise it upwards too.
    amplitude_factor_upper_bound: float = 1.0
    amplitude_floor: float = 0.0
    # In real units. Needed by 'anchor_at_reference' and 'local_expansion'.
    reference_point: Optional[list[float]] = None
    # States that the deviation vanishes at the reference point, d(x0) = 0, by conditioning the
    # prior on it. For objectives that are defined relative to their value at x0. Do not also give
    # the optimiser the datum (x0, f(x0)): that would state the same thing twice.
    anchor_at_reference: bool = False
    local_expansion: Optional[LocalExpansionSettings] = None

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:

        # States saved before a setting existed do not contain it; it then takes its default,
        # which is chosen so that the behaviour of such a state is unchanged

        unexpected_fields = set(saved_state) - {field.name for field in fields(cls)}

        assert not unexpected_fields, (
            f"Fields {sorted(unexpected_fields)} from saved state not expected for {cls.__name__}"
        )
        assert 'bound_value' in saved_state, f"Field 'bound_value' not found in saved state for {cls.__name__}"

        return cls(**_with_local_expansion_settings(saved_state))


def _with_local_expansion_settings(
        settings: Mapping[str, Any]
) -> dict[str, Any]:

    # The nested settings arrive as a plain dictionary, from user input and from saved states

    settings = dict(settings)

    if isinstance(settings.get('local_expansion'), dict):

        _validate_typed_dict(
            typed_dict=settings['local_expansion'],
            expected_typed_dict_class=LocalExpansionSettingsInputDict,
            object_name='local_expansion',
        )

        settings['local_expansion'] = LocalExpansionSettings(**settings['local_expansion'])

    return settings


class ProxyPrior(SavableClass):

    name = 'proxy_prior'

    def __init__(
            self,
            mean_function: ProxyMeanFunction,
            settings: ProxyPriorSettings
    ) -> None:

        assert settings.bound_value > 0.0, "'bound_value' must be positive."
        assert settings.bound_in_n_sigmas > 0.0, "'bound_in_n_sigmas' must be positive."
        assert 0.0 < settings.amplitude_factor_lower_bound < 1.0, (
            "'amplitude_factor_lower_bound' must be between 0 and 1."
        )
        assert settings.amplitude_factor_upper_bound >= 1.0, (
            "'amplitude_factor_upper_bound' must be at least 1, so that the stated bound itself is allowed."
        )

        if settings.anchor_at_reference or settings.local_expansion is not None:
            assert settings.reference_point is not None, (
                "Must specify 'reference_point' (in real units) to use 'anchor_at_reference' or 'local_expansion'."
            )

        if settings.reference_point is not None:
            n_variables = mean_function.n_variables

            assert _is_list_of_numbers(settings.reference_point) and len(settings.reference_point) == n_variables, (
                f"'reference_point' must be a list with one number per variable ({n_variables}). "
                f"Received {settings.reference_point}."
            )

        if settings.local_expansion is not None:
            # Built once here as well, so that a bad setting fails at construction
            local_expansion_matrices(settings=settings.local_expansion, n_variables=mean_function.n_variables)

        self.mean_function = mean_function
        self.settings = settings

    @classmethod
    def from_mean_function_and_settings(
            cls,
            mean_function: ProxyMeanFunction,
            settings: Mapping[str, Any]
    ) -> Self:

        _validate_typed_dict(
            typed_dict=settings,
            expected_typed_dict_class=ProxyPriorSettingsInputDict,
            object_name=cls.name,
        )

        assert 'bound_value' in settings, (
            "Must specify 'bound_value' in the proxy prior settings, e.g. {'bound_value': 0.01} "
            "if the objective deviates at most 1% from the proxy."
        )

        return cls(
            mean_function=mean_function,
            settings=ProxyPriorSettings(**_with_local_expansion_settings(settings))
        )

    def gather_dicts_to_save(self) -> dict:
        return {
            'name': self.name,
            'state': {
                'mean_function': self.mean_function.gather_dicts_to_save(),
                'settings': self.settings.gather_dicts_to_save()
            }
        }

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:

        mean_function = rehydrate_object(
            superclass=ProxyMeanFunction,  # type: ignore[type-abstract]  # rehydration finds a concrete subclass
            name=saved_state['mean_function']['name'],
            saved_state=saved_state['mean_function']['state']
        )

        settings = ProxyPriorSettings.from_saved_state(
            saved_state=saved_state['settings']
        )

        return cls(
            mean_function=mean_function,
            settings=settings
        )


def _is_list_of_numbers(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(element, (int, float)) for element in value)


def _symmetric_matrix(
        value: Any,
        n_variables: int,
        setting_name: str
) -> torch.Tensor:

    # Lists only: an array or a tensor would run but not save as JSON, and only fail at checkpoint time

    assert isinstance(value, list) and all(_is_list_of_numbers(row) for row in value), (
        f"'{setting_name}' must be a list of lists of numbers (use .tolist() on an array)."
    )

    matrix = torch.tensor(value, dtype=torch.get_default_dtype())

    assert list(matrix.shape) == [n_variables, n_variables], (
        f"'{setting_name}' must have shape [n_variables, n_variables] = [{n_variables}, {n_variables}] "
        f"but has shape {list(matrix.shape)}."
    )

    assert torch.allclose(matrix, matrix.T), f"'{setting_name}' must be symmetric."

    # Symmetrised exactly, so that the matrix checked for definiteness (which reads one triangle)
    # is the matrix used
    return 0.5 * (matrix + matrix.T)


def local_expansion_matrices(
        settings: LocalExpansionSettings,
        n_variables: int
) -> tuple[torch.Tensor, torch.Tensor]:

    # Validates the settings and returns (gradient covariance C, curvature metric H). Everything the
    # kernel needs is checked here and nowhere else, so a kernel built directly is validated too.

    assert settings.gradient_standard_deviation is None or settings.gradient_covariance is None, (
        "Specify either 'gradient_standard_deviation' or 'gradient_covariance' for the local expansion, not both."
    )

    for name in ('shared_curvature_standard_deviation', 'general_curvature_standard_deviation'):
        assert getattr(settings, name) >= 0.0, f"'{name}' must be non-negative."

    if settings.shared_curvature_standard_deviation > 0.0 or settings.general_curvature_standard_deviation > 0.0:
        assert settings.curvature_metric is not None, (
            "Must specify 'curvature_metric' (the proxy's curvature at the reference point) "
            "to use 'shared_curvature_standard_deviation' or 'general_curvature_standard_deviation'."
        )

    if settings.train_amplitude_factor:
        # Strictly: a factor starting on a bound of its interval has an infinite raw value
        assert 0.0 < settings.amplitude_factor_lower_bound < 1.0 < settings.amplitude_factor_upper_bound, (
            "'amplitude_factor_lower_bound' and 'amplitude_factor_upper_bound' of the local expansion "
            "must strictly enclose 1 when 'train_amplitude_factor' is set."
        )

    tolerance = 1e-10

    if settings.gradient_covariance is not None:

        covariance = _symmetric_matrix(settings.gradient_covariance, n_variables, 'gradient_covariance')
        smallest_eigenvalue = float(torch.linalg.eigvalsh(covariance).min())

        assert smallest_eigenvalue >= -tolerance * float(covariance.abs().max()), (
            f"'gradient_covariance' must be positive semi-definite but has the eigenvalue {smallest_eigenvalue}."
        )

    else:

        value = settings.gradient_standard_deviation if settings.gradient_standard_deviation is not None else 0.0

        assert isinstance(value, (int, float)) or _is_list_of_numbers(value), (
            "'gradient_standard_deviation' must be a number or a list of numbers."
        )

        standard_deviations = torch.tensor(value, dtype=torch.get_default_dtype()).expand(n_variables) \
            if isinstance(value, (int, float)) else torch.tensor(value, dtype=torch.get_default_dtype())

        assert list(standard_deviations.shape) == [n_variables], (
            f"'gradient_standard_deviation' must be a number or have one value per variable ({n_variables}) "
            f"but has {list(standard_deviations.shape)}."
        )
        assert bool((standard_deviations >= 0.0).all()), "'gradient_standard_deviation' must be non-negative."

        covariance = torch.diag(standard_deviations ** 2)

    if settings.curvature_metric is None:
        metric = torch.zeros(n_variables, n_variables)

    elif isinstance(settings.curvature_metric, (int, float)):
        metric = abs(settings.curvature_metric) * torch.eye(n_variables)

    else:

        # The magnitude of the curvature: a negative semi-definite Hessian (an optimum of a
        # maximised proxy) is negated, an indefinite one is refused
        metric = _symmetric_matrix(settings.curvature_metric, n_variables, 'curvature_metric')
        eigenvalues = torch.linalg.eigvalsh(metric)
        scale = tolerance * float(metric.abs().max())

        if float(eigenvalues.max()) <= scale:
            metric = -metric

        assert float(eigenvalues.min()) >= -scale or float(eigenvalues.max()) <= scale, (
            f"'curvature_metric' must be positive or negative semi-definite (the curvature at an optimum) "
            f"but has eigenvalues from {float(eigenvalues.min())} to {float(eigenvalues.max())}."
        )

    return covariance, metric


def make_single_column_objective_normaliser(
        normaliser_objectives: NormalisationFunction,
        objective_index: int,
        n_objectives: int
) -> NormalisationFunction:

    # Adapts a normaliser working on [n_points, n_objectives] to a single objective's column.
    #   - Assumes the normaliser works element-wise per objective (true for all current normalisers)

    def normalise_single_column(
            objective_values: torch.Tensor
    ) -> torch.Tensor:

        expanded_values = objective_values.unsqueeze(-1).expand(*objective_values.shape, n_objectives)

        return normaliser_objectives(expanded_values)[..., objective_index]

    return normalise_single_column


def _apply_pointwise(
        function: NormalisationFunction,
        variable_values: torch.Tensor
) -> torch.Tensor:

    # botorch queries with batched points [..., n_points, n_variables]; the normalisers and the
    # proxy take a plain matrix, so flatten and restore around the call

    flattened_values = variable_values.reshape(-1, variable_values.shape[-1])

    return function(flattened_values).reshape(variable_values.shape)


class ProxyMean(gpytorch.means.Mean):  # type: ignore[misc]

    _functions_not_set_message = (
        "The proxy prior's normalisation functions must be set before the model is used. "
        "(This should happen automatically when the optimiser updates its predictor.)"
    )

    def __init__(
            self,
            mean_function: ProxyMeanFunction
    ) -> None:

        super().__init__()

        self.mean_function = mean_function

        self._unnormaliser_variables: Optional[NormalisationFunction] = None
        self._normaliser_objectives: Optional[NormalisationFunction] = None
        self._normaliser_variables: Optional[NormalisationFunction] = None

    def update_normalisation_functions(
            self,
            unnormaliser_variables: NormalisationFunction,
            normaliser_objectives: NormalisationFunction,
            normaliser_variables: Optional[NormalisationFunction] = None
    ) -> None:
        self._unnormaliser_variables = unnormaliser_variables
        self._normaliser_objectives = normaliser_objectives
        self._normaliser_variables = normaliser_variables

    def unnormalise_variables(
            self,
            variable_values: torch.Tensor
    ) -> torch.Tensor:

        # T_x^{-1}(x), keeping gradients and any batch dimensions

        assert self._unnormaliser_variables is not None, self._functions_not_set_message

        return _apply_pointwise(self._unnormaliser_variables, variable_values)

    def normalise_variables(
            self,
            variable_values: torch.Tensor
    ) -> torch.Tensor:

        # T_x(x), for components that are specified at a point given in real units

        assert self._normaliser_variables is not None, (
            "This feature of the proxy prior needs the normaliser of the variables, "
            "which 'update_normalisation_functions' was not given."
        )

        return _apply_pointwise(self._normaliser_variables, variable_values)

    def evaluate_proxy_real_units(
            self,
            variable_values: torch.Tensor
    ) -> torch.Tensor:

        # g(T_x^{-1}(x)): unnormalise the query points, evaluate the proxy in real units

        assert self._unnormaliser_variables is not None, self._functions_not_set_message

        # botorch queries the mean with batched points [..., n_points, n_variables];
        # the proxy contract is a plain 2D matrix, so flatten and restore around the call
        output_shape = variable_values.shape[:-1]
        flattened_values = variable_values.reshape(-1, variable_values.shape[-1])

        # The proxy is evaluated without gradients. This is safe with the current derivative-free
        # acquisition optimisers but would silence gradients through the prior mean if a
        # gradient-based acquisition optimiser is ever added.
        with torch.no_grad():

            proxy_values = self.mean_function(
                variable_values=self._unnormaliser_variables(flattened_values)
            )

        return proxy_values.reshape(output_shape)

    def scale_objectives_to_normalised(
            self,
            objective_values: torch.Tensor
    ) -> torch.Tensor:

        # Transforms a *deviation* (rather than a value) into normalised space, i.e. applies
        # only the scale of the normalisation and not the shift.

        assert self._normaliser_objectives is not None, self._functions_not_set_message

        with torch.no_grad():

            transformed_values = self._normaliser_objectives(objective_values)
            transformed_zeros = self._normaliser_objectives(torch.zeros_like(objective_values))

            return transformed_values - transformed_zeros

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        # m(x) = T_y(g(T_x^{-1}(x))): the prior mean is the proxy, mapped through the
        # current normalisations

        assert self._normaliser_objectives is not None, self._functions_not_set_message

        proxy_values_real_units = self.evaluate_proxy_real_units(
            variable_values=x
        )

        with torch.no_grad():

            return self._normaliser_objectives(proxy_values_real_units)


class _AmplitudeFactor:

    # A scalar factor on a kernel, fixed at 1 or trained within an interval. Shared by the two
    # proxy prior kernels.

    _amplitude_factor_is_trained: bool = False

    def _register_amplitude_factor(
            self,
            train: bool,
            lower_bound: float,
            upper_bound: float,
            initial_value: float
    ) -> None:

        self._amplitude_factor_is_trained = train

        if not train:
            return

        self.register_parameter(  # type: ignore[attr-defined]
            name='raw_amplitude_factor',
            parameter=torch.nn.Parameter(torch.zeros(()))
        )

        self.register_constraint(  # type: ignore[attr-defined]
            param_name='raw_amplitude_factor',
            constraint=Interval(
                lower_bound=lower_bound,
                upper_bound=upper_bound
            )
        )

        # The constraint transform is infinite on the bounds themselves (zero gradient, 'inf' in
        # the saved state), so the start is kept inside the interval
        margin = 0.01 * (upper_bound - lower_bound)

        self.amplitude_factor = min(max(initial_value, lower_bound + margin), upper_bound - margin)

    @property
    def amplitude_factor(self) -> torch.Tensor:

        if self._amplitude_factor_is_trained:
            constraint = self.raw_amplitude_factor_constraint  # type: ignore[attr-defined]
            return constraint.transform(self.raw_amplitude_factor)  # type: ignore[attr-defined]

        else:
            return torch.tensor(1.0)

    @amplitude_factor.setter
    def amplitude_factor(
            self,
            value: Union[float, torch.Tensor]
    ) -> None:

        assert self._amplitude_factor_is_trained, (
            "Cannot set the amplitude factor when 'train_amplitude_factor' is off."
        )

        constraint = self.raw_amplitude_factor_constraint  # type: ignore[attr-defined]

        self.initialize(  # type: ignore[attr-defined]
            raw_amplitude_factor=constraint.inverse_transform(torch.as_tensor(value))
        )


class LocalExpansionKernel(_AmplitudeFactor, gpytorch.kernels.Kernel):  # type: ignore[misc]

    # Covariance of the deviation's first- and second-order Taylor terms about the reference
    # point (see LocalExpansionSettings). With u = x - x0 in real units, e(u) = u^T H u / 2:
    #
    #   k(x, x') = u^T C u'  +  beta_s^2 e(u) e(u')  +  (beta_a^2 / 4) (u^T H u')^2
    #
    # Every term is the kernel of a finite set of polynomial features, so the sum is positive
    # semi-definite, it vanishes at the reference point, and its prior variance is
    #
    #   k(x, x) = u^T C u + (beta_s^2 + beta_a^2) e(u)^2

    has_lengthscale = False

    def __init__(
            self,
            proxy_mean: ProxyMean,
            reference_point: list[float],
            settings: LocalExpansionSettings
    ) -> None:

        super().__init__()

        self.proxy_mean = proxy_mean
        self.settings = settings

        gradient_covariance, curvature_metric = local_expansion_matrices(
            settings=settings,
            n_variables=len(reference_point)
        )

        # Plain attributes rather than buffers: they are derived from the settings, which are
        # saved with the prior, so the state dict holds only the trained factor
        self._reference_point = torch.tensor(reference_point, dtype=torch.get_default_dtype())
        self._gradient_covariance = gradient_covariance
        self._curvature_metric = curvature_metric

        self._register_amplitude_factor(
            train=settings.train_amplitude_factor,
            lower_bound=settings.amplitude_factor_lower_bound,
            upper_bound=settings.amplitude_factor_upper_bound,
            initial_value=1.0
        )

    def _displacement_real_units(
            self,
            variable_values: torch.Tensor
    ) -> torch.Tensor:

        # u = T_x^{-1}(x) - x0

        return self.proxy_mean.unnormalise_variables(variable_values) - self._reference_point

    def forward(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            diag: bool = False,
            **params: Any
    ) -> torch.Tensor:

        if params.pop('last_dim_is_batch', False):
            raise NotImplementedError(f"'last_dim_is_batch' is not supported by {self.__class__.__name__}.")

        u1 = self._displacement_real_units(x1)
        u2 = self._displacement_real_units(x2)

        metric_u1 = u1 @ self._curvature_metric
        metric_u2 = u2 @ self._curvature_metric

        harmonic_1 = 0.5 * (metric_u1 * u1).sum(dim=-1)
        harmonic_2 = 0.5 * (metric_u2 * u2).sum(dim=-1)

        if diag:
            gradient_term = ((u1 @ self._gradient_covariance) * u2).sum(dim=-1)
            shared_term = harmonic_1 * harmonic_2
            general_term = 0.25 * (metric_u1 * u2).sum(dim=-1) ** 2

        else:
            gradient_term = (u1 @ self._gradient_covariance) @ u2.transpose(-1, -2)
            shared_term = harmonic_1.unsqueeze(-1) * harmonic_2.unsqueeze(-2)
            general_term = 0.25 * (metric_u1 @ u2.transpose(-1, -2)) ** 2

        values_real_units = (
            gradient_term +
            self.settings.shared_curvature_standard_deviation ** 2 * shared_term +
            self.settings.general_curvature_standard_deviation ** 2 * general_term
        )

        # A covariance transforms with the square of the objective normalisation's scale
        objective_scale = self.proxy_mean.scale_objectives_to_normalised(
            objective_values=torch.ones((), dtype=values_real_units.dtype)
        )

        return (self.amplitude_factor * objective_scale) ** 2 * values_real_units


class ProxyScaledKernel(_AmplitudeFactor, gpytorch.kernels.Kernel):  # type: ignore[misc]

    # k(x, x') = a(x) a(x') k_b(x, x'), the deviation band around the proxy (see '_scaling'),
    #   - optionally conditioned on the deviation vanishing at the reference point,
    #   - optionally plus the local expansion about that point. The band is then what remains
    #     after the expansion's low-order terms, and should be set accordingly.

    has_lengthscale = False

    # With a hard cap the factor starts just inside it (the constraint transform is infinite on
    # the bound); otherwise at the stated bound itself
    initial_amplitude_factor_under_hard_cap = 0.99

    def __init__(
            self,
            base_kernel: gpytorch.kernels.Kernel,
            proxy_mean: ProxyMean,
            settings: ProxyPriorSettings
    ) -> None:

        super().__init__()

        self.base_kernel = base_kernel
        self.proxy_mean = proxy_mean
        self.settings = settings

        self._reference_point: Optional[torch.Tensor] = None

        if settings.reference_point is not None:
            self._reference_point = torch.tensor(settings.reference_point, dtype=torch.get_default_dtype())

        self.local_expansion: Optional[LocalExpansionKernel] = None

        if settings.local_expansion is not None:

            assert settings.reference_point is not None, "The local expansion needs a reference point."

            self.local_expansion = LocalExpansionKernel(
                proxy_mean=proxy_mean,
                reference_point=settings.reference_point,
                settings=settings.local_expansion
            )

        if settings.anchor_at_reference:
            assert settings.reference_point is not None, "Anchoring needs a reference point."

        # The deviation bound is an upper bound on the band, so with the default cap of 1 the
        # amplitude may shrink with training but never exceed the bound
        self._register_amplitude_factor(
            train=settings.train_amplitude_factor,
            lower_bound=settings.amplitude_factor_lower_bound,
            upper_bound=settings.amplitude_factor_upper_bound,
            initial_value=(
                self.initial_amplitude_factor_under_hard_cap if settings.amplitude_factor_upper_bound == 1.0 else 1.0
            )
        )

    def _deviation_bound_real_units(
            self,
            variable_values: torch.Tensor
    ) -> torch.Tensor:

        # epsilon(x): the promised pointwise deviation band, in real objective units

        if self.settings.bound_type == 'relative':

            proxy_values = self.proxy_mean.evaluate_proxy_real_units(
                variable_values=variable_values
            )

            return (self.settings.bound_value * proxy_values.abs()).clamp(
                min=self.settings.amplitude_floor
            )

        elif self.settings.bound_type == 'absolute':

            return torch.full(
                size=variable_values.shape[:-1],
                fill_value=self.settings.bound_value
            )

        else:
            raise ValueError(f"Unknown bound type: '{self.settings.bound_type}'")

    def _sigma(
            self,
            variable_values: torch.Tensor
    ) -> torch.Tensor:

        # sigma(x) = epsilon(x) / kappa, transformed to normalised objective units

        sigma_real_units = self._deviation_bound_real_units(variable_values) / self.settings.bound_in_n_sigmas

        return self.proxy_mean.scale_objectives_to_normalised(
            objective_values=sigma_real_units
        )

    def _scaling(
            self,
            variable_values: torch.Tensor,
            **params: Any
    ) -> torch.Tensor:

        # a(x) = c * sigma(x) / sqrt(k_b(x, x))
        #   - dividing by the base kernel's diagonal makes it a correlation kernel, so that
        #     the prior variance is exactly c^2 sigma(x)^2 even for base kernels with
        #     k_b(x, x) != 1 (e.g. sums of kernels, spectral mixtures)

        base_diagonal = self._evaluate_base_kernel(
            variable_values, variable_values, diag=True, **params
        ).clamp(min=1e-12).sqrt()

        return self.amplitude_factor * self._sigma(variable_values) / base_diagonal

    def _evaluate_base_kernel(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            diag: bool,
            **params: Any
    ) -> torch.Tensor:

        values = self.base_kernel(x1, x2, diag=diag, **params)

        if not isinstance(values, torch.Tensor):
            # Materialising lazy kernel tensors; fine at the small data sizes veropt is built for
            values = values.to_dense()

        return values

    @staticmethod
    def _outer(
            values_1: torch.Tensor,
            values_2: torch.Tensor,
            diag: bool
    ) -> torch.Tensor:

        if diag:
            return values_1 * values_2

        else:
            return values_1.unsqueeze(-1) * values_2.unsqueeze(-2)

    def _reference_point_normalised(
            self,
            like: torch.Tensor
    ) -> torch.Tensor:

        # x0 in the model's coordinates, shaped [..., 1, n_variables] to match the batch of 'like'

        assert self._reference_point is not None, "This feature needs a reference point."

        reference_point = self.proxy_mean.normalise_variables(self._reference_point.unsqueeze(0).to(like.dtype))

        return reference_point.expand(*like.shape[:-2], 1, like.shape[-1])

    def _anchoring_correction(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            scaling_1: torch.Tensor,
            scaling_2: torch.Tensor,
            diag: bool,
            **params: Any
    ) -> torch.Tensor:

        # Conditioning the band on d(x0) = 0:  k(x, x') - k(x, x0) k(x0, x') / k(x0, x0)
        #   - This is the covariance of the process given its value at x0, so it stays positive
        #     semi-definite
        #   - sigma(x0) cancels between numerator and denominator, leaving
        #     a(x) a(x') k_b(x, x0) k_b(x0, x') / k_b(x0, x0), which is computed directly
        #   - Except where sigma(x0) = 0 exactly (a relative bound around a proxy that is zero
        #     there): then d(x0) = 0 already and says nothing about the rest, so no correction

        reference_1 = self._reference_point_normalised(like=x1)
        reference_2 = self._reference_point_normalised(like=x2)

        base_1 = self._evaluate_base_kernel(x1, reference_1, diag=False, **params).squeeze(-1)
        base_2 = self._evaluate_base_kernel(x2, reference_2, diag=False, **params).squeeze(-1)
        base_at_reference = self._evaluate_base_kernel(reference_1, reference_1, diag=True, **params)

        correction = (
            self._outer(scaling_1 * base_1, scaling_2 * base_2, diag) /
            (base_at_reference if diag else base_at_reference.unsqueeze(-1))
        )

        is_anchored_already = self._sigma(reference_1) <= 0.0

        if not diag:
            is_anchored_already = is_anchored_already.unsqueeze(-1)

        return torch.where(is_anchored_already, torch.zeros_like(correction), correction)

    def forward(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            diag: bool = False,
            **params: Any
    ) -> torch.Tensor:

        if params.pop('last_dim_is_batch', False):
            raise NotImplementedError(f"'last_dim_is_batch' is not supported by {self.__class__.__name__}.")

        # k(x, x') = a(x) a(x') k_b(x, x')

        scaling_1 = self._scaling(x1, **params)
        scaling_2 = self._scaling(x2, **params)

        values = self._outer(scaling_1, scaling_2, diag) * self._evaluate_base_kernel(x1, x2, diag=diag, **params)

        if self.settings.anchor_at_reference:
            values = values - self._anchoring_correction(x1, x2, scaling_1, scaling_2, diag=diag, **params)

        if self.local_expansion is not None:
            values = values + self.local_expansion.forward(x1, x2, diag=diag)

        return values

    def gradient_covariance_at_reference_real_units(
            self,
            relative_step: Optional[float] = None
    ) -> torch.Tensor:

        # Cov(grad d(x0)) of the complete prior, in (objective units / variable units)^2
        #   - The band contributes to it as well as the local expansion's gradient term, so this,
        #     and not the expansion's setting alone, is what the prior says about how far the
        #     objective's gradient at x0 may differ from the proxy's
        #   - It exists only if the band is mean-square differentiable, i.e. the base kernel is
        #     twice differentiable at zero distance: a Matern kernel needs nu >= 1.5, and at
        #     nu = 1.5 the estimate converges only linearly in the step. Kernels below that are
        #     refused rather than given a step-size artefact.
        #   - By central differences of the kernel, d^2 k / dx_i dx'_j at (x0, x0) -- which is
        #     Cov(D_i, D_j) for the central-difference gradient estimator D, so it is symmetric
        #     and positive semi-definite by construction. The step is a fraction of the smallest
        #     lengthscale (normalised units) and the estimate is checked at half the step.

        for module in self.base_kernel.modules():
            if isinstance(module, gpytorch.kernels.MaternKernel) and module.nu < 1.5:
                raise ValueError(
                    f"The prior's gradient covariance does not exist for a Matern base kernel with "
                    f"nu = {module.nu}: its sample paths are not differentiable."
                )

        if relative_step is None:
            lengthscales = [
                float(module.lengthscale.detach().min()) for module in self.base_kernel.modules()
                if getattr(module, 'has_lengthscale', False)
            ]
            relative_step = 1e-3 * min(lengthscales + [1.0])

        assert self._reference_point is not None, "This feature needs a reference point."

        n_variables = len(self._reference_point)
        reference_point = self._reference_point_normalised(like=torch.zeros(1, n_variables)).squeeze(0)

        def stencil(step: float) -> torch.Tensor:
            steps = step * torch.eye(n_variables, dtype=reference_point.dtype)
            return torch.cat([reference_point + steps, reference_point - steps])

        def second_difference(step: float) -> torch.Tensor:
            values = self.forward(stencil(step), stencil(step))
            n = n_variables
            return (values[:n, :n] - values[:n, n:] - values[n:, :n] + values[n:, n:]) / (2.0 * step) ** 2

        with torch.no_grad():

            coarse, fine = second_difference(relative_step), second_difference(0.5 * relative_step)
            disagreement = float(torch.linalg.norm(fine - coarse) / torch.linalg.norm(fine).clamp(min=1e-300))

            if disagreement > 1e-2:
                warnings.warn(
                    f"The prior's gradient covariance at the reference point changed by {disagreement:.1%} when "
                    f"the finite-difference step ({relative_step:.1e}, normalised units) was halved; "
                    f"the estimate is not converged."
                )

            # Chain rule back to real units: variables through the (diagonal) Jacobian of T_x^{-1},
            # the objective through the scale of T_y
            real_stencil = self.proxy_mean.unnormalise_variables(stencil(relative_step))
            variable_scales = (
                (real_stencil[:n_variables] - real_stencil[n_variables:]).diagonal() / (2.0 * relative_step)
            )
            objective_scale = self.proxy_mean.scale_objectives_to_normalised(torch.ones((), dtype=fine.dtype))

            return fine / (torch.outer(variable_scales, variable_scales) * objective_scale ** 2)
