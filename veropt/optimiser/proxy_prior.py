import abc
from dataclasses import dataclass
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


class ProxyPriorSettingsInputDict(TypedDict, total=False):
    bound_value: float
    bound_type: Literal['relative', 'absolute']
    bound_in_n_sigmas: float
    train_amplitude_factor: bool
    amplitude_factor_lower_bound: float
    amplitude_floor: float


@dataclass
class ProxyPriorSettings(SavableDataClass):
    bound_value: float
    bound_type: Literal['relative', 'absolute'] = 'relative'
    bound_in_n_sigmas: float = 2.0
    train_amplitude_factor: bool = True
    amplitude_factor_lower_bound: float = 0.1
    amplitude_floor: float = 0.0


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
            settings=ProxyPriorSettings(**settings)
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


class ProxyMean(gpytorch.means.Mean):  # type: ignore[misc]

    def __init__(
            self,
            mean_function: ProxyMeanFunction
    ) -> None:

        super().__init__()

        self.mean_function = mean_function

        self._unnormaliser_variables: Optional[NormalisationFunction] = None
        self._normaliser_objectives: Optional[NormalisationFunction] = None

    def update_normalisation_functions(
            self,
            unnormaliser_variables: NormalisationFunction,
            normaliser_objectives: NormalisationFunction
    ) -> None:
        self._unnormaliser_variables = unnormaliser_variables
        self._normaliser_objectives = normaliser_objectives

    def evaluate_proxy_real_units(
            self,
            variable_values: torch.Tensor
    ) -> torch.Tensor:

        # g(T_x^{-1}(x)): unnormalise the query points, evaluate the proxy in real units

        assert self._unnormaliser_variables is not None, (
            "The proxy prior's normalisation functions must be set before the model is used. "
            "(This should happen automatically when the optimiser updates its predictor.)"
        )

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

        assert self._normaliser_objectives is not None, (
            "The proxy prior's normalisation functions must be set before the model is used. "
            "(This should happen automatically when the optimiser updates its predictor.)"
        )

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

        assert self._normaliser_objectives is not None, (
            "The proxy prior's normalisation functions must be set before the model is used. "
            "(This should happen automatically when the optimiser updates its predictor.)"
        )

        proxy_values_real_units = self.evaluate_proxy_real_units(
            variable_values=x
        )

        with torch.no_grad():

            return self._normaliser_objectives(proxy_values_real_units)


class ProxyScaledKernel(gpytorch.kernels.Kernel):  # type: ignore[misc]

    has_lengthscale = False

    # Initialising just inside the upper bound since the constraint transform is infinite at the boundary
    initial_amplitude_factor = 0.99

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

        if settings.train_amplitude_factor:

            self.register_parameter(
                name='raw_amplitude_factor',
                parameter=torch.nn.Parameter(torch.zeros(()))
            )

            # The deviation bound is an upper bound, so the amplitude may shrink with training
            # but can never exceed the bound.
            self.register_constraint(
                param_name='raw_amplitude_factor',
                constraint=Interval(
                    lower_bound=settings.amplitude_factor_lower_bound,
                    upper_bound=1.0
                )
            )

            self.amplitude_factor = torch.tensor(self.initial_amplitude_factor)

    @property
    def amplitude_factor(self) -> torch.Tensor:

        if self.settings.train_amplitude_factor:
            return self.raw_amplitude_factor_constraint.transform(self.raw_amplitude_factor)

        else:
            return torch.tensor(1.0)

    @amplitude_factor.setter
    def amplitude_factor(
            self,
            value: Union[float, torch.Tensor]
    ) -> None:

        assert self.settings.train_amplitude_factor, (
            "Cannot set the amplitude factor when 'train_amplitude_factor' is off."
        )

        value = torch.as_tensor(value)

        self.initialize(
            raw_amplitude_factor=self.raw_amplitude_factor_constraint.inverse_transform(value)
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

        base_values = self._evaluate_base_kernel(x1, x2, diag=diag, **params)

        if diag:
            return scaling_1 * scaling_2 * base_values

        else:
            return scaling_1.unsqueeze(-1) * scaling_2.unsqueeze(-2) * base_values
