from typing import Literal

import botorch
import torch

from veropt import ProxyMeanFunction, bayesian_optimiser
from veropt.optimiser.practice_objectives import Hartmann


# When a fast, approximate version of the objective exists — and we know how far the true
# objective can deviate from it — the surrogate model can start from the proxy instead of
# from a flat prior, so the expensive evaluations only have to learn the residual.
#
# Here the "expensive" objective is the Hartmann function and the proxy is a distorted
# version of it. The distortion is at most 1% of the objective value, so we can honestly
# promise the optimiser 'bound_value': 0.01.
#
# Notes:
#   - The proxy is evaluated in real units (real variable values in, real objective values out)
#   - The proxy is called many times per optimisation step, so it should be fast (much less
#     than a second per call)
#   - With an informative prior, it pays to start model training as early as possible, hence
#     'n_points_before_fitting' is set to a single batch below


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

        # The proxy can receive values a floating point rounding error outside the bounds,
        # so functions with a strictly validated domain (like botorch's) should clamp
        variable_values = variable_values.clamp(0.0, 1.0)

        true_values = self.function(variable_values)

        return true_values * (1.0 + 0.01 * torch.sin(6.0 * torch.pi * variable_values[:, 0]))


n_variables: Literal[6] = 6
n_evaluations_per_step = 4

objective = Hartmann(
    n_variables=n_variables
)

proxy = DistortedHartmannProxy(
    n_variables=n_variables
)

optimiser = bayesian_optimiser(
    n_initial_points=8,
    n_bayesian_points=32,
    n_evaluations_per_step=n_evaluations_per_step,
    objective=objective,
    model={
        'kernels': 'matern',
        'proxy_prior': proxy,
        'proxy_prior_settings': {
            'bound_value': 0.01,  # the objective is within 1% of the proxy...
            'bound_in_n_sigmas': 2.0,  # ...interpreted as two standard deviations of the prior
            'amplitude_floor': 0.01  # keeps the prior variance away from zero where the proxy is ~0
        }
    },
    n_points_before_fitting=n_evaluations_per_step
)

for step in range(4):
    optimiser.run_optimisation_step()

best_objective_value = float(optimiser.evaluated_objectives_real_units.max())

print(f"\nBest objective value found: {best_objective_value:.4f} (the optimum is 3.3224)")
