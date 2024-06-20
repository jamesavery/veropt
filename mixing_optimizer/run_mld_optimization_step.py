from veropt import BayesOptimiser
from veropt.obj_funcs.mld_objective_function_2d_parallel import *
from veropt.gui import veropt_gui
from veropt.experiment import *
import datetime as dt

n_init_points = 10
n_bayes_points = 30
n_evals_per_step = 2
experiment = "mld_experiment7"
# criterion = 'rho'

params = ["c_k", "c_eps"]
param_list = f'/data/ocean/veropt_results/{experiment}/param_list_{experiment}.txt'

with open(param_list, "w") as file:
    file.write(f"{params[0]},{params[1]},obj_func")

now = dt.datetime.now()

obj_func = MLD1ObjFun(param_list)

optimiser = BayesOptimiser(n_init_points, n_bayes_points, obj_func, n_evals_per_step=n_evals_per_step)

# veropt_gui.run(optimiser)

optimiser.run_all_opt_steps()

optimiser.save_optimiser(f'nbayes_{n_bayes_points}_{experiment}_{now.day}-{now.month}-{now.year}_1deg_2d_parallel_opt.pkl')