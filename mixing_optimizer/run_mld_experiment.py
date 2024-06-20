from veropt import BayesOptimiser, load_optimiser
from veropt.obj_funcs.mld_obj_function_aegir import *
from veropt.experiment import *
import datetime as dt

rootdir = "/groups/ocean/mmroz/mld_experiments_aegir"
experiment = "mld_test_experiment"

n_init_points = 1
n_bayes_points = 1
n_evals_per_step = 1
n_points = n_init_points + n_bayes_points
opt_steps = n_points // n_evals_per_step

n_params = 4
param_names = ["c_k", "c_eps", "alpha_tke", "kappaM_min"]
bounds_lower = [.05, .05, .05, 1e-7]
bounds_upper = [1., 1., 1e4, 1e-3]
param_list_filename = f'{rootdir}/{experiment}/param_list_{experiment}.txt'
param_str = ""
for p in param_names:
    param_str += f"{p},"

with open(param_list_filename, "w") as file:
    file.write(f"{param_str}obj_func")

now = dt.datetime.now()

obj_func = MLD1ObjFun(param_names, param_list_filename, n_params, bounds_lower, bounds_upper, 
    rootdir=rootdir, experiment=experiment, float_type="float32", 
    ncores=1, ncores_nx=1, ncores_ny=1, ncycles=1, res="4deg", lat_range=(-52,50),
    target="observations", target_filename="IFREMER_mld_rho_50S50N_4deg")

optimiser = BayesOptimiser(n_init_points, n_bayes_points, obj_func, n_evals_per_step=n_evals_per_step)
optimiser.model.constraint_dict_list[0]["covar_module"]["raw_lengthscale"] = [0.1, 5]
optimiser.set_acq_func_params("beta", 0.8)

# optimiser.run_all_opt_steps()

for i in range(opt_steps):
    if i == 0:
        pass
    else:
        optimiser_to_load = f'{rootdir}/{experiment}/{experiment}_opt_step_{i-1}.pkl'
        optimiser = load_optimiser(optimiser_to_load)

    optimiser.run_opt_step()
    optimiser_savename = f'{rootdir}/{experiment}/{experiment}_opt_step_{i}.pkl'
    optimiser.save_optimiser(optimiser_savename)