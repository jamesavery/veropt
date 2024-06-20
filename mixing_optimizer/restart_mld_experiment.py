from veropt import BayesOptimiser, load_optimiser
from veropt.obj_funcs.mld_obj_function import *
from veropt.gui import veropt_gui
from veropt.experiment import *
import datetime as dt

rootdir = "/groups/ocean/mmroz"
experiment = "mld_experiment6"

last_opt_step = 2
opt_steps = 21
optimiser_to_load = f'{rootdir}/{experiment}/{experiment}_opt_step_{last_opt_step}.pkl'

optimiser = load_optimiser(optimiser_to_load)

for i in range(last_opt_step+1, opt_steps):
    if i == last_opt_step+1:
        pass
    else:
        optimiser_to_load = f'{rootdir}/{experiment}/{experiment}_opt_step_{i-1}.pkl'
        optimiser = load_optimiser(optimiser_to_load)

    optimiser.run_opt_step()
    optimiser_savename = f'{rootdir}/{experiment}/{experiment}_opt_step_{i}.pkl'
    optimiser.save_optimiser(optimiser_savename)