from veropt import BayesOptimiser, load_optimiser
from veropt.obj_funcs.mld_obj_function_remoteslurm import *
from veropt.experiment import *
import sys, json, datetime as dt

# Call as: pytho3 run_mld_remote.py <experiment_name> <remote_server_name>
# E.g.: python3 run_mld_remote.py 4deg_tke_eke_wind_observations lumi
try: 
    experiment_name, remote_server_name, optimizer_config_name = sys.argv[1:4]
except:
    print(f"Syntax: python3 run_mld_remote.py <experiment_name> <remote_server_name> <optimizer_config_name>")
    sys.exit(1)

local_cfg = {
 'datadir':  "/data/ocean/",
 'outputdir':"/data/ocean/veropt_results/"
}



############ COMPUTE SERVER SETTINGS - ############
# TODO: All the server settings should live in a json file
servers = {
    "aegir":{
        # SSH access setup - needs to be set up in ~/.ssh/config
        'hostname': "aegir",   # Name of the server in ~/.ssh/config

        # Slurm setup
        'account': 'nn9297k',  
        'partition': 'aegir',  # Slurm-partition
        'constraints': "v1",   # Slurm-constraints: desired CPU arch for running Veros on Aegir
        'max_time':'23:59:59',
        'n_cores':16, 'n_cores_nx':4, 'n_cores_ny':4,

        # Veros run settings
        'float_type':"float64",
        'remote_outputdir':"/groups/ocean/mmroz",
        'device_type':"cpu",
        'backend': "numpy",
        'n_cycles': 6,

        # VerOpt settings
        'n_evals_per_step': 16 # Number of parallel trials per opt step. TODO: Possibly make more flexible
#TODO: template for slurm batch job
    },

    "lumi": {
        # SSH access setup - needs to be set up in ~/.ssh/config
        'hostname': "lumi",
        
        # Slurm setup
        'account': 'project_465000815',
        'partition': 'small-g',
        'constraints': None, # All LUMI small-g compute nodes are identical
        'max_time':'71:59:59',
        'n_cores':1, 'n_cores_nx':1, 'n_cores_ny':1,

        # Veros run settings
        'remote_outputdir':"~/ocean/veropt_results/",
        'float_type':"float64",
        'device_type':"gpu",
        'backend': "jax",
        'n_cycles': "All",

        # VerOpt settings
        'n_evals_per_step': 2
#TODO: template for slurm batch job        
    }
}

server_cfg = servers[remote_server_name]
###################################################

# READ IN EXPERIMENT CONFIG
experiment_dir      = f"{local_cfg['outputdir']}/{experiment_name}"
try:
    with open(f"{experiment_dir}/experiment.json","r") as f:
        expt_cfg = json.load(f)
except:
    raise FileNotFoundError(f"Experiment config file {experiment_dir}/experiment.json not found")
 
# READ IN OPTIMIZER CONFIG
try: 
    with open(f"{experiment_dir}/{optimizer_config_name}.json","r") as f:
        opt_cfg = json.load(f)
except: 
    raise FileNotFoundError(f"Optimizer config file {experiment_dir}/{optimizer_config_name}.json not found")

print(f"Experiment config: {expt_cfg}")
print(f"Server config: {server_cfg}")
print(f"Local machine config: {local_cfg}")
print(f"Optimisation config: {opt_cfg}")

# SET UP OPTIMIZER 
obj_func = MLD1ObjFun(expt_cfg,server_cfg,local_cfg)

n_init_rounds    = opt_cfg['n_init_rounds']
n_bayes_rounds   = opt_cfg['n_bayes_rounds']
n_evals_per_step = server_cfg['n_evals_per_step']

optimiser = BayesOptimiser(n_init_rounds*n_evals_per_step, n_bayes_rounds*n_evals_per_step, obj_func, n_evals_per_step=n_evals_per_step)
optimiser.model.constraint_dict_list[0]["covar_module"]["raw_lengthscale"] = opt_cfg['raw_lengthscale']
optimiser.set_acq_func_params("beta", opt_cfg['beta'])

# optimiser.run_all_opt_steps()

# RUN ALL OPTIMISATION STEPS WITH CHECKPOINTING
opt_steps = n_init_rounds + n_bayes_rounds
for i in range(opt_steps):
    if i == 0:
        pass
    else:
        optimiser_to_load = f"{local_cfg['outputdir']}/{experiment_name}/{experiment_name}_opt_step_{i-1}.pkl"
        optimiser = load_optimiser(optimiser_to_load)

    optimiser.run_opt_step()
    optimiser_savename = f"{local_cfg['outputdir']}/{experiment_name}/{experiment_name}_opt_step_{i}.pkl"
    optimiser.save_optimiser(optimiser_savename)
