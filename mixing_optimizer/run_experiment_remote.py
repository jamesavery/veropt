from veropt import BayesOptimiser, load_optimiser
from veropt.obj_funcs.mld_obj_function_remoteslurm import *
from veropt.experiment import *
import sys, pyjson5 as json, datetime as dt


# Call as: pytho3 run_experiment_remote.py <experiment_name> <remote_server_name>
# E.g.: python3 run_experiment_remote.py 4deg_tke_eke_wind_observations lumi
try: 
    experiment_name, remote_server_name = sys.argv[1:3]
except:
    print(f"Syntax: python3 run_mld_remote.py <experiment_name> <remote_server_name> [resume_from_step] [resume_extra_steps]")
    sys.exit(1)

if(len(sys.argv)>3):
    resume_from_step   = int(sys.argv[3])
else:
    resume_from_step   = None
    
if(len(sys.argv)>4):
    resume_extra_steps = int(sys.argv[4])
else:
    resume_extra_steps = 0


local_cfg = {
 'datadir':  "/data/ocean/",
 'outputdir':"/data/ocean/veropt_results/",
 'source_dir': "/opt/code/ocean/veropt/" 
}

## TODO: Possibly move to ObjectFunction constructor
# READ IN SERVER CONFIG
try:
    # TODO: restructure as servers/{server_name}/server.json?        
    with open(f"{local_cfg['source_dir']}/servers/servers.json","r") as f:
        servers = json.load(f)
    server_cfg = servers[remote_server_name]
except Exception as e:
    print(f"While reading {local_cfg['source_dir']}/servers/servers.json:\n\t{e}")
    sys.exit(1)

# READ IN EXPERIMENT CONFIG
experiment_dir      = f"{local_cfg['outputdir']}/{experiment_name}"
try:
    with open(f"{experiment_dir}/experiment.json","r") as f:
        expt_cfg = json.load(f)
except Exception as e:
    print(f"While reading {experiment_dir}/experiment.json:\n\t{e}")
    sys.exit(2)

# READ IN OR CREATE EXPERIMENT STATE
state_filename = f"{experiment_dir}/experiment_state.json"
if os.path.exists(state_filename):
    try: 
        with open(state_filename,"r") as f:
            expt_state = json.load(f)
    except Exception as e:
        print(f"While reading {state_filename}:\n\t{e}")
        sys.exit(3)
else: 
    expt_state = {'experiment_name': experiment_name,
                 'experiment_dir': experiment_dir,
                 'next_point': 0,
                 'points': {}
                 }
    with open(state_filename,"wb") as f:
        json.encode_io(expt_state,f)
        

# READ IN OPTIMIZER CONFIG
try: 
    with open(f"{experiment_dir}/optimizer.json","r") as f:
        opt_cfg = json.load(f)
except Exception as e: 
    print(f"While reading {experiment_dir}/optimizer.json:\n\t{e}")
    sys.exit(4)    


print(f"\n\nExperiment config: {expt_cfg}\n")
print(f"Server config: {server_cfg}\n")
print(f"Local machine config: {local_cfg}\n")
print(f"Optimisation config: {opt_cfg}\n\n")

# SET UP OPTIMIZER 
obj_func = MLD1ObjFun(expt_state, expt_cfg,server_cfg,local_cfg, opt_cfg)

n_init_rounds    = opt_cfg['n_init_rounds']
n_bayes_rounds   = opt_cfg['n_bayes_rounds']
n_evals_per_step = opt_cfg['n_evals_per_step']

if "random_seed" in opt_cfg:
    random_seed      = opt_cfg['random_seed']
    print(f"Reproducible run: random_seed={random_seed}")
else:
    random_seed = None

if resume_from_step is None:
    optimiser = BayesOptimiser(n_init_rounds*n_evals_per_step, n_bayes_rounds*n_evals_per_step, obj_func, n_evals_per_step=n_evals_per_step, random_seed=random_seed)
    if 'raw_lengthscale' in opt_cfg: optimiser.model.constraint_dict_list[0]["covar_module"]["raw_lengthscale"] = opt_cfg['raw_lengthscale']
    if 'beta'            in opt_cfg: optimiser.set_acq_func_params("beta", opt_cfg['beta'])
    resume_from_step = 0
    opt_steps        = n_init_rounds + n_bayes_rounds        
else:
    print(f"Loading optimizer from checkpoint no. {resume_from_step} for {resume_extra_steps} additional steps")
    opt_steps = resume_from_step + resume_extra_steps
    
# optimiser.run_all_opt_steps()

# RUN ALL OPTIMISATION STEPS WITH CHECKPOINTING
for i in range(resume_from_step,opt_steps):
    if i == 0:
        pass
    else:
        optimiser_to_load = f"{local_cfg['outputdir']}/{experiment_name}/{experiment_name}_opt_step_{i-1}.pkl"
        optimiser = load_optimiser(optimiser_to_load)

    optimiser.run_opt_step()
    optimiser_savename = f"{local_cfg['outputdir']}/{experiment_name}/{experiment_name}_opt_step_{i}.pkl"
    optimiser.save_optimiser(optimiser_savename)
