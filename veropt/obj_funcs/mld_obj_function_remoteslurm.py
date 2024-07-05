import os, sys, json, stat, time, shutil, subprocess
from dataclasses import dataclass, field
import xarray as xr, numpy as np
from veropt import ObjFunction


def _write_batch_script(file_path_name: str, string: str) -> None:
    with open(file_path_name, 'w+') as file:
        file.write(string)

    st = os.stat(file_path_name)
    os.chmod(file_path_name, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH )


def correct_coords(optimized_filename, res):
    ds = xr.open_dataset(f"{optimized_filename}.nc")
    if res == "4deg":
        new_xu_coords = np.concatenate([np.arange(90., 360., 4.), np.arange(2., 90., 4.)])
        new_xt_coords = np.concatenate([np.arange(88., 360., 4.), np.arange(0., 88., 4.)])
    elif res == "1deg":
        new_xu_coords = np.concatenate([np.arange(90., 360., 1.), np.arange(0., 90., 1.)])
        new_xt_coords = np.concatenate([np.arange(90.5, 360.5, 1.), np.arange(0.5, 90.5, 1.)])
    else:
        raise ValueError("Resolution should either be '4deg' or '1deg'.")
    
    ds.coords['xt'] = new_xt_coords
    ds = ds.sortby(ds.xt)
    ds.coords['xu'] = new_xu_coords
    ds = ds.sortby(ds.xu)
    ds.to_netcdf(f'{optimized_filename}_corr_coords.nc')


def calc_y(optimized_dataset, target_dataset, lat_range):  
    lat_min, lat_max = lat_range  
    mld = optimized_dataset.mld.isel(Time=-1).sel(yt=slice(lat_min, lat_max)).values
    mld = np.where(mld == 0., -1., mld)

    target_mld = target_dataset.mld.values
    target_mld = np.where(target_mld == 0., -1., target_mld)
    
    MSE = np.nanmean(np.square((mld-target_mld)/target_mld))

    return -np.sqrt(MSE)


class MLD1ObjFun(ObjFunction):
    def __init__(self, 
                 expt_cfg, server_cfg, local_cfg): 

        self.expt_cfg   = expt_cfg
        self.server_cfg = server_cfg
        self.local_cfg  = local_cfg         

        # location, name, latitude range and resolution of the target mixed layer depth (MLD) map
        #target_type: "simulated_target" | "ifremer"
        #experiment_type: "mixed_layer_depth" | "..."
        local_outputdir = local_cfg['outputdir']
        local_datadir   = local_cfg['datadir']
        target_type     = expt_cfg['target_type']
        experiment_type = expt_cfg['experiment_type']
        experiment_name = expt_cfg['experiment_name']

        self.target_directory = f"{local_datadir}/{target_type}/{experiment_type}"
        self.target_dataset = xr.open_dataset(f"{self.target_directory}/{expt_cfg['target_filename']}.nc")

        # optimiser setup
        self.param_list_filename = f'{local_outputdir}/{experiment_name}/param_list_{experiment_name}.txt'
        param_names = expt_cfg['param_names']
        param_str   = ",".join(param_names)

        with open(self.param_list_filename, "w") as file:
            file.write(f"{param_str}obj_func")        

        bounds = [expt_cfg['bounds_lower'], expt_cfg['bounds_upper']]
        n_objs = expt_cfg['n_objectives']
        
        # TODO: Potentially not none - should live in... experiment.json 
        init_vals = None
        stds = None

        # slurm setup 
        # TODO: Gradually move to server configuation
        # TODO: Also clean up to reflect structure
        self.sleeping_time: int = 10 # in seconds
        self.experiment_name: str = experiment_name
        self.source_file_path: str = f"{local_outputdir}/{self.experiment_name}/experiment.py"
        self.assets_file_path: str = f"{local_outputdir}/{self.experiment_name}/assets.json"
        self.startup_jobs: dict = field(default_factory=dict)

        super().__init__(self.mldObjFunc, bounds, len(param_names), n_objs, init_vals, stds)


    def mldObjFunc(self, new_x):

        experiment_name = self.expt_cfg['experiment_name']
        local_outputdir = self.local_cfg['outputdir']
        ncycles = self.server_cfg['n_cycles']
        
        n_points = len(new_x.numpy()[0])
        n_params = len(new_x.numpy()[0][0])
        setup_args = ()
        y = []
        param_val_strings = []
        param_names = self.expt_cfg['param_names']

        for i in range(n_points):
            # in new_x.numpy()[i][j][k]
            # index i: objective index (as default, n_obj = 1)
            # index j: index of point within opt step
            # index k: param index
            param_val_string = ""
            setup_string = ""
            for j in range(n_params):
                param_name, param_val = param_names[j], new_x.numpy()[0][i][j]
                param_val_string += f"{param_val},"
                setup_string += f"{param_name}={param_val}="
            setup_args += (setup_string[:-1],)
            param_val_strings.append(param_val_string)

        print(setup_args)
        self.setup_runs(setup_args)
        print("Finished setting up runs")
        self.transfer_jobs(setup_args)
        print("Finished transferring jobs")
        sys.exit(0)

        self.start_jobs(setup_args)

        for i in range(ncycles):
            self.check_jobs_status(setup_args)

        setup_names = [f"{experiment_name}={arg}" for arg in setup_args]

        for setup, param_val_string in zip(setup_names, param_val_strings):
            optimized_filepath = f"{local_outputdir}/{experiment_name}/{setup}"
            optimized_filename = f"/{setup}.{str(ncycles-1).zfill(4)}.averages"

            target_type = self.expt_cfg['target_type']
            if target_type == "ifremer":
                correct_coords(f"{optimized_filepath}/{optimized_filename}", self.expt_cfg['map_resolution'])
                optimized_dataset = xr.open_dataset(f"{optimized_filepath}/{optimized_filename}_corr_coords.nc")
            elif target_type == "simulated_target": 
                optimized_dataset = xr.open_dataset(f"{optimized_filepath}/{optimized_filename}.nc")
            else:
                raise ValueError("Target type should be either 'ifremer' or 'simulated_target'.")

            new_y = calc_y(optimized_dataset, self.target_dataset, lat_range=self.expt_cfg['lat_range'])
            y.append(new_y)

            with open(self.param_list_filename, "a") as file:
                file.write(f"\n{param_val_string},{new_y}")

        if len(y) == 0:
            y = y[0]
        else: pass

        return y

    
    def setup_runs(self, setup_args) -> None:
        experiment_name = self.expt_cfg['experiment_name']
        local_outputdir = self.local_cfg['outputdir']     

        setup_names = [f"{experiment_name}={arg}" for arg in setup_args] #  "mld_experiment1=c_k=0.1", "mld_experiment1-c_k-0.2",

        for setup in setup_names:
            setup_dir = f"{local_outputdir}/{experiment_name}/{setup}"

            if not os.path.exists(setup_dir):
                os.makedirs(setup_dir)
                print(f"\nDirectory created: {setup_dir}")
            else:
                print(f"\nDirectory exists: {setup_dir}")

            if not os.path.isfile(f"{setup_dir}/{setup}.py"):
                shutil.copy(self.source_file_path, f"{setup_dir}/experiment.py")
                print(f"    File {self.source_file_path}.py was copied to: {setup_dir}")
            else:
                print(f"    File exists: {setup_dir}/{setup}.py")

            if not os.path.isfile(f"{setup_dir}/assets.json"):
                shutil.copy(self.assets_file_path, f"{setup_dir}/assets.json")
                print(f"    File assets.json was copied to: {setup_dir}")
            else:
                print(f"    File exists: {setup_dir}/assets.json")

            batch_script_str = self.make_batch_script(setup)
            _write_batch_script(f"{setup_dir}/run_veros.slurm", batch_script_str)


    def transfer_jobs(self, setup_args) -> None:
        experiment_name = self.expt_cfg['experiment_name']
        local_outputdir = self.local_cfg['outputdir']        

        setup_names = [f"{experiment_name}={arg}" for arg in setup_args] #  "mld_experiment1=c_k=0.1", "mld_experiment1-c_k-0.2",

        hostname = self.server_cfg['hostname']

        for setup in setup_names:
            setup_dir  = f"{local_outputdir}/{experiment_name}/{setup}"
            remote_dir = f"{hostname}:ocean/veropt_results/{experiment_name}/"

            print(f"Attempting: scp -Cr {setup_dir} {remote_dir}/")
            # TODO: Use python fabric module for error handling
            pipe = subprocess.Popen(["scp","-Cr", setup_dir, f"{remote_dir}/"],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True)
            result = pipe.stdout.read()
            stderr = pipe.stderr.read()
            if not stderr:
                print(f"Transferred {setup_dir} to {remote_dir} with result: {result}")
            else:
                print(f"Error transferring {setup_dir} to {remote_dir}")
                print(f"Error: {stderr}")
                print(f"Aborting.\n\n")
                sys.exit(1)
        

    def start_jobs(self, setup_args) -> None:
        experiment_name = self.expt_cfg['experiment_name']
        local_outputdir = self.local_cfg['outputdir']                
        #self.startup_jobs = {} # remove

        setup_names = [f"{experiment_name}={arg}" for arg in setup_args] #  "mld_experiment1-c_k-0.1", "mld_experiment1-c_k-0.2",
        hostname = self.server_cfg['hostname']

        for setup in setup_names:
            print(f"\nSubmitting job {hostname}:ocean/{experiment_name}/{setup}/veros_batch.sh")            

            (f"ssh {hostname} 'cd ocean/{experiment_name}/{setup} && sbatch --parsable veros_batch.sh'")

            pipe = subprocess.Popen(["ssh", hostname, 
                                     f"'cd ocean/{experiment_name}/{setup} && sbatch --parsable veros_batch.sh'"],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True)
            jobid = int(pipe.stdout.read())
            stderr = pipe.stdout.read()

            if not stderr:
                print(f"Submitted {jobid} for setup {setup}")
                #self.startup_jobs.update({setup: jobid}) # remove
                #os.system('squeue -p aegir -j ' + jobid + ' > squeue.out')
            else:
                print(f"Setup {setup} was not submitted, aborting...")
                sys.exit(1)


    def check_jobs_status(self, setup_args) -> None:
        """JobId=43848356
           JobName=mld_experiment1
           JobState=RUNNING/PENDING
        """
        experiment_name = self.expt_cfg['experiment_name']
        local_outputdir = self.local_cfg['outputdir']                

        setup_names = [f"{experiment_name}={arg}" for arg in setup_args] #  "mld_experiment1-c_k-0.1", "mld_experiment1-c_k-0.2",
        submitted_jobs = setup_names.copy()

        while submitted_jobs:

            for setup in submitted_jobs:

                log_file = f"{local_outputdir}/{experiment_name}/{setup}/{setup}.out"
                if os.path.isfile(log_file) and os.stat(log_file).st_size != 0:
                    with open(log_file, "r") as f:
                        jobid = f.readline().split("=")[-1].strip()
                else:
                    print(f"\nSetup {setup} has not been started yet")                    
                    # break
                    continue

                pipe = subprocess.Popen(["scontrol", "show", "job", jobid],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True)
                stdout = pipe.stdout.read()
                stderr = pipe.stderr.read()

                if stdout:
                    job_dict = {}
                    for item in stdout.split():
                        key, *value = item.split('=')
                        job_dict[key] = value[0] if value else None  # Using None or '' as the default value

                    print("Job {jd[JobName]}({jd[JobId]}) status: {jd[JobState]}".format(jd=job_dict))

                if "slurm_load_jobs error: Invalid job id specified" in stderr:
                    print(f"{jobid} status: COMPLETED")
                    submitted_jobs.remove(setup)

            if submitted_jobs:
                print("\nThe following jobs are still pending or running: " + ", ".join(submitted_jobs))
                time.sleep(self.sleeping_time)


    def make_batch_script(self, setup_name: str) -> str:
        experiment_name = self.expt_cfg['experiment_name']
        local_outputdir = self.local_cfg['outputdir']                
        local_sourcedir = self.local_cfg['source_dir']
        server_hostname = self.server_cfg['hostname']

        dummy = setup_name.split("=")[1:]
        setup_args_dict = {dummy[i]: float(dummy[i + 1]) for i in range(0, len(dummy), 2)}
        with open(f"{local_outputdir}/{experiment_name}/{setup_name}/setup_args.txt", "w") as f:
            json.dump(setup_args_dict, f)

        with open(f"{local_sourcedir}/servers/{server_hostname}/run_veros.slurm", "r") as f:
            batch_script_str = f.read()
        
        self.server_cfg['setup_name'] = setup_name
        return batch_script_str.format(**self.server_cfg)
