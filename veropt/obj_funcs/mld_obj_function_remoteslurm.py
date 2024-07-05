import os, sys, json, stat, time, tqdm, shutil, subprocess
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
        self.source_file_path: str = f"{local_outputdir}/{experiment_name}/experiment.py"
        self.assets_file_path: str = f"{local_outputdir}/{experiment_name}/assets.json"

        super().__init__(self.mldObjFunc, bounds, len(param_names), n_objs, init_vals, stds)


    def mldObjFunc(self, new_x):

        experiment_name = self.expt_cfg['experiment_name']
        local_outputdir = self.local_cfg['outputdir']
        n_cycles = self.server_cfg['n_cycles']
        
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
        jobids = self.start_jobs(setup_args)
        print(f"Finished starting jobs, job ids: {jobids}") 

        for i in range(n_cycles):
            self.check_jobs_status(setup_args, jobids)
        
        self.transfer_results(setup_args)

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


    #TODO: Tidy up.    
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
        experiment_name  = self.expt_cfg  ['experiment_name']
        local_outputdir  = self.local_cfg ['outputdir']  
        remote_outputdir = self.server_cfg['outputdir']      

        setup_names = [f"{experiment_name}={arg}" for arg in setup_args] #  "mld_experiment1=c_k=0.1", "mld_experiment1-c_k-0.2",

        hostname = self.server_cfg['hostname']
        remote_dir = f"{remote_outputdir}/ocean/veropt_results/{experiment_name}/"

        for setup in setup_names:
            setup_dir  = f"{local_outputdir}/{experiment_name}/{setup}"

            print(f"Attempting: scp -Cr {setup_dir} {hostname}:{remote_dir}/")
            # TODO: Use python fabric module for error handling
            pipe = subprocess.Popen(["scp","-Cr", setup_dir, f"{hostname}:{remote_dir}/"],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True)
            result = pipe.stdout.read()
            stderr = pipe.stderr.read()
            if not stderr:
                print(f"Transferred {setup_dir} to {remote_dir} with result: {result}")
            else:
                print(f"Error transferring {setup_dir} to {remote_dir}")
                print(f"Error output was:\n{stderr}")
                print(f"Aborting.\n\n")
                sys.exit(4)

    def transfer_results(self, setup_args) -> None:
        experiment_name  = self.expt_cfg  ['experiment_name']
        local_outputdir  = self.local_cfg ['outputdir']  
        remote_outputdir = self.server_cfg['outputdir']      

        setup_names = [f"{experiment_name}={arg}" for arg in setup_args]
        hostname    = self.server_cfg['hostname']
        remote_dir  = f"{remote_outputdir}/ocean/veropt_results/{experiment_name}/"

        for setup in setup_names:
            setup_dir = f"{local_outputdir}/{experiment_name}/{setup}"

            print(f"Attempting: scp -Cr {hostname}:{remote_dir}/* {setup_dir}/")
            pipe = subprocess.Popen(["scp","-Cr", f"{hostname}:{remote_dir}/\*", f"{setup_dir}/"],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True)
            result = pipe.stdout.read()
            stderr = pipe.stderr.read()

            if not stderr:
                print(f"Transferred {remote_dir}/* to {setup_dir} with result: {result}")
            else:
                print(f"Error transferring {remote_dir}/* to {setup_dir}")
                print(f"Error output was:\n{stderr}")
                print(f"Aborting.\n\n")
                sys.exit(5)
        

    def start_jobs(self, setup_args) -> None:
        experiment_name = self.expt_cfg['experiment_name']
        remote_outputdir = self.server_cfg['outputdir']

        setup_names = [f"{experiment_name}={arg}" for arg in setup_args] #  "mld_experiment1-c_k-0.1", "mld_experiment1-c_k-0.2",
        hostname = self.server_cfg['hostname']

        jobids = []
        for setup in setup_names:
            remote_dir = f"{remote_outputdir}/ocean/veropt_results/{experiment_name}/{setup}/"
            command    = f"cd {remote_dir} && sbatch --parsable run_veros.slurm"

            print(f"\nSubmitting job {remote_dir}/run_veros.slurm")            

            print(f"ssh {hostname} '{command}'")

            pipe = subprocess.Popen(["ssh", hostname, command],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True)
            stdout = pipe.stdout.read()
            stderr = pipe.stderr.read()

            if not stderr:
                print(f"Got {stdout} after submitting setup {setup}")
                print(f"Output was:\n{stdout}")
                jobids += [int(l) for l in stdout.split('\n') if l.isdigit()]
            else:
                print(f"Submission of {setup} failed.")
                print(f"Error output was:\n{stderr}")
                print(f"Aborting.\n\n")
                sys.exit(5)
        
        return jobids


    def check_jobs_status(self, setup_args, jobids) -> None:
        """JobId=43848356
           JobName=mld_experiment1
           JobState=RUNNING/PENDING
        """
        experiment_name  = self.expt_cfg  ['experiment_name']
        local_outputdir  = self.local_cfg ['outputdir']                
        remote_outputdir = self.server_cfg['outputdir']
        hostname         = self.server_cfg['hostname']

        setup_names = [f"{experiment_name}={arg}" for arg in setup_args] #  "mld_experiment1-c_k-0.1", "mld_experiment1-c_k-0.2",

        # TODO: 
        #  - For each jobid in jobids
        #  -- Check if the job is: queued, running, completing or completed
        #  -- If running, print the progress from slurm-{jobid}.out
        #  -- If completed, remove the corresponding setup from submitted_jobs

        # TODO: 
        #  try/except: On error, cancel all jobs and clean up.

        pending_jobs = 0
        for i in range(len(jobids)):
            pending_jobs |= (1 << i)

        while pending_jobs:
            for i in range(len(jobids)):
                jobid, setup = jobids[i], setup_names[i]

                remote_dir = f"{remote_outputdir}/ocean/veropt_results/{experiment_name}/{setup}/"
                log_file   = f"{remote_dir}/slurm-{jobid}.out"               



                pipe = subprocess.Popen(["ssh", hostname, f"scontrol show job {jobid}"],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True)
                stdout = pipe.stdout.read()
                stderr = pipe.stderr.read()

                if stdout:
                    job_dict = {}
                    for item in stdout.split():
                        key, *value = item.split('=')
                        job_dict[key] = value[0] if value else None  # Using None or '' as the default value

                    print("Job {jd[JobId]}/{jd[JobName]} status: {jd[JobState]} (Reason: {jd[Reason]}).".format(jd=job_dict))

                if job_dict['JobState'] == "COMPLETED" or "slurm_load_jobs error: Invalid job id specified" in stderr:
                    print(f"{jobid} status: COMPLETED")
                    pending_jobs &= ~(1 << i)

                if job_dict['JobState'] == "RUNNING":
                    pipe = subprocess.Popen(["ssh", hostname, f"tail -n 3 {log_file}"],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            text=True)
                    stdout = pipe.stdout.read()
                    stderr = pipe.stderr.read()
                    print(f"Progress of {jobid}:\n----------------------------------\n{stdout}\n----------------------------------\n")

            if pending_jobs:
                print("\nThe following jobs are still pending or running: " )
                for i in range(len(jobids)):
                    if pending_jobs & (1 << i):
                        print(f"{jobids[i]}")

                remote_poll_delay = 3*60; # should be in server/experiment config
                for i in tqdm.tqdm(range(remote_poll_delay), "Time until next server poll"):
                    time.sleep(1)
    


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
        
        template_substitutions = self.server_cfg | self.expt_cfg | {"setup_name": setup_name}
        #print(f"template_substitutions: {template_substitutions}")

        return batch_script_str.format(**template_substitutions)
