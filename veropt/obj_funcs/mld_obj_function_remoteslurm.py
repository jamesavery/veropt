import os, sys, stat, time, tqdm, shutil, subprocess
from dataclasses import dataclass, field
import xarray as xr, numpy as np, pyjson5 as json
from veropt import ObjFunction

# TODO: Veropt change: we should be minimizing instead of maximizing, since that's the canonical thing to do.
# DONE: identifier should be same as experiment.json's experiment_name. Fix so that it is read from experiment.json and written to setup_args.txt
# TODO: Return partially completed set of points (opt_cfg['required_completed_fraction'] = 0.5) instead of waiting for slowest
# TODO: Make and automatically transfer scripts to process all remote points in one go (to reduce network overhead)
# TODO: Maybe even process results on the server, and only copy back all files if in the top N? (Specify what to retain if not in top N)
# TODO: Also generate plots on the fly for debugging and monitoring. Plots per point, and optimization progress plots.
# TODO: Sleep time and slurm-maxtime proportional to resolution and simulation time. 
# TODO: Make checkpointing robust
#        - Check what Ida already has
#        - Replay old points in state-file
#        - Deal with partially or fully completed but unprocessed points (from prior crash)
#        - DONE Return both x and y (allowing points to be returned in different order than requested). Think about asyncronous scheme.
#        - Keep track of best N points. Whenever a best-point is pushed out of the list, extract "keep for all" data and delete rest.

def write_json(filename, data):
    with open(filename,"wb") as f:
        json.encode_io(data,f)

def read_json(filename):
    with open(filename,"rb") as f:
        return json.decode_io(f)

def write_batch_script(file_path_name: str, string: str) -> None:
    with open(file_path_name, 'w+') as file:
        file.write(string)

    st = os.stat(file_path_name)
    os.chmod(file_path_name, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH )

def point_xs(point_id, param_list, expt_state):
    point_params = expt_state['points'][point_id]['params']
    return [point_params[param] for param in param_list]

def correct_coords(optimized_filename, res):
    if os.path.exists(f"{optimized_filename}_corr_coords.nc"):
        print(f"correct_coords called on already coord-corrected: {optimized_filename}.nc")
        return
    
    ds = xr.open_dataset(f"{optimized_filename}.nc")
    if   res == "4deg":
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
    os.remove(f"{optimized_filename}.nc")


def calc_y(optimized_dataset, target_dataset, lat_range):  
    lat_min, lat_max = lat_range  
    mld = optimized_dataset.mld.isel(Time=-1).sel(yt=slice(lat_min, lat_max)).values
    mld = np.where(mld == 0., -1., mld)

    target_mld = target_dataset.mld.values
    target_mld = np.where(target_mld == 0., -1., target_mld)
    
    MSE = np.nanmean(np.square((mld-target_mld)/target_mld))

    return -np.sqrt(MSE)


class MLD1ObjFun(ObjFunction):
    def __init__(self, expt_state,
                 expt_cfg, server_cfg, local_cfg, opt_cfg): 

        self.expt_state = expt_state
        self.expt_cfg   = expt_cfg
        self.server_cfg = server_cfg
        self.local_cfg  = local_cfg
        self.opt_cfg    = opt_cfg
        
        # location, name, latitude range and resolution of the target mixed layer depth (MLD) map
        #target_type: "simulated_target" | "ifremer"
        #experiment_type: "mixed_layer_depth" | "..."
        local_outputdir = local_cfg['outputdir']
        local_datadir   = local_cfg['datadir']
        target_type     = expt_cfg['target_type']
        experiment_type = expt_cfg['experiment_type']
        experiment_name = expt_cfg['experiment_name']

        self.target_directory = f"{local_datadir}/{target_type}/{experiment_type}"
        self.target_dataset   = xr.open_dataset(f"{self.target_directory}/{expt_cfg['target_filename']}.nc")
        self.state_filename   = f"{local_outputdir}/{experiment_name}/experiment_state.json"        
        self.source_file_path = f"{local_outputdir}/{experiment_name}/experiment.py"
        self.assets_file_path = f"{local_outputdir}/{experiment_name}/assets.json"

        # optimiser setup
        self.param_list_filename = f'{local_outputdir}/{experiment_name}/param_list_{experiment_name}.txt'
        param_names = expt_cfg['param_names']
        param_str   = ",".join(param_names)

        with open(self.param_list_filename, "w") as file:
            file.write(f"{param_str}obj_func")        

        bounds = [expt_cfg['bounds_lower'], expt_cfg['bounds_upper']]
        n_objs = expt_cfg['n_objectives']
        
        # TODO: Potentially not none - should live in... experiment.json 
        # TODO: Ask Marta what exactly these do again.
        init_vals = None # Are these the already evaluated points? In that case, we should get these from expt_state['points'].
        stds = None      # Where do we obtain the standard deviations from?

        self.sleeping_time: int = 10 # in seconds. TODO: Move to config (server or experiment?)
        year_in_seconds = 24*60*60*360 # TODO: Can we get veros to do 365 day years? 
        if(self.expt_cfg['map_resolution'] == "1deg"):
            self.remote_poll_delay = int(1200*self.expt_cfg["simulation_time"]/year_in_seconds); # should be in server/experiment config
        else: # TODO: 4deg
            self.remote_poll_delay = int(  12*self.expt_cfg["simulation_time"]/year_in_seconds)  # should be in server/experiment config            

        super().__init__(self.mldObjFunc, bounds, len(param_names), n_objs, init_vals, stds)

    def mldObjFunc(self, new_x):

        print(f"Running objective function with new_x: {new_x}")
        n_cycles = self.server_cfg['n_cycles']
        
        n_evals_per_step   = self.opt_cfg['n_evals_per_step']
        max_parallel_evals = self.server_cfg['max_parallel_evals']
        n_parallel_evals   = min(n_evals_per_step, max_parallel_evals)        

        n_points = len(new_x.numpy()[0])
        n_params = len(new_x.numpy()[0][0])
        setup_args = ()
        param_names = self.expt_cfg['param_names']

        if(n_points != n_evals_per_step):
            print(f"n_points={n_points} != n_evals_per_step={n_evals_per_step}.")
            print(f"new_x = {new_x}")

        # TODO: Make a function param_val_string from dict to string, and param_string_val from string to dict instead.
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
            # param_val_strings.append(param_val_string)

        # TODO: Deal with partially or fully completed but unprocessed points (from prior crash)

        job_ids = []
        step_points = self.setup_runs(setup_args)  # Setup runs for new points. TODO: setup_args from new_x, get rid of setup_args here.
        for i in range(0, n_points, n_parallel_evals):
            parallel_points = step_points[i*n_parallel_evals:(i+1)*n_parallel_evals]
            self.transfer_jobs(parallel_points)          # Transfer new jobs (state=='setup_created') to remote server. TODO: Use expt_state['points'] instead of step_points
            parallel_job_ids = self.start_jobs(parallel_points)   # Start new slurm jobs (state=='job_transferred') on remote server. TODO: Use expt_state['points'] instead of step_points
        
            print(f"Finished starting jobs, job ids: {dict(zip(parallel_points,parallel_job_ids))}") # TODO: Use expt_state['points'] instead of step_points

            #for i in range(n_cycles): # TODO: Figure out how to handle n_cycles better.
            ready_points = self.check_jobs_status(parallel_job_ids)
        
            self.transfer_results(parallel_points)
            job_ids += parallel_job_ids

        print(f"ready_points = {ready_points}")
        completed_x = np.array([[point_xs(p, self.expt_cfg['param_names'], self.expt_state)] for p in ready_points])
        completed_y = np.array(self.process_results(ready_points)) # Process results from prior cycles in expt_state

        print(f"completed_x = {completed_x} ")
        print(f"completed_y = {completed_y} ")
        not_nan = ~np.isnan(completed_y)
        print(f"not_nan = {not_nan} ")

        if(len(not_nan) == 0):
            return [[]], [[]]
        else: 
            new_x = completed_x[not_nan]
            new_y = completed_y[not_nan]
            print(f"Returning {new_x}, {new_y}")
            return new_x, new_y
        

    def process_results(self, ready_points):
        # Extract completed and back-transferred results from expt_state
        local_outputdir = self.local_cfg['outputdir']
        experiment_name = self.expt_cfg ['experiment_name']

        points = self.expt_state['points']
        n_objs           = self.expt_cfg['n_objectives']
        n_ready_points   = len(ready_points)
        y = np.zeros((n_ready_points,n_objs))
        
        for i in range(n_ready_points): 
            point_id = ready_points[i]

            optimized_filepath = f"{local_outputdir}/{experiment_name}/point={point_id}/"
            optimized_filename = f"{self.expt_cfg['experiment_name']}.averages"
            #optimized_filename = f"/{setup}.{str(n_cycles-1).zfill(4)}.averages"

            target_type = self.expt_cfg['target_type']
            try: 
                if target_type == "ifremer":
                    correct_coords(f"{optimized_filepath}/{optimized_filename}", self.expt_cfg['map_resolution'])
                    optimized_dataset = xr.open_dataset(f"{optimized_filepath}/{optimized_filename}_corr_coords.nc")                
                elif target_type == "simulated_target": 
                    optimized_dataset = xr.open_dataset(f"{optimized_filepath}/{optimized_filename}.nc")
                else:
                    raise ValueError("Target type should be either 'ifremer' or 'simulated_target'.")
            except FileNotFoundError as e:
                print(f"VerOS point {point_id}: File {optimized_filename}.nc not found: {e}.")
                y[i] = np.nan

                continue

            # Check if calculation completed or faile
            simulation_time = optimized_dataset.Time
            
            if((len(simulation_time) == 0) or (abs(float(simulation_time[-1])/1e9 - self.expt_cfg['simulation_time']) > 86400)):
                print(f"VerOS point {point_id} failed before completion: {simulation_time}")
                y[i] = np.nan
            else:
                y[i] = calc_y(optimized_dataset, self.target_dataset, lat_range=self.expt_cfg['lat_range'])
                self.expt_state['points'][point_id]['value'] = y[i]

            points[point_id]['state'] = 'result_processed' #TODO: completed?
#            with open(self.param_list_filename, "a") as file:
#                file.write(f"\n{param_val_string},{new_y}")

        write_json(self.state_filename,self.expt_state)

        if len(y) == 1:
            return y[0]
        else:
            return y

    def setup_runs(self, setup_args) -> None:
        experiment_name = self.expt_cfg['experiment_name']
        local_outputdir = self.local_cfg['outputdir']     

        setup_names = [f"{experiment_name}={arg}" for arg in setup_args] #  "mld_experiment1=c_k=0.1", "mld_experiment1-c_k-0.2",

        step_points = []
        for i, setup_name in enumerate(setup_names):
            next_point = self.expt_state['next_point']

            setup_dir = f"{local_outputdir}/{experiment_name}/point={next_point}"
            os.makedirs(setup_dir, exist_ok=True)

            # Write out parameters and expt setup as JSON to setup_args.txt
            kv = setup_name.split("=")[1:] # TODO: '_' splits kv-pairs, '=' splits k and v
            params_dict = {kv[i]: float(kv[i + 1]) for i in range(0, len(kv), 2)}
            setup_args_dict = params_dict | self.expt_cfg
            write_json(f"{setup_dir}/setup_args.txt", setup_args_dict)

            shutil.copy(self.source_file_path, f"{setup_dir}/experiment.py")
            shutil.copy(self.assets_file_path, f"{setup_dir}/assets.json")

            batch_script_str = self.make_batch_script(setup_name, next_point)
            write_batch_script(f"{setup_dir}/run_veros.slurm", batch_script_str)

            # Assign parameters to the point number in the experiment state.
            # This allows easy reproduction of results, rerunning failed point runs, plotting, etc.
            self.expt_state['points'][next_point] = {'params':params_dict, 'state':'setup_created'}
            self.expt_state['next_point'] += 1        
            step_points += [next_point]
                
        write_json(self.state_filename,self.expt_state)
        
        return step_points

    def transfer_jobs(self, step_points) -> None:
        experiment_name  = self.expt_cfg  ['experiment_name']
        local_outputdir  = self.local_cfg ['outputdir']  
        remote_outputdir = self.server_cfg['outputdir']      

        hostname = self.server_cfg['hostname']
        remote_dir = f"{remote_outputdir}/ocean/veropt_results/{experiment_name}/"

        for point_id in step_points:
            setup_dir  = f"{local_outputdir}/{experiment_name}/point={point_id}"

            success = False
            tries = 0
            max_tries = 3 # TODO: Move to config
            while(not success and tries < max_tries):
                print(f"Attempting: scp -Cr {setup_dir} {hostname}:{remote_dir}/")
                # TODO: Use python fabric module for error handling
                pipe = subprocess.Popen(["scp","-Cr", setup_dir, f"{hostname}:{remote_dir}/"],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True)
                result = pipe.stdout.read()
                stderr = pipe.stderr.read()
                if not stderr:
                    print(f"Transferred {setup_dir} to {remote_dir} with result: {result}")
                    self.expt_state['points'][point_id]['state'] = 'job_transferred'
                    success = True
                else:
                    print(f"Error transferring {setup_dir} to {remote_dir}")
                    print(f"Error output was:\n{stderr}")
                    tries += 1
                    print(f"Retrying in {tries*60} seconds.")
                    time.sleep(tries*60) # TODO: Move to config
                    #print(f"Aborting.\n\n")
                    #sys.exit(5)

        write_json(self.state_filename,self.expt_state)

    def transfer_results(self, step_points) -> None:
        experiment_name  = self.expt_cfg  ['experiment_name']
        local_outputdir  = self.local_cfg ['outputdir']  
        remote_outputdir = self.server_cfg['outputdir']      

        hostname    = self.server_cfg['hostname']        

        for point_id in step_points:
            local_dir  = f"{local_outputdir}/{experiment_name}/point={point_id}"
            remote_dir = f"{remote_outputdir}/ocean/veropt_results/{experiment_name}/point={point_id}"            

#TODO: rsync instead of scp
            success = False
            tries = 0
            max_tries = 3 # TODO: Move to config
            while(not success and tries < max_tries):
                print(f"Attempting: scp -Cr {hostname}:{remote_dir}/* {local_dir}/")
                pipe = subprocess.Popen(["scp","-Cr", f"{hostname}:{remote_dir}/*", f"{local_dir}/"],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True)
                result = pipe.stdout.read()
                stderr = pipe.stderr.read()

                if not stderr:
                    print(f"Transferred {remote_dir}/* to {local_dir} with result: {result}")
                    self.expt_state['points'][point_id]['state'] = 'result_transferred'
                    success = True
                # Don't delete the whole home directory by accident
                    assert(len(remote_dir) > len(self.server_cfg['outputdir'])+10) 
                    print(f"Attempting remote cleanup: ssh {hostname} 'rm -rf {remote_dir}'")
                    pipe = subprocess.Popen(["ssh",hostname, f"rm -rf {remote_dir}"],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True)
                    result = pipe.stdout.read()
                    stderr = pipe.stderr.read()

                    if not stderr:
                        print(f"Remote cleanup of {remote_dir} successful: {result}")
                    else:
                        print(f"Error cleaning up {remote_dir}")
                        print(f"Error output was:\n{stderr}")
                else:
                    print(f"Error transferring {remote_dir}/* to {remote_dir}")
                    print(f"Error output was:\n{stderr}")
                    tries += 1
                    print(f"Retrying in {tries*60} seconds.")
                    time.sleep(tries*60) # TODO: Move to config
                    #print(f"Aborting.\n\n")
                    #sys.exit(6)
        
        write_json(self.state_filename,self.expt_state)
        

    def start_jobs(self, step_points) -> None:
        experiment_name  = self.expt_cfg['experiment_name']
        remote_outputdir = self.server_cfg['outputdir']
        hostname         = self.server_cfg['hostname']

        job_ids = []
        for point_id in step_points:
            remote_dir = f"{remote_outputdir}/ocean/veropt_results/{experiment_name}/point={point_id}/"
            command    = f"cd {remote_dir} && sbatch --parsable run_veros.slurm"

            print(f"\nSubmitting job {remote_dir}/run_veros.slurm")            

            print(f"ssh {hostname} '{command}'")

            success = False
            tries = 0
            max_tries = 3 # TODO: Move to config
            while(not success and tries < max_tries):
                pipe = subprocess.Popen(["ssh", hostname, command],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True)
                stdout = pipe.stdout.read()
                stderr = pipe.stderr.read()

                if not stderr and stdout.strip().isdigit():
                    print(f"Point {point_id} submitted successfully. Slurm Job ID: {stdout.strip()}")
                    job_id = int(stdout.strip())
                    self.expt_state['points'][point_id]['job_id'] = job_id
                    self.expt_state['points'][point_id]['state']  = "submitted"
                    job_ids += [job_id]
                    success = True
                else:
                    print(f"Submission of point={point_id} failed.")
                    print(f"Error output was:\n{stderr}")
                    tries += 1
                    print(f"Retrying in {tries*60} seconds.")
                    time.sleep(tries*60) # TODO: Move to config
                    #print(f"Aborting.\n\n")
                    #sys.exit(7)

        write_json(self.state_filename,self.expt_state)

        return job_ids


    def check_jobs_status(self, job_ids) -> None:
        """JobId=43848356
           JobName=mld_experiment1
           JobState=RUNNING/PENDING
        """
        experiment_name  = self.expt_cfg  ['experiment_name']
        remote_outputdir = self.server_cfg['outputdir']
        hostname         = self.server_cfg['hostname']

        # TODO: 
        #  try/except: On error, cancel all jobs and clean up.
        
        # Let's check the status even of orphaned submitted jobs (say, if we've crashed)
        pt_dict = self.expt_state['points']
        submitted_points = [p for p in pt_dict 
                            if pt_dict[p]['state'] == 'submitted' 
                            or pt_dict[p]['state'] == 'running']
        job_ids = [pt_dict[point_id]['job_id'] for point_id in submitted_points]

        pending_jobs = 0
        for i in range(len(job_ids)):
            pending_jobs |= (1 << i)

        

        while pending_jobs:
            ready_points = []
            for i in range(len(job_ids)):
                point_id, job_id = submitted_points[i], job_ids[i]

                remote_dir = f"{remote_outputdir}/ocean/veropt_results/{experiment_name}/point={point_id}/"
                log_file   = f"{remote_dir}/slurm-{job_id}.out"               

                pipe = subprocess.Popen(["ssh", hostname, f"scontrol show job {job_id}"],
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

                if job_dict['JobState'] == "COMPLETED" or job_dict['JobState'] == "COMPLETING" or "slurm_load_jobs error: Invalid job id specified" in stderr:
                    print(f"{job_id} status: COMPLETED")
                    pending_jobs &= ~(1 << i)
                    self.expt_state['points'][point_id]['state'] = 'job_completed'
                    ready_points += [point_id]

                if job_dict['JobState'] == "RUNNING":
                    pipe = subprocess.Popen(["ssh", hostname, f"tail -n 3 {log_file}"],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            text=True)
                    stdout = pipe.stdout.read()
                    stderr = pipe.stderr.read()
                    self.expt_state['points'][point_id]['state'] = 'job_running'
                    print(f"Progress of Point {point_id}/{job_id}:\n----------------------------------\n{stdout}\n----------------------------------\n")

                if stderr and "slurm_load_jobs error: Invalid job id specified" not in stderr:
                    print(f"Error checking job {job_id}: {stderr}")
                    print(f"Continuing in 60 seconds.")
                    time.sleep(60) # TODO: Move to config
                    continue

            if pending_jobs:
                print("\nThe following jobs are still pending or running: " )
                for i in range(len(job_ids)):
                    if pending_jobs & (1 << i):
                        print(f"Point {submitted_points[i]}, Slurm Job ID {job_ids[i]}")

                write_json(self.state_filename,self.expt_state)

                for i in tqdm.tqdm(range(self.remote_poll_delay), "Time until next server poll"):
                    time.sleep(1)

        return list(set(ready_points))
    

    # Just builds the batch script string, no unexpected side effects like writing to files.
    def make_batch_script(self, setup_name: str, point_id: int) -> str:
        local_sourcedir = self.local_cfg['source_dir']
        server_hostname = self.server_cfg['hostname']

        with open(f"{local_sourcedir}/servers/{server_hostname}/run_veros.slurm", "r") as f:
            batch_script_str = f.read()
        
        template_substitutions = self.server_cfg | self.expt_cfg | {"setup_name": setup_name, 'point_id': point_id}

        return batch_script_str.format(**template_substitutions)
