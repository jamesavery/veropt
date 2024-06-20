import os
import sys
import json
import stat
import time
import shutil
import subprocess
from dataclasses import dataclass, field
import xarray as xr
import numpy as np
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
    def __init__(self, param_names, param_list_filename, n_params, bounds_lower, bounds_upper,
                 rootdir="/groups/ocean/mmroz/mld_experiments_aegir",
                 experiment="mld_experiment1", float_type="float32", 
                 ncores=16, ncores_nx=4, ncores_ny=4, ncycles=6, res="4deg", lat_range=(-23,23),
                 target="observations", target_filename="IFREMER_mld_rho_tropics"):

        # location, name, latitude range and resolution of the target mixed layer depth (MLD) map
        self.target = target
        self.target_filename = target_filename 
        self.lat_range = lat_range
        self.res = res

        # optimiser setup
        self.param_list_filename: str = param_list_filename
        self.param_names: str = param_names
        bounds = [bounds_lower, bounds_upper]
        n_objs = 1
        init_vals = None
        stds = None

        # slurm setup
        self.sleeping_time: int = 10 # in seconds
        self.experiment: str = experiment
        self.rootdir: str = rootdir
        self.source_file_path: str = f"{self.rootdir}/{self.experiment}/{self.experiment}.py"
        self.assets_file_path: str = f"{self.rootdir}/{self.experiment}/assets.json"
        self.n_cores: int = ncores
        self.n_cores_nx: int = ncores_nx
        self.n_cores_ny: int = ncores_ny
        self.partition_name: str = "aegir"
        self.constraint: str = "v1"
        self.ncycles: int = ncycles
        self.cycle_length: int = 31104000 # - year  # 2592000 - month
        self.backend: str = "jax"
        self.float_type: str = float_type
        self.startup_jobs: dict = field(default_factory=dict)

        super().__init__(self.mldObjFunc, bounds, n_params, n_objs, init_vals, stds)


    def mldObjFunc(self, new_x):

        n_points = len(new_x.numpy()[0])
        n_params = len(new_x.numpy()[0][0])
        setup_args = ()
        y = []
        param_val_strings = []

        target_filepath = f"{self.rootdir}/target_dataset"
        target_dataset = xr.open_dataset(f"{target_filepath}/{self.target_filename}.nc")

        for i in range(n_points):
            # in new_x.numpy()[i][j][k]
            # index i: objective index (as default, n_obj = 1)
            # index j: index of point within opt step
            # index k: param index
            param_val_string = ""
            setup_string = ""
            for j in range(n_params):
                param_name, param_val = self.param_names[j], new_x.numpy()[0][i][j]
                param_val_string += f"{param_val},"
                setup_string += f"{param_name}={param_val}="
            setup_args += (setup_string[:-1],)
            param_val_strings.append(param_val_string)

        print(setup_args)
        self.setup_runs(setup_args)
        self.start_jobs(setup_args)
        for i in range(self.ncycles):
            self.check_jobs_status(setup_args)

        setup_names = [f"{self.experiment}={arg}" for arg in setup_args]

        for setup, param_val_string in zip(setup_names, param_val_strings):
            optimized_filepath = f"{self.rootdir}/{self.experiment}/{setup}"
            optimized_filename = f"/{setup}.{str(self.ncycles-1).zfill(4)}.averages"

            if self.target == "observations":
                correct_coords(f"{optimized_filepath}/{optimized_filename}", self.res)
                optimized_dataset = xr.open_dataset(f"{optimized_filepath}/{optimized_filename}_corr_coords.nc")
            else: 
                optimized_dataset = xr.open_dataset(f"{optimized_filepath}/{optimized_filename}.nc")

            new_y = calc_y(optimized_dataset, target_dataset, lat_range=self.lat_range)
            y.append(new_y)

            with open(self.param_list_filename, "a") as file:
                file.write(f"\n{param_val_string},{new_y}")

        if len(y) == 0:
            y = y[0]
        else: pass

        return y

    
    def setup_runs(self, setup_args) -> None:
        setup_names = [f"{self.experiment}={arg}" for arg in setup_args] #  "mld_experiment1-c_k-0.1", "mld_experiment1-c_k-0.2",

        for setup in setup_names:
            setup_dir = f"{self.rootdir}/{self.experiment}/{setup}"

            if not os.path.exists(setup_dir):
                os.makedirs(setup_dir)
                print(f"\nDirectory created: {setup_dir}")
            else:
                print(f"\nDirectory exists: {setup_dir}")

            if not os.path.isfile(f"{setup_dir}/{setup}.py"):
                shutil.copy(self.source_file_path, f"{setup_dir}/{setup}.py")
                print(f"    File {setup}.py was copied to: {setup_dir}")
            else:
                print(f"    File exists: {setup_dir}/{setup}.py")

            if not os.path.isfile(f"{setup_dir}/assets.json"):
                shutil.copy(self.assets_file_path, f"{setup_dir}/assets.json")
                print(f"    File assets.json was copied to: {setup_dir}")
            else:
                print(f"    File exists: {setup_dir}/assets.json")

            batch_script_str = self.make_batch_script(setup)
            _write_batch_script(f"{setup_dir}/veros_batch.sh", batch_script_str)


    def start_jobs(self, setup_args) -> None:
        #self.startup_jobs = {} # remove

        setup_names = [f"{self.experiment}={arg}" for arg in setup_args] #  "mld_experiment1-c_k-0.1", "mld_experiment1-c_k-0.2",

        for setup in setup_names:
            os.chdir(f"{self.rootdir}/{self.experiment}/{setup}")
            print(f"\nSubmitting job {self.rootdir}/{self.experiment}/{setup}/veros_batch.sh")

            pipe = subprocess.Popen(["sbatch", "--parsable", "veros_batch.sh"],
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

        setup_names = [f"{self.experiment}={arg}" for arg in setup_args] #  "mld_experiment1-c_k-0.1", "mld_experiment1-c_k-0.2",
        submitted_jobs = setup_names.copy()

        while submitted_jobs:

            for setup in submitted_jobs:

                log_file = f"{self.rootdir}/{self.experiment}/{setup}/{setup}.out"
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
        dummy = setup_name.split("=")[1:]
        setup_args_dict = {dummy[i]: float(dummy[i + 1]) for i in range(0, len(dummy), 2)}
        with open(f"{self.rootdir}/{self.experiment}/{setup_name}/setup_args.txt", "w") as f:
            json.dump(setup_args_dict, f)

        return f"""#!/bin/bash -l
#SBATCH -p {self.partition_name}
#SBATCH -A ocean
#SBATCH --job-name={setup_name}
#SBATCH --time=23:59:59
#SBATCH --constraint={self.constraint}
#SBATCH --nodes=1
#SBATCH --ntasks={self.n_cores}
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
##SBATCH --threads-per-core=1
##SBATCH --exclusive
#SBATCH --output={setup_name}.out

if [ X"$SLURM_STEP_ID" = "X" -a X"$SLURM_PROCID" = "X"0 ]
then
    echo "SLURM_JOB_ID = $SLURM_JOB_ID"
fi

export OMP_NUM_THREADS=2

command="ml list"
search_string="miniconda/python3.11"

if $command | grep -q "$search_string"; then
    echo "Miniconda has already been loaded"
else
    echo "miniconda has not been loaded yet; loading --->"
    ml load miniconda/python3.11
fi

#conda init bash
#source $HOME/.bashrc
if [ "$CONDA_DEFAULT_ENV" = "veropt" ]; then
    conda deactivate
fi

conda activate veros_jax_cpu

veros resubmit -i {setup_name} -n {self.ncycles} -l {self.cycle_length} \
-c 'mpiexec -n {self.n_cores} -- veros run {setup_name}.py -b {self.backend} -n {self.n_cores_nx} {self.n_cores_ny} --float-type {self.float_type}' \
--callback 'sbatch veros_batch.sh'
"""
