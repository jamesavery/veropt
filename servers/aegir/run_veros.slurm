#!/bin/bash -l
#SBATCH --job-name={setup_name}
#SBATCH -p {partition}
#SBATCH -A {account}
#SBATCH --time={max_time} 
#SBATCH --nodes=1
#SBATCH --ntasks={n_cores}
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=0
#SBATCH --mem=0

if [ X"$SLURM_STEP_ID" = "X" -a X"$SLURM_PROCID" = "X"0 ]
then
    echo "SLURM_JOB_ID = $SLURM_JOB_ID"
fi

# JA: Why NUM_THREADS=2, given allocation of 1 CPU core per task?
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

veros resubmit -i {setup_name} -n {n_cycles} -l {cycle_length} \
-c 'mpiexec -n {n_cores} -- veros run {setup_name}.py \
-b {backend} -n {n_cores_nx} {n_cores_ny} \
--device {device_type} --float-type {float_type}' \
--callback 'sbatch run_veros.slurm'