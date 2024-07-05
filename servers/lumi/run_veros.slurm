#!/bin/bash
#SBATCH --job-name={setup_name}
#SBATCH -p {partition}
#SBATCH -A {account}
#SBATCH --time={max_time}
#SBATCH --nodes=1              # Total number of nodes 
#SBATCH --ntasks={n_cores}     # Number of MPI-tasks per job
#SBATCH --gpus-per-task=1      # VerOS supports only 1 GPU per MPI task

ml load LUMI/23.09 partition/G
ml load EasyBuild-user
ml load singularity-bindings/system-cpeGNU-22.08
ml load craype-accel-amd-gfx90a
ml load rocm

export VEROS_SIF_IMAGE=/projappl/{account}/ocean/leap_15.4_rocm-5.3_veros.sif
export MPICH_GPU_SUPPORT_ENABLED=1
export MPI4JAX_USE_HIP_MPI=1
export CPU_BIND="map_cpu:x48,56,16,24,1,8,32,40"

# Move run-directory to scratch space
SCRATCH_DIR=/dev/shm/veropt/{setup_name}
RESULT_DIR=$PWD

mkdir -p ${SCRATCH_DIR}
cp -r ./* ${SCRATCH_DIR}
cd ${SCRATCH_DIR}

cat << EOF > select_gpu
#!/bin/bash

##export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/hip/lib
export TF_CPP_MIN_LOG_LEVEL=0
export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
exec \$*
EOF
chmod +x ./select_gpu

srun singularity exec $VEROS_SIF_IMAGE ./select_gpu veros run --backend {backend} --device {device_type} --float-type {float_type} ./experiment.py

rm -rf ./select_gpu

# Copy back results from scratch space and clean up
cp -r ${SCRATCH_DIR}/* ${RESULT_DIR}/
rm -rf ${SCRATCH_DIR}
cd ${RESULT_DIR}

echo "Job $SLURM_JOB_ID ($SLURM_JOB_NAME) finished"

