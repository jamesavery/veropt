#!/bin/bash
#SBATCH --job-name={experiment_name}_P{point_id}
#SBATCH -p {partition}
#SBATCH -A {account}
#SBATCH --time={max_time}
#SBATCH --nodes=1              # Total number of nodes 
#SBATCH --ntasks={n_cores}     # Number of MPI-tasks per job
#SBATCH --gpus-per-task=1      # VerOS supports only 1 GPU per MPI task
#SBATCH --cpus-per-task=1

module load LUMI/23.09 partition/G
module load craype-accel-amd-gfx90a
module load rocm

export VEROS_SIF_IMAGE=/projappl/{account}/ocean/leap_15.4_rocm-5.3_veros.sif
export MPICH_GPU_SUPPORT_ENABLED=1
export MPI4JAX_USE_HIP_MPI=1
export CPU_BIND="map_cpu:x48,56,16,24,1,8,32,40"

# Move run-directory to scratch space
#export SCRATCH_DIR=/dev/shm/veropt/{experiment_name}/point={point_id}
# /dev/shm is failing for some reason right not, fake it on disk until I have time to debug
#export SCRATCH_DIR=/scratch/{account}/scratch/{experiment_name}/point={point_id}
#export RESULT_DIR=$PWD

#echo "Copying job files for point {point_id} to scratch memory file system $SCRATCH_DIR"
#mkdir -p $SCRATCH_DIR
#cp -r ./* $SCRATCH_DIR
#rm $SCRATCH_DIR/slurm-$SLURM_JOB_ID.out
#cd $SCRATCH_DIR

cat << EOF > ./select_gpu
#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=0
export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
exec \$*
EOF
chmod +x ./select_gpu

echo $PWD
ls -lSrh 

echo "Running VerOS on {experiment_name} point {point_id}"
singularity exec -B /scratch/{account}/ $VEROS_SIF_IMAGE ./select_gpu veros run --backend {backend} --device {device_type} --float-type {float_type} ./experiment.py

echo "Contents of scratch directory before copying back:"
echo $PWD
ls -lSrh 

#echo "Copying back results for point {point_id} from scratch space and cleaning up"
#cp -r $SCRATCH_DIR/* $RESULT_DIR/
#rm -rf $SCRATCH_DIR
#cd $RESULT_DIR

echo "Job $SLURM_JOB_ID ($SLURM_JOB_NAME) finished"

