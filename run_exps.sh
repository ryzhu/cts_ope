#!/bin/bash
SBATCH --partition=next
SBATCH --time=06:00:00
SBATCH --nodes=10
SBATCH --ntasks-per-node=1
#SBATCH --mem=8192

# only use the following on partition with GPU
####SBATCH --gres=gpu:1

# Memory per node specification is in MB. It is optional.
# The default limit is 3000MB per core.
#SBATCH --job-name="sample"
#SBATCH --output=sample.out

# only use the following if you want email notification
####SBATCH --mail-user=youremailaddress
####SBATCH --mail-type=ALL

# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# sample job
NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
echo NPROCS=$NPROCS
echo "Done"
