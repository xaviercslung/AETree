#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu
#SBATCH --job-name=TrainAETree

module purge

singularity exec --nv \
	    --overlay /home/sc8635/pytorch1.8.0-cuda11.1.ext3:ro \
	    /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python -u train_ae_tree.py"



