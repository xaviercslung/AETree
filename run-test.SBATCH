#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=200:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu
#SBATCH --job-name=TrainAETree_64boxes

module purge

## python <weight_leaf,weight_type,learning rate,test_number,data_file_name,number of boxes>
## python -u train_ae_tree.py 0.01 2 0.05 01 AETree_64.pickle 64
## current version of the model only takes LR > 1e-6
singularity exec --nv \
	    --overlay /home/sc8635/pytorch1.8.0-cuda11.1.ext3:ro \
	    /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c \
	    "source /ext3/env.sh; python -u train_ae_tree.py 0.05 2 1e-6 21 AETree_64.pickle 64"



