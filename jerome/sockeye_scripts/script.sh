#!/bin/bash

#SBATCH --account=st-sielmann-1-gpu
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=4
#SBATCH --job-name=blueberries
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jeromejjcho@gmail.com
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --time=8:00:00

module load gcc python miniconda3 cuda cudnn
source ~/.bashrc
londa activate bluberries

cd $SLURM_SUBMIT_DIR


