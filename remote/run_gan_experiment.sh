#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=23:59:59
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB


# Notification configuration
#SBATCH --mail-user=avraham.chapman@adelaide.edu.au  	# Email to which notifications will be sent

#loading modules
module load CUDA/9.0.176
module load Python/3.6.1-foss-2016b
source ../../virtualenvs/pytorch/bin/activate

echo Starting GAN Experiment ${NAME} in ${OUTPUT_DIR}...
python -u ../scripts/train.py --dataroot ../part_A_train.json --name ${NAME} --model ${MODEL}