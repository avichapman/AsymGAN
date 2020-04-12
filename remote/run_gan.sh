#!/bin/bash

startJob() {
	export NAME=$1
	export MODEL=$2
	export NET_DIRNAME=$3
	
	mkdir -p $NET_DIRNAME
	cd $NET_DIRNAME
	export OUTPUT_DIR=$PWD
	
	echo Starting GAN Experiment ${NAME} in ${OUTPUT_DIR}...
	
	command="sbatch -J $NET_DIRNAME ../run_gan_experiment.sh"
	result=$($command)
	jobId=$(echo $result | awk '{print $4}')
	echo Job Id = $jobId
	
	command="sbatch -J $NET_DIRNAME --dependency=afterany:$jobId ../run_gan_experiment.sh"
	result=$($command)
	jobId=$(echo $result | awk '{print $4}')
	echo Job Id = $jobId
	
	command="sbatch -J $NET_DIRNAME --dependency=afterany:$jobId ../run_gan_experiment.sh"
	result=$($command)
	jobId=$(echo $result | awk '{print $4}')
	echo Job Id = $jobId
	
	command="sbatch -J $NET_DIRNAME --dependency=afterany:$jobId ../run_gan_experiment.sh"
	result=$($command)
	jobId=$(echo $result | awk '{print $4}')
	echo Job Id = $jobId
	
	command="sbatch -J $NET_DIRNAME --dependency=afterany:$jobId ../run_gan_experiment.sh"
	result=$($command)
	jobId=$(echo $result | awk '{print $4}')
	echo Job Id = $jobId
	
	command="sbatch -J $NET_DIRNAME --dependency=afterany:$jobId ../run_gan_experiment.sh"
	result=$($command)
	jobId=$(echo $result | awk '{print $4}')
	echo Job Id = $jobId
	
	command="sbatch -J $NET_DIRNAME --dependency=afterany:$jobId ../run_gan_experiment.sh"
	result=$($command)
	jobId=$(echo $result | awk '{print $4}')
	echo Job Id = $jobId
	
	cd ..
}

startJob CycleGAN cycle_gan gan_cycle
startJob AsymGAN asym_gan gan_asym

# show dependencies in squeue output:
./check_status.sh
