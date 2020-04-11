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

startJob 8 8 3 dtect83
startJob 16 16 3 dtect163
startJob 16 16 4 dtect164
startJob 32 32 3 dtect323
startJob 32 32 4 dtect324
startJob 32 32 5 dtect325
startJob 64 64 3 dtect643
startJob 64 64 4 dtect644
startJob 64 64 5 dtect645
startJob 64 64 6 dtect646

# show dependencies in squeue output:
./check_status.sh
