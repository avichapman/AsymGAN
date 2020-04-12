#!/bin/bash

IFS=

expected_job_names=(gan_cycle gan_asym)

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

printf "      Name Epoch Status  Jobs"
echo 
for name in "${expected_job_names[@]}"
do
	cd ${name}

	# Read epochs...
	readarray EPOCHS < <(grep -r "^epoch " | cut -d ' ' -f 2)

	MIN_EPOCH=0
	for EPOCH in "${EPOCHS[@]}"
	do
		EPOCH="$(echo -e "${EPOCH}" | tr -d '[:space:]')"
		if (( $(echo "$EPOCH > $MIN_EPOCH" | bc -l) )); then
			MIN_EPOCH="$EPOCH"
		fi
	done
	
	cd ..

	is_waiting=true
	result=$(squeue -u ${USER} | grep ${name})
	if [ ${#result} -eq 0 ]
	then
		is_waiting=false
		printf "%10s %8s %8s %5s ${RED}MISSING${NC} " ${name} ${MIN_EPOCH}
	else
		statuses=$(echo $result | awk '{print $5}')
		while read -r status; do
			if [ ${status} == "R" ]
			then
				is_waiting=false
				printf "%10s %8s %8s %5s ${GREEN}RUNNING${NC} " ${name} ${MIN_EPOCH}
			fi
		done <<< "$statuses"
	fi
	
	if [ "$is_waiting" = true ]
	then
		printf "%10s %8s %8s %5s WAITING " ${name} ${MIN_EPOCH}
	fi
	
	printf "$result" | tr " " "\n" | grep -c "${name}"
done
