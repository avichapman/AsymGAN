#!/bin/bash

IFS=

expected_job_names=(dtect83 dtect163 dtect164 dtect323 dtect324 dtect325 dtect643 dtect644 dtect645 dtect646)

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

printf "      Name Max Loss Min Loss Epoch Status  Jobs"
echo 
for name in "${expected_job_names[@]}"
do
	cd ${name}
	
	# Read Losses
	readarray LOSSES < <(grep -r " \* MAE " | cut -d '*' -f 2 | cut -d 'E' -f 2)

	MIN_LOSS=99999.0
	MAX_LOSS=0.0
	for LOSS in "${LOSSES[@]}"
	do
		LOSS="$(echo -e "${LOSS}" | tr -d '[:space:]')"
		if (( $(echo "$LOSS < $MIN_LOSS" | bc -l) )); then
			MIN_LOSS="$LOSS"
		fi
		if (( $(echo "$LOSS > $MAX_LOSS" | bc -l) )); then
			MAX_LOSS="$LOSS"
		fi
	done
	
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
		printf "%10s %8s %8s %5s ${RED}MISSING${NC} " ${name} ${MAX_LOSS} ${MIN_LOSS} ${MIN_EPOCH}
	else
		statuses=$(echo $result | awk '{print $5}')
		while read -r status; do
			if [ ${status} == "R" ]
			then
				is_waiting=false
				printf "%10s %8s %8s %5s ${GREEN}RUNNING${NC} " ${name} ${MAX_LOSS} ${MIN_LOSS} ${MIN_EPOCH}
			fi
		done <<< "$statuses"
	fi
	
	if [ "$is_waiting" = true ]
	then
		printf "%10s %8s %8s %5s WAITING " ${name} ${MAX_LOSS} ${MIN_LOSS} ${MIN_EPOCH}
	fi
	
	printf "$result" | tr " " "\n" | grep -c "${name}"
done
