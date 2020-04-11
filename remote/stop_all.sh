#!/bin/bash

IFS=

result=$(squeue -u ${USER} | grep dtect)
jobIds=$(echo $result | awk '{print $1}')

while read -r jobId; do
    scancel $jobId
done <<< "$jobIds"
