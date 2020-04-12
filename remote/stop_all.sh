#!/bin/bash

IFS=

result=$(squeue -u ${USER} | grep gan_)
jobIds=$(echo $result | awk '{print $1}')

while read -r jobId; do
    scancel $jobId
done <<< "$jobIds"
