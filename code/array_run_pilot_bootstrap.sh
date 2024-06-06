#!/bin/bash

# Submit subjects to be run through. all jobs will share the
# same JOBID, only will differentiate by their array number.
# Example output file: slurm-<JOBID>_<ARRAY>.out

# bash array_run_pilot_bootstrap.sh 


#subjs=($(ls /om2/user/jsmentch/projects/nat_img/sourcedata/data/cneuromod/friends.fmriprep/sub-0*/ses-* -d))
# excluding pilots
#subjs=($(ls sub-leap[0-9]* -d))

# Declare an empty array
declare -a boot

# Loop from 1 to 500
for i in {1..500}
do
  # Add each number to the array
  boot+=($i)
done



# take the length of the array
# this will be useful for indexing later
len=$(expr ${#boot[@]} - 1) # len - 1

echo Spawning ${#boot[@]} sub-jobs.

sbatch --array=0-$len%500 run_pilot_bootstrap.sh ${boot[@]}