#!/bin/bash

#SBATCH --job-name=boots
#SBATCH --output=/om2/user/jsmentch/nat_asd_logs/%A_%a.out 
#SBATCH --error=/om2/user/jsmentch/nat_asd_logs/%A_%a.err 
#SBATCH --partition=normal 
#SBATCH --time=3:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=22G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jsmentch@mit.edu

source /om2/user/jsmentch/anaconda/etc/profile.d/conda.sh

conda activate hbn_asd

#TASK_ID=${roi_id[$SLURM_ARRAY_TASK_ID]}

# while read sub; do sbatch run_pilot.sh $sub; done < pilots_ru_dm_list.txt


args=($@)
#subjs=(${args[@]:1})

# index slurm array to grab subject
#SLURM_ARRAY_TASK_ID=0
boot=${args[${SLURM_ARRAY_TASK_ID}]}

echo "running bootstrap ${boot}"
#boot=$1
sub=NDARHJ830RXD

#parcel='A1'

#python pilot.py "${sub}" "${parcel}"

python pilot.py -s $sub -p all -f cochresnet50pca1 -d 7 -b $boot
#python pilot.py -s $sub -p all -f manual -d 7 -l



#python pilot.py -s NDARHJ830RXD -p all -f manual -d 7 -l
