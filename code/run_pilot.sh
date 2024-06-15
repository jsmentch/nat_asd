#!/bin/bash

#SBATCH --job-name=crn50pcal5rev
#SBATCH --output=/om2/user/jsmentch/nat_asd_logs/%x_%j.out 
#SBATCH --error=/om2/user/jsmentch/nat_asd_logs/%x_%j.err 
#SBATCH --partition=normal 
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=50G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jsmentch@mit.edu

source /om2/user/jsmentch/anaconda/etc/profile.d/conda.sh

conda activate hbn_asd

#TASK_ID=${roi_id[$SLURM_ARRAY_TASK_ID]}

# while read sub; do sbatch run_pilot.sh $sub; done < good_pilots_ru_dm_list.txt

sub=$1
#sub=NDARHJ830RXD

#parcel='A1'

#python pilot.py "${sub}" "${parcel}"

python pilot.py -s $sub -p auditory -f cochresnet50PCAlocal5rev -d 7
#python pilot.py -s $sub -p all -f manual -d 7 -l



#python pilot.py -s NDARHJ830RXD -p all -f manual -d 7 -l
