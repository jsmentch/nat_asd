#!/bin/bash

#SBATCH --job-name=pilot_enc
#SBATCH --output=/om2/user/jsmentch/nat_asd_logs/%x_%j.out 
#SBATCH --error=/om2/user/jsmentch/nat_asd_logs/%x_%j.err 
#SBATCH --partition=normal 
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jsmentch@mit.edu

source /om2/user/jsmentch/anaconda/etc/profile.d/conda.sh

conda activate hbn_asd

#TASK_ID=${roi_id[$SLURM_ARRAY_TASK_ID]}

#echo "Processing: $TASK_ID"

sub="NDARHJ830RXD"

parcel='A1'

#python pilot.py "${sub}" "${parcel}"

python pilot.py