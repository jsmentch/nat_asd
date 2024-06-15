#!/bin/bash

#SBATCH --job-name=shpilots
#SBATCH --output=/om2/user/jsmentch/nat_asd_logs/%x_%j.out 
#SBATCH --error=/om2/user/jsmentch/nat_asd_logs/%x_%j.err 
#SBATCH --partition=normal 
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jsmentch@mit.edu

source /om2/user/jsmentch/anaconda/etc/profile.d/conda.sh

conda activate hbn_asd

#TASK_ID=${roi_id[$SLURM_ARRAY_TASK_ID]}

# while read sub; do sbatch run_pilot.sh $sub; done < pilots_ru_dm_list.txt
# while read sub; do sbatch run_pilots_short.sh $sub; done < good_pilots_ru_dm_list.txt

sub=$1
#sub=NDARHJ830RXD

#parcel='A1'

#python pilot.py "${sub}" "${parcel}"
# "manual" 'manuallow' 'audioset' 
str_list=("cochresnet50pca1" "cochresnet50pca200" "cochresnet50pca5" 
             "cochresnet50pca10" "cochresnet50pca50" "cochresnet50pca100" 
             "cochresnet50pcafull1" "cochresnet50pcafull200" "cochresnet50pcafull5" 
             "cochresnet50pcafull10" "cochresnet50pcafull50" "cochresnet50pcafull100" 
             "cochresnet50pcac2" "cochresnet50PCAlocal1" 
             "cochresnet50PCAlocal1mean" "cochresnet50PCAlocal10mean")

# Loop through the list
for feat in "${str_list[@]}"; do
    python pilot.py -s $sub -p auditory -f $feat -d 7
    #python pilot.py -s $sub -p all -f manual -d 7 -l
    echo "$feat"
done


python pilot.py -s $sub -p auditory -f manualhrf -d 0
python pilot.py -s $sub -p auditory -f manualhrfpca1 -d 0
python pilot.py -s $sub -p auditory -f manualhrfpca10 -d 0


#python pilot.py -s NDARHJ830RXD -p all -f manual -d 7 -l
