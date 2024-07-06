#!/bin/bash

#SBATCH --job-name=a4a5hbn
#SBATCH --output=/om2/user/jsmentch/nat_asd_logs/%x_%j.out 
#SBATCH --error=/om2/user/jsmentch/nat_asd_logs/%x_%j.err 
#SBATCH --partition=normal 
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jsmentch@mit.edu

source /om2/user/jsmentch/anaconda/etc/profile.d/conda.sh

conda activate hbn_asd

#TASK_ID=${roi_id[$SLURM_ARRAY_TASK_ID]}

# while read sub; do sbatch run_pilot-cnm-length.sh $sub; done < good_pilots_ru_dm_list.txt

sub=$1
#sub=NDARHJ830RXD


python pilot-cnm-length.py -s $sub -p a4a5 -f 1 -t 5 -d 0
python pilot-cnm-length.py -s $sub -p a4a5 -f 1 -t 10 -d 0
python pilot-cnm-length.py -s $sub -p a4a5 -f 1 -t 15 -d 0
python pilot-cnm-length.py -s $sub -p a4a5 -f 1 -t 23 -d 0

python pilot-cnm-length.py -s $sub -p a4a5 -f 10 -t 5 -d 0
python pilot-cnm-length.py -s $sub -p a4a5 -f 10 -t 10 -d 0
python pilot-cnm-length.py -s $sub -p a4a5 -f 10 -t 15 -d 0
python pilot-cnm-length.py -s $sub -p a4a5 -f 10 -t 23 -d 0

python pilot-cnm-length.py -s $sub -p a4a5 -f 100 -t 5 -d 0
python pilot-cnm-length.py -s $sub -p a4a5 -f 100 -t 10 -d 0
python pilot-cnm-length.py -s $sub -p a4a5 -f 100 -t 15 -d 0
python pilot-cnm-length.py -s $sub -p a4a5 -f 100 -t 23 -d 0



#python pilot.py -s $sub -p all -f manual -d 7 -l
# python pilot.py -s $sub -p earlyvisual -f resnet50pca1hrf -d 0
# python pilot.py -s $sub -p earlyvisual -f resnet50pca5hrf -d 0
# python pilot.py -s $sub -p earlyvisual -f resnet50pca10hrf -d 0
#python pilot.py -s $sub -p earlyvisual -f resnet50pca500hrf -d 0

# python pilot.py -s $sub -p ventralvisual -f video_slowfastr50pca1hrf -d 0
# python pilot.py -s $sub -p ventralvisual -f video_slowfastr50pca10hrf -d 0
# python pilot.py -s $sub -p ventralvisual -f video_slowfastr50pca100hrf -d 0

# python pilot.py -s $sub -p ventralvisual -f video_resnet50pca1hrf -d 0
# python pilot.py -s $sub -p ventralvisual -f video_resnet50pca10hrf -d 0
# python pilot.py -s $sub -p ventralvisual -f video_resnet50pca100hrf -d 0

# python pilot.py -s $sub -p ventralvisual -f resnet50pca1hrf -d 0
# python pilot.py -s $sub -p ventralvisual -f resnet50pca10hrf -d 0
# python pilot.py -s $sub -p ventralvisual -f resnet50pca100hrf -d 0

#python pilot.py -s NDARHJ830RXD -p all -f manual -d 7 -l
