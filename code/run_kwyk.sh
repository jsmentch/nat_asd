#!/bin/bash

#SBATCH --job-name=kwykgpu
#SBATCH --output=/om2/user/jsmentch/nat_asd_logs/%x_%j.out 
#SBATCH --error=/om2/user/jsmentch/nat_asd_logs/%x_%j.err 
#SBATCH --partition=normal 
#SBATCH --time=1:45:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jsmentch@mit.edu
#SBATCH --gres=gpu:1

#source /om2/user/jsmentch/anaconda/etc/profile.d/conda.sh

#conda activate hbn_asd

#TASK_ID=${roi_id[$SLURM_ARRAY_TASK_ID]}

# while read sub; do sbatch run_kwyk.sh $sub; done < pilots_ru_dm_list.txt

module load openmind/singularity/3.9.5

sub=$1
#sub=NDARHJ830RXD

#parcel='A1'
cd /om2/scratch/tmp/jsmentch/kwyk

# #HBN
# singularity run -B /nese -B /om2 -B $(pwd):/data -W /data --nv /om2/scratch/tmp/jsmentch/kwyk.simg -m bvwn_multi_prior -n 2 \
#   --save-variance --save-entropy /nese/mit/group/sig/projects/hbn/hbn_bids/sub-${sub}/ses-HBNsiteRU/anat/sub-${sub}_ses-HBNsiteRU_acq-HCP_T1w.nii.gz ${sub}

# #CNEURoMOD
singularity run -B /nese -B /om2 -B $(pwd):/data -W /data --nv /om2/scratch/tmp/jsmentch/kwyk.simg -m bvwn_multi_prior -n 2 \
  --save-variance --save-entropy /nese/mit/group/sig/projects/cneuromod/cneuromod/anat/sub-${sub}/ses-001/anat/sub-${sub}_ses-001_T1w.nii.gz ${sub}





#python pilot.py "${sub}" "${parcel}"

#python pilot.py -s $sub -p auditory -f both_hrf -d 0



