#!/bin/bash

#SBATCH --job-name=kwykstrag
#SBATCH --output=/om2/user/jsmentch/nat_asd_logs/%x_%j.out 
#SBATCH --error=/om2/user/jsmentch/nat_asd_logs/%x_%j.err 
#SBATCH --partition=normal 
#SBATCH --time=6:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jsmentch@mit.edu
#SBATCH --gres=gpu:1
#SBATCH --array=0-43

#source /om2/user/jsmentch/anaconda/etc/profile.d/conda.sh

#conda activate hbn_asd

#TASK_ID=${roi_id[$SLURM_ARRAY_TASK_ID]}

# OLD # while read sub; do sbatch run_kwyk.sh $sub; done < pilots_ru_dm_list.txt

#sub=$1

#to run:  sbatch --array=0-43 run_kwyk.sh

# sub=$(sed -n "${SLURM_ARRAY_TASK_ID}p" t1_subjects.txt)
sub=$(sed -n "${SLURM_ARRAY_TASK_ID}p" kwyk_stragglers.txt)
# sub=$(sed -n "${SLURM_ARRAY_TASK_ID}p" t1_fmriprep_subjects.txt)

echo "Running job with SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running sub: ${sub}"

#sub=NDARHJ830RXD

module load openmind/singularity/3.9.5


#parcel='A1'
cd /om2/scratch/tmp/jsmentch/kwyk

# #HBN
for file in /nese/mit/group/sig/projects/hbn/hbn_bids/sub-${sub}/ses-*/anat/sub-${sub}_ses-*_T1w.nii.gz; do
    if [[ -f "$file" ]]; then
        echo "Processing $file"
        singularity run -B /nese -B /om2 -B $(pwd):/data -W /data --nv /om2/user/jsmentch/kwyk.simg -m bvwn_multi_prior -n 2 --save-entropy /nese/mit/group/sig/projects/hbn/hbn_bids/sub-${sub}/ses-*/anat/sub-${sub}_ses-*_T1w.nii.gz ${sub}
    else 
        echo "$file not found"
    fi
    break
done



# #CNEURoMOD
# singularity run -B /nese -B /om2 -B $(pwd):/data -W /data --nv /om2/user/jsmentch/kwyk.simg -m bvwn_multi_prior -n 2 \
#   --save-variance --save-entropy /nese/mit/group/sig/projects/cneuromod/cneuromod/anat/sub-${sub}/ses-001/anat/sub-${sub}_ses-001_T1w.nii.gz ${sub}

# #HBN brain masked
# singularity run -B /nese -B /om2 -B $(pwd):/data -W /data --nv /om2/user/jsmentch/kwyk.simg -m bvwn_multi_prior -n 2 --save-entropy /om2/scratch/tmp/jsmentch/kwyk/brain_masked_t1s/sub-${sub}_masked-t1.nii.gz ${sub}



#python pilot.py "${sub}" "${parcel}"

#python pilot.py -s $sub -p auditory -f both_hrf -d 0



