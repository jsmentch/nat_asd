#!/bin/bash

module load openmind8/apptainer/1.1.7
module load openmind8/connectome-workbench/1.5.0 
module load openmind/freesurfer/6.0.0


export FS_LICENSE='/nese/mit/group/sig/projects/hbn/hbn_bids/code/license.txt'
export SUBJECT="sub-NDARHJ830RXD"
export SUBJECTS_DIR='/nese/mit/group/sig/projects/hbn/hbn_bids/derivatives/freesurfer_7.3.2'
export HEMI="lh"
export INPUT="/nese/mit/group/sig/projects/hbn/hbn_bids/derivatives/freesurfer_7.3.2/sub-NDARHJ830RXD/label/lh.V1_exvivo.label"
export FREESURFER_IMG='/om2/user/jsmentch/containers/images/bids/bids-freesurfer--7.4.1-unstable.sing'
## CONVERT FROM NATIVE TO FSAVERAGE

export FSAVG_OUTPUT='/om2/scratch/tmp/jsmentch/test_lh.label'


# apptainer exec -B /om2,/nese --containall "$FREESURFER_IMG" \
# env SUBJECTS_DIR=$SUBJECTS_DIR \
# env FS_LICENSE=$FS_LICENSE \
# env SUBJECT=$SUBJECT \
# env SUBJECTS_DIR=$SUBJECTS_DIR \
# env HEMI=$HEMI \
# env INPUT=$INPUT \
# env FSAVG_OUTPUT=$FSAVG_OUTPUT \
# mri_label2label --srcsubject "$SUBJECT" --trgsubject fsaverage --hemi "$HEMI" \
#     --srclabel "$INPUT" --trglabel "$FSAVG_OUTPUT" --regmethod surface

# CONVERT FROM .label to .label.gii

GII_OUTPUT='/om2/scratch/tmp/jsmentch/test_lh.label.gii'

apptainer exec -B /om2,/nese --nv "$FREESURFER_IMG" \
env SUBJECTS_DIR=$SUBJECTS_DIR \
env FS_LICENSE=$FS_LICENSE \
env SUBJECT=$SUBJECT \
env SUBJECTS_DIR=$SUBJECTS_DIR \
env HEMI=$HEMI \
env INPUT=$INPUT \
env FSAVG_OUTPUT=$FSAVG_OUTPUT \
env GII_OUTPUT=$GII_OUTPUT \
mris_convert --label "${SUBJECTS_DIR}/${SUBJECT}/label/${HEMI}.V1_exvivo.label" \
V1 "${SUBJECTS_DIR}/${SUBJECT}/surf/${HEMI}.white" "$GII_OUTPUT"



#old
#mris_convert --label "$FSAVG_OUTPUT" V1 "${SUBJECTS_DIR}/fsaverage/surf/lh.white" "$GII_OUTPUT"


# Use this instead:

# label_name="V1"
# label_in="${sub}_${hemi_fs}.V1_exvivo.label.gii"
# apptainer exec --containall -e \
# -B /om2/scratch/tmp/yibei/friends/data:/data \
# -B /om2/user/smeisler/license.txt:/license.txt \
# --env FS_LICENSE=/license.txt \
# /om2/user/smeisler/freesurfer_7.4.1.img \
# mris_convert  --label /data/anat.freesurfer/${sub}/label/${hemi_fs}.${label_name}_exvivo.label ${label_name} \
# /data/anat.freesurfer/${sub}/surf/${hemi_fs}.midthickness \
# /data/subj_fsLR/${label_in}







hemi_fs="lh" # hemisphere in freesurfer naming conventions
hemi_wb="L" # hemisphere in connectome workbench naming conventions
density_in="164k" # fsaverage density
density_out="32k" # cifti fsLR density
freesurfer_path=$SUBJECTS_DIR
sub=$SUBJECT
outpath="/om2/scratch/tmp/jsmentch"
label_in=$GII_OUTPUT # or other label
label_out="$outpath/test_lh_out.label.gii"
fsaverage_resample_dir="/om2/user/jsmentch/HCPpipelines/global/templates/standard_mesh_atlases/resample_fsaverage" # has these files https://github.com/Washington-University/HCPpipelines/tree/master/global/templates/standard_mesh_atlases/resample_fsaverage

wb_shortcuts -freesurfer-resample-prep \
${freesurfer_path}/${sub}/surf/${hemi_fs}.white \
${freesurfer_path}/${sub}/surf/${hemi_fs}.pial \
${freesurfer_path}/${sub}/surf/${hemi_fs}.sphere.reg \
${fsaverage_resample_dir}/fs_LR-deformed_to-fsaverage.${hemi_wb}.sphere.${density_out}_fs_LR.surf.gii \
${outpath}/${sub}_${hemi_fs}.midthickness.surf.gii \
${outpath}/${sub}.${hemi_wb}.midthickness.${density_in}_fs_LR.surf.gii \
${outpath}/${sub}_${hemi_fs}.sphere.reg.surf.gii

# wb_shortcuts -freesurfer-resample-prep \
# ${freesurfer_path}/${sub}/surf/${hemi_fs}.white \ # white surface
# ${freesurfer_path}/${sub}/surf/${hemi_fs}.pial \ # pial surface
# ${freesurfer_path}/${sub}/surf/${hemi_fs}.sphere.reg \ # sphere
# ${fsaverage_resample_dir}/fs_LR-deformed_to-fsaverage.${hemi_wb}.sphere.${density_out}_fs_LR.surf.gii \ # fsLR sphere
# ${outpath}/${sub}_${hemi_fs}.midthickness.surf.gii \ # output freesurfer midthickness gifti
# ${outpath}/${sub}.${hemi_wb}.midthickness.${density_in}_fs_LR.surf.gii \ # output fsLR midthickness gifti
# ${outpath}/${sub}_${hemi_fs}.sphere.reg.surf.gii  # output freesurfer sphere gifti


wb_command -label-resample \
${label_in} \
${outpath}/${sub}_${hemi_fs}.sphere.reg.surf.gii \
${fsaverage_resample_dir}/fs_LR-deformed_to-fsaverage.${hemi_wb}.sphere.${density_out}_fs_LR.surf.gii \
ADAP_BARY_AREA \
${label_out} \
-area-surfs \
${outpath}/${sub}_${hemi_fs}.midthickness.surf.gii \
${outpath}/${sub}.${hemi_wb}.midthickness.${density_in}_fs_LR.surf.gii








# wb_command -label-resample \
# ${label_in} \
# ${outpath}/${sub}_${hemi_fs}.sphere.reg.surf.gii \ # fsaverage sphere from previous command
# ${fsaverage_resample_dir}/fs_LR-deformed_to-fsaverage.${hemi_wb}.sphere.${density_out}_fs_LR.surf.gii \ # fsLR sphere
# ADAP_BARY_AREA \
# ${label_out} \
# -area-surfs \
# ${outpath}/${sub}_${hemi_fs}.midthickness.surf.gii \ # freesurfer midthickness from previous command
# ${outpath}/${sub}.${hemi_wb}.midthickness.${density_in}_fs_LR.surf.gii # fsLR midthickness from previous command