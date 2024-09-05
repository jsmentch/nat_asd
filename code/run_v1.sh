#!/bin/bash

module load openmind8/apptainer/1.1.7
module load openmind8/connectome-workbench/1.5.0 
module load openmind/freesurfer/6.0.0



# Path to the text file containing subject IDs
subjects_file="/om2/scratch/tmp/jsmentch/nat_asd/code/pilots_ru_dm_list.txt"

# Loop through each line in the file
while IFS= read -r sub_id
do
    # Use $sub_id for your processing
    echo "Processing subject: $sub_id"
    export SUB_ID=$sub_id

    export FS_LICENSE='/nese/mit/group/sig/projects/hbn/hbn_bids/code/license.txt'
    export FS_DIR='/nese/mit/group/sig/projects/hbn/hbn_bids/derivatives/freesurfer_7.3.2'
    export FS_IMG='/om2/user/jsmentch/containers/images/bids/bids-freesurfer--7.4.1-unstable.sing'
    export fsaverage_resample_dir="/om2/user/jsmentch/HCPpipelines/global/templates/standard_mesh_atlases/resample_fsaverage" # has these files 
    
    # export SUB_ID="NDARHJ830RXD"
    export SUBJECT="sub-${SUB_ID}"
    
    export outpath="/om2/scratch/tmp/jsmentch/nat_asd/outputs/v1_roi"

    check="$outpath/${SUBJECT}_V1.dlabel.nii"
    if [ -f "$check" ]; then
        echo "File $check exists, skipping..."
        continue
    fi


    for hemi in lh rh; do
        if [ "$hemi" = "lh" ]; then
            export HEMI="lh"
            export HEMI_WB="L"
        else
            export HEMI="rh"
            export HEMI_WB="R"
        fi
    
        # Display the current values for debugging
        echo "HEMI: $HEMI, HEMI_WB: $HEMI_WB"
        # export HEMI="lh"
        # export HEMI_WB="L" # hemisphere in connectome workbench naming conventions
        export density_in="164k" # fsaverage density
        export density_out="32k" # cifti fsLR density
        export label_out="$outpath/${SUBJECT}_V1_${HEMI_WB}.label.gii"
        
        ## CONVERT FROM NATIVE TO FSAVERAGE
        
        export GII_OUTPUT='/om2/scratch/tmp/jsmentch/nat_asd/outputs/v1_roi/temp.label.gii'
        
        apptainer exec -B /om2,/nese --nv "$FS_IMG" \
        env FS_DIR=$FS_DIR \
        env FS_LICENSE=$FS_LICENSE \
        env SUBJECT=$SUBJECT \
        env FS_DIR=$FS_DIR \
        env HEMI=$HEMI \
        env GII_OUTPUT=$GII_OUTPUT \
        mris_convert --label "${FS_DIR}/${SUBJECT}/label/${HEMI}.V1_exvivo.label" \
        V1 "${FS_DIR}/${SUBJECT}/surf/${HEMI}.white" "$GII_OUTPUT"
        
        wb_shortcuts -freesurfer-resample-prep \
        ${FS_DIR}/${SUBJECT}/surf/${HEMI}.white \
        ${FS_DIR}/${SUBJECT}/surf/${HEMI}.pial \
        ${FS_DIR}/${SUBJECT}/surf/${HEMI}.sphere.reg \
        ${fsaverage_resample_dir}/fs_LR-deformed_to-fsaverage.${HEMI_WB}.sphere.${density_out}_fs_LR.surf.gii \
        ${outpath}/${SUBJECT}_${HEMI}.midthickness.surf.gii \
        ${outpath}/${SUBJECT}.${HEMI_WB}.midthickness.${density_in}_fs_LR.surf.gii \
        ${outpath}/${SUBJECT}_${HEMI}.sphere.reg.surf.gii
        
        wb_command -label-resample \
        ${GII_OUTPUT} \
        ${outpath}/${SUBJECT}_${HEMI}.sphere.reg.surf.gii \
        ${fsaverage_resample_dir}/fs_LR-deformed_to-fsaverage.${HEMI_WB}.sphere.${density_out}_fs_LR.surf.gii \
        ADAP_BARY_AREA \
        ${label_out} \
        -area-surfs \
        ${outpath}/${SUBJECT}_${HEMI}.midthickness.surf.gii \
        ${outpath}/${SUBJECT}.${HEMI_WB}.midthickness.${density_in}_fs_LR.surf.gii
        
        rm $GII_OUTPUT
        rm ${outpath}/${SUBJECT}_${HEMI}.midthickness.surf.gii
        rm ${outpath}/${SUBJECT}_${HEMI}.sphere.reg.surf.gii
        rm ${outpath}/${SUBJECT}.${HEMI_WB}.midthickness.${density_in}_fs_LR.surf.gii

    done
    left_label="$outpath/${SUBJECT}_V1_L.label.gii"
    right_label="$outpath/${SUBJECT}_V1_R.label.gii"
    out_label="$outpath/${SUBJECT}_V1.dlabel.nii"
    wb_command -cifti-create-label $out_label \
    -left-label $left_label \
    -right-label $right_label

done < "$subjects_file"




# wb_command -cifti-create-label /om2/scratch/tmp/jsmentch/nat_asd/outputs/v1_roi/test.dlabel.nii \
# -left-label /om2/scratch/tmp/jsmentch/nat_asd/outputs/v1_roi/sub-NDARDU853XZ6_V1_L.label.gii \
# -right-label /om2/scratch/tmp/jsmentch/nat_asd/outputs/v1_roi/sub-NDARDU853XZ6_V1_R.label.gii