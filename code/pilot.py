import pandas as pd
import nibabel as nb
import os
import numpy as np
import glob
import h5py
import hcp_utils as hcp
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA

import numpy as np
from stacking_fmri import stacking_CV_fmri, stacking_fmri
from ridge_tools import R2
import matplotlib.pyplot as plt
import seaborn as sns

import time
import nat_asd_utils

start_time = time.time()


atlas,atlas_data=nat_asd_utils.load_glasser()


sub="NDARHJ830RXD"

parcel='A1'


parcels=[
    'A1']

atlas_indices,indices,parcel_names=nat_asd_utils.get_parcel_indices(atlas,parcels)



atlas_indices_indices=np.where( (atlas_data==204) | (atlas_data==24) )[0]




delay=6

im_file = f'/nese/mit/group/sig/projects/hbn/hbn_bids/derivatives/xcp_d_0.7.1/sub-{sub}/ses-HBNsiteRU/func/sub-{sub}_ses-HBNsiteRU_task-movieDM_space-fsLR_den-91k_desc-denoisedSmoothed_bold.dtseries.nii'
img = nb.load(im_file)
img_y = img.get_fdata()
Y=img_y[delay:,atlas_indices_indices]

all_layers=['input_after_preproc',
 'conv1_relu1',
 'maxpool1',
 'layer1',
 'layer2',
 'layer3',
 'layer4',
 'avgpool']

# all_layers=['input_after_preproc',
#  'conv1_relu1',
#  'maxpool1']
X=nat_asd_utils.load_audio_features_RAW('DM',delay,all_layers)
#X=load_audio_features('DM',delay,all_layers)

X = [array[:Y.shape[0], :] for array in X]
Y= Y[:X[0].shape[0],:]


r2s, stacked_r2s, r2s_weighted, _, _, S_average = stacking_CV_fmri(Y, X, method = 'cross_val_ridge',n_folds = 5,score_f=R2)

elapsed_time=time.time() - start_time
print(elapsed_time)

np.savez(f'../data/test_output_{sub}', r2s=r2s, stacked_r2s=stacked_r2s, r2s_weighted=r2s_weighted, S_average=S_average, elapsed_time=elapsed_time)