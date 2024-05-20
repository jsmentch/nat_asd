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
#pd.set_option('display.max_rows', None)


#use the conda environment hbn_asd


def load_audio_features_SRP(stim,delay,all_layers,n_components):
    #dimensionality reduction to 50 components
    transformer = SparseRandomProjection(n_components=n_components)
    print(f'loading features {n_components} SRP components')


    save_features_dir = f'../data/{stim}_clips_cochresnet50/'
    
    X=[]
    # X_train=[]
    # X_test=[]
    #print('CochResNet50 time-averaged')
    # Open the file 'myfile.h5' in read-only mode
    file = h5py.File(f'{save_features_dir}cochresnet50_activations.h5', 'r')
    for layer in all_layers:
    # # Now you can access datasets within the file
        data = file[layer]
        #print(data.shape, layer)
        X.append(  transformer.fit_transform(  np.array(data)[:(-1*delay),:]  )  )
        # X_train.append(np.array(data)[:600,:])
        # X_test.append(np.array(data)[600:,:])
    
    # # Don't forget to close the file when you're done
    file.close()
    return(X)

def load_audio_features_PCA(stim,delay,all_layers,n_components):
    #dimensionality reduction to 50 components
    transformer = PCA(n_components=n_components)
    print(f'loading features {n_components} PCA components')

    save_features_dir = f'../data/{stim}_clips_cochresnet50/'
    
    X=[]
    # X_train=[]
    # X_test=[]
    #print('CochResNet50 time-averaged')
    # Open the file 'myfile.h5' in read-only mode
    file = h5py.File(f'{save_features_dir}cochresnet50_activations.h5', 'r')
    for layer in all_layers:
    # # Now you can access datasets within the file
        data = file[layer]
        #print(data.shape, layer)
        X.append(  transformer.fit_transform(  np.array(data)[:(-1*delay),:]  )  )
        # X_train.append(np.array(data)[:600,:])
        # X_test.append(np.array(data)[600:,:])
    
    # # Don't forget to close the file when you're done
    file.close()
    return(X)
    

def load_video_features(stim,delay,all_layers):
    #dimensionality reduction to 50 components
    transformer = SparseRandomProjection(n_components=50)
    save_path = f'../data/{stim}_frames_resnet50/'
    #print('ResNet50 frame embeddings')
    # for emb_f in emb_list:
    #     emb = np.load(f'{emb_f}')
    #     print(emb['frame_0000.jpg'].shape, os.path.splitext(os.path.basename(emb_f))[0])
    X=[]
    for layer in all_layers:
        X_layer=[]
        emb = np.load(f'{save_path}{layer}.npz')
        for k in list(emb.keys()):
            X_layer.append(emb[k].flatten())
        X.append(  transformer.fit_transform(  np.array(X_layer)[:(-1*delay),:]  )  )
    return(X)




def load_fmri_data(im_file,delay,indices):
    # load fmri data for a subject
    img = nb.load(im_file)
    img_y = img.get_fdata()
    Y=img_y[delay:,indices]
    return(Y)
    # Y_train=Y[:600,:]
    # Y_test=Y[600:,:]

def get_subject_list():
    pilot_subjects=pd.read_csv('../data/pilots_ru_dm.csv') # load pilot subjects
    
    subjects=[]
    for sub in list(pilot_subjects['Identifiers_y']):
        
        im_file = f'/nese/mit/group/sig/projects/hbn/hbn_bids/derivatives/xcp_d_0.7.1/sub-{sub}/ses-HBNsiteRU/func/sub-{sub}_ses-HBNsiteRU_task-movieDM_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii'
        try:
            img = nb.load(im_file)
            #print('loaded')
            subjects.append(sub)
        except:
            print(f'missing {sub}')
    print(f'loaded {len(subjects)} subjects')
    return(subjects)
    
# def get_parcel_indices(parcels):
#     patternR = '|'.join(['Right_' + parcel for parcel in parcels])
#     patternL = '|'.join(['Left_' + parcel for parcel in parcels])
    
#     # get a boolean series where True indicates a match
#     matches = atlas['label'].str.contains(patternR) | atlas['label'].str.contains(patternL)
    
#     # get the indices that match
#     atlas_indices = atlas[matches]['index'].tolist()
#     indices = matches[matches].index.tolist()
#     parcel_names=atlas[matches]['label'].tolist()
#     return(atlas_indices,indices,parcel_names)

# def get_parcel_indices(atlas,parcels):
#     patternR = '|'.join(['Right_' + parcel for parcel in parcels])
#     patternL = '|'.join(['Left_' + parcel for parcel in parcels])
    
#     # get a boolean series where True indicates a match
#     matches = atlas['label'].str.contains(patternR) | atlas['label'].str.contains(patternL)
    
#     # get the indices that match
#     indices = matches[matches].index.tolist()
#     parcel_names=atlas[matches]['label'].tolist()
#     return(indices,parcel_names)

def get_parcel_indices(atlas, parcels):
    patternR = '|'.join(['Right_' + parcel for parcel in parcels])
    patternL = '|'.join(['Left_' + parcel for parcel in parcels])
    
    # get a boolean series where True indicates a match
    matches = atlas['label'].str.contains(patternR) | atlas['label'].str.contains(patternL)
    
    # get the indices that match
    atlas_indices = atlas[matches]['index'].tolist()
    indices = matches[matches].index.tolist()
    parcel_names=atlas[matches]['label'].tolist()
    return(atlas_indices,indices,parcel_names)
    
def load_audio_features_RAW(stim,delay,all_layers):
    #dimensionality reduction to 50 components
    #transformer = SparseRandomProjection(n_components=50)
    

    save_features_dir = f'../data/{stim}_clips_cochresnet50/'
    
    X=[]
    # X_train=[]
    # X_test=[]
    #print('CochResNet50 time-averaged')
    # Open the file 'myfile.h5' in read-only mode
    file = h5py.File(f'{save_features_dir}cochresnet50_activations.h5', 'r')
    for layer in all_layers:
    # # Now you can access datasets within the file
        data = file[layer]
        print(data.shape, layer)
        #X.append(  transformer.fit_transform(  np.array(data)[:(-1*delay),:]  )  )
        X.append( np.array(data)[:(-1*delay),:]  )

        # X_train.append(np.array(data)[:600,:])
        # X_test.append(np.array(data)[600:,:])
    
    # # Don't forget to close the file when you're done
    file.close()
    return(X)

def load_glasser():
    atlas_dlabel='/om2/user/jsmentch/atlases/atlas-Glasser/atlas-Glasser_space-fsLR_den-91k_dseg.dlabel.nii'
    img = nb.load(atlas_dlabel)
    atlas_data=img.get_fdata()
    atlas_data=atlas_data[0,:]
    atlas=pd.read_csv('/om2/user/jsmentch/atlases/atlas-Glasser/atlas-Glasser_dseg.tsv', sep='\t')
    return(atlas,atlas_data)
    