import pandas as pd
import nibabel as nb
import os
import numpy as np
import glob
import h5py
import hcp_utils as hcp
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import numpy as np
from stacking_fmri import stacking_CV_fmri, stacking_fmri
from ridge_tools import R2
import matplotlib.pyplot as plt
import seaborn as sns

import time
#pd.set_option('display.max_rows', None)
#use the conda environment hbn_asd

def standardscale(X_raw):
    from sklearn.preprocessing import StandardScaler
    X=[]
    for xx in X_raw:
        scaler = StandardScaler()
        X.append( scaler.fit_transform(X=xx,y=None) )
    return(X)

def apply_pca(X, n_components):
    from sklearn.decomposition import PCA
    
    X_pca=[]
    for xx in X:
        pca = PCA(n_components=n_components)
        X_pca.append(     pca.fit_transform( xx )   )
    return(X_pca)

def apply_zscore(X):
    from scipy.stats import zscore
    X_z=[]
    for x in X:
        X_z.append(zscore(x))
    return(X_z)

def load_audio_features(stim,all_layers):
    save_features_dir = f'../data/{stim}_clips_cochresnet50/'
    X=[]
    # Open the file 'myfile.h5' in read-only mode
    file = h5py.File(f'{save_features_dir}cochresnet50_activations.h5', 'r')
    for layer in all_layers:
    # # Now you can access datasets within the file
        data = file[layer]
        #print(data.shape, layer)
        #X.append(  np.array(data)[:(-1*delay),:]   )
        X.append(  np.array(data)   )

        # X_train.append(np.array(data)[:600,:])
        # X_test.append(np.array(data)[600:,:])
    
    # # Don't forget to close the file when you're done
    file.close()
    return(X)


def load_audio_features_processed(filename,all_layers):
    #load the features that already had PCA applied to them ,etc, from h5 files
    save_features_dir = f'../data/features/'
    X=[]
    file = h5py.File(f'{save_features_dir}{filename}', 'r')
    for layer in all_layers:
        X.append(  np.array(file[layer])   )
    file.close()
    return(X)

def load_audio_features_manual_hrf(stim,features):
    #features=['rms','chroma', 'mfcc', 'mfs', 'as_embed', 'as_scores']
    import hrf_tools
    from scipy.signal import resample
    X=[]
    for f in features:
        feature=np.load(f'../data/features/{stim}_{f}.npy')
        scaler = StandardScaler()
        feature = scaler.fit_transform(X=feature,y=None)
        hz=feature.shape[0]/600
        feature=hrf_tools.apply_optimal_hrf_10hz(feature,hz)
        feature=resample(feature, 750, axis=0) #resample to 1hz for now 
        X.append(feature)
    return(X)


def load_both_features_hrf(stim):
    features_manual=['as_embed', 'as_scores']
    features_cochresnet=['input_after_preproc',
                    'conv1_relu1',
                    'maxpool1',
                    'layer1',
                    'layer2',
                    'layer3',
                    'layer4',
                    'avgpool']
    from sklearn.decomposition import PCA
    import hrf_tools
    
    features=features_cochresnet
    X_raw=load_audio_features(stim,features)
    X=standardscale(X_raw)
    X=apply_pca(X, 1)
    for xx in X:
        hz=xx.shape[0]/600
        hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    x1shape=X[0].shape[0]
    X2=load_audio_features_manual_hrf(stim,features_manual)
    for xx in X2:
        X.append(xx)
    x2shape=X[-1].shape[0]
    X = [array[:min([x2shape,x1shape]), :] for array in X]
    return(X)


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

def load_audio_features_PCA(stim,all_layers,n_components):
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
        #X.append(  transformer.fit_transform(  np.array(data)[:(-1*delay),:]  )  )
        X.append(  transformer.fit_transform(  np.array(data))  )

        # X_train.append(np.array(data)[:600,:])
        # X_test.append(np.array(data)[600:,:])
    
    # # Don't forget to close the file when you're done
    file.close()
    return(X)


def load_audio_features_PCAc2(stim,all_layers):
    #dimensionality reduction to 2 components and discard the first component
    transformer = PCA(n_components=2)
    #print(f'loading features 2 PCA components')

    save_features_dir = f'../data/{stim}_clips_cochresnet50/'
    
    X=[]
    file = h5py.File(f'{save_features_dir}cochresnet50_activations.h5', 'r')
    for layer in all_layers:
    # # Now you can access datasets within the file
        data = file[layer]
        #print(transformer.fit_transform(  np.array(data))[:,1:].shape)
        X.append(  transformer.fit_transform(  np.array(data))[:,1:]  )
    file.close()
    return(X)
    

def load_video_features(stim,all_layers):
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
        #X.append(  transformer.fit_transform(  np.array(X_layer)[:(-1*delay),:]  )  )
        X.append(  transformer.fit_transform(  np.array(X_layer)  )  )

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
    #parcels MUST be a list of strings
    patternR = '|'.join(['Right_' + parcel for parcel in parcels])
    patternL = '|'.join(['Left_' + parcel for parcel in parcels])
    
    # get a boolean series where True indicates a match
    matches = atlas['label'].str.contains(patternR) | atlas['label'].str.contains(patternL)
    
    # get the indices that match
    atlas_indices = atlas[matches]['index'].tolist()
    indices = matches[matches].index.tolist()
    parcel_names=atlas[matches]['label'].tolist()
    return(atlas_indices,indices,parcel_names)
    
def load_audio_features_RAW(stim,all_layers):
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
        #X.append( np.array(data)[:(-1*delay),:]  )
        X.append( np.array(data) )


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
    