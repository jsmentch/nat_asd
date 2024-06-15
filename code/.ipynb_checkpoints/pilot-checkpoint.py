import pandas as pd
import nibabel as nb
import os
import numpy as np
import glob
#import h5py
import hcp_utils as hcp
# from sklearn.random_projection import SparseRandomProjection
# from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import numpy as np
from stacking_fmri import stacking_CV_fmri, stacking_fmri
from ridge_tools import R2
import matplotlib.pyplot as plt
import seaborn as sns
import time
import nat_asd_utils
import sys
import argparse
import hrf_tools

"""
script to ...
Standard usage: python pilot.py -s NDARHJ830RXD -p auditory -f cochresnet50pca1 -d 7 -l
"""
# Inputs:
# - subject
# - parcels: audio, video, audiovideo, all, custom
# - features [which features and PCA?] and which layers?
# - delay length or HRF or FIR, etc

# Outputs:
# - saved results
#   - including list of parcels and list of feautres
#   - including how long it took to run
# - optional overview plot of run

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-s-", "--subject", type=str, help="input subject(s)", nargs='+', required=True)
    parser.add_argument("-p", "--parcels", type=str, help="parcels: auditory, visual, audiovisual, all_select, all, custom (WIP)", required=True)
#    parser.add_argument("-c", "--customparcels", type=str, help="parcels: audio, video, audiovideo, all_select, custom")
    parser.add_argument("-f", "--features", type=str, help="cochresnet50, cochresnet50pca1, cochresnet50pca200, manual", required=True)
    parser.add_argument("-d", "--delay", type=int, help="parcels: audio, video, audiovideo, all, custom", required=True)
    parser.add_argument("-b", "--bootstrap", type=int, help="bootstrap: which permutation it is", default=None)
    parser.add_argument('-l', '--plot', help="to make a plot or not", action='store_true')  # on/off flag
    parser.add_argument('-z', '--zscore', help="to zscore or not", action='store_true')  # on/off flag

    args = parser.parse_args()

    sub=args.subject[0]
    delay=args.delay

    unique_name=f'sub-{sub}_roi-{args.parcels}_feat-{args.features}_delay-{delay}'
    print(f'running subject {sub}')

    if args.parcels=='all':
        Y=load_sub_brain_all(sub,delay) #load the whole brain
        parcels='all'
    else:
        parcels=select_parcels(args.parcels) # load parcel set
        atlas_indices_indices=extract_parcels(parcels) # get indices of parcels
        Y=load_sub_brain(sub,delay,atlas_indices_indices) #load brain data
    
    X,features=load_features(args.features) #load X

    X = [array[:Y.shape[0], :] for array in X]
    if args.zscore:
        X=nat_asd_utils.apply_zscore(X)
        unique_name = unique_name + f'_z'
    Y= Y[:X[0].shape[0],:]

    if args.bootstrap is None:
        print("No value was passed to args.bootstrap")
    else:
        print(f"The value passed to args.bootstrap is {args.bootstrap}, randomly permuting features X")
        for i in range(len(X)):
            np.random.shuffle(X[i])
        unique_name = unique_name + f'_bootstrap-{args.bootstrap}'
    
    print(f'starting regression')
    
    r2s, stacked_r2s, r2s_weighted, _, _, S_average = stacking_CV_fmri(Y, X, method = 'cross_val_ridge',n_folds = 5,score_f=R2)
    
    elapsed_time=time.time() - start_time
    print(elapsed_time)
    
    print(f'saving results')
    
    binary_parcels = [np.void(s.encode('utf-8')) for s in parcels]
    binary_features = [np.void(s.encode('utf-8')) for s in features]
    output_directory_name='good_pilots'
    np.savez(f'../{output_directory_name}/{unique_name}', r2s=r2s, stacked_r2s=stacked_r2s, r2s_weighted=r2s_weighted, S_average=S_average, elapsed_time=elapsed_time, binary_parcels=binary_parcels, binary_features=binary_features)

    if args.plot:
        plot_violins(r2s, stacked_r2s, S_average, features, unique_name)






def load_features(feat_set):
    features_manual=['rms','chroma', 'mfcc', 'mfs', 'as_embed', 'as_scores']
    features_cochresnet=['input_after_preproc',
                    'conv1_relu1',
                    'maxpool1',
                    'layer1',
                    'layer2',
                    'layer3',
                    'layer4',
                    'avgpool']
    features_cochresnet_short=['input_after_preproc',
                    'conv1_relu1',
                    'maxpool1',
                    'layer1',
                    'layer2',
                    'layer3',
                    'layer4']
    if feat_set=="manual":
        from scipy.signal import resample
        X=[]
        features=['rms','chroma', 'mfcc', 'mfs', 'as_embed', 'as_scores']
        for f in features:
            feature=np.load(f'../data/features/DM_{f}.npy')
            #print(feature.shape)
            # transformer = PCA(n_components=n_components)
            scaler = StandardScaler()
            # feature=transformer.fit_transform(feature)
            feature = scaler.fit_transform(X=feature,y=None)
            #print(feature.shape)
            feat_x = resample(feature, 750, axis=0) #resample to 1hz for now 
            X.append(feat_x)
    elif feat_set=='manuallow':
        from scipy.signal import resample
        X=[]
        features=['rms','chroma', 'mfcc', 'mfs']
        for f in features:
            feature=np.load(f'../data/features/DM_{f}.npy')
            #print(feature.shape)
            # transformer = PCA(n_components=n_components)
            scaler = StandardScaler()
            # feature=transformer.fit_transform(feature)
            feature = scaler.fit_transform(X=feature,y=None)
            #print(feature.shape)
            feat_x = resample(feature, 750, axis=0) #resample to 1hz for now 
            X.append(feat_x)
    elif feat_set=='audioset':
        from scipy.signal import resample
        X=[]
        features=['as_embed', 'as_scores']
        for f in features:
            feature=np.load(f'../data/features/DM_{f}.npy')
            #print(feature.shape)
            # transformer = PCA(n_components=n_components)
            scaler = StandardScaler()
            # feature=transformer.fit_transform(feature)
            feature = scaler.fit_transform(X=feature,y=None)
            #print(feature.shape)
            feat_x = resample(feature, 750, axis=0) #resample to 1hz for now 
            X.append(feat_x)
    elif feat_set=='manualhrf':
        features=['rms','chroma', 'mfcc', 'mfs', 'as_embed', 'as_scores']
        X=nat_asd_utils.load_audio_features_manual_hrf('DM',features)
    elif feat_set=='manualhrfpca1':
        features=['rms','chroma', 'mfcc', 'mfs', 'as_embed', 'as_scores']
        X=nat_asd_utils.load_audio_features_manual_hrf('DM',features)
        from sklearn.decomposition import PCA
        X_pca=[]
        transformer = PCA(n_components=1)
        for x in X:
            X_pca.append(   transformer.fit_transform(x)   ) 
        X=X_pca
    elif feat_set=='manualhrfpca10':
        features=['chroma', 'mfcc', 'mfs', 'as_embed', 'as_scores']
        X=nat_asd_utils.load_audio_features_manual_hrf('DM',features)
        from sklearn.decomposition import PCA
        X_pca=[]
        transformer = PCA(n_components=10)
        for x in X:
            X_pca.append(   transformer.fit_transform(x)   ) 
        X=X_pca
    elif feat_set=="cochresnet50":
        features=features_cochresnet
        X=nat_asd_utils.load_audio_features('DM',features)




    elif feat_set=="cochresnet50pca1hrfssfirst":
        from sklearn.decomposition import PCA
        features=features_cochresnet
        X_raw=nat_asd_utils.load_audio_features('DM',features)
        X=nat_asd_utils.standardscale(X_raw)
        X=nat_asd_utils.apply_pca(X, 1)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)

    elif feat_set=="cochresnet50pca20hrfssfirst":
        from sklearn.decomposition import PCA
        features=features_cochresnet
        X_raw=nat_asd_utils.load_audio_features('DM',features)
        X=nat_asd_utils.standardscale(X_raw)
        X=nat_asd_utils.apply_pca(X, 20)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)

    
    elif feat_set=="cochresnet50pca1hrf":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-mean_PCA-1.hdf5'
        X=nat_asd_utils.load_audio_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="cochresnet50pca1":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-mean_PCA-1.hdf5'
        X=nat_asd_utils.load_audio_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pca200":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-mean_PCA-200.hdf5'
        X=nat_asd_utils.load_audio_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pca5":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-mean_PCA-5.hdf5'
        X=nat_asd_utils.load_audio_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pca10":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-mean_PCA-10.hdf5'
        X=nat_asd_utils.load_audio_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pca50":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-mean_PCA-50.hdf5'
        X=nat_asd_utils.load_audio_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pca100":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-mean_PCA-100.hdf5'
        X=nat_asd_utils.load_audio_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pcafull1":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-full_PCA-1.hdf5'
        X=nat_asd_utils.load_audio_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pcafull200":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-full_PCA-200.hdf5'
        X=nat_asd_utils.load_audio_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pcafull5":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-full_PCA-5.hdf5'
        X=nat_asd_utils.load_audio_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pcafull10":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-full_PCA-10.hdf5'
        X=nat_asd_utils.load_audio_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pcafull50":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-full_PCA-50.hdf5'
        X=nat_asd_utils.load_audio_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pcafull100":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-full_PCA-100.hdf5'
        X=nat_asd_utils.load_audio_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pcac2":
        features=features_cochresnet
        X=nat_asd_utils.load_audio_features_PCAc2('DM',features)
    elif feat_set=="cochresnet50PCAlocal1":
        features=features_cochresnet_short
        feature_filename='DM_cochresnet50_activations-full_PCA-local-1.hdf5'
        X=nat_asd_utils.load_audio_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50PCAlocal10":
        features=features_cochresnet_short
        feature_filename='DM_cochresnet50_activations-full_PCA-local-10.hdf5'
        X=nat_asd_utils.load_audio_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50PCAlocal1mean":
        features=features_cochresnet_short
        feature_filename='DM_cochresnet50_activations-full_PCA-local-1_mean.hdf5'
        X=nat_asd_utils.load_audio_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50PCAlocal10mean":
        features=features_cochresnet_short
        feature_filename='DM_cochresnet50_activations-full_PCA-local-10_mean.hdf5'
        X=nat_asd_utils.load_audio_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50PCAlocal1rev":
        features=features_cochresnet_short
        feature_filename='DM_cochresnet50_activations-full_PCA-local-1_rev.hdf5'
        X=nat_asd_utils.load_audio_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50PCAlocal5rev":
        features=features_cochresnet_short
        feature_filename='DM_cochresnet50_activations-full_PCA-local-5_rev.hdf5'
        X=nat_asd_utils.load_audio_features_processed(feature_filename,features)

    return(X,features)


def plot_violins(r2s, stacked_r2s, S_average, features, output_name):
    plot_data=np.concatenate((r2s, stacked_r2s.reshape(1, -1)), axis=0).T
    fig, axs = plt.subplots(2, figsize=(7, 10))
    axs = axs.flatten()
    
    #plt.figure(figsize=(6,4))
    sns.violinplot(data=plot_data,ax=axs[0])
    #    plt.xticks(np.arange(24), ['Subject'+str(k+1) for k in range(24)])
    #plt.title(f'R2 for each feature space')
    axs[0].set_title(f'R2 for each feature space')
    
    #features=all_layers
    #features=['rms','chroma', 'mfcc', 'mfs', 'as_embed', 'as_scores']
    
    # Add 'stacked' to the end of your features list
    labels = features + ['stacked']
    
    # Set the xticklabels of your plot
    axs[0].set_xticks(range(len(labels)))
    axs[0].set_xticklabels(labels,rotation=45)
    
    plt.figure(figsize=(6,4))
    sns.violinplot(data=S_average,ax=axs[1])
    #    plt.xticks(np.arange(24), ['Subject'+str(k+1) for k in range(24)])
    #plt.title(f'Violin plot of the subjects by delay for brain region {parcel_names[j]}')
    labels = features
    axs[1].set_title(f'Stacking weights')
    
    # Set the xticklabels of your plot
    axs[1].set_xticks(range(len(labels)))
    axs[1].set_xticklabels(labels,rotation=45)

    fig.savefig(f'../plots/pilots/{output_name}.png')

def load_sub_brain(sub,delay,atlas_indices_indices):    
    im_file = f'/nese/mit/group/sig/projects/hbn/hbn_bids/derivatives/xcp_d_0.7.1/sub-{sub}/ses-HBNsiteRU/func/sub-{sub}_ses-HBNsiteRU_task-movieDM_space-fsLR_den-91k_desc-denoisedSmoothed_bold.dtseries.nii'
    img = nb.load(im_file)
    img_y = img.get_fdata()
    Y=img_y[delay:,atlas_indices_indices]
    print(f'loaded brain data')
    return(Y)
    
def load_sub_brain_all(sub,delay):    
    im_file = f'/nese/mit/group/sig/projects/hbn/hbn_bids/derivatives/xcp_d_0.7.1/sub-{sub}/ses-HBNsiteRU/func/sub-{sub}_ses-HBNsiteRU_task-movieDM_space-fsLR_den-91k_desc-denoisedSmoothed_bold.dtseries.nii'
    img = nb.load(im_file)
    img_y = img.get_fdata()
    Y=img_y[delay:,:]
    print(f'loaded brain data')
    return(Y)



def extract_parcels(parcels):
    atlas,atlas_data=nat_asd_utils.load_glasser()
    atlas_indices,indices,parcel_names=nat_asd_utils.get_parcel_indices(atlas,parcels)
    atlas_indices_indices = np.where(np.isin(atlas_data, atlas_indices))[0]
    print(f'loaded parcels {parcels}')
    return(atlas_indices_indices)

def select_parcels(parcel_selection):
    #parcels: auditory, visual, audiovideo, all, custom
    if parcel_selection == 'auditory':
        parcels=[
        'A1',
        'LBelt',
        'MBelt',
        'PBelt',
        'A4',
        'TA2',
        'A5']
    elif parcel_selection == 'visual':
        parcels=[
                'V1',
                'V2',
                'V3',
                'V4',
                'MT',
                'MST',
                'V4t',
                'FST',
                'FFC',
                'V8',
                'PIT',
                'VVC',
                'VMV1',
                'VMV2',
                'VMV3',
                'V3A',
                'V3B',
                'V6',
                'V6A',
                'V7']
    elif parcel_selection == 'audiovisual':
        parcels=[
                'IPS1',
                'IFSa',
                'IFSp',
                'IFJa',
                'IFJp',
                'FEF',
                'STSvp',
                'STSdp',
                'STSva',
                'STSda',
                'STGa',
                'STV',
                'TPOJ1',
                'TPOJ2',
                'TPOJ3']
    elif parcel_selection == 'all_select':
        parcels=[
                'V1',
                'V2',
                'V3',
                'V4',
                'MT',
                'MST',
                'V4t',
                'FST',
                'FFC',
                'V8',
                'PIT',
                'VVC',
                'VMV1',
                'VMV2',
                'VMV3',
                'V3A',
                'V3B',
                'V6',
                'V6A',
                'V7',
                'IPS1',
                'IFSa',
                'IFSp',
                'IFJa',
                'IFJp',
                'FEF',
                'STSvp',
                'STSdp',
                'STSva',
                'STSda',
                'STGa',
                'STV',
                'TPOJ1',
                'TPOJ2',
                'TPOJ3',
                'A1',
                'LBelt',
                'MBelt',
                'PBelt',
                'A4',
                'TA2',
                'A5']
    return(parcels)

if __name__ == "__main__":
    main()
