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
script to run pilot encoding models
Standard usage: python pilot.py -s NDARHJ830RXD -p auditory -f cochresnet50pca1 -d 7 -l
"""
# Inputs:
# - subject: subject id from hbn
# - parcels: a subset of the MMP parcels eg: audio, video, audiovideo, all, custom[custom not implemented yet]
# - features: the features to predict brain data from.
# - delay: length in TRs (0.8s) to account for HRF. If using an hrf feature, set to 0
# - bootstrap: the permutation count. if a number is given here it will randomly permute features and add the number to the output filename
# - plot: to plot summary figures or not.
# - zscore: to apply zscore before regression

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
    parser.add_argument("-f", "--features", type=str, help="eg cochresnet50, cochresnet50pca1, cochresnet50pca200, manual", required=True)
    parser.add_argument("-d", "--delay", type=int, help="parcels: audio, video, audiovideo, all, custom", required=True)
    parser.add_argument("-b", "--bootstrap", type=int, help="bootstrap: which permutation it is", default=None)
    parser.add_argument('-l', '--plot', help="to make a plot or not", action='store_true')  # on/off flag
    parser.add_argument('-z', '--zscore', help="to zscore or not", action='store_true')  # on/off flag
    parser.add_argument('-y', '--zscorey', help="to zscore brain data or not", action='store_true')  # on/off flag
    parser.add_argument('-g', '--himalaya', help="to run group banded regression from himalaya instead of stacked regression", action='store_true')  # on/off flag
    parser.add_argument('-r', '--ridgecv', help="to run simple ridgecv", action='store_true')  # on/off flag
    parser.add_argument('-e', '--elasticnetcv', help="to run simple elasticnetcv", action='store_true')  # on/off flag
    parser.add_argument('-a', '--lassocv', help="to run simple elasticnetcv", action='store_true')  # on/off flag

    parser.add_argument('-t', '--friendstask', help="the friends task if doing friends", default=None)  # on/off flag

    args = parser.parse_args()

    sub=args.subject[0]
    delay=args.delay

    unique_name=f'sub-{sub}_roi-{args.parcels}_feat-{args.features}_delay-{delay}' # for filename saving output
    print(f'running subject {sub}')

    if args.parcels=='all':
        Y=load_sub_brain_all(sub,delay) #load the whole brain
        parcels='all'
    else:
        parcels=select_parcels(args.parcels) # load parcel set
        atlas_indices_indices=extract_parcels(parcels) # get indices of parcels
        if args.friendstask is None:
            Y=load_sub_brain(sub,delay,atlas_indices_indices) #load brain data from selected parcels
        else:
            Y=load_sub_brain_friends(sub,args.friendstask,delay,atlas_indices_indices)
            unique_name=unique_name+'_friends'
    X,features=load_features(args.features) #load X
    if args.zscorey:
        from scipy.stats import zscore
        print('zscoring brain data')    
        Y=zscore(Y)
        unique_name=unique_name+'_yz'

    if args.ridgecv:
        print('run ridgecv') #skip the array shaping here and do differently later
    elif args.elasticnetcv:
        print('run elasticnetcv') #skip the array shaping here and do differently later
    elif args.lassocv:
        print('run lassocv') #skip the array shaping here and do differently later
    else:
        X = [array[:Y.shape[0], :] for array in X] #trim X features to the same length as the Y brain data since sometimes the run was cut short
        if args.zscore:
            X=nat_asd_utils.apply_zscore(X)
            unique_name = unique_name + f'_z'
        Y=Y[:X[0].shape[0],:]

    if args.bootstrap is None:
        print("No value was passed to args.bootstrap")
    else:
        print(f"The value passed to args.bootstrap is {args.bootstrap}, randomly permuting features X")
        for i in range(len(X)):
            np.random.shuffle(X[i])
        unique_name = unique_name + f'_bootstrap-{args.bootstrap}'

    if args.himalaya:
        #run himalaya banded regression
        from himalaya.ridge import GroupRidgeCV
        from stacking_fmri import get_cv_indices

        unique_name = unique_name + f'_himalaya'

        n_time=Y.shape[0]
        n_folds=5
        ind = get_cv_indices(n_time, n_folds=n_folds)
        data=np.copy(Y)
        #features=np.copy(X)
        feats=X
        r2_list=[]
        coef_list=[]
        for ind_num in range(n_folds):
            # split data into training and testing sets
            train_ind = ind != ind_num
            test_ind = ind == ind_num
            train_data = data[train_ind]
            train_features = [F[train_ind] for F in feats]
            test_data = data[test_ind]
            test_features = [F[test_ind] for F in feats]

            banded_ridge= GroupRidgeCV(groups="input",cv=5)
            banded_ridge.fit(train_features, train_data)
            score = banded_ridge.score(test_features, test_data)
            print("R^2 Score: ", np.mean(score))
            r2_list.append(score)
            coef_list.append(banded_ridge.coef_)
        elapsed_time=time.time() - start_time
        print(elapsed_time)
        
        print(f'saving results')
        
        S_average=np.mean(coef_list,axis=0)
        banded_r2s=np.mean(r2_list,axis=0)

        binary_parcels = [np.void(s.encode('utf-8')) for s in parcels]
        binary_features = [np.void(s.encode('utf-8')) for s in features]
        output_directory_name='good_pilots'
        np.savez(f'../{output_directory_name}/{unique_name}', banded_r2s=banded_r2s, S_average=S_average, elapsed_time=elapsed_time, binary_parcels=binary_parcels, binary_features=binary_features)


    elif args.ridgecv:
        from stacking_fmri import get_cv_indices
        from sklearn.linear_model import RidgeCV

        unique_name = unique_name + f'_ridgecv'

        X = X[:,:Y.shape[0]]
        Y= Y[:X.shape[1],:]
        
        #trim first 20 TRs
        X = X.T[20:,:]
        Y= Y[20:,:]
        
        n_time=Y.shape[0]
        n_folds=5
        ind = get_cv_indices(n_time, n_folds=n_folds)
        data=np.copy(Y)
        feats=np.copy(X)
        
        test_r2_list=[]
        train_r2_list=[]
        coef_list=[]
        for ind_num in range(n_folds):
            # split data into training and testing sets
            train_ind = ind != ind_num
            test_ind = ind == ind_num
            train_data = data[train_ind]
            train_features = feats[train_ind]#[F[train_ind] for F in features]
            test_data = data[test_ind]
            test_features = feats[test_ind]#[F[test_ind] for F in features]
        
            ridge=RidgeCV()
            ridge.fit(train_features, train_data)
            test_score = ridge.score(test_features, test_data)
            train_score= ridge.score(train_features, train_data)
            print(f"fold {ind_num} test R^2 Score: ", format(np.mean(test_score), '.2f'))
            print(f"fold {ind_num} train R^2 Score: ", format(np.mean(train_score), '.2f'))
        
            test_r2_list.append(test_score)
            train_r2_list.append(train_score)
        elapsed_time=time.time() - start_time
        print(elapsed_time)
        
        print(f'saving results')
        print("MEAN test R^2 Score: ", format(np.mean(test_r2_list), '.2f'))
        print("MEAN train R^2 Score: ", format(np.mean(train_r2_list), '.2f'))
        binary_parcels = [np.void(s.encode('utf-8')) for s in parcels]
        binary_features = [np.void(s.encode('utf-8')) for s in features]
        output_directory_name='good_pilots'
        np.savez(f'../{output_directory_name}/{unique_name}', test_r2_list=test_r2_list, train_r2_list=train_r2_list, elapsed_time=elapsed_time, binary_parcels=binary_parcels, binary_features=binary_features)

    elif args.elasticnetcv:
        from stacking_fmri import get_cv_indices
        from sklearn.linear_model import MultiTaskElasticNetCV

        unique_name = unique_name + f'_elasticnetcv'

        X = X[:,:Y.shape[0]]
        Y= Y[:X.shape[1],:]
        
        #trim first 20 TRs
        X = X.T[20:,:]
        Y= Y[20:,:]
        
        n_time=Y.shape[0]
        n_folds=5
        ind = get_cv_indices(n_time, n_folds=n_folds)
        data=np.copy(Y)
        feats=np.copy(X)
        
        test_r2_list=[]
        train_r2_list=[]
        coef_list=[]
        for ind_num in range(n_folds):
            # split data into training and testing sets
            train_ind = ind != ind_num
            test_ind = ind == ind_num
            train_data = data[train_ind]
            train_features = feats[train_ind]#[F[train_ind] for F in features]
            test_data = data[test_ind]
            test_features = feats[test_ind]#[F[test_ind] for F in features]
        
            elasticnet=MultiTaskElasticNetCV()
            elasticnet.fit(train_features, train_data)
            test_score = elasticnet.score(test_features, test_data)
            train_score= elasticnet.score(train_features, train_data)
            print(f"fold {ind_num} test R^2 Score: ", format(np.mean(test_score), '.2f'))
            print(f"fold {ind_num} train R^2 Score: ", format(np.mean(train_score), '.2f'))
        
            test_r2_list.append(test_score)
            train_r2_list.append(train_score)
        elapsed_time=time.time() - start_time
        print(elapsed_time)
        
        print(f'saving results')
        print("MEAN test R^2 Score: ", format(np.mean(test_r2_list), '.2f'))
        print("MEAN train R^2 Score: ", format(np.mean(train_r2_list), '.2f'))
        binary_parcels = [np.void(s.encode('utf-8')) for s in parcels]
        binary_features = [np.void(s.encode('utf-8')) for s in features]
        output_directory_name='good_pilots'
        np.savez(f'../{output_directory_name}/{unique_name}', test_r2_list=test_r2_list, train_r2_list=train_r2_list, elapsed_time=elapsed_time, binary_parcels=binary_parcels, binary_features=binary_features)


    elif args.lassocv:
        from stacking_fmri import get_cv_indices
        #from sklearn.linear_model import MultiTaskLassoCV
        from sklearn.linear_model import LassoCV
        unique_name = unique_name + f'_lassocv'

        X = X[:,:Y.shape[0]]
        Y= Y[:X.shape[1],:]
        
        #trim first 20 TRs
        X = X.T[20:,:]
        Y= Y[20:,:]
        
        n_time=Y.shape[0]
        n_folds=5
        ind = get_cv_indices(n_time, n_folds=n_folds)
        data=np.copy(Y)
        feats=np.copy(X)
        
        test_r2_list=[]
        train_r2_list=[]
        coef_list=[]
        for ind_num in range(n_folds):
            # split data into training and testing sets
            train_ind = ind != ind_num
            test_ind = ind == ind_num
            train_data = data[train_ind]
            train_features = feats[train_ind]#[F[train_ind] for F in features]
            test_data = data[test_ind]
            test_features = feats[test_ind]#[F[test_ind] for F in features]
            test_r2_list_list=[]
            train_r2_list_list=[]
            for i in range(Y.shape[1]):
                lasso = LassoCV(max_iter=10000,tol=0.001)
                lasso.fit(train_features, train_data[:, i])
                test_score = lasso.score(test_features, test_data[:, i])
                train_score= lasso.score(train_features, train_data[:, i])
                # print(f"fold {ind_num} test R^2 Score: ", format(np.mean(test_score), '.2f'))
                # print(f"fold {ind_num} train R^2 Score: ", format(np.mean(train_score), '.2f'))
                test_r2_list_list.append(test_score)
                train_r2_list_list.append(train_score)
            test_r2_list.append(np.asanyarray(test_r2_list_list))
            train_r2_list.append(np.asanyarray(train_r2_list_list))
        elapsed_time=time.time() - start_time
        print(elapsed_time)
        
        print(f'saving results')
        print("MEAN test R^2 Score: ", format(np.mean(test_r2_list), '.2f'))
        print("MEAN train R^2 Score: ", format(np.mean(train_r2_list), '.2f'))
        binary_parcels = [np.void(s.encode('utf-8')) for s in parcels]
        binary_features = [np.void(s.encode('utf-8')) for s in features]
        output_directory_name='good_pilots'
        np.savez(f'../{output_directory_name}/{unique_name}', test_r2_list=test_r2_list, train_r2_list=train_r2_list, elapsed_time=elapsed_time, binary_parcels=binary_parcels, binary_features=binary_features)

    else:
        #run stacked regression
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
    features_slowfast=['blocks.1_fast',
                    'blocks.1_slow',
                    'blocks.2_fast',
                    'blocks.2_slow',
                    'blocks.3_fast',
                    'blocks.3_slow',
                    'blocks.4_fast',
                    'blocks.4_slow',
                    'blocks.5',
                    'blocks.6']
    features_resnet=['relu','maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
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
    elif feat_set=='concatspeech':
        X=[]
        feature_data=[]
        X1,features=load_features('cochresnet50pca1hrfssfirst')
        X1 = [x[:,0] for x in X1]
        X.append(X1[4])
        X.append(X1[5])
        
        X1,feats=load_features('manualhrfpca10')
        X.append(X1[3][:-1,1])
        features.append('as_embed_pca2')
        X1,feats=load_features('audioset')
        Xx = [x[:-1,:] for x in X1]
        for xx in Xx:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
        
        X.append(Xx[1][:,0])
        X.append(Xx[1][:,132])
        features.append('as-Speech')
        features.append('as-Music')
        X=np.asanyarray(X)

    elif feat_set=='cochresnet50mean_input_after_preproc_hrf':
        features=['input_after_preproc']
        X=nat_asd_utils.load_audio_features('DM',features)[0]
        hz=X.shape[0]/600
        hrf_tools.apply_optimal_hrf_10hz(X,hz)
    elif feat_set=='cochresnet50mean_conv1_relu1_hrf':
        features=['conv1_relu1']
        X=nat_asd_utils.load_audio_features('DM',features)[0]
        hz=X.shape[0]/600
        hrf_tools.apply_optimal_hrf_10hz(X,hz)
    elif feat_set=='cochresnet50mean_maxpool1_hrf':
        features=['maxpool1']
        X=nat_asd_utils.load_audio_features('DM',features)[0]
        hz=X.shape[0]/600
        hrf_tools.apply_optimal_hrf_10hz(X,hz)
    elif feat_set=='cochresnet50mean_layer1_hrf':
        features=['layer1']
        X=nat_asd_utils.load_audio_features('DM',features)[0]
        hz=X.shape[0]/600
        hrf_tools.apply_optimal_hrf_10hz(X,hz)
    elif feat_set=='cochresnet50mean_layer2_hrf':
        features=['layer2']
        X=nat_asd_utils.load_audio_features('DM',features)[0]
        hz=X.shape[0]/600
        hrf_tools.apply_optimal_hrf_10hz(X,hz)
    elif feat_set=='cochresnet50mean_layer3_hrf':
        features=['layer3']
        X=nat_asd_utils.load_audio_features('DM',features)[0]
        hz=X.shape[0]/600
        hrf_tools.apply_optimal_hrf_10hz(X,hz)
    elif feat_set=='cochresnet50mean_layer4_hrf':
        features=['layer4']
        X=nat_asd_utils.load_audio_features('DM',features)[0]
        hz=X.shape[0]/600
        hrf_tools.apply_optimal_hrf_10hz(X,hz)
    elif feat_set=='cochresnet50mean_avgpool_hrf':
        features=['avgpool']
        X=nat_asd_utils.load_audio_features('DM',features)[0]
        hz=X.shape[0]/600
        hrf_tools.apply_optimal_hrf_10hz(X,hz)

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
    elif feat_set=="cochresnet50pca10hrfssfirst":
        from sklearn.decomposition import PCA
        features=features_cochresnet
        X_raw=nat_asd_utils.load_audio_features('DM',features)
        X=nat_asd_utils.standardscale(X_raw)
        X=nat_asd_utils.apply_pca(X, 10)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)            
    elif feat_set=="cochresnet50pca100hrfssfirst":
        from sklearn.decomposition import PCA
        features=features_cochresnet
        X_raw=nat_asd_utils.load_audio_features('DM',features)
        X=nat_asd_utils.standardscale(X_raw)
        X=nat_asd_utils.apply_pca(X, 100)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    
    
    elif feat_set=="cochresnet50srp05hrfssfirst":
        from sklearn.decomposition import PCA
        features=features_cochresnet
        X_raw=nat_asd_utils.load_audio_features('DM',features)
        X=nat_asd_utils.standardscale(X_raw)
        X=nat_asd_utils.apply_srp(X,0.5)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="cochresnet50srp01hrfssfirst":
        from sklearn.decomposition import PCA
        features=features_cochresnet
        X_raw=nat_asd_utils.load_audio_features('DM',features)
        X=nat_asd_utils.standardscale(X_raw)
        X=nat_asd_utils.apply_srp(X,0.1)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)

    elif feat_set=="cochresnet50pca1hrf":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-mean_PCA-1.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="resnet50pca1hrf":
        features=features_resnet
        feature_filename='DM_resnet50_activations-PCA-1.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="resnet50pca5hrf":
        features=features_resnet
        feature_filename='DM_resnet50_activations-PCA-5.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="resnet50pca10hrf":
        features=features_resnet
        feature_filename='DM_resnet50_activations-PCA-10.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz) 
    elif feat_set=="resnet50pca50hrf":
        features=features_resnet
        feature_filename='DM_resnet50_activations-PCA-50.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz) 
    elif feat_set=="resnet50pca100hrf":
        features=features_resnet
        feature_filename='DM_resnet50_activations-PCA-100.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz) 
    elif feat_set=="resnet50pca200hrf":
        features=features_resnet
        feature_filename='DM_resnet50_activations-PCA-200.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz) 
    elif feat_set=="resnet50pca500hrf":
        features=features_resnet
        feature_filename='DM_resnet50_activations-PCA-500.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)

    elif feat_set=="cochresnet50pca1hrffriends_s01e02a":
        features=features_cochresnet
        feature_filename='friends_s01e02a_cochresnet50_activations-mean_PCA-1.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/703
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="cochresnet50pca10hrffriends_s01e02a":
        features=features_cochresnet
        feature_filename='friends_s01e02a_cochresnet50_activations-mean_PCA-10.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/703
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="cochresnet50pca100hrffriends_s01e02a":
        features=features_cochresnet
        feature_filename='friends_s01e02a_cochresnet50_activations-mean_PCA-100.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/703
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="cochresnet50pca1hrffriends_s01e02b":
        features=features_cochresnet
        feature_filename='friends_s01e02b_cochresnet50_activations-mean_PCA-1.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/703
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="cochresnet50pca10hrffriends_s01e02b":
        features=features_cochresnet
        feature_filename='friends_s01e02b_cochresnet50_activations-mean_PCA-10.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/703
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="cochresnet50pca100hrffriends_s01e02b":
        features=features_cochresnet
        feature_filename='friends_s01e02b_cochresnet50_activations-mean_PCA-100.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/703
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)

    elif feat_set=="cochresnet50srp01hrffriends_s01e02a":
        features=features_cochresnet
        X=nat_asd_utils.load_audio_features_SRP('friends_s01e02a',features,0.1)
        for xx in X:
            hz=xx.shape[0]/703
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="cochresnet50srp01hrffriends_s01e02b":
        features=features_cochresnet
        X=nat_asd_utils.load_audio_features_SRP('friends_s01e02b',features,0.1)
        for xx in X:
            hz=xx.shape[0]/703
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)    
    elif feat_set=="cochresnet50srp05hrffriends_s01e02a":
        features=features_cochresnet
        X=nat_asd_utils.load_audio_features_SRP('friends_s01e02a',features,0.5)
        for xx in X:
            hz=xx.shape[0]/703
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="cochresnet50srp05hrffriends_s01e02b":
        features=features_cochresnet
        X=nat_asd_utils.load_audio_features_SRP('friends_s01e02b',features,0.5)
        for xx in X:
            hz=xx.shape[0]/703
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)    
    
    elif feat_set=="video_resnet50pca1hrf":
        features=features_resnet
        feature_filename='DM_videos_resnet50-PCA-1.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="video_resnet50pca10hrf":
        features=features_resnet
        feature_filename='DM_videos_resnet50-PCA-10.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="video_resnet50pca100hrf":
        features=features_resnet
        feature_filename='DM_videos_resnet50-PCA-100.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="video_slowfastr50pca1hrf":
        features=features_slowfast
        feature_filename='DM_videos_slowfast_r50-PCA-1.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="video_slowfastr50pca10hrf":
        features=features_slowfast
        feature_filename='DM_videos_slowfast_r50-PCA-10.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="video_slowfastr50pca100hrf":
        features=features_slowfast
        feature_filename='DM_videos_slowfast_r50-PCA-100.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
        for xx in X:
            hz=xx.shape[0]/600
            hrf_tools.apply_optimal_hrf_10hz(xx,hz)
    elif feat_set=="cochresnet50pca1":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-mean_PCA-1.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pca200":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-mean_PCA-200.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pca5":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-mean_PCA-5.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pca10":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-mean_PCA-10.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pca50":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-mean_PCA-50.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pca100":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-mean_PCA-100.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)

    elif feat_set=="cochresnet50pca1friends_s01e02a":
        features=features_cochresnet
        feature_filename='friends_s01e02a_cochresnet50_activations-mean_PCA-1.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pca10friends_s01e02a":
        features=features_cochresnet
        feature_filename='friends_s01e02a_cochresnet50_activations-mean_PCA-10.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pca100friends_s01e02a":
        features=features_cochresnet
        feature_filename='friends_s01e02a_cochresnet50_activations-mean_PCA-100.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pca1friends_s01e02b":
        features=features_cochresnet
        feature_filename='friends_s01e02b_cochresnet50_activations-mean_PCA-1.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pca10friends_s01e02b":
        features=features_cochresnet
        feature_filename='friends_s01e02b_cochresnet50_activations-mean_PCA-10.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pca100friends_s01e02b":
        features=features_cochresnet
        feature_filename='friends_s01e02b_cochresnet50_activations-mean_PCA-100.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)

    
    elif feat_set=="cochresnet50pcafull1":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-full_PCA-1.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pcafull200":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-full_PCA-200.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pcafull5":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-full_PCA-5.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pcafull10":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-full_PCA-10.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pcafull50":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-full_PCA-50.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pcafull100":
        features=features_cochresnet
        feature_filename='DM_cochresnet50_activations-full_PCA-100.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50pcac2":
        features=features_cochresnet
        X=nat_asd_utils.load_audio_features_PCAc2('DM',features)
    elif feat_set=="cochresnet50PCAlocal1":
        features=features_cochresnet_short
        feature_filename='DM_cochresnet50_activations-full_PCA-local-1.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50PCAlocal10":
        features=features_cochresnet_short
        feature_filename='DM_cochresnet50_activations-full_PCA-local-10.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50PCAlocal1mean":
        features=features_cochresnet_short
        feature_filename='DM_cochresnet50_activations-full_PCA-local-1_mean.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50PCAlocal10mean":
        features=features_cochresnet_short
        feature_filename='DM_cochresnet50_activations-full_PCA-local-10_mean.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50PCAlocal1rev":
        features=features_cochresnet_short
        feature_filename='DM_cochresnet50_activations-full_PCA-local-1_rev.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="cochresnet50PCAlocal5rev":
        features=features_cochresnet_short
        feature_filename='DM_cochresnet50_activations-full_PCA-local-5_rev.hdf5'
        X=nat_asd_utils.load_features_processed(feature_filename,features)
    elif feat_set=="both_hrf":
        X=nat_asd_utils.load_both_features_hrf('DM')
        features=['input_after_preproc',
                  'conv1_relu1',
                  'layer1',
                  'layer2',
                  'layer3',
                  'layer4',
                  'avgpool',
                  'as_embed', 
                  'as_scores']
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


def load_sub_brain_friends(sub,task,delay,atlas_indices_indices):        
    fmriprep_folder='/nese/mit/group/sig/projects/cneuromod/friends/postproc_fmriprep'
    pattern=f'{fmriprep_folder}/sub-{sub}/sub-{sub}_ses-*_task-{task}_space-fsLR_den-91k_bold_smoothed.dtseries.nii'
    im_file = glob.glob(pattern)[0]
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
    elif parcel_selection == 'a4a5':
        parcels=[
        'A4',
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
    elif parcel_selection == 'earlyvisual':
        parcels=[
                'V1',
                'V2',
                'V3',
                'V4',
                'MT']

    elif parcel_selection == 'ventralvisual':
        parcels=[
                'FFC',
                'V8',
                'PIT',
                'VVC',
                'VMV1',
                'VMV2',
                'VMV3']
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
