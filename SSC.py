import h5py
import scipy
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import metrics
from math import sqrt
from scipy import linalg
import time
from sklearn.cluster import SpectralClustering

seed = 78638924
np.random.seed(seed)

if __name__ == "__main__":
    #Setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('dataset', default='usps', choices=[, 'eyaleb', 'arfaces', 'coil20'])
    parser.add_argument('--n_clusters', default=10, type=int)
    args = parser.parse_args()
    print(args)

    n_clusters = args.n_clusters

    #Loading dataset
    from datasets import load_processed_eyaleb, load_arfaces, load_coil20
    elif args.dataset == 'eyaleb':
        x, y = load_processed_eyaleb()
    elif args.dataset == 'coil20':
        x, y = load_coil20()
    elif args.dataset == 'arfaces':
        x, y = load_arfaces()
        
    print("X shape ",x.shape)
    print("Y shape ",y.shape) 
    x=np.transpose(x)
    nsamples = x.shape[1]

    import time
    start_time = time.time()

    print("Calculating C by subspace clustering for X")
    tau = 0.1   
    clist = []
    for j in range(nsamples):
        x_j = x[:,j]
        x_i_c = x.copy()
        x_i_c[:,j] = 0
        c_i,resid,grad = ista(x_i_c,x_j)
        clist.append(c_i)    
        if(j%500==0):
            print("Value of j ",j)   
    C = np.concatenate(clist, axis=0).reshape(nsamples,nsamples).transpose()
    idx = np.arange(nsamples)
    C[idx,idx]=0
    print("C shape ",C.shape)

    #Computing NMI and ARI
    C_aff = np.add(C,np.transpose(C))
    clustering = SpectralClustering(assign_labels='discretize', n_clusters=args.n_clusters,random_state=0,affinity='precomputed').fit(C_aff)
    x_pred = clustering.labels_
    print('nmi = %.4f, ari = %.4f' % (metrics.nmi(y, x_pred), metrics.ari(y, x_pred)))
    print("Confusion Matrix ",confusion_matrix(y,x_pred))

    print("--- %s seconds ---" % (time.time() - start_time))