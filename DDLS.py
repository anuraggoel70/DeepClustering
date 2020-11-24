# # USPS Dataset (16*16 images 7291 train images 2007 test images) 256*7291
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
from datasets import load_mnist, load_usps
import matplotlib.pyplot as plt
import metrics
#from spgl1 import spgl1, spg_lasso
from math import sqrt
from scipy import linalg
import time
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

seed = 78638924
np.random.seed(seed)

def soft_thresh(x, l):
    return np.sign(x) * np.maximum(np.abs(x) - l, 0.)

def ista(A, b, l=0.5, maxit=1):
    x = np.zeros(A.shape[1])
    pobj = []
    L = linalg.norm(A, ord=2) ** 2  # Lipschitz constant
    time0 = time.time()
    for _ in range(maxit):
        x = soft_thresh(x + np.dot(A.T, b - A.dot(x)) / L, l / L)
        this_pobj = 0.5 * linalg.norm(A.dot(x) - b) ** 2 + l * linalg.norm(x, 1)
        pobj.append((time.time() - time0, this_pobj))

    times, pobj = map(np.array, zip(*pobj))
    return x, pobj, times

def findclusterlabels(x, y, n_clusters):
    x = np.transpose(x) # 256*7291
    nsamples = x.shape[1]
    # Initialization of parameters
    k1=512
    k2=256
    k3=128
    num_input_features = x.shape[0]
    num_samples = x.shape[1]
    n = 11 #number of iterations
    mu = 0.5

    d1 = np.random.randn(num_input_features,k1)
    d2 = np.random.randn(k1,k2)
    d3 = np.random.randn(k2,k3)
    z = np.random.randn(k3,num_samples)

    # H initialization
    C = np.random.randn(nsamples,nsamples)
    idx = np.arange(nsamples)
    C[idx,idx] = 0

    cost_list=[]
    cost_list1=[]
    cost_list2=[]
    d1_list = []
    d2_list = []
    d3_list = []
    z_list = []
    
    z[z<0]=0

    bestnmi = 0
    bestari = 0
    bestsilhoutti = -2
    for i in range(n):
        print("Calculating d1 d2 d3")
        #Solving d1,d2,d3 via pseudo-inverse
        P = np.dot(d3,z)
        Q = np.dot(d2,P)

        P[P<0]=0
        Q[Q<0]=0
        
        #Saving old d1,d2,d3 values
        d1_pre = d1
        d2_pre = d2
        d3_pre = d3
        z_pre = z

        d1 = np.dot(x,(np.linalg.pinv(Q)))  #d1 shape num_input_features*k1

        d2 = np.dot((np.linalg.pinv(d1)),np.dot(x,np.linalg.pinv(P))) # d2 shape k1*k2
        #d2[d2<0]=0

        W = np.linalg.pinv(np.dot(d1,d2))
        d3 = np.dot(W, np.dot(x, np.linalg.pinv(z))) #d3 shape k2*k3
        #d3[d3<0]=0

        #Calculating error of d1,d2,d3
        d1_error = np.linalg.norm((d1 - d1_pre), 'fro')
        d2_error = np.linalg.norm((d2 - d2_pre), 'fro')
        d3_error = np.linalg.norm((d3 - d3_pre), 'fro')

        print("Solving for z...")
        #Solving z via sylvester equation of form ax+xb=c
        a = np.dot((np.transpose(np.dot(d1,np.dot(d2,d3)))),(np.dot(d1,np.dot(d2,d3))))
        b = mu*(np.dot(np.transpose(np.identity(C.shape[0])-np.transpose(C)),(np.identity(C.shape[0])-C)))
        c = np.dot((np.transpose(np.dot(d1,np.dot(d2,d3)))),x)
        print("Calculating z")
        z = scipy.linalg.solve_sylvester(a, b, c)

        z[z<0]=0

        z_error = np.linalg.norm((z - z_pre), 'fro')

        print("Calculating C by subspace clustering")
        tau = 0.1   
        clist = []
        for j in range(nsamples):
            z_j = z[:,j]
            z_i_c = z.copy()
            z_i_c[:,j] = 0
            #c_i,resid,grad,info = spg_lasso(z_i_c, z_j, tau, verbosity=0)
            c_i,resid,grad = ista(z_i_c,z_j)
            clist.append(c_i)
        C = np.concatenate(clist, axis=0).reshape(nsamples,nsamples).transpose()
        idx = np.arange(nsamples)
        C[idx,idx]=0
        print("C shape ",C.shape)

        #Computing ARI and NMI
        C_aff = np.add(C,np.transpose(C))
        clustering = SpectralClustering(assign_labels='discretize', n_clusters=n_clusters,random_state=0,affinity='precomputed').fit(C_aff)
        z_pred = clustering.labels_

        nmi_cur = metrics.nmi(y, z_pred)
        ari_cur = metrics.ari(y, z_pred)


        print('acc = %.4f, nmi = %.4f, ari = %.4f' % (metrics.acc(y, z_pred), metrics.nmi(y, z_pred), metrics.ari(y, z_pred)))
        print("Confusion Matrix ",confusion_matrix(y,z_pred))

        cursilhoutti = silhouette_score(x.T, z_pred)

        if bestnmi < nmi_cur:
            bestnmi = nmi_cur
        if bestari < ari_cur:
            bestari = ari_cur
        if cursilhoutti > bestsilhoutti:
            bestsilhoutti = cursilhoutti

        print("Calculating cost")
        #Cost Calculation
        z_cost = z
        P = np.dot(d3,z_cost)
        P[P<0]=0
        Q = np.dot(d2,P)
        Q[Q<0]=0
        cost1 = (x - (np.dot(d1,Q)))

        result1 = np.sum(cost1**2)
        cost1f = np.linalg.norm(cost1, 'fro')
        cost1f = np.square(cost1f)

        cost2 = (z - (np.dot(z,C)))
        cost2f = np.linalg.norm(cost2, 'fro')
        cost2f = np.square(cost2f)

        costf = cost1f + (mu*cost2f) + np.sum(C)

        if (i>0):
            cost_list.append(costf)
            cost_list1.append(cost1f)
            cost_list2.append(cost2f)
            d1_list.append(d1_error)
            d2_list.append(d2_error)
            d3_list.append(d3_error)
            z_list.append(z_error)

        print("Result 1 : ", result1)
        print("Cost 1 : ", cost1f)
        print("Cost 2 : ", cost2f)
        print("Final Cost ", costf)
        print("End of %d iteration",i)
        print("==================================")
    print("cost list 1 : ",cost_list1)
    print("cost list 2 : ",cost_list2)
    print("final cost list : ",cost_list)
    print("d1 list : ",d1_list)
    print("d2 list : ",d2_list)
    print("d3 list : ",d3_list)
    print("Z list : ",z_list)
    print('bestnmi = %.4f, bestari = %.4f', bestnmi, bestari)
    print('best sillhouti = %.4f', bestsilhoutti)

    #Plotting cost function
    fig = plt.figure()
    plt.plot(list(range(1,n)), cost_list1, '-r')
    fig.suptitle("Error Cost 1 Plot ||X-D1D2D3Z||")
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost')
    fig.savefig('Cost1.png')

    fig = plt.figure()
    plt.plot(list(range(1,n)), cost_list2, '-b')
    fig.suptitle("Error Cost 2 Plot ||Z-Z.H'(H.H')-1.H||")
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost')
    fig.savefig('Cost2.png')

    fig = plt.figure()
    plt.plot(list(range(1,n)), cost_list, '-r')
    fig.suptitle("DDLS")
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    fig.savefig('FinalCost.png')

    fig = plt.figure()
    plt.plot(list(range(1,n)), d1_list, '-b')
    fig.suptitle("D1 Loss Plot")
    plt.xlabel('No. of iterations')
    plt.ylabel('D1 Loss')
    fig.savefig('d1loss.png')

    fig = plt.figure()
    plt.plot(list(range(1,n)), d2_list, '-b')
    fig.suptitle("D2 Loss Plot")
    plt.xlabel('No. of iterations')
    plt.ylabel('D2 Loss')
    fig.savefig('d2loss.png')

    fig = plt.figure()
    plt.plot(list(range(1,n)), d3_list, '-b')
    fig.suptitle("D3 Loss Plot")
    plt.xlabel('No. of iterations')
    plt.ylabel('D3 Loss')
    fig.savefig('d3loss.png')

    fig = plt.figure()
    plt.plot(list(range(1,n)), z_list, '-b')
    fig.suptitle("Z Loss Plot")
    plt.xlabel('No. of iterations')
    plt.ylabel('Z Loss')
    fig.savefig('zloss.png')

    return 

if __name__ == "__main__":
    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('dataset', default='usps', choices=['mnist', 'usps', 'mnist-test', 'eyaleb', 'arfaces', 'coil20', 'zygote'])
    parser.add_argument('--n_clusters', default=10, type=int)
    #parser.add_argument('--batch_size', default=256, type=int)
    #parser.add_argument('--maxiter', default=2e4, type=int)
    #parser.add_argument('--gamma', default=0.1, type=float, help='coefficient of clustering loss')
    #parser.add_argument('--update_interval', default=140, type=int)
    #parser.add_argument('--tol', default=0.001, type=float))
    args = parser.parse_args()
    print(args)

    n_clusters = args.n_clusters

    # load dataset
    from datasets import load_mnist, load_usps, load_processed_eyaleb, load_arfaces, load_zygote, load_coil20
    if args.dataset == 'mnist':
        x, y = load_mnist()
        x, y = x[0:8000], y[0:8000]
    elif args.dataset == 'usps':
        x, y = load_usps('data/usps')
        #x, y = x[0:2000], y[0:2000]
    elif args.dataset == 'mnist-test':
        x, y = load_mnist()
        x, y = x[63000:], y[63000:]
    elif args.dataset == 'eyaleb':
        x, y = load_processed_eyaleb()
    elif args.dataset == 'coil20':
        x, y = load_coil20()
    elif args.dataset == 'arfaces':
        x, y = load_arfaces()
    elif args.dataset == 'zygote':
        x, y = load_zygote()

    #x = np.reshape(x,(x.shape[0],x.shape[1]*x.shape[2]))
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
        #c_i,resid,grad,info = spg_lasso(z_i_c, z_j, tau, verbosity=0)
        c_i,resid,grad = ista(x_i_c,x_j)
        clist.append(c_i)    
        if(j%500==0):
            print("Value of j ",j)   
    C = np.concatenate(clist, axis=0).reshape(nsamples,nsamples).transpose()
    idx = np.arange(nsamples)
    C[idx,idx]=0
    print("C shape ",C.shape)

    #Computing ARI and NMI
    C_aff = np.add(C,np.transpose(C))
    clustering = SpectralClustering(assign_labels='discretize', n_clusters=args.n_clusters,random_state=0,affinity='precomputed').fit(C_aff)
    x_pred = clustering.labels_
    #print('acc = %.4f, nmi = %.4f, ari = %.4f' % (metrics.acc(y, x_pred), metrics.nmi(y, x_pred), metrics.ari(y, x_pred)))
    print('nmi = %.4f, ari = %.4f' % (metrics.nmi(y, x_pred), metrics.ari(y, x_pred)))
    print("Confusion Matrix ",confusion_matrix(y,x_pred))
    print("sillhouti score for ssc ", silhouette_score(x.T, x_pred))

    # n_clusters = args.n_clusters

    # findclusterlabels(np.transpose(x), y, n_clusters)

    print("--- %s seconds ---" % (time.time() - start_time))