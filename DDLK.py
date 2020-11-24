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
from scipy.io import loadmat
import metrics
from eyaleb import EYaleB
import torch
from sklearn.metrics import silhouette_score

seed = 78638924
np.random.seed(seed)

def findclusterlabels(x, y, n_clusters):
    x = np.transpose(x) # 256*7291

    # Initialization of parameters
    k1=64
    k2=32
    k3=16
    num_input_features = x.shape[0]
    num_samples = x.shape[1]
    n = 21 #number of iterations
    mu = 0.4

    d1 = np.random.randn(num_input_features,k1)
    d2 = np.random.randn(k1,k2)
    d3 = np.random.randn(k2,k3)
    z = np.random.randn(k3,num_samples)

    # H initialization
    xlabels = KMeans(n_clusters = args.n_clusters, init="k-means++").fit_predict(np.transpose(x))
    print("Length of xlabels ",xlabels.shape)

    silhouette_avg = silhouette_score(x.T, xlabels)
    print("Kmeans Silhoutte index ",silhouette_avg)

    onehotencoder = preprocessing.OneHotEncoder()
    h = onehotencoder.fit_transform(np.transpose(xlabels).reshape(-1,1)).toarray()
    # print("H shape ",h.shape)
    # hh = h.sum(axis=1)
    # print("Sum over cols of h ",hh.sum(axis=0))
    h = np.transpose(h)

    #print('k means acc = %.4f, nmi = %.4f, ari = %.4f' % (metrics.acc(y, xlabels), metrics.nmi(y, xlabels), metrics.ari(y, xlabels)))
    print('nmi = %.4f, ari = %.4f' % (metrics.nmi(y, xlabels), metrics.ari(y, xlabels)))
    print("Confusion Matrix  = ",confusion_matrix(y,xlabels))

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
        hterm = np.dot(np.transpose(h),np.dot(np.linalg.inv(np.dot(h,np.transpose(h))),h))
        #print("hterm shape",hterm.shape)
        #print("Calculating a")
        a = np.dot((np.transpose(np.dot(d1,np.dot(d2,d3)))),(np.dot(d1,np.dot(d2,d3))))
        #print("Calculating b")
        b = mu*(np.dot(np.transpose(np.identity(hterm.shape[0])-np.transpose(hterm)),(np.identity(hterm.shape[0])-hterm)))
        #print("Calculating c")
        c = np.dot((np.transpose(np.dot(d1,np.dot(d2,d3)))),x)
        print("Calculating z")
        z = scipy.linalg.solve_sylvester(a, b, c)

        z[z<0]=0

        z_error = np.linalg.norm((z - z_pre), 'fro')

        print("Calculating h by kmeans")
        z_pred = KMeans(n_clusters = args.n_clusters, init="k-means++").fit_predict(np.transpose(z))

        nmi_cur = metrics.nmi(y, z_pred)
        ari_cur = metrics.ari(y, z_pred)
        #print('acc = %.4f, nmi = %.4f, ari = %.4f' % (metrics.acc(y, z_pred), metrics.nmi(y, z_pred), metrics.ari(y, z_pred)))
        print('nmi = %.4f, ari = %.4f' % (nmi_cur, ari_cur))
        print("Confusion Matrix ",confusion_matrix(y,z_pred))
        # print("NMI Score ", normalized_mutual_info_score(y, z_pred))
        # print("ARI Score ", adjusted_rand_score(y, z_pred))

        cursilhoutti = silhouette_score(x.T, z_pred)

        if bestnmi < nmi_cur:
            bestnmi = nmi_cur
        if bestari < ari_cur:
            bestari = ari_cur
        if cursilhoutti > bestsilhoutti:
            bestsilhoutti = cursilhoutti

        h_pre = h
        h = onehotencoder.fit_transform(np.transpose(z_pred).reshape(-1,1)).toarray()

        hh = h.sum(axis=1)
        print("Sum over cols of h ",hh.sum(axis=0))

        hhh = h-np.transpose(h_pre)
        # print("h-h_pre shape",hhh.shape)
        # print("Difference in h ",hhh)

        h = np.transpose(h)

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

        hterm = np.dot(np.transpose(h),np.dot(np.linalg.inv(np.dot(h,np.transpose(h))),h))
        cost2 = (z - (np.dot(z,hterm)))
        cost2f = np.linalg.norm(cost2, 'fro')
        cost2f = np.square(cost2f)

        costf = cost1f + (mu*cost2f)

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
    fig.suptitle("DDLK")
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
    parser.add_argument('dataset', default='usps', choices=['mnist', 'usps', 'mnist-test', 'fashion-mnist', 'eyaleb', 'olivetti', 'cifar10', 'coil20', 'arp10', 'yale', 'arfaces', 'zygote'])
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
    from datasets import load_mnist, load_usps, load_eyaleb, load_fashion_mnist, load_processed_eyaleb, load_cifar10, load_olivetti, load_coil20, load_arp10, load_yale, load_zygote, load_arfaces
    if args.dataset == 'mnist':
        x, y = load_mnist()
        x, y = x[0:8000], y[0:8000]
    elif args.dataset == 'usps':
        x, y = load_usps('data/usps')
        x, y = x[0:1000], y[0:1000]
    elif args.dataset == 'mnist-test':
        x, y = load_mnist()
        x, y = x[63000:], y[63000:]
    elif args.dataset == 'eyaleb':
        x, y = load_processed_eyaleb()
    elif args.dataset == 'olivetti':
        x, y = load_olivetti()
    elif args.dataset == 'fashion-mnist':
        x, y = load_fashion_mnist()
        x, y = x[0:8000], y[0:8000]
    elif args.dataset == 'cifar10':
        x, y = load_cifar10()
    elif args.dataset == 'coil20':
        x, y = load_coil20()
    elif args.dataset == 'arp10':
        x, y = load_arp10()
    elif args.dataset == 'yale':
        x, y = load_yale()
    elif args.dataset == 'arfaces':
        x, y = load_arfaces()
    elif args.dataset == 'zygote':
        x, y = load_zygote()
        
    print("X shape ",x.shape)
    print("Y shape ",y.shape) 

    # sllist = []
    # for i in range(2,15):
    #     xlabels = KMeans(n_clusters = i, init="k-means++").fit_predict(np.transpose(x))
    #     silhouette_avg = silhouette_score(x.T, xlabels)
    #     sllist.append(silhouette_avg)

    # #Plotting cost function
    # fig = plt.figure()
    # plt.plot(list(range(2,15)), sllist, '-r')
    # fig.suptitle("KMeans Silhoutte Score")
    # plt.xlabel('No. of Clusters')
    # plt.ylabel('Silhoutte index')
    # plt.show()
    # print(sllist)

    import time
    start_time = time.time()

    findclusterlabels(x, y, n_clusters)

    print("--- %s seconds ---" % (time.time() - start_time))
    