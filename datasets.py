import numpy as np
import h5py
from scipy.io import loadmat
import torch

def load_processed_eyaleb(data_path='./datasets/EYALEB_DATA.mat'):
    EYALEB = loadmat(data_path)
    x = EYALEB['EYALEB_DATA']
    y = EYALEB['EYALEB_LABEL']
    x = np.transpose(x)
    y = np.transpose(y)
    y_len = y.shape[0]
    y2 = y.reshape((y_len,))
    return x, y2

def load_coil20(data_path='./datasets/COIL20.mat'):
    coil20 = loadmat(data_path)
    x = coil20['fea']
    y = coil20['gnd']
    y2 = y.reshape((y.shape[0],))
    return x,y2

def load_arfaces(data_path='./datasets/AR.mat'):
    arfaces = loadmat(data_path)
    x = arfaces['train_img']
    y = arfaces['train_lbl']
    x = np.transpose(x)
    y = np.transpose(y)
    y2 = [np.where(r==1)[0][0] for r in y]
    y2 = np.array(y2)
    return x,y2