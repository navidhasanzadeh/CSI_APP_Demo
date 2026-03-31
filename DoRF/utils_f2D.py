# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:10:43 2024

@author: 16478
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from scipy.signal import find_peaks, peak_prominences

import scipy.io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.metrics import f1_score
import os
import random
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import random
from sklearn.preprocessing import OneHotEncoder
from itertools import chain, combinations
import pandas as pd
from hampel import hampel
from matplotlib import pyplot as plt
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import scipy.signal as ss
import operator
from tqdm import tqdm
from scipy.signal import savgol_filter

from sklearn.preprocessing import OneHotEncoder
# os.environ["CUDA_VISIBLE_DEVICES"]="2" # second gpu
from sklearn.linear_model import LinearRegression
import numpy as np
import scipy.linalg as LA
import scipy.signal as ss
from scipy.signal import welch
from itertools import chain, combinations
import time
from scipy import signal
import cv2
from sklearn.covariance import empirical_covariance, LedoitWolf, MinCovDet, OAS
import pickle
# from hampel import hampel
import copy
from scipy.signal import savgol_filter
from scipy.signal import medfilt
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import LocalOutlierFactor
# from tslearn.clustering import TimeSeriesKMeans
from itertools import chain, combinations
import joblib
from scipy import stats
from scipy.signal import find_peaks
import numpy as np
import scipy.linalg as LA
import scipy.signal as ss
from scipy.signal import welch
from itertools import chain, combinations
import time
from scipy import signal
import cv2
from scipy import signal
from math import remainder, tau
import doatools.model as model
import doatools.estimation as estimation


from scipy.signal import firwin, lfilter
import matplotlib
# matplotlib.use('Agg')
def apply_fir_lowpass_filter(signal, cutoff, fs, numtaps):
    """
    Apply FIR low-pass filter to a 1D signal.
    
    Parameters:
    - signal (array-like): The input signal to be filtered.
    - cutoff (float): The cutoff frequency of the low-pass filter in Hz.
    - fs (float): The sampling frequency of the signal in Hz.
    - numtaps (int): The number of taps in the filter (higher means sharper cut-off).
    
    Returns:
    - filtered_signal (ndarray): The filtered signal.
    """
    # Design the FIR filter
    fir_coeff = firwin(numtaps, cutoff / (0.5 * fs), window="hamming")
    
    # Apply the FIR filter to the signal
    filtered_signal = lfilter(fir_coeff, 1.0, signal)
    
    return filtered_signal

def ant_processing(x):
    x = x.T.copy()
    x_bar = x.T.copy()
    for i in range(len(x)):
        mag = np.abs(x[i,:])
        phase = np.unwrap(np.angle(x[i,:]))
        phase_unwrap = phase
        x_k = np.arange(len(phase)).reshape(-1,)
        z1 = np.polyfit(x_k, phase_unwrap.reshape(-1,), 1)
        p1 = np.poly1d(z1)        
        y_k = p1(x_k)
        phase_unwrap = phase_unwrap - y_k
        phase_new = phase_unwrap
        x[i,:] = P2R_unit(phase_new)
    x = x.T
    x_bar = x_bar.T
    for i in range(len(x)):
        phase = np.angle(x[i,:])
        mag = np.abs(x_bar[:,i])
        phase_unwrap = phase
        b, a = signal.butter(2, 0.05)
        phase_unwrap = signal.filtfilt(b, a, phase_unwrap.T, padlen=150).T

        # phase_unwrap = np.unwrap(phase)
        # x_k = np.arange(len(phase_unwrap)).reshape(-1,)
        # z = np.polyfit(x_k, phase_unwrap.reshape(-1,), 1)
        # p = np.poly1d(z)
        # phase_unwrap = phase_unwrap - p(x_k)
        # phase_unwrap = hampel(phase_unwrap, window_size=8).filtered_data            
        # phase_unwrap_ = savgol_filter(phase_unwrap, 8 , 2)
        # phase_unwrap_ = moving_average(phase_unwrap_, 8)
        x[i,:] = P2R(mag, phase_unwrap)
    return x

def array_response_vector(array,velocity, freq_ref, freq_num):
    N = array.shape
    # print((1+freq_ref)/freq_num)
    v = np.exp(-1j*4*np.pi*(2.4*10**9 + (freq_ref - freq_num//2) * 312.5 * 000)/(3*10**8)*0.01*array*velocity)
    return v/np.sqrt(N)

def music(CovMat,L,N,array,Angles, freq_ref=0, freq_num=1):
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    # array holds the positions of antenna elements
    # Angles are the grid of directions in the azimuth angular domain
    _,V = LA.eig(CovMat)
    Qn  = V[:,L:N]
    numAngles = Angles.size
    pspectrum = np.zeros(numAngles)
    for i in range(numAngles):
        av = array_response_vector(array,Angles[i], freq_ref, freq_num)
        pspectrum[i] = 1/LA.norm((Qn.conj().transpose()@av))
    psindB    = np.log10(10*pspectrum/pspectrum.min())
    # DoAsMUSIC,_= ss.find_peaks(psindB,height=1.35, distance=1.5)
    DoAsMUSIC,_= ss.find_peaks(psindB)

    return DoAsMUSIC,pspectrum,psindB

def esprit(CovMat,L,N):
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    _,U = LA.eig(CovMat)
    # L = np.where(np.abs(LA.eig(CovMat)[0])<0.01 * np.max(LA.eig(CovMat)[0]))[0][0]-1
    S = U[:,0:L]
    Phi = LA.pinv(S[0:N-1]) @ S[1:N] # the original array is divided into two subarrays [0,1,...,N-2] and [1,2,...,N-1]
    eigs,_ = LA.eig(Phi)
    DoAsESPRIT_angle = np.arcsin(np.angle(eigs)/np.pi)
    DoAsESPRIT_abs = np.abs(eigs)

    return DoAsESPRIT_angle, DoAsESPRIT_abs

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def moving_average(x, w):
    l = len(x)
    x = np.concatenate([x[::-1], x, x[::-1]])
    return np.convolve(x, np.ones(w), 'same')[l:-l] / w

def calc_bands_power(x, dt, bands):
    f, psd = welch(x, fs=1. / dt,nfft=10*len(x))
    power =  [np.mean(psd[np.where((f >= lf) & (f <= lf+0.5))]) for lf in bands]
    return power

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def P2R_unit(angles):
    return 1 * np.exp(1j*angles)

def P2R(radii, angles):
    return radii * np.exp(1j*angles)

def R2P(x):
    return np.abs(x), np.angle(x)

def add_white_noise(x_volts, start, length, target_noise_db = 20, seed=10):
    x_volts = np.copy(x_volts)
    x_watts = x_volts ** 2
    mean_noise = 0
    np.random.seed(seed)
    target_noise_watts = 10 ** (target_noise_db / 10)
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(target_noise_watts), len(x_watts))
    y_volts = x_volts
    y_volts[start:start+length] = x_volts[start:start+length] + noise_volts[start:start+length]
    return y_volts

def add_bias_noise(x_volts, start, length, seed=10):
    y_volts = np.copy(x_volts)
    np.random.seed(seed)
    y_volts[start:start+length] += 2 * np.pi * np.random.rand() - np.pi
    return y_volts

def MUSIC_CSI_org(sample_data):
    sig_padded = np.zeros([len(sample_data),len(sample_data.T)+100],dtype=np.complex128)
    sig_padded[:,50:-50] = sample_data
    sig_padded[:,:50] = sample_data[:,50:0:-1]
    sig_padded[:,-50:] = sample_data[:,-1:-51:-1]
    Speeds = np.linspace(-0.05,0.05,512)

    v = np.zeros([Speeds.size,len(sample_data.T)])  
    print(v.shape)
    # MUSIC
    for w in range(50,len(sample_data.T)-50): 
        # print(w)                         
        sig_window = sig_padded[:, w-16:w+16]
        H = sig_window
        H = H.transpose()
        CovMat = H@H.conj().transpose()
        CovMat = np.nan_to_num(CovMat)
        U_abs,U = LA.eig(CovMat)
        U_abs = np.abs(U_abs)
        # L = np.where(U_abs>U_abs[0]*0.01)[0][-1] + 1
        L = 1
        N = len(CovMat)
        array = np.linspace(0,N-1,N)                    
        DoAsMUSIC, _, psindB = music(CovMat,L,len(CovMat),array,Speeds)
        v[:,w-25] = psindB
        doppler_vector = np.argmax(v.transpose(),1)
    return Speeds[doppler_vector], v


# In[6]:


def Root_MUSIC_CSI(sample_data, do_COV_processing = False, L=1):
    sig_padded = np.zeros([len(sample_data),len(sample_data.T)+100],dtype=np.complex128)
    sig_padded[:,50:-50] = sample_data
    sig_padded[:,:50] = sample_data[:,50:0:-1]
    sig_padded[:,-50:] = sample_data[:,-1:-51:-1]
    doppler_vector = []
    for w in range(50,len(sample_data.T)+50):                        
        sig_window = sig_padded[:, w-16:w+16]
        H = sig_window
        H = H.transpose()
        if do_COV_processing == False:
            CovMat = H@H.conj().transpose()
            CovMat = np.nan_to_num(CovMat)
        else:
            CovMats = []
            for kkk in range(1,4):
                CovMats_ = []
                for ii in range(kkk, len(H.transpose())):
                    H_1 = H[:,ii-kkk].reshape(-1,1)
                    H_2 = H[:,ii].reshape(-1,1)
                    # H_1 = H_1 / (1e-9 + H_1[0])
                    # H_2 = H_2 / (1e-9 + H_2[0])
                    H_ = (H_2 / (H_1 + 0.000001 * np.abs(np.mean(H_1))))** (1/kkk)
                    CovMat = H_@H_.conj().transpose()
                    CovMat = np.nan_to_num(CovMat)
                    CovMats_.append(CovMat)
                CovMats.append(np.median(CovMats_,0))    
            CovMat = np.mean(CovMats, 0)
        # CovMat = estimation.spatial_smooth(CovMat, 1, True)
        U_abs,U = LA.eig(CovMat)
        U_abs = np.abs(U_abs)
        # L = np.where(U_abs>U_abs[0]*0.01)[0][-1] + 1
        N = len(CovMat)
        # print(CovMat.shape)
        # CovMat_smooth = estimation.spatial_smooth(CovMat, 1, True)
        estimator = estimation.RootMUSIC1D(1.0)
        resolved, estimates = estimator.estimate(CovMat, L)
        doppler_vector.append(estimates.locations)
    return doppler_vector


def MUSIC_CSI_(sample_data, scale):
    sig_padded = np.zeros([len(sample_data),550],dtype=np.complex128)
    sig_padded[:,25:-25] = sample_data
    sig_padded[:,:25] = sample_data[:,25:0:-1]
    sig_padded[:,-25:] = sample_data[:,-1:-26:-1]
    max_scale = np.max(np.abs(scale)) * 5
    Speeds = np.linspace(-1 * max_scale, 1* max_scale, 32)

    v = np.zeros([2048*4,500])   
    # MUSIC
    vel = []
    for w in range(25,525):                        
        sig_window = sig_padded[:, w-16:w+16]
        H = sig_window
        H = H.transpose()
        CovMat = H@H.conj().transpose()
        CovMat = np.nan_to_num(CovMat)
        U_abs,U = LA.eig(CovMat)
        U_abs = np.abs(U_abs)
        # L = np.where(U_abs>U_abs[0]*0.01)[0][-1] + 1
        L = 1
        max_speed_w = np.max(np.abs(scale[w-25])) * 5
        Speeds_ = np.linspace(-1 * max_speed_w, 1* max_speed_w, 32)
        N = len(CovMat)
        array = np.linspace(0,N-1,N)                    
        DoAsMUSIC, _, psindB = music(CovMat,L,len(CovMat),array,Speeds_)
        vel.append(Speeds_[np.argmax(psindB)])
        sss = int(max_speed_w / max_scale *1024*4)
        f = signal.resample(psindB, 2*sss)
        f = (f - f.min()) / (f.max()-f.min()+1e-12)
        v[1024*4-sss:1024*4+sss,w-25] = f
        v[:1024*4-sss,w-25] = np.min(f)
        v[1024*4+sss:,w-25] = np.min(f)
    return moving_average(np.array(vel),8), v

def MUSIC_CSI_scale(sample_data):
    sig_padded = np.zeros([len(sample_data),550],dtype=np.complex128)
    sig_padded[:,25:-25] = sample_data
    sig_padded[:,:25] = sample_data[:,25:0:-1]
    sig_padded[:,-25:] = sample_data[:,-1:-26:-1]
    Speeds = np.concatenate([-1 * np.geomspace(1,1e-6,11), 1 * np.geomspace(1,1e-6,11),[0]])
    v = np.zeros([Speeds.size,500])   
    # MUSIC
    for w in range(25,525):                        
        sig_window = sig_padded[:, w-8:w+8]
        H = sig_window
        H = H.transpose()
        CovMat = H@H.conj().transpose()
        CovMat = np.nan_to_num(CovMat)
        U_abs,U = LA.eig(CovMat)
        U_abs = np.abs(U_abs)
        L = 1
        N = len(CovMat)
        array = np.linspace(0,N-1,N)                    
        DoAsMUSIC, _, psindB = music(CovMat,L,len(CovMat),array,Speeds)
        v[:,w-25] = psindB
    doppler_vector = np.argmax(v.transpose(),1)
    return Speeds[doppler_vector], v

def MUSIC_CSI(sample_data):
    vel_scale, _ = MUSIC_CSI_scale(sample_data)
    vel, v = MUSIC_CSI_(sample_data, vel_scale)
    return vel, v
    
def getDopplerVel_MUSIC(subs):
    vel,v = MUSIC_CSI(subs)
    return moving_average(vel, 50)

def MSE(x, y):
    x = x.copy()
    y = y.copy()
    x = (x-x.mean())
    x = x / x.std()
    y = (y-y.mean())
    y = y / y.std()
    return np.mean((x-y)**2)

def MSE_sign(x, y):
    pos_sign = MSE(x,y)
    neg_sign = MSE(x,-1*y)
    if pos_sign<neg_sign:
        return 1 * pos_sign
    return -1 * neg_sign

def getBestCCSet(x):
    ants_power_set = list(powerset(range(len(x))))
    ants_power_set_cc = {}
    for ant_set in ants_power_set:
        ants_vels = np.array(x).copy()
        for ant_i in ant_set:
            ants_vels[ant_i] = -1 * ants_vels[ant_i]
        vels_all_AP_cc = []
        all_pair_vels_all_AP= list(combinations(ants_vels, 2))
        for p1, p2 in all_pair_vels_all_AP:
            vels_all_AP_cc.append(MSE(p1[50:-50], p2[50:-50]))

        ants_power_set_cc[ant_set] = np.sum(vels_all_AP_cc)
        
    neg_ants = list(ants_power_set_cc.keys())[np.argmin(list(ants_power_set_cc.values()))]
    ants_vels = np.array(x).copy()
    for ant_i in neg_ants:
        ants_vels[ant_i] = -1 * ants_vels[ant_i]
    return ants_vels


def getBinaryClusteredMean(x, x_ref = None):
    vels = np.array(x)
    if x_ref is not None:
        for i in range(len(vels)):
            cc_ref_x = MSE_sign(vels[i][50:-50], x_ref[50:-50])
            vels[i] = vels[i] * np.sign(cc_ref_x)
            
    clf = LocalOutlierFactor(n_neighbors=int(len(vels)/2)+1, contamination=0.1)
    vels_outliers = clf.fit_predict(vels)
    vels__ = vels[vels_outliers==1] 
    vels = vels[vels_outliers==1]
    clustering = TimeSeriesKMeans(n_clusters=2, metric="euclidean", max_iter=100,
                          # max_iter_barycenter=5,
                          random_state=0)
    pred = clustering.fit_predict(vels__)
    vels_1 = vels[pred==0]
    if len(vels_1) > 2:
        clf = LocalOutlierFactor(n_neighbors=int(len(vels_1)/2)+1, contamination="auto")
        vels_1_outliers = clf.fit_predict(vels_1)
        vels_1 = vels_1[vels_1_outliers==1]   
         
    vels_2 = vels[pred==1]
    if len(vels_2) > 2:
        clf = LocalOutlierFactor(n_neighbors=int(len(vels_2)/2)+1, contamination="auto")
        vels_2_outliers = clf.fit_predict(vels_2)
        vels_2 = vels_2[vels_2_outliers==1]   
    
    vel_1_cc = []
    all_pair_vels2= list(combinations(vels_1, 2))
    for p1, p2 in all_pair_vels2:
        vel_1_cc.append(np.corrcoef(p1, p2)[0][1])
    vel_1_score = np.median(vel_1_cc)
    vel_1_score_ = vel_1_score
    vel_2_cc = []
    all_pair_vels2= list(combinations(vels_2, 2))
    for p1, p2 in all_pair_vels2:
        vel_2_cc.append(np.corrcoef(p1, p2)[0][1])
    vel_2_score = np.median(vel_2_cc)
    vel_2_score_ = vel_2_score
    vel_1_score_ /= (vel_1_score + vel_2_score)
    vel_2_score_ /= (vel_1_score + vel_2_score)
    if len(vels_1) <2:
        vel1 = np.mean(vels_2, 0)
    elif len(vels_2)<2:
        vel1 = np.mean(vels_1, 0)
    else:
        vel1 = vel_1_score_ * np.median(vels_1, 0) + np.sign(MSE_sign(np.median(vels_1, 0), np.median(vels_2, 0))) * vel_2_score_ * np.median(vels_2, 0)
    return vel1
    
def getAdvancedMUSIC(ant):    
    vels = []
    ant_subs =[]
    for i in range(4, ant.shape[1]-4):
        ant_subs.append(ant[:,i-4:i+4].T)
    vels = joblib.Parallel(n_jobs=12, verbose=0)(joblib.delayed(getDopplerVel_MUSIC)(i) for i in ant_subs)
    vels = np.array(vels)
    vel1_ref = getBinaryClusteredMean(vels)
    vel1 = getBinaryClusteredMean(vels, vel1_ref)
    return vel1

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    
def AWGN_SNR(x, target_snr_db):
    x_ = x.copy()
    for i in range(len(x_)):
        x_volts = x_[i]
        x_watts = x_volts ** 2
        # Calculate signal power and convert to dB 
        sig_avg_watts = np.mean(x_watts)
        sig_avg_db = 10 * np.log10(sig_avg_watts)
        # Calculate noise according to [2] then convert to watts
        noise_avg_db = sig_avg_db - target_snr_db
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        # Generate an sample of white noise
        mean_noise = 0
        noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
        # Noise up the original signal
        x_[i] = x_volts + noise_volts
    return x_
    
def antenna_pair_score(x, y):
    ccs = []
    for i in range(x.shape[1]):
        X_ = np.array([x[:,i],y[:,i]]).T
        clustering = SpectralClustering(n_clusters=2,
                                        assign_labels='discretize',
                                        random_state=0).fit(X_)
        # plt.figure()
        # plt.scatter(X_[:,0], X_[:,1],  
        #             c = clustering.fit_predict(X_), cmap =plt.cm.winter) 
        
        pred = clustering.fit_predict(X_)
        ind_0 = np.where(pred==0)[0]
        ind_1 = np.where(pred==1)[0]
        
        clf = LocalOutlierFactor(n_neighbors=np.min([10, 2 + int(len(ind_0)/2)]), contamination=0.1)
        y_pred_0 = clf.fit_predict(X_[:,:][ind_0])
        
        clf = LocalOutlierFactor(n_neighbors=np.min([10, 2 + int(len(ind_0)/2)]), contamination=0.1)
        y_pred_1 = clf.fit_predict(X_[:,:][ind_1])
        
        ind_0 = ind_0[np.where(y_pred_0==1)[0]]
        ind_1 = ind_1[np.where(y_pred_1==1)[0]]
        ind_0 = [k for k in ind_0 if k not in np.where(y_pred_0==-1)[0]]
        ind_1 = [k for k in ind_1 if k not in np.where(y_pred_1==-1)[0]]
    
        cc0 = np.corrcoef(X_[:,0][ind_0], X_[:,1][ind_0])[0][1]
        cc1 = np.corrcoef(X_[:,0][ind_1], X_[:,1][ind_1])[0][1]
        cc = (np.abs(cc0) + np.abs(cc1))/2
        ccs.append(cc)
        # plt.title(cc)
        # plt.show()

    return np.median(ccs)
    


# In[7]:

from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
def savgol_CSI(x):
        window_size_sg = 32  # Must be odd, typically larger than poly_order
        poly_order_sg = 3
        savgol_filtered_signal = savgol_filter(np.concatenate([x[::-1],x,x[::-1]]), window_size_sg, poly_order_sg)
        savgol_filtered_signal = savgol_filtered_signal[len(x):2*len(x)]
        return savgol_filtered_signal
def ant_processing(x):
    # x /= (np.abs(x)+0.000000001)
    x = x.copy()
    x_bar = x.copy()
    for i in range(len(x.T)):
        mag = np.abs(x[:,i])
        phase = np.unwrap(np.angle(x[:,i]))
        phase_unwrap = phase
        x_k = np.arange(len(phase)).reshape(-1,)
        # z1 = np.polyfit([5, 19, 32, 46], phase_unwrap.reshape(-1,)[[5, 19, 32, 46]], 1)

        # z1 = np.polyfit(x_k, phase_unwrap.reshape(-1,), 1)
        
        # p1 = np.poly1d(z1)        
        # y_k = p1(x_k)
        ransac = RANSACRegressor( random_state=42)
        ransac.fit(x_k.reshape(-1, 1),  phase_unwrap.reshape(-1,))
        y_k = ransac.predict(x_k.reshape(-1, 1))
        phase_unwrap = phase_unwrap - y_k
        phase_new = phase_unwrap
        x[:,i] = P2R(mag, phase_new)
        # plt.figure()
        # plt.plot(phase)
        # plt.plot(y_k)
        # plt.show()
    # x = x.T
    # x_bar = x_bar.T
    for i in range(len(x)):
        phase = np.angle(x[i,:])
        mag = np.abs(x[i,:])
        # phase = savgol_CSI(phase)
        # mag = savgol_CSI(mag)
        # phase_unwrap = phase
        # b, a = signal.butter(2, 0.1)
        # phase_unwrap = signal.filtfilt(b, a, phase_unwrap.T, padlen=150, method='gust').T
        # mag_f = signal.filtfilt(b, a, mag.T, padlen=150, method='gust').T

        # phase_unwrap = np.unwrap(phase)
        # x_k = np.arange(len(phase_unwrap)).reshape(-1,)
        # z = np.polyfit(x_k, phase_unwrap.reshape(-1,), 1)
        # p = np.poly1d(z)
        # phase_unwrap = phase_unwrap - p(x_k)
        phase_unwrap = hampel(phase, window_size=4).filtered_data            
        # phase_unwrap_ = savgol_filter(phase_unwrap, 8 , 2)
        # phase_unwrap_ = moving_average(phase_unwrap_, 8)
        x[i,:] = P2R(mag, phase_unwrap)
    return x
def ant_pair_processing(x, ant_best_pair_filter=None):
    x = x.T.copy()
    x /= (np.abs(x)+0.000000001)
    x_bar = x.T.copy()
    # for i in range(len(x)):
    #     mag = np.abs(x[i,:])
    #     phase = np.unwrap(np.angle(x[i,:]))
    #     phase_unwrap = phase
    #     x_k = np.arange(len(phase)).reshape(-1,)
    #     z1 = np.polyfit(x_k, phase_unwrap.reshape(-1,), 1)
    #     p1 = np.poly1d(z1)        
    #     y_k = p1(x_k)
    #     phase_unwrap = phase_unwrap - y_k
    #     phase_new = phase_unwrap
    #     x[i,:] = P2R_unit(phase_new)
    x = x.T
    x_bar = x_bar.T
    
    for i in range(len(x)):
        phase = np.angle(x[i,:])
        mag = np.abs(x_bar[:,i])
        phase_unwrap = phase
        # phase_unwrap = np.unwrap(phase)
        # print(phase_unwrap.shape)
        # phase_unwrap = hampel(phase_unwrap.T, window_size=15).filtered_data.T
        # if True:
        #     y_0 = 0 
        #     y = np.diff(phase_unwrap) * ant_best_pair_filter
        #     y = np.cumsum(y)
        #     y = np.append([y_0], y).reshape(-1) + phase_unwrap[0]
        # phase_unwrap = y
        # y[np.where(np.abs(y)>0.2)] = 0
        # phase_unwrap = np.cumsum(y)
        b, a = signal.butter(2, 0.05)
        phase_unwrap = signal.filtfilt(b, a, phase_unwrap, padlen=150).T
        # phase_unwrap_ = savgol_filter(phase_unwrap, 8 , 2)
        # phase_unwrap = moving_average(phase_unwrap, 8)
        # x_k = np.arange(len(phase_unwrap)).reshape(-1,)
        # z = np.polyfit(x_k, phase_unwrap.reshape(-1,), 1)
        # p = np.poly1d(z)
        # phase_unwrap = phase_unwrap - p(x_k)
        x[i,:] = P2R_unit(phase_unwrap)
    return x


def ant_pair_processing_2(x, ant_best_pair_filter=None):
    x = x.copy()
    x_bar = x.copy()
    for i in range(len(x)-1):
        x[i,:] = x[i,:] / x[i+1,:]
        phase = np.angle(x[i,:])
        mag = np.abs(x[i,:])
        angle_pp = moving_average(phase,1)
        angle_pp = hampel(angle_pp, window_size=15).filtered_data.T
        angle_pp = moving_average(angle_pp,32)
        mag_pp = moving_average(mag,1)
        mag_pp = hampel(mag_pp, window_size=15).filtered_data.T
        mag_pp = moving_average(mag_pp,32)
        
        x_bar[i,:] = P2R(mag_pp,angle_pp)
    x_bar = x_bar[:-1]
    return x_bar
# In[8]:

# In[16]:


import doatools.estimation as estimation
def MUSIC_CSI_org(sample_data, L=1, freq_ref = 0, number_of_freq=1):
    # number_of_freq = len(sample_data)
    # print(number_of_freq / (1+freq_ref))
    sig_padded = np.zeros([len(sample_data),550],dtype=np.complex128)
    sig_padded[:,25:-25] = sample_data
    sig_padded[:,:25] = sample_data[:,25:0:-1]
    sig_padded[:,-25:] = sample_data[:,-1:-26:-1]
    Speeds = np.linspace(-1,1,1024)

    v = np.zeros([Speeds.size,500])
    eigens = []
    # MUSIC
    all_L = []
    for w in range(25,525):                        
        sig_window = sig_padded[:, w-8:w+8]
        H = sig_window
        H = H.transpose()
        CovMat = H@H.conj().transpose()
        CovMat = np.nan_to_num(CovMat)
        # CovMat = estimation.spatial_smooth(CovMat, 1, True)
        # print(CovMat.shape)
        U_abs,U = LA.eig(CovMat)
        U_abs = np.abs(U_abs)
        eigens.append(U_abs)
        # L = np.where(U_abs>U_abs[0]*0.1)[0][-1] + 1
        # all_L.append(L)
        # L = 8
        N = len(CovMat)
        array = np.linspace(0,N-1,N)                    
        DoAsMUSIC, _, psindB = music(CovMat,L,len(CovMat),array,Speeds, freq_ref, number_of_freq)
        psindB = psindB 
        v[:,w-25] = psindB
        doppler_vector = np.argmax(v.transpose(),1)
    return Speeds[doppler_vector], v, eigens, all_L


import matplotlib.font_manager
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
def antenna_pair_score_vis(ants):
    
    # plt.figure(figsize=(4,1))
    all_ccs = []
    for ii, ant in enumerate(ants):
        # ax = plt.subplot(1,3,ii+1)
        x = ant[0]
        y = ant[1]
        ccs = []
        for i in range(x.shape[1]):
            X_ = np.array([x[:,i],y[:,i]]).T
            clustering = SpectralClustering(n_clusters=2,
                                            assign_labels='discretize',
                                            random_state=0).fit(X_)
            # plt.figure()
            # plt.scatter(X_[:,0], X_[:,1],  
            #             c = "k", s = 2)             
            # plt.xlabel("Ant {}".format([(2,1),(3,2),(1,3)][ii]),  weight='bold', fontsize=13,fontproperties=matplotlib.font_manager.FontProperties(fname="timesbf.ttf"))
            # # plt.ylabel("Ant. {}".format([2,3,1][ii]))
            # if ii==0:
            #     plt.ylabel("Phase",  weight='bold', fontsize=12,fontproperties=matplotlib.font_manager.FontProperties(fname="timesbf.ttf"))
            # else:
            #     plt.yticks([])
            # for tick in ax.get_xticklabels():
            #     tick.set_fontproperties(matplotlib.font_manager.FontProperties(fname="times.ttf"))
            # for tick in ax.get_yticklabels():
            #     tick.set_fontproperties(matplotlib.font_manager.FontProperties(fname="times.ttf"))  
            pred = clustering.fit_predict(X_)
            ind_0 = np.where(pred==0)[0]
            ind_1 = np.where(pred==1)[0]
            
            clf = LocalOutlierFactor(n_neighbors=np.min([10, 2 + int(len(ind_0)/2)]), contamination=0.1)
            y_pred_0 = clf.fit_predict(X_[:,:][ind_0])
            
            clf = LocalOutlierFactor(n_neighbors=np.min([10, 2 + int(len(ind_0)/2)]), contamination=0.1)
            y_pred_1 = clf.fit_predict(X_[:,:][ind_1])
            
            ind_0 = ind_0[np.where(y_pred_0==1)[0]]
            ind_1 = ind_1[np.where(y_pred_1==1)[0]]
            ind_0 = [k for k in ind_0 if k not in np.where(y_pred_0==-1)[0]]
            ind_1 = [k for k in ind_1 if k not in np.where(y_pred_1==-1)[0]]
        
            cc0 = np.corrcoef(X_[:,0][ind_0], X_[:,1][ind_0])[0][1]
            cc1 = np.corrcoef(X_[:,0][ind_1], X_[:,1][ind_1])[0][1]
            cc = (np.abs(cc0) + np.abs(cc1))/2
            ccs.append(cc)
            # plt.title(np.round(cc,3))
            # plt.show()
        all_ccs.append(np.median(ccs))
    # plt.savefig("ants_ratio.pdf", bbox_inches = 'tight')
    return all_ccs


# In[11]:


def array_response_vector2D(array,velocity,angle, angle_z):
    N = array.shape
    # v1 = np.exp(-1j*4*np.pi*(2.4*10**9)/(3*10**8)*0.01*array*velocity*np.sin(angle)*np.sin(angle_z))
    v2 = np.exp(-1j*4*np.pi*(2.4*10**9)/(3*10**8)*0.01*array*velocity*np.cos(angle)*np.sin(angle_z))
    # v3 = np.exp(-1j*4*np.pi*(2.4*10**9)/(3*10**8)*0.01*array*velocity*np.cos(angle_z))
    # v = (np.abs(v1**2 + v2**2 + v3**2)**0.5
    # v = (v1 + v2 + v3) / (np.abs(v1 + v2 + v3))
    return v2/np.sqrt(N)

def music2D(CovMat,L,N,array,Vels,Angles, Angles_z):
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    # array holds the positions of antenna elements
    # Angles are the grid of directions in the azimuth angular domain
    _,V = LA.eig(CovMat)
    Qn  = V[:,L:N]
    numAngles = (Vels.size, Angles.size, Angles_z.size)
    pspectrum = np.zeros(numAngles)
    for i in range(numAngles[0]):
        for j in range(numAngles[1]):
            for k in range(numAngles[2]):
                av = array_response_vector2D(array,Vels[i],Angles[j],Angles_z[k])
                pspectrum[i][j][k] = 1/LA.norm((Qn.conj().transpose()@av))
    psindB    = np.log10(10*pspectrum/pspectrum.min())
    # DoAsMUSIC,_= ss.find_peaks(psindB,height=1.35, distance=1.5)
    # DoAsMUSIC,_= ss.find_peaks(psindB)

    return pspectrum,psindB
def plot3D(x, y, z, action=-1):
    fig = plt.figure(figsize=(10,5))
    color = list(np.arange(0,1,1/500))
    ax = fig.add_subplot(projection='3d')

    # Plot a sin curve using the x and y axes.

    x = x - x[0]
    y = y - y[0]
    z = z - z[0]
    plt.xlim([np.min([x,y,z]), np.max([x,y,z])])
    plt.ylim([np.min([x,y,z]), np.max([x,y,z])])
    ax.set_zlim(np.min([x,y,z]), np.max([x,y,z]))
    ax.scatter(x, y, z ,c=color,s=50)
    ax.scatter(x[0], y[0], z[0], label='Start',c='red',s=80)
    ax.scatter(x[-1], y[-1], z[-1], label='End',c='black',s=80)
    
    plt.title("CSI Action: {}".format(["circle", "up-down", "left-right", "push-pull", "no action", ""][action]))
    plt.legend()
    
def MUSIC_CSI_2D(sample_data, L=1):
    sig_padded = np.zeros([len(sample_data),550],dtype=np.complex128)
    sig_padded[:,25:-25] = sample_data
    sig_padded[:,:25] = sample_data[:,25:0:-1]
    sig_padded[:,-25:] = sample_data[:,-1:-26:-1]
    Speeds = np.linspace(-0.05,0.05,128)
    # Angles = np.array([0.0])
    Angles = np.linspace(0 * np.pi, +1 * np.pi,16)
    Angles_z = np.array([np.pi/2])
    # Angles_z = np.linspace(0, +1 * np.pi/2,8)
    # Speeds = np.concatenate([-1 * np.geomspace(1,1e-6,11), 1 * np.geomspace(1,1e-6,11),[0]])
    v = np.zeros([Speeds.size,Angles.size, Angles_z.size,500])  
    v_x = []
    v_y = []
    v_z = []
    eigens = []
    # MUSIC
    for w in tqdm(range(25,525)):                        
        sig_window = sig_padded[:, w-16:w+16]
        H = sig_window
        H = H.transpose()
        CovMat = H@H.conj().transpose()
        CovMat = np.nan_to_num(CovMat)
        CovMat = estimation.spatial_smooth(CovMat, 1, True)
        U_abs,U = LA.eig(CovMat)
        U_abs = np.abs(U_abs)
        eigens.append(U_abs)
        # L = np.where(U_abs>U_abs[0]*0.01)[0][-1] + 1
        # L = 1
        N = len(CovMat)
        array = np.linspace(0,N-1,N)    
        _, psindB = music2D(CovMat,L,len(CovMat),array,Speeds, Angles, Angles_z)
        ve_max, ang_max, ang_z_max = np.unravel_index(np.argmax(psindB), np.array(psindB).shape)
        # print(Angles[ang_max])
        v_x.append(Speeds[ve_max] * np.cos(Angles[ang_max]) * np.sin(Angles_z[ang_z_max]))
        v_y.append(Speeds[ve_max] * np.sin(Angles[ang_max]) * np.sin(Angles_z[ang_z_max]))
        v_z.append(Speeds[ve_max] * np.cos(Angles_z[ang_z_max]))
        v[:,:,:,w-25] = psindB
    v_x = np.array(v_x)    
    v_y = np.array(v_y)
    v_z = np.array(v_z)
    v_x_ = v_x - v_x.mean()
    v_y_ = v_y - v_y.mean()
    v_z_ = v_z - v_z.mean()
    # plot3D(moving_average(np.cumsum(v_x),16), moving_average(np.cumsum(v_y),16), moving_average(np.cumsum(v_z_),16))
    doppler_vector = np.argmax(v.transpose(),1)
    return [v_x_, v_y_, v_z_], v, eigens


import math

def geometric_mean(numbers):
    # Check if the list is empty
    if len(numbers) == 0:
        return None
    
    # Calculate the product of all numbers in the list
    product = 1
    for num in numbers:
        product *= num
    
    # Calculate the geometric mean
    geo_mean = math.pow(product, 1/len(numbers))
    
    return geo_mean


# In[24]:


def getMDL(eigenvalues, d, K, N):
    MDL = -K * (N-d)
    MDL *= np.log(geometric_mean(eigenvalues[d:]) / (1e-10 + np.mean(eigenvalues[d:])))
    MDL += 0.5 * d * (2*N - d) * np.log(K)
    return MDL


# In[21]:


import numpy as np

def calculate_mdl(eigenvalues, num_snapshots, num_antennas):
    """
    Calculate the Minimum Description Length (MDL) for estimating the number of sources in an AoA estimation problem.

    Parameters:
    eigenvalues (np.array): Array of eigenvalues of the covariance matrix, sorted in descending order.
    num_snapshots (int): Number of snapshots or observations.
    num_antennas (int): Number of antennas in the array.

    Returns:
    int: Estimated number of sources.
    np.array: MDL values for each possible number of sources.
    """

    num_eigenvalues = len(eigenvalues)
    mdl_values = np.zeros(num_antennas)

    for k in range(num_antennas):
        # Sum of the smallest eigenvalues
        sum_smallest_eigenvalues = np.sum(eigenvalues[k:])
        
        # Geometric mean of the smallest eigenvalues
        geometric_mean = np.prod(eigenvalues[k:])**(1 / (num_eigenvalues - k))
        
        # Arithmetic mean of the smallest eigenvalues
        arithmetic_mean = sum_smallest_eigenvalues / (num_eigenvalues - k)
        
        # MDL formula
        mdl_values[k] = -num_snapshots * (num_eigenvalues - k) * np.log(arithmetic_mean / geometric_mean) + \
                        0.5 * k * (2 * num_eigenvalues - k) * np.log(num_snapshots)

    # Find the minimum MDL value
    estimated_sources = np.argmin(mdl_values)

    return estimated_sources, mdl_values

def add_average_padding(signal, pad_length):
    """
    Adds padding to the beginning and end of a signal using the average 
    of the first 10% and the last 10% of the signal.

    Parameters:
    signal (list or numpy array): The input signal to be padded.
    pad_length (int): The number of elements to pad at both the beginning and the end.

    Returns:
    numpy array: The padded signal.
    """
    # Convert the input signal to a numpy array if it is not already
    signal = np.array(signal)
    signal_length = len(signal)
    
    # Calculate the number of elements corresponding to 10% of the signal
    ten_percent_length = max(1, int(0.1 * signal_length))

    # Calculate the average of the first and last 10% of the signal
    start_avg = np.mean(signal[:ten_percent_length])
    end_avg = np.mean(signal[-ten_percent_length:])

    # Create the padding arrays with the averages
    start_padding = np.full(pad_length, start_avg)
    end_padding = np.full(pad_length, end_avg)

    # Concatenate the padding to the beginning and end of the signal
    padded_signal = np.concatenate([start_padding, signal, end_padding])

    return padded_signal
def normalize_signal(signal):
    """
    Normalize the signal to have zero mean and unit variance.
    """
    mean = np.mean(signal)
    std = np.std(signal)
    normalized_signal = (signal - mean) / std
    return normalized_signal

def mean_squared_error(signal1, signal2):
    """
    Calculate the mean squared error (MSE) between two signals.
    The signals are first normalized to have zero mean and unit variance.
    """
    # Normalize the signals
    signal1_normalized = normalize_signal(signal1)
    signal2_normalized = normalize_signal(signal2)
    
    # Calculate MSE
    mse = np.mean((signal1_normalized - signal2_normalized) ** 2)
    return mse


def circular_shift(signal, shift_amount):
    """
    Apply a circular shift to a given signal.

    Parameters:
    signal (array-like): The input signal to be shifted.
    shift_amount (int): The number of positions to shift the signal. Positive values shift to the right, 
                        and negative values shift to the left.

    Returns:
    np.array: The circularly shifted signal.
    """
    signal = np.array(signal)
    shift_amount = shift_amount % len(signal)  # Ensure shift amount is within the signal length
    return np.roll(signal, shift_amount)


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-((x - mu)**2) / (2 * sigma**2))
def get_jumps_sig(ss):
    ss_d = []
    prev = 0
    for i in ss:
        if i != prev and i != 0:
            prev = i
        ss_d.append(prev)
    ss_d_old = ss_d    
    ss_d = np.array(ss_d)
    ss_d_std = ss_d.std()
    ss_d = ss_d - ss_d.min()
    ss_d = ss_d / ss_d.std()
    x = np.linspace(0, len(ss), len(ss))
    # params, _ = curve_fit_with_multiple_initials(gaussian, x, ss_d, p0=[[1, 400, 10]], maxfev = 10000, method = 'trf')
    params, covariance = curve_fit_with_multiple_initials(gaussian, x, ss_d, initial_guesses = [[1, 1 + 50 * i, 10] for i in range(len(ss)//50-1)], plot_all=False, maxfev = 10000, method = 'trf')
    # Extract the fitting parameters
    amp, mu, sigma = params
    # Generate the fitted curve
    y_fit = gaussian(x, *params)
    r2 = r2_score(np.array(ss_d), y_fit)
    mse = mean_squared_error(np.array(ss_d), y_fit)

    y_fit = ss_d_std * y_fit
    ss_d = ss_d * ss_d_std
    # print(f'R-squared: {r2:.2f}')
    # ss_d = hampel(np.array(ss_d), window_size=8).filtered_data
    # ss_d = moving_average(savgol_filter(ss_d, 8 , 2),8)
    # print(mu)
    return ss_d, y_fit, np.abs(mu), 10e0 * (np.abs(sigma) + ((np.abs(sigma)-40.0) ** 2)**0.5 + ((np.abs(mu)-200.0) ** 2)**0.5) * mse

from scipy.optimize import curve_fit

def curve_fit_with_multiple_initials(func, xdata, ydata, initial_guesses, plot_all=False, **kwargs):
    best_params = None
    best_covariance = None
    best_residuals = np.inf  # Set the initial best residuals to infinity
    all_fits = []  # To store all fits if plot_all is True

    for initial_guess in initial_guesses:
        try:
            params, covariance = curve_fit(func, xdata, ydata, p0=initial_guess, **kwargs)
            residuals = np.sum((ydata - func(xdata, *params))**2)

            if plot_all:
                all_fits.append((params, residuals))

            if residuals < best_residuals:
                best_residuals = residuals
                best_params = params
                best_covariance = covariance

        except RuntimeError:
            # If curve fitting fails for this initial guess, continue to the next
            continue

    if best_params is not None:
        # Plotting the data and the best fit
        # plt.figure(figsize=(10, 6))
        # plt.scatter(xdata, ydata, label='Data', color='black')

        # # Plot all fits if requested
        # if plot_all:
        #     for i, (params, residuals) in enumerate(all_fits):
        #         plt.plot(xdata, func(xdata, *params), linestyle='--', label=f'Fit {i+1}: Residuals={residuals:.2f}')

        # # Plot the best fit
        # plt.plot(xdata, func(xdata, *best_params), label='Best Fit', color='red', linewidth=2)

        # plt.legend()
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('Curve Fitting with Multiple Initial Guesses')
        # plt.show()

        return best_params, best_covariance
    else:
        raise RuntimeError("Failed to converge to a solution with all initial guesses.")

def shift_to_middle(signal_, padding_length=300):
    if len(np.shape(signal_)) == 1:
        signal = signal_
    else:
        signal = signal_[1]
        # signal = 0 * signal_[0]
        # for i in range(len(signal_)):
        #     signal += signal_[i]
    ss = np.zeros(len(signal))
    
    # Finding peaks and prominences for positive signal
    peaks1, _ = find_peaks(signal, prominence=0.0 * np.max(np.abs(signal)))
    prominences1 = peak_prominences(signal, peaks1)[0]
    ss[peaks1] = prominences1

    # Finding peaks and prominences for negative signal
    peaks2, _ = find_peaks(-1 * signal, prominence=0.0 * np.max(np.abs(signal)))
    prominences2 = peak_prominences(-1 * signal, peaks2)[0]
    ss[peaks2] = prominences2

    peaks = list(peaks1) + list(peaks2)

    # Assuming get_jumps_sig is a predefined function
    ss_d, y_fit, mean, ss_d_error = get_jumps_sig(ss)

    # Padding and circular shifting
    if len(np.shape(signal_)) == 1:
        signal_padded = add_average_padding(signal, padding_length)
        shifted_signal_ = circular_shift(signal_padded, -1 * (int(np.round(mean) - (len(signal) // 2))))[padding_length:-padding_length]
    else:
        shifted_signal_ = []
        for i in range(len(signal_)):
            signal_padded = add_average_padding(signal_[i], padding_length)
            shifted_signal = circular_shift(signal_padded, -1 * (int(np.round(mean) - (len(signal) // 2))))[padding_length:-padding_length]
            shifted_signal_.append(shifted_signal)
        shifted_signal_ = np.array(shifted_signal_)
    return shifted_signal_

from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution

def align_signals(signal1, signal2, best_stretch=None, best_shift=None, best_sign=None):
    signal2_ = copy.copy(signal2)
    # Normalize the signals to have zero mean and unit variance
    signal1 = (signal1 - np.mean(signal1)) / np.std(signal1)
    signal2 = (signal2 - np.mean(signal2)) / np.std(signal2)
    
    # Define the error function to minimize
    def alignment_error(params):
        stretch, shift, sign = params
        
        # Apply stretch, shift, and sign to signal2
        length1 = len(signal1)
        x2 = np.linspace(0, 1, len(signal2))
        x2_new = (x2 - shift) / stretch
        
        # Interpolate signal2 to align with signal1
        f = interp1d(x2_new, sign * signal2, bounds_error=False, fill_value=0)
        
        # Create the interpolated version of signal2 with the same length as signal1
        x1 = np.linspace(0, 1, length1)
        signal2_aligned = f(x1)
        
        # Calculate the mean squared error
        return np.mean((signal1 - signal2_aligned) ** 2)

    # Use differential evolution for global optimization
    bounds = [(0.5, 2.0), (-0.5, 0.5), (-1, 1)]  # Bounds for stretch, shift, and sign
    best_stretch_, best_shift_, best_sign_ = None, None, None
    if best_stretch==None or best_shift==None or best_sign==None:
        result = differential_evolution(alignment_error, bounds)
        best_stretch_, best_shift_, best_sign_ = result.x
    # Extract the best stretch, shift, and sign values
    
    if best_stretch==None:
        best_stretch = best_stretch_
    if best_shift==None:
        best_shift = best_shift_
    if best_sign==None:
        best_sign = best_sign_
    # Now apply the best stretch, shift, and sign to signal2 to get the aligned signal
    length1 = len(signal1)
    x2 = np.linspace(0, 1, len(signal2))
    x2_new = (x2 - best_shift) / best_stretch
    
    # Interpolate signal2 using the optimal parameters
    f = interp1d(x2_new, best_sign * signal2_, bounds_error=False, fill_value=0)
    x1 = np.linspace(0, 1, length1)
    signal2_aligned = f(x1)
    
    return best_stretch, best_shift, best_sign, signal1, signal2_aligned

def align_signals_2(signal1, signal2, best_stretch=None, best_shift=None, best_sign=None):
    signal2_ = copy.copy(signal2)
    # Normalize the signals to have zero mean and unit variance
    signal1 = (signal1 - np.mean(signal1)) / np.std(signal1)
    signal2 = (signal2 - np.mean(signal2)) / np.std(signal2)
    
    # Define the error function to minimize
    def alignment_error(params):
        stretch, shift, sign = params
        
        # Apply stretch, shift, and sign to signal2
        length1 = len(signal1)
        x2 = np.linspace(0, 1, len(signal2))
        x2_new = (x2 - shift) / stretch
        
        # Interpolate signal2 to align with signal1
        f = interp1d(x2_new, sign * signal2, bounds_error=False, fill_value=0)
        
        # Create the interpolated version of signal2 with the same length as signal1
        x1 = np.linspace(0, 1, length1)
        signal2_aligned = f(x1)
        
        # Calculate the mean squared error
        return np.mean((signal1 - signal2_aligned) ** 2)

    # Use differential evolution for global optimization
    bounds = [(0.5, 2.0), (-0.5, 0.5), (-1, 1), (-1,1)]  # Bounds for stretch, shift, and sign
    result = differential_evolution(alignment_error, bounds)
    
    # Extract the best stretch, shift, and sign values
    best_stretch_, best_shift_, best_sign_ = result.x
    if best_stretch==None:
        best_stretch = best_stretch_
    if best_shift==None:
        best_shift = best_shift_
    if best_sign==None:
        best_sign = best_sign_
    # Now apply the best stretch, shift, and sign to signal2 to get the aligned signal
    length1 = len(signal1)
    x2 = np.linspace(0, 1, len(signal2))
    x2_new = (x2 - best_shift) / best_stretch
    
    # Interpolate signal2 using the optimal parameters
    f = interp1d(x2_new, best_sign * signal2_, bounds_error=False, fill_value=0)
    x1 = np.linspace(0, 1, length1)
    signal2_aligned = f(x1)
    
    return best_stretch, best_shift, best_sign, signal1, signal2_aligned


def align_signals_all(signal1, signal2, best_stretch=None, best_shift=None, best_sign=None):
    # Define the error function to minimize      
    signal1 = (signal1 - np.mean(signal1)) / np.std(signal1)
    signal2 = (signal2 - np.mean(signal2)) / np.std(signal2)
    def alignment_error(params):     
        stretch, shift, sign_x, sign_y, sign_z, bias_x, bias_y, bias_z, coef_x, coef_y, coef_z = params
        sum_error_all = 0
        for i in range(len(signal1)):
            signal1_ = signal1[i]
            signal2_ = signal2[i]    
            signal1_ = (signal1_ - np.mean(signal1_)) / np.std(signal1_)
            signal2_ = (signal2_ - np.mean(signal2_)) / np.std(signal2_)
            sum_sig_video = signal2 * 0.0
            for j in range(len(signal2_)):
                signal2_[j] = (signal2_[j] - np.mean(signal2_[j])) / np.std(signal2_[j])
                # Apply stretch, shift, and sign to signal2
                length1 = len(signal1_)
                x2 = np.linspace(0, 1, len(signal2_[j]))
                x2_new = (x2 - shift) / stretch
                x1 = np.linspace(0, 1, length1)
                # Interpolate signal2 to align with signal1
                if i%3 == 0:
                    f = interp1d(x2_new, sign_x * signal2_[j], bounds_error=False, fill_value=0)
                    
                    signal2_aligned = f(x1)
                    sum_sig_video += bias_x + coef_x * signal2_aligned
                if i%3 == 1:
                    f = interp1d(x2_new, sign_y * signal2_[j], bounds_error=False, fill_value=0)
                    signal2_aligned = f(x1)
                    sum_sig_video += bias_y + coef_y * signal2_aligned
                if i%3 == 2:
                    f = interp1d(x2_new, sign_z * signal2_[j], bounds_error=False, fill_value=0)  
                    signal2_aligned = f(x1)
                    sum_sig_video += bias_z + coef_z * signal2_aligned
            sum_error_all += np.mean((signal1_ - sum_sig_video) ** 2) 
        # Calculate the mean squared error
        print(sum_error_all)
        return sum_error_all

    # Normalize the signals to have zero mean and unit variance

    

    # Use differential evolution for global optimization
    bounds = [(0.5, 2.0), (-0.5, 0.5), (-1, 1), (-1, 1), (-1, 1), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5)]  # Bounds for stretch, shift, and sign
    result = differential_evolution(alignment_error, bounds, disp=True)
    
    stretch, shift, sign_x, sign_y, sign_z, bias_x, bias_y, bias_z, coef_x, coef_y, coef_z = result.x
    
    length1 = len(signal1[0])
    x2 = np.linspace(0, 1, len(signal2[0][0]))
    x2_new = (x2 - shift) / stretch
    x1 = np.linspace(0, 1, length1)
    # Interpolate signal2 using the optimal parameters

    signal2_out__ = []
    for i in range(len(signal2)):
        signal__ = []
        for j in range(len(signal2[i])): 
            if j %3 ==0:
                f = interp1d(x2_new, sign_x * signal2[i][j], bounds_error=False, fill_value=0)        
                signal2_out = bias_x + coef_x * f(x1)
            if j %3 ==1:
                f = interp1d(x2_new, sign_y * signal2[i][j], bounds_error=False, fill_value=0)        
                signal2_out = bias_y + coef_y * f(x1) 
            if j %3 ==2:
                f = interp1d(x2_new, sign_z * signal2[i][j], bounds_error=False, fill_value=0)        
                signal2_out = bias_z + coef_z * f(x1) 
            signal__.append(signal2_out)
        signal2_out__.append(signal__)
        
    return result, np.array(signal2_out__)

import torch
import torch.nn as nn
import torch.optim as optim

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x has shape (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)  # lstm_out has shape (batch_size, sequence_length, hidden_dim)
        out = self.fc(lstm_out)  # Final output has shape (batch_size, sequence_length, output_dim)
        return out

# Training function
def train_lstm_model(input_signal, target_signal, learning_rate=0.001, epochs=100, hidden_dim=128, num_layers=2):
    # Convert input and target signals to tensors
    input_signal_tensor = torch.tensor(input_signal, dtype=torch.float32)
    target_signal_tensor = torch.tensor(target_signal, dtype=torch.float32)

    # Transpose the input signal to match LSTM input shape: (batch_size, sequence_length, input_dim)
    input_signal_tensor = input_signal_tensor.permute(0, 2, 1)  # Shape: (80, 500, 9)

    # Get the dimensions of the input and output
    batch_size, sequence_length, num_channels = input_signal_tensor.shape
    _, output_sequence_length = target_signal_tensor.shape

    # Initialize the model
    model = LSTMModel(input_dim=num_channels, hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers)
    
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        
        optimizer.zero_grad()  # Zero gradients
        
        # Forward pass
        output = model(input_signal_tensor)  # Output shape: (batch_size, sequence_length, output_dim)
        
        # Reshape the output to match the target (batch_size, sequence_length)
        output = output.squeeze(-1)
        
        # Compute loss
        loss = criterion(output, target_signal_tensor)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')
    
    return model

def evaluate_model(model, input_signal):
    model.eval()  # Set the model to evaluation mode

    # Convert input signal to tensor and transpose it
    input_signal_tensor = torch.tensor(input_signal, dtype=torch.float32)
    input_signal_tensor = input_signal_tensor.permute(0, 2, 1)  # Shape: (batch_size, sequence_length, input_dim)

    with torch.no_grad():  # Disable gradient computation for evaluation
        output = model(input_signal_tensor)  # Output shape: (batch_size, sequence_length, output_dim)
        output = output.squeeze(-1)  # Squeeze to get shape (batch_size, sequence_length)
    
    return output.numpy()  # Convert output tensor back to NumPy array

from torch.utils.data import TensorDataset, random_split

def split_signals(input_signal, target_signal, train_ratio=0.8):
    """
    Splits the input and target signals into training and test sets.

    Args:
        input_signal (torch.Tensor): The input signal of shape (batch_size, num_channels, sequence_length).
        target_signal (torch.Tensor): The target signal of shape (batch_size, sequence_length).
        train_ratio (float): Proportion of the data to use for training (default is 0.8).

    Returns:
        train_input (torch.Tensor): Training set input signals.
        train_target (torch.Tensor): Training set target signals.
        test_input (torch.Tensor): Test set input signals.
        test_target (torch.Tensor): Test set target signals.
    """
    # Combine input and target signals into a dataset
    dataset = TensorDataset(input_signal, target_signal)
    
    # Calculate the sizes for training and test sets
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    
    # Split the dataset into training and test sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Unpack the input and target signals from the datasets
    train_input, train_target = zip(*train_dataset)
    test_input, test_target = zip(*test_dataset)
    
    # Convert the tuples back to tensors
    train_input = torch.stack(train_input)
    train_target = torch.stack(train_target)
    test_input = torch.stack(test_input)
    test_target = torch.stack(test_target)
    
    return train_input, train_target, test_input, test_target


# Define the Bidirectional LSTM model
#from tslearn.metrics import SoftDTWLossPyTorch
class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Multiply by 2 for bidirectional

    def forward(self, x):
        # x has shape (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)  # lstm_out has shape (batch_size, sequence_length, hidden_dim*2)
        out = self.fc(lstm_out)  # Final output has shape (batch_size, sequence_length, output_dim)
        return out

# Training function
def train_bilstm_model(input_signal, target_signal, learning_rate=0.001, epochs=100, hidden_dim=128, num_layers=2):
    # Convert input and target signals to tensors
    input_signal_tensor = torch.tensor(input_signal, dtype=torch.float32)
    target_signal_tensor = torch.tensor(target_signal, dtype=torch.float32)

    # Transpose the input signal to match LSTM input shape: (batch_size, sequence_length, input_dim)
    input_signal_tensor = input_signal_tensor.permute(0, 2, 1)  # Shape: (80, 500, 9)

    # Get the dimensions of the input and output
    batch_size, sequence_length, num_channels = input_signal_tensor.shape
    _, output_sequence_length = target_signal_tensor.shape

    # Initialize the bidirectional LSTM model
    model = BiLSTMModel(input_dim=num_channels, hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers)
    
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    # criterion = SoftDTWLossPyTorch(gamma=0.2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        
        optimizer.zero_grad()  # Zero gradients
        
        # Forward pass
        output = model(input_signal_tensor)  # Output shape: (batch_size, sequence_length, output_dim)
        
        # Reshape the output to match the target (batch_size, sequence_length)
        output = output.squeeze(-1)
        
        # Compute loss
        # print(output.shape)
        # print(target_signal_tensor.shape)
        # loss = criterion(output, target_signal_tensor.unsqueeze(2)).mean()
        loss = criterion(output, target_signal_tensor)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')
    
    return model


##############
# Attention LSTM

import torch
import torch.nn as nn
import torch.optim as optim

# Attention Layer over input features
class FeatureAttention(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(FeatureAttention, self).__init__()
        # Project hidden states to match input dimension
        self.hidden_proj = nn.Linear(hidden_dim * 2, input_dim)
        self.softmax = nn.Softmax(dim=2)  # Softmax over the feature dimension

    def forward(self, hidden_states, inputs):
        # hidden_states: (batch_size, sequence_length, hidden_dim * 2)
        # inputs: (batch_size, sequence_length, input_dim)

        # Project hidden states to input dimension
        hidden_proj = self.hidden_proj(hidden_states)  # Shape: (batch_size, sequence_length, input_dim)

        # Compute attention scores
        attn_scores = inputs * hidden_proj  # Element-wise multiplication

        # Compute attention weights
        attn_weights = self.softmax(attn_scores)  # Softmax over input_dim

        # Compute attended inputs
        attended_inputs = (inputs * attn_weights).sum(dim=2)  # Sum over input_dim

        return attended_inputs, attn_weights

# Define the Bidirectional LSTM with Attention model
class BiLSTMWithAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(BiLSTMWithAttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.attention = FeatureAttention(hidden_dim, input_dim)
        self.fc = nn.Linear(hidden_dim * 2 + 1, output_dim)  # Add 1 for the attended input

    def forward(self, x):
        # x has shape (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, sequence_length, hidden_dim * 2)
        
        # Apply attention over the input features
        attended_inputs, attn_weights = self.attention(lstm_out, x)  # attended_inputs: (batch_size, sequence_length)

        # Combine attended inputs with LSTM outputs
        attended_inputs = attended_inputs.unsqueeze(2)  # Shape: (batch_size, sequence_length, 1)
        combined = torch.cat((lstm_out, attended_inputs), dim=2)  # Shape: (batch_size, sequence_length, hidden_dim * 2 + 1)
        
        # Pass through final fully connected layer
        out = self.fc(combined).squeeze(-1)  # Final output shape: (batch_size, sequence_length)

        return out, attn_weights

# Training function
def train_bilstm_attention_model(input_signal, target_signal, learning_rate=0.001, epochs=100, hidden_dim=128, num_layers=2):
    # Convert input and target signals to tensors
    input_signal_tensor = torch.tensor(input_signal, dtype=torch.float32)
    target_signal_tensor = torch.tensor(target_signal, dtype=torch.float32)

    # Transpose the input signal to match LSTM input shape: (batch_size, sequence_length, input_dim)
    input_signal_tensor = input_signal_tensor.permute(0, 2, 1)  # Shape: (80, 500, 9)

    # Get the dimensions of the input and output
    batch_size, sequence_length, input_dim = input_signal_tensor.shape

    # Initialize the bidirectional LSTM with attention model
    model = BiLSTMWithAttentionModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers)
    
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        
        optimizer.zero_grad()  # Zero gradients
        
        # Forward pass
        output, attn_weights = model(input_signal_tensor)  # Output shape: (batch_size, sequence_length)
        
        # Compute loss
        loss = criterion(output, target_signal_tensor)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')
    
    return model

# Evaluation function
def evaluate_model_with_attention(model, input_signal):
    model.eval()  # Set the model to evaluation mode

    # Convert input signal to tensor and transpose to match LSTM input shape: (batch_size, sequence_length, input_dim)
    input_signal_tensor = torch.tensor(input_signal, dtype=torch.float32).permute(0, 2, 1)  # Shape: (batch_size, sequence_length, input_dim)
    
    with torch.no_grad():  # Disable gradient calculation during evaluation
        output, attn_weights = model(input_signal_tensor)  # Output shape: (batch_size, sequence_length)
    
    return output.numpy(), attn_weights.numpy()  # Return both the predictions and attention weights

# # Example usage:
# # Assuming input_signal has shape (80, 9, 500) and target_signal has shape (80, 500)
# input_signal = torch.randn(80, 9, 500)  # Example input signal
# target_signal = torch.randn(80, 500)    # Example target signal

# # Train the bidirectional LSTM with attention model
# model = train_bilstm_attention_model(input_signal, target_signal, epochs=100)

# # Evaluate the model on new input data and get attention weights
# new_input_signal = torch.randn(80, 9, 500)  # Example new input signal for evaluation
# predicted_output, attention_weights = evaluate_model_with_attention(model, new_input_signal)

# # Display the predicted output and attention weights
# print(predicted_output)
# print(attention_weights)

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def visualize_attention_weights_3d(attention_weights, sample_idx=0):
    """
    Visualize the attention weights over time and input features in 3D.
    
    Parameters:
    attention_weights (numpy array): The attention weights of shape (batch_size, sequence_length, input_dim).
    sample_idx (int): Index of the sample in the batch to visualize.
    """
    # Extract the attention weights for the specific sample
    attention_sample = attention_weights[sample_idx]  # Shape: (sequence_length, input_dim)
    
    # Define the meshgrid for plotting
    sequence_length = attention_sample.shape[0]
    input_dim = attention_sample.shape[1]
    
    time_steps = np.arange(sequence_length)  # X-axis (time steps)
    features = np.arange(input_dim)  # Y-axis (input features)
    
    # Create a meshgrid for plotting
    time_steps, features = np.meshgrid(time_steps, features)
    
    # Transpose the attention weights to fit the meshgrid shape
    attention_z = attention_sample.T  # Shape: (input_dim, sequence_length)
    
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    ax.plot_surface(time_steps, features, attention_z, cmap='viridis')

    # Set plot labels and title
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Input Feature')
    ax.set_zlabel('Attention Weight')
    ax.set_title(f'Attention Weights for Sample {sample_idx}')

    # Show the plot
    plt.show()


import torch
import torch.nn as nn
import torch.optim as optim

# Multi-Head Attention Layer
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_dim, input_dim, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query_proj = nn.Linear(hidden_dim * 2, hidden_dim)  # From LSTM output
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, hidden_states, inputs):
        batch_size, seq_length, _ = inputs.size()
        
        # Project the hidden states, keys, and values
        queries = self.query_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)
        keys = self.key_proj(inputs).view(batch_size, seq_length, self.num_heads, self.head_dim)
        values = self.value_proj(inputs).view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Compute attention scores: (batch_size, seq_length, num_heads, head_dim)
        attn_scores = torch.einsum('bqhd,bkhd->bhqk', queries, keys) / self.head_dim ** 0.5
        
        # Apply softmax to get attention weights
        attn_weights = self.softmax(attn_scores)
        attn_weights = self.dropout(attn_weights)  # Apply dropout to attention weights

        # Compute attention outputs
        attn_outputs = torch.einsum('bhqk,bkhd->bqhd', attn_weights, values)
        attn_outputs = attn_outputs.reshape(batch_size, seq_length, -1)  # Combine heads

        # Final projection and layer normalization
        attn_outputs = self.out_proj(attn_outputs)
        attn_outputs = self.layer_norm(attn_outputs + inputs)  # Apply residual connection

        return attn_outputs, attn_weights


# Bidirectional LSTM with Multi-Head Attention Model
class BiLSTMWithMultiHeadAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout_rate=0.1):
        super(BiLSTMWithMultiHeadAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.attention = MultiHeadAttention(num_heads, hidden_dim, input_dim, dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2 + input_dim, output_dim)  # Add input_dim for attended input

    def forward(self, x):
        # x has shape (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, sequence_length, hidden_dim * 2)

        # Apply multi-head attention over the input features
        attn_outputs, attn_weights = self.attention(lstm_out, x)  # attn_outputs: (batch_size, sequence_length, input_dim)

        # Combine attended inputs with LSTM outputs
        combined = torch.cat((lstm_out, attn_outputs), dim=2)  # Shape: (batch_size, sequence_length, hidden_dim * 2 + input_dim)
        
        # Pass through final fully connected layer
        out = self.fc(combined).squeeze(-1)  # Final output shape: (batch_size, sequence_length)

        return out, attn_weights


# Training function
def train_bilstm_multihead_attention(input_signal, target_signal, learning_rate=0.001, epochs=100, hidden_dim=128, num_layers=2, num_heads=4, dropout_rate=0.1):
    # Convert input and target signals to tensors
    input_signal_tensor = torch.tensor(input_signal, dtype=torch.float32)
    target_signal_tensor = torch.tensor(target_signal, dtype=torch.float32)

    # Transpose the input signal to match LSTM input shape: (batch_size, sequence_length, input_dim)
    input_signal_tensor = input_signal_tensor.permute(0, 2, 1)  # Shape: (80, 500, 9)

    # Get the dimensions of the input and output
    batch_size, sequence_length, input_dim = input_signal_tensor.shape

    # Initialize the bidirectional LSTM with multi-head attention model
    model = BiLSTMWithMultiHeadAttention(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, 
                                         num_layers=num_layers, num_heads=num_heads, dropout_rate=dropout_rate)
    
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        
        optimizer.zero_grad()  # Zero gradients
        
        # Forward pass
        output, attn_weights = model(input_signal_tensor)  # Output shape: (batch_size, sequence_length)
        
        # Compute loss
        loss = criterion(output, target_signal_tensor)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')
    
    return model


# Evaluation function
def evaluate_model_with_multihead_attention(model, input_signal):
    model.eval()  # Set the model to evaluation mode

    # Convert input signal to tensor and transpose to match LSTM input shape: (batch_size, sequence_length, input_dim)
    input_signal_tensor = torch.tensor(input_signal, dtype=torch.float32).permute(0, 2, 1)  # Shape: (batch_size, sequence_length, input_dim)
    
    with torch.no_grad():  # Disable gradient calculation during evaluation
        output, attn_weights = model(input_signal_tensor)  # Output shape: (batch_size, sequence_length)
    
    return output.numpy(), attn_weights.numpy()  # Return both the predictions and attention weights


import torch
import torch.nn as nn
import torch.optim as optim

# Multi-Head Attention Layer with Temporal Regularization
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_dim, input_dim, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query_proj = nn.Linear(hidden_dim * 2, hidden_dim)  # From LSTM output
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, hidden_states, inputs):
        batch_size, seq_length, _ = inputs.size()
        
        # Project the hidden states, keys, and values
        queries = self.query_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)
        keys = self.key_proj(inputs).view(batch_size, seq_length, self.num_heads, self.head_dim)
        values = self.value_proj(inputs).view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Compute attention scores: (batch_size, seq_length, num_heads, head_dim)
        attn_scores = torch.einsum('bqhd,bkhd->bhqk', queries, keys) / self.head_dim ** 0.5
        
        # Apply softmax to get attention weights
        attn_weights = self.softmax(attn_scores)
        attn_weights = self.dropout(attn_weights)  # Apply dropout to attention weights

        # Compute attention outputs
        attn_outputs = torch.einsum('bhqk,bkhd->bqhd', attn_weights, values)
        attn_outputs = attn_outputs.reshape(batch_size, seq_length, -1)  # Combine heads

        # Final projection and layer normalization
        attn_outputs = self.out_proj(attn_outputs)
        attn_outputs = self.layer_norm(attn_outputs + inputs)  # Apply residual connection

        return attn_outputs, attn_weights


# Bidirectional LSTM with Multi-Head Attention Model with Smooth Attention
class BiLSTMWithSmoothAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout_rate=0.1):
        super(BiLSTMWithSmoothAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.attention = MultiHeadAttention(num_heads, hidden_dim, input_dim, dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2 + input_dim, output_dim)  # Add input_dim for attended input

    def forward(self, x):
        # x has shape (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, sequence_length, hidden_dim * 2)

        # Apply multi-head attention over the input features
        attn_outputs, attn_weights = self.attention(lstm_out, x)  # attn_outputs: (batch_size, sequence_length, input_dim)

        # Combine attended inputs with LSTM outputs
        combined = torch.cat((lstm_out, attn_outputs), dim=2)  # Shape: (batch_size, sequence_length, hidden_dim * 2 + input_dim)
        
        # Pass through final fully connected layer
        out = self.fc(combined).squeeze(-1)  # Final output shape: (batch_size, sequence_length)

        return out, attn_weights


# Smoothness regularization function
def smooth_attention_loss(attn_weights):
    """
    Penalizes large changes in attention weights between consecutive time steps.
    This will encourage the attention weights to change smoothly over time.
    """
    diff = attn_weights[:, 1:, :] - attn_weights[:, :-1, :]  # Difference between consecutive time steps
    smoothness_loss = torch.sum(diff ** 2)  # L2 penalty on the differences
    return smoothness_loss


# Training function with smooth attention loss
def train_bilstm_smooth_attention(input_signal, target_signal, learning_rate=0.001, epochs=100, hidden_dim=128, num_layers=2, num_heads=4, dropout_rate=0.1, smoothness_lambda=0.1):
    # Convert input and target signals to tensors
    input_signal_tensor = torch.tensor(input_signal, dtype=torch.float32)
    target_signal_tensor = torch.tensor(target_signal, dtype=torch.float32)

    # Transpose the input signal to match LSTM input shape: (batch_size, sequence_length, input_dim)
    input_signal_tensor = input_signal_tensor.permute(0, 2, 1)  # Shape: (80, 500, 9)

    # Get the dimensions of the input and output
    batch_size, sequence_length, input_dim = input_signal_tensor.shape

    # Initialize the bidirectional LSTM with smooth attention model
    model = BiLSTMWithSmoothAttention(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, 
                                      num_layers=num_layers, num_heads=num_heads, dropout_rate=dropout_rate)
    
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        
        optimizer.zero_grad()  # Zero gradients
        
        # Forward pass
        output, attn_weights = model(input_signal_tensor)  # Output shape: (batch_size, sequence_length)
        
        # Compute the primary loss (MSE loss)
        loss = criterion(output, target_signal_tensor)
        
        # Compute the smoothness loss on the attention weights
        smooth_loss = smooth_attention_loss(attn_weights)
        
        # Total loss is the combination of the main loss and the smoothness regularization
        total_loss = loss + smoothness_lambda * smooth_loss
        
        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}, Smoothness Loss: {smooth_loss.item()}')
    
    return model


# Evaluation function
def evaluate_model_with_smooth_attention(model, input_signal):
    model.eval()  # Set the model to evaluation mode

    # Convert input signal to tensor and transpose to match LSTM input shape: (batch_size, sequence_length, input_dim)
    input_signal_tensor = torch.tensor(input_signal, dtype=torch.float32).permute(0, 2, 1)  # Shape: (batch_size, sequence_length, input_dim)
    
    with torch.no_grad():  # Disable gradient calculation during evaluation
        output, attn_weights = model(input_signal_tensor)  # Output shape: (batch_size, sequence_length)
    
    return output.numpy(), attn_weights.numpy()  # Return both the predictions and attention weights


# # Example usage:
# # Assuming input_signal has shape (80, 9, 500) and target_signal has shape (80, 500)
# input_signal = torch.randn(80, 9, 500)  # Example input signal
# target_signal = torch.randn(80, 500)    # Example target signal

# # Train the bidirectional LSTM with smooth attention model
# model = train_bilstm_smooth_attention(input_signal, target_signal, epochs=100)

# # Evaluate the model on new input data and get attention weights
# new_input_signal = torch.randn(80, 9, 500)  # Example new input signal for evaluation
# predicted_output, attention_weights = evaluate_model_with_smooth_attention(model, new_input_signal)

# # Display the predicted output and attention weights
# print(predicted_output)
# print(attention_weights)

import torch
import torch.nn as nn
import torch.optim as optim

# Attention Layer over input features
class FeatureAttention(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(FeatureAttention, self).__init__()
        # Project hidden states to match input dimension
        self.hidden_proj = nn.Linear(hidden_dim * 2, input_dim)
        self.softmax = nn.Softmax(dim=2)  # Softmax over the feature dimension

    def forward(self, hidden_states, inputs):
        # hidden_states: (batch_size, sequence_length, hidden_dim * 2)
        # inputs: (batch_size, sequence_length, input_dim)

        # Project hidden states to input dimension
        hidden_proj = self.hidden_proj(hidden_states)  # Shape: (batch_size, sequence_length, input_dim)

        # Compute attention scores
        attn_scores = inputs * hidden_proj  # Element-wise multiplication

        # Compute attention weights
        attn_weights = self.softmax(attn_scores)  # Softmax over input_dim

        # Compute attended inputs
        attended_inputs = (inputs * attn_weights).sum(dim=2)  # Sum over input_dim

        return attended_inputs, attn_weights

# Define the Bidirectional LSTM with Attention model
class BiLSTMWithAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(BiLSTMWithAttentionModel, self).__init__()
        self.lstm = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.attention = FeatureAttention(hidden_dim, input_dim)
        self.fc = nn.Linear(hidden_dim * 2 + 1, output_dim)  # Add 1 for the attended input

    def forward(self, x):
        # x has shape (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, sequence_length, hidden_dim * 2)
        
        # Apply attention over the input features
        attended_inputs, attn_weights = self.attention(lstm_out, x)  # attended_inputs: (batch_size, sequence_length)

        # Combine attended inputs with LSTM outputs
        attended_inputs = attended_inputs.unsqueeze(2)  # Shape: (batch_size, sequence_length, 1)
        combined = torch.cat((lstm_out, attended_inputs), dim=2)  # Shape: (batch_size, sequence_length, hidden_dim * 2 + 1)
        
        # Pass through final fully connected layer
        out = self.fc(combined).squeeze(-1)  # Final output shape: (batch_size, sequence_length)

        return out, attn_weights

# Smoothness Regularization: Penalizes large differences between consecutive attention weights
def smoothness_regularization(attn_weights):
    # attn_weights shape: (batch_size, sequence_length, input_dim)
    diff = attn_weights[:, 1:, :] - attn_weights[:, :-1, :]  # Compute differences between consecutive time steps
    smoothness_loss = torch.mean(diff ** 2)  # L2 loss to penalize large differences
    return smoothness_loss

# Training function with smooth attention regularization and GPU support
def train_bilstm_attention_model(input_signal, target_signal, learning_rate=0.001, epochs=100, hidden_dim=128, num_layers=2, smoothness_lambda=0.1):
    # Check for GPU availability and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert input and target signals to tensors and move them to the device
    input_signal_tensor = torch.tensor(input_signal, dtype=torch.float32).to(device)
    target_signal_tensor = torch.tensor(target_signal, dtype=torch.float32).to(device)

    # Transpose the input signal to match LSTM input shape: (batch_size, sequence_length, input_dim)
    input_signal_tensor = input_signal_tensor.permute(0, 2, 1)  # Shape: (80, 500, 9)

    # Get the dimensions of the input and output
    batch_size, sequence_length, input_dim = input_signal_tensor.shape

    # Initialize the bidirectional LSTM with attention model and move it to the device
    model = BiLSTMWithAttentionModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers).to(device)
    
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        
        optimizer.zero_grad()  # Zero gradients
        
        # Forward pass
        output, attn_weights = model(input_signal_tensor)  # Output shape: (batch_size, sequence_length)
        
        # Compute the main loss (MSE)
        mse_loss = criterion(output, target_signal_tensor)
        
        # Compute smoothness regularization loss
        smoothness_loss = smoothness_regularization(attn_weights)
        
        # Total loss is MSE loss + smoothness regularization loss
        total_loss = mse_loss + smoothness_lambda * smoothness_loss
        
        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {total_loss.item()}, MSE Loss: {mse_loss.item()}, Smoothness Loss: {smoothness_loss.item()}')
    
    return model

# Evaluation function with GPU support
def evaluate_model_with_attention(model, input_signal):
    model.eval()  # Set the model to evaluation mode

    # Check for GPU availability and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert input signal to tensor, transpose it, and move it to the device
    input_signal_tensor = torch.tensor(input_signal, dtype=torch.float32).permute(0, 2, 1).to(device)  # Shape: (batch_size, sequence_length, input_dim)
    
    with torch.no_grad():  # Disable gradient calculation during evaluation
        output, attn_weights = model(input_signal_tensor)  # Output shape: (batch_size, sequence_length)
    
    # Move the output and attention weights back to the CPU for further processing
    return output.cpu().numpy(), attn_weights.cpu().numpy()  # Return both the predictions and attention weights

# # Example usage:
# # Assuming input_signal has shape (80, 9, 500) and target_signal has shape (80, 500)
# input_signal = torch.randn(80, 9, 500)  # Example input signal
# target_signal = torch.randn(80, 500)    # Example target signal

# # Train the bidirectional LSTM with attention model and smooth attention regularization, with GPU support
# model = train_bilstm_attention_model(input_signal, target_signal, epochs=100, smoothness_lambda=0.1)

# # Evaluate the model on new input data and get attention weights, with GPU support
# new_input_signal = torch.randn(80, 9, 500)  # Example new input signal for evaluation
# predicted_output, attention_weights = evaluate_model_with_attention(model, new_input_signal)

# # Display the predicted output and attention weights
# print(predicted_output)
# print(attention_weights)


import matplotlib.pyplot as plt
import numpy as np

# Feature labels (you can customize these labels depending on your application)
feature_labels = [f"Feature {i+1}" for i in range(9)]
def plot_attention_weights_and_output(attention_weights, model_output, sample_idx=0, feature_labels=None):
    """
    Plots attention weights over time for all features using imshow, alongside the model output.
    
    Parameters:
    - attention_weights: numpy array of attention weights, shape (batch_size, sequence_length, input_dim)
    - model_output: numpy array of model output, shape (batch_size, sequence_length)
    - sample_idx: index of the sample in the batch to visualize
    - feature_labels: list of feature labels for the y-axis (optional)
    """
    # Select the attention weights and model output for the given sample
    attn_weights_sample = attention_weights[sample_idx]  # Shape: (sequence_length, input_dim)
    model_output_sample = model_output[sample_idx]  # Shape: (sequence_length)
    
    # Transpose attention weights to have features on the y-axis and time steps on the x-axis
    attn_weights_sample = attn_weights_sample.T  # Shape: (input_dim, sequence_length)
    
    # Create the figure and subplots
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Subplot 1: Attention weights heatmap
    cax = ax[0].imshow(attn_weights_sample, aspect='auto', cmap='hot', interpolation='nearest')
    
    # Set x and y labels for the heatmap
    ax[0].set_xlabel('Time Step')
    ax[0].set_ylabel('Feature')
    
    # Add feature labels to the y-axis, if provided
    if feature_labels:
        ax[0].set_yticks(np.arange(len(feature_labels)))
        ax[0].set_yticklabels(feature_labels)
    
    # Add a colorbar for the attention weights
    # fig.colorbar(cax, ax=ax[0], label='Attention Weight')
    
    # Set the title for the attention heatmap
    ax[0].set_title(f'Attention Weights Over Time for Sample {sample_idx}')
    
    # Subplot 2: Model output line plot
    ax[1].plot(model_output_sample, label='Model Output')
    
    # Set labels for the model output plot
    ax[1].set_xlabel('Time Step')
    ax[1].set_ylabel('Output Value')
    ax[1].set_title('Model Output Signal')
    
    # Add a tight layout to improve spacing
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
import torch
import torch.nn as nn
import torch.optim as optim

# Attention Layer over input features
class FeatureAttention(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(FeatureAttention, self).__init__()
        # Project CNN features to match input dimension
        self.hidden_proj = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)  # Softmax over the feature dimension (channels)

    def forward(self, features, inputs):
        # features: (batch_size, hidden_dim, sequence_length)
        # inputs: (batch_size, input_dim, sequence_length)
        
        # Project features to input dimension
        hidden_proj = self.hidden_proj(features)  # Shape: (batch_size, input_dim, sequence_length)

        # Compute attention scores
        attn_scores = inputs * hidden_proj  # Element-wise multiplication

        # Compute attention weights
        attn_weights = self.softmax(attn_scores)  # Softmax over input_dim (channels)

        # Compute attended inputs
        attended_inputs = (inputs * attn_weights).sum(dim=1)  # Sum over input_dim (channels)

        return attended_inputs, attn_weights

# Define the CNN with Attention model
class CNNWithAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CNNWithAttentionModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # Additional convolutional layers can be added here
        self.attention = FeatureAttention(hidden_dim, input_dim)
        self.fc = nn.Linear(hidden_dim + 1, output_dim)  # +1 for attended input

    def forward(self, x):
        # x has shape (batch_size, input_dim, sequence_length)
        features = self.conv1(x)  # features shape: (batch_size, hidden_dim, sequence_length)
        features = self.relu(features)

        # Apply attention over the input features
        attended_inputs, attn_weights = self.attention(features, x)  # attended_inputs: (batch_size, sequence_length)

        # Combine attended inputs with features
        attended_inputs = attended_inputs.unsqueeze(1)  # Shape: (batch_size, 1, sequence_length)
        combined = torch.cat((features, attended_inputs), dim=1)  # Shape: (batch_size, hidden_dim + 1, sequence_length)

        # Pass through final fully connected layer
        combined = combined.permute(0, 2, 1)  # Shape: (batch_size, sequence_length, hidden_dim + 1)
        out = self.fc(combined).squeeze(-1)  # Final output shape: (batch_size, sequence_length)

        return out, attn_weights

# Smoothness Regularization: Penalizes large differences between consecutive attention weights
def smoothness_regularization(attn_weights):
    # attn_weights shape: (batch_size, input_dim, sequence_length)
    attn_weights = attn_weights.permute(0, 2, 1)  # Shape: (batch_size, sequence_length, input_dim)
    diff = attn_weights[:, 1:, :] - attn_weights[:, :-1, :]  # Compute differences between consecutive time steps
    smoothness_loss = torch.mean(diff ** 2)  # L2 loss to penalize large differences
    return smoothness_loss

# Training function with smooth attention regularization and GPU support
def train_cnn_attention_model(input_signal, target_signal, learning_rate=0.001, epochs=100, hidden_dim=128, smoothness_lambda=0.1):
    # Check for GPU availability and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert input and target signals to tensors and move them to the device
    input_signal_tensor = torch.tensor(input_signal, dtype=torch.float32).to(device)
    target_signal_tensor = torch.tensor(target_signal, dtype=torch.float32).to(device)

    # Input signal shape should be (batch_size, input_dim, sequence_length)
    # No need to permute since input_signal is already in this shape (80, 9, 500)

    # Initialize the CNN with attention model and move it to the device
    input_dim = input_signal_tensor.shape[1]
    model = CNNWithAttentionModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1).to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()  # Zero gradients

        # Forward pass
        output, attn_weights = model(input_signal_tensor)  # Output shape: (batch_size, sequence_length)

        # Compute the main loss (MSE)
        mse_loss = criterion(output, target_signal_tensor)

        # Compute smoothness regularization loss
        smoothness_loss = smoothness_regularization(attn_weights)

        # Total loss is MSE loss + smoothness regularization loss
        total_loss = mse_loss + smoothness_lambda * smoothness_loss

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {total_loss.item()}, MSE Loss: {mse_loss.item()}, Smoothness Loss: {smoothness_loss.item()}')

    return model

# Evaluation function with GPU support
def evaluate_model_with_attention(model, input_signal):
    model.eval()  # Set the model to evaluation mode

    # Check for GPU availability and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert input signal to tensor and move it to the device
    input_signal_tensor = torch.tensor(input_signal, dtype=torch.float32).to(device)  # Shape: (batch_size, input_dim, sequence_length)

    with torch.no_grad():  # Disable gradient calculation during evaluation
        output, attn_weights = model(input_signal_tensor)  # Output shape: (batch_size, sequence_length)

    # Move the output and attention weights back to the CPU for further processing
    return output.cpu().numpy(), attn_weights.cpu().numpy()  # Return both the predictions and attention weights

# # Example usage:
# # Assuming input_signal has shape (80, 9, 500) and target_signal has shape (80, 500)
# input_signal = torch.randn(80, 9, 500)  # Example input signal
# target_signal = torch.randn(80, 500)    # Example target signal

# # Train the CNN with attention model and smooth attention regularization, with GPU support
# model = train_cnn_attention_model(input_signal, target_signal, epochs=100, smoothness_lambda=0.1)

# # Evaluate the model on new input data and get attention weights, with GPU support
# new_input_signal = torch.randn(80, 9, 500)  # Example new input signal for evaluation
# predicted_output, attention_weights = evaluate_model_with_attention(model, new_input_signal)

# # Display the predicted output and attention weights
# print(predicted_output)
# print(attention_weights)

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy  # For saving the best model

# Attention Layer over input features
class FeatureAttention(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(FeatureAttention, self).__init__()
        # Project hidden states to match input dimension
        self.hidden_proj = nn.Linear(hidden_dim * 2, input_dim)
        self.softmax = nn.Softmax(dim=2)  # Softmax over the feature dimension

    def forward(self, hidden_states, inputs):
        # hidden_states: (batch_size, sequence_length, hidden_dim * 2)
        # inputs: (batch_size, sequence_length, input_dim)

        # Project hidden states to input dimension
        hidden_proj = self.hidden_proj(hidden_states)  # Shape: (batch_size, sequence_length, input_dim)

        # Compute attention scores
        attn_scores = inputs * hidden_proj  # Element-wise multiplication

        # Compute attention weights
        attn_weights = self.softmax(attn_scores)  # Softmax over input_dim

        # Compute attended inputs
        attended_inputs = (inputs * attn_weights).sum(dim=2)  # Sum over input_dim

        return attended_inputs, attn_weights

# Define the Bidirectional LSTM with Attention model
class BiLSTMWithAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(BiLSTMWithAttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.attention = FeatureAttention(hidden_dim, input_dim)
        self.fc = nn.Linear(hidden_dim * 2 + 1, output_dim)  # Add 1 for the attended input

    def forward(self, x):
        # x has shape (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, sequence_length, hidden_dim * 2)
        
        # Apply attention over the input features
        attended_inputs, attn_weights = self.attention(lstm_out, x)  # attended_inputs: (batch_size, sequence_length)

        # Combine attended inputs with LSTM outputs
        attended_inputs = attended_inputs.unsqueeze(2)  # Shape: (batch_size, sequence_length, 1)
        combined = torch.cat((lstm_out, attended_inputs), dim=2)  # Shape: (batch_size, sequence_length, hidden_dim * 2 + 1)
        
        # Pass through final fully connected layer
        out = self.fc(combined).squeeze(-1)  # Final output shape: (batch_size, sequence_length)

        return out, attn_weights

# Smoothness Regularization: Penalizes large differences between consecutive attention weights
def smoothness_regularization(attn_weights):
    # attn_weights shape: (batch_size, sequence_length, input_dim)
    diff = attn_weights[:, 1:, :] - attn_weights[:, :-1, :]  # Compute differences between consecutive time steps
    smoothness_loss = torch.mean(diff ** 2)  # L2 loss to penalize large differences
    return smoothness_loss

# Training function with validation set, model checkpointing, and loss plotting
def train_bilstm_attention_model(input_signal, target_signal, validation_input, validation_target, 
                                 learning_rate=0.001, epochs=100, hidden_dim=128, num_layers=2, 
                                 smoothness_lambda=0.1):
    # Check for GPU availability and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert input and target signals to tensors and move them to the device
    input_signal_tensor = torch.tensor(input_signal, dtype=torch.float32).to(device)
    target_signal_tensor = torch.tensor(target_signal, dtype=torch.float32).to(device)

    validation_input_tensor = torch.tensor(validation_input, dtype=torch.float32).to(device)
    validation_target_tensor = torch.tensor(validation_target, dtype=torch.float32).to(device)

    # Transpose the input signal to match LSTM input shape: (batch_size, sequence_length, input_dim)
    input_signal_tensor = input_signal_tensor.permute(0, 2, 1)  # Shape: (80, 500, 9)
    validation_input_tensor = validation_input_tensor.permute(0, 2, 1)  # Shape: (80, 500, 9)

    # Get the dimensions of the input and output
    batch_size, sequence_length, input_dim = input_signal_tensor.shape

    # Initialize the bidirectional LSTM with attention model and move it to the device
    model = BiLSTMWithAttentionModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers).to(device)
    
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_model_wts = copy.deepcopy(model.state_dict())  # To store the best model weights
    best_val_loss = float('inf')  # To track the best validation loss

    # Lists to store training and validation loss for each epoch
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        
        optimizer.zero_grad()  # Zero gradients
        
        # Forward pass on training set
        output, attn_weights = model(input_signal_tensor)  # Output shape: (batch_size, sequence_length)
        
        # Compute the main loss (MSE)
        mse_loss = criterion(output, target_signal_tensor)
        
        # Compute smoothness regularization loss
        smoothness_loss = smoothness_regularization(attn_weights)
        
        # Total loss is MSE loss + smoothness regularization loss
        total_loss = mse_loss + smoothness_lambda * smoothness_loss
        
        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        # Validation step
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            val_output, _ = model(validation_input_tensor)
            val_loss = criterion(val_output, validation_target_tensor)
        
        # Track the losses
        train_losses.append(total_loss.item())
        val_losses.append(val_loss.item())
        
        # Check if this is the best model so far, based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())  # Save the best model weights

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Training Loss: {total_loss.item()}, Validation Loss: {val_loss.item()}')

    # Load the best model weights before returning
    model.load_state_dict(best_model_wts)

    # Plot the training and validation loss vs. epoch
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), train_losses, label='Training Loss')
    plt.plot(range(epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model

# Evaluation function with GPU support
def evaluate_model_with_attention(model, input_signal):
    model.eval()  # Set the model to evaluation mode

    # Check for GPU availability and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert input signal to tensor, transpose it, and move it to the device
    input_signal_tensor = torch.tensor(input_signal, dtype=torch.float32).permute(0, 2, 1).to(device)  # Shape: (batch_size, sequence_length, input_dim)
    
    with torch.no_grad():  # Disable gradient calculation during evaluation
        output, attn_weights = model(input_signal_tensor)  # Output shape: (batch_size, sequence_length)
    
    # Move the output and attention weights back to the CPU for further processing
    return output.cpu().numpy(), attn_weights.cpu().numpy()  # Return both the predictions and attention weights

# # Example usage:
# # Assuming input_signal has shape (80, 9, 500) and target_signal has shape (80, 500)
# input_signal = torch.randn(80, 9, 500)  # Example input signal
# target_signal = torch.randn(80, 500)    # Example target signal

# # Assuming validation_input has shape (80, 9, 500) and validation_target has shape (80, 500)
# validation_input = torch.randn(80, 9, 500)  # Example validation input
# validation_target = torch.randn(80, 500)    # Example validation target

# # Train the bidirectional LSTM with attention model and smooth attention regularization, with GPU support
# model = train_bilstm_attention_model(input_signal, target_signal, validation_input, validation_target, epochs=100, smoothness_lambda=0.1)

# # Evaluate the model on new input data and get attention weights, with GPU support
# new_input_signal = torch.randn(80, 9, 500)  # Example new input signal for evaluation
# predicted_output, attention_weights = evaluate_model_with_attention(model, new_input_signal)

# # Display the predicted output and attention weights
# print(predicted_output)
# print(attention_weights)

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy  # For saving the best model

# Attention Layer over input features
class FeatureAttention(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(FeatureAttention, self).__init__()
        # Project hidden states to match input dimension
        self.hidden_proj = nn.Linear(hidden_dim * 2, input_dim)
        self.softmax = nn.Softmax(dim=2)  # Softmax over the feature dimension

    def forward(self, hidden_states, inputs):
        # hidden_states: (batch_size, sequence_length, hidden_dim * 2)
        # inputs: (batch_size, sequence_length, input_dim)

        # Project hidden states to input dimension
        hidden_proj = self.hidden_proj(hidden_states)  # Shape: (batch_size, sequence_length, input_dim)

        # Compute attention scores
        attn_scores = inputs * hidden_proj  # Element-wise multiplication

        # Compute attention weights
        attn_weights = self.softmax(attn_scores)  # Softmax over input_dim

        # Compute attended inputs
        attended_inputs = (inputs * attn_weights).sum(dim=2)  # Sum over input_dim

        return attended_inputs, attn_weights

# Define the Bidirectional LSTM with Attention model
class BiLSTMWithAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(BiLSTMWithAttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.attention = FeatureAttention(hidden_dim, input_dim)
        self.fc = nn.Linear(hidden_dim * 2 + 1, output_dim)  # Add 1 for the attended input

    def forward(self, x):
        # x has shape (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, sequence_length, hidden_dim * 2)
        
        # Apply attention over the input features
        attended_inputs, attn_weights = self.attention(lstm_out, x)  # attended_inputs: (batch_size, sequence_length)

        # Combine attended inputs with LSTM outputs
        attended_inputs = attended_inputs.unsqueeze(2)  # Shape: (batch_size, sequence_length, 1)
        combined = torch.cat((lstm_out, attended_inputs), dim=2)  # Shape: (batch_size, sequence_length, hidden_dim * 2 + 1)
        
        # Pass through final fully connected layer (output_dim = 9 for multi-feature output)
        out = self.fc(combined)  # Final output shape: (batch_size, sequence_length, output_dim)

        return out, attn_weights

# Smoothness Regularization: Penalizes large differences between consecutive attention weights
def smoothness_regularization(attn_weights):
    # attn_weights shape: (batch_size, sequence_length, input_dim)
    diff = attn_weights[:, 1:, :] - attn_weights[:, :-1, :]  # Compute differences between consecutive time steps
    smoothness_loss = torch.mean(diff ** 2)  # L2 loss to penalize large differences
    return smoothness_loss

# Training function with validation set, model checkpointing, and loss plotting
def train_bilstm_attention_model(input_signal, target_signal, validation_input, validation_target, 
                                 learning_rate=0.001, epochs=100, hidden_dim=128, num_layers=2, 
                                 smoothness_lambda=0.1):
    # Check for GPU availability and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert input and target signals to tensors and move them to the device
    input_signal_tensor = torch.tensor(input_signal, dtype=torch.float32).to(device)
    target_signal_tensor = torch.tensor(target_signal, dtype=torch.float32).to(device)

    validation_input_tensor = torch.tensor(validation_input, dtype=torch.float32).to(device)
    validation_target_tensor = torch.tensor(validation_target, dtype=torch.float32).to(device)

    # Transpose the input signal to match LSTM input shape: (batch_size, sequence_length, input_dim)
    input_signal_tensor = input_signal_tensor.permute(0, 2, 1)  # Shape: (80, 500, 9)
    validation_input_tensor = validation_input_tensor.permute(0, 2, 1)  # Shape: (80, 500, 9)

    # Get the dimensions of the input and output
    batch_size, sequence_length, input_dim = input_signal_tensor.shape
    output_dim = target_signal.shape[2]  # Multi-feature output with 9 features

    # Initialize the bidirectional LSTM with attention model and move it to the device
    model = BiLSTMWithAttentionModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device)
    
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_model_wts = copy.deepcopy(model.state_dict())  # To store the best model weights
    best_val_loss = float('inf')  # To track the best validation loss

    # Lists to store training and validation loss for each epoch
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        
        optimizer.zero_grad()  # Zero gradients
        
        # Forward pass on training set
        output, attn_weights = model(input_signal_tensor)  # Output shape: (batch_size, sequence_length, output_dim)
        
        # Compute the main loss (MSE)
        mse_loss = criterion(output, target_signal_tensor)
        
        # Compute smoothness regularization loss
        smoothness_loss = smoothness_regularization(attn_weights)
        
        # Total loss is MSE loss + smoothness regularization loss
        total_loss = mse_loss + smoothness_lambda * smoothness_loss
        
        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        # Validation step
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            val_output, _ = model(validation_input_tensor)
            val_loss = criterion(val_output, validation_target_tensor)
        
        # Track the losses
        train_losses.append(total_loss.item())
        val_losses.append(val_loss.item())
        
        # Check if this is the best model so far, based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())  # Save the best model weights

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Training Loss: {total_loss.item()}, Validation Loss: {val_loss.item()}')

    # Load the best model weights before returning
    model.load_state_dict(best_model_wts)

    # Plot the training and validation loss vs. epoch
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), train_losses, label='Training Loss')
    plt.plot(range(epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model

# Evaluation function with GPU support
def evaluate_model_with_attention(model, input_signal):
    model.eval()  # Set the model to evaluation mode

    # Check for GPU availability and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert input signal to tensor, transpose it, and move it to the device
    input_signal_tensor = torch.tensor(input_signal, dtype=torch.float32).permute(0, 2, 1).to(device)  # Shape: (batch_size, sequence_length, input_dim)
    
    with torch.no_grad():  # Disable gradient calculation during evaluation
        output, attn_weights = model(input_signal_tensor)  # Output shape: (batch_size, sequence_length, output_dim)
    
    # Move the output and attention weights back to the CPU for further processing
    return output.cpu().numpy(), attn_weights.cpu().numpy()  # Return both the predictions and attention weights


import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.ndimage import shift
import numpy as np
import matplotlib.pyplot as plt
def traverse_single_time(doppler_data, start_time, num_neighbors=1):
    """
    Traverse through each subcarrier starting from the top (subcarrier 0) at the specified start_time.
    The algorithm selects the closest neighbor (in terms of value) by comparing with the median of the previous five subcarriers.
    
    Parameters:
    doppler_data (2D np.array): A 2D matrix representing Doppler velocities (rows=subcarriers, columns=time).
    start_time (int): The time step where the traversal should start from subcarrier 0.
    num_neighbors (int): The number of neighboring time steps to consider when finding the next point.
    
    Returns:
    list: A list of (subcarrier, time) coordinates that represent the traversal path.
    """
    num_subcarriers, num_time_steps = doppler_data.shape
    
    # Ensure start_time is within bounds
    if start_time < 0 or start_time >= num_time_steps:
        raise ValueError("Start time is out of range")

    # Initialize an empty list to store the path coordinates
    path = []

    # Start from subcarrier 0 at the specified start time
    current_subcarrier = 0
    current_time = start_time
    path.append((current_subcarrier, current_time))

    # Traverse through the subcarriers
    while current_subcarrier < num_subcarriers - 1:
        # Get the values of the last up to 5 subcarriers (or fewer if there are less than 5)
        previous_subcarriers = [doppler_data[sub, time] for sub, time in path[-5:]]
        median_value = np.median(previous_subcarriers)
        
        # Look at the neighbors in time around the current time
        time_range = range(max(0, current_time - num_neighbors), min(num_time_steps, current_time + num_neighbors + 1))
        
        # Find the closest neighbor in time by comparing with the median of the previous subcarriers
        closest_time = min(time_range, key=lambda t: abs(doppler_data[current_subcarrier + 1, t] - median_value))
        
        # Move to the next subcarrier and selected closest time
        current_subcarrier += 1
        current_time = closest_time
        path.append((current_subcarrier, current_time))

    return path

def plot_doppler_with_path(doppler_data, path):
    """
    Plot the Doppler data with the path points overlaid as red scatter points.
    
    Parameters:
    doppler_data (2D np.array): The Doppler velocity data.
    path (list): List of (subcarrier, time) coordinates for the path.
    """
    # Plot the Doppler data
    plt.imshow(doppler_data, aspect='auto', origin='lower', cmap='viridis')
    
    # Extract subcarrier and time indices from the path
    subcarriers, times = zip(*path)
    
    # Overlay the path as red scatter points
    plt.scatter(times, subcarriers, color='red', label='Selected Path', s=10)
    
    plt.colorbar(label='Doppler Velocity')
    plt.xlabel('Time')
    plt.ylabel('Subcarrier')
    plt.title('Doppler Data with Selected Path')
    plt.legend()
    plt.show()

import pulp
# Function to optimize paths with neighbor limitation
def optimize_paths(doppler_data, num_neighbors=1):
    num_subcarriers, num_time_steps = doppler_data.shape
    
    # Define the linear programming problem
    prob = pulp.LpProblem("DopplerPathOptimization", pulp.LpMinimize)
    
    # Define binary decision variables: x[i][j][k] = 1 if point (i,j) is in path k, 0 otherwise
    x = pulp.LpVariable.dicts("x", 
                              ((i, j, k) for i in range(num_subcarriers) 
                                           for j in range(num_time_steps) 
                                           for k in range(num_time_steps)),
                              cat='Binary')
    
    # Objective: minimize the total error across all paths
    objective = []
    for k in range(num_time_steps):  # for each path
        for i in range(num_subcarriers - 1):  # traverse all subcarriers except the last
            for j in range(num_time_steps):  # check all time points
                # Look at the neighboring time points within the range of num_neighbors
                for nj in range(max(0, j - num_neighbors), min(num_time_steps, j + num_neighbors + 1)):
                    # Calculate the error (difference)
                    error = abs(doppler_data[i+1][nj] - doppler_data[i][j])
                    # Add the term (error * decision variable) to the objective
                    # Instead of multiplying two variables, just handle one decision at a time
                    objective.append(error * x[i, j, k])
    
    # Set the objective function to minimize
    prob += pulp.lpSum(objective)
    
    # Constraints: each (i, j) point must be assigned to exactly one path
    for i in range(num_subcarriers):
        for j in range(num_time_steps):
            prob += pulp.lpSum(x[i, j, k] for k in range(num_time_steps)) == 1
    
    # Constraints: each path must contain exactly one point for each subcarrier
    for k in range(num_time_steps):
        for i in range(num_subcarriers):
            prob += pulp.lpSum(x[i, j, k] for j in range(num_time_steps)) == 1
    
    # Solve the problem
    prob.solve()

    # Extract the solution paths
    paths = []
    for k in range(num_time_steps):
        path = []
        for i in range(num_subcarriers):
            for j in range(num_time_steps):
                if pulp.value(x[i, j, k]) == 1:
                    path.append((i, j))
                    break
        paths.append(path)

    return paths

import numpy as np

def custom_ifft(x):
    N = len(x)
    result = np.zeros(N, dtype=complex)
    
    # Iterate over each output element
    for k in range(N):
        sum_value = 0.0
        # Sum over each input element
        for n in range(N):
            angle = 2 * np.pi * k * n / N
            sum_value += x[n] * np.exp(1j * angle)
        result[k] = sum_value / N
        
    return result

def pad_2d_array_along_axis(array, pad_width, axis, pad_value=0):
    """
    Pads a 2D numpy array along a specified axis.

    Parameters:
        array (numpy.ndarray): Input 2D array to pad.
        pad_width (tuple): Number of elements to pad on both sides (before, after) along the axis.
        axis (int): Axis to pad along (0 for rows, 1 for columns).
        pad_value: Value to use for padding. Default is 0.

    Returns:
        numpy.ndarray: Padded array.
    """
    if array.ndim != 2:
        raise ValueError("Input array must be 2D.")

    if axis not in [0, 1]:
        raise ValueError("Axis must be 0 (rows) or 1 (columns).")

    if len(pad_width) != 2:
        raise ValueError("pad_width must be a tuple of two integers.")

    pad_before, pad_after = pad_width

    if axis == 0:
        # Pad along rows
        padding = ((pad_before, pad_after), (0, 0))
    elif axis == 1:
        # Pad along columns
        padding = ((0, 0), (pad_before, pad_after))

    return np.pad(array, padding, constant_values=pad_value)

from scipy.signal.windows import kaiser
from sklearn.linear_model import LinearRegression
import numpy as np

def inv_nudft(t, freqs, X, scale="amplitude", weights=None,
              method="lstsq", lam=0.0, rcond=None):
    """
    Inverse NUDFT via linear solve / least squares.

    Forward (your NUDFT) was:
        X[m] = sum_n w[n] * x[n] * exp(-j 2π f_m t_n) / S
    where S depends on `scale`:
        scale="none"     -> S = 1
        scale="unitary"  -> S = sqrt(N)
        scale="amplitude"-> S = N
        scale="power"    -> S = N**2

    Here we solve for x given (t, freqs, X) by forming:
        E[m,n] = exp(-j 2π f_m t_n)          (shape M×N)
        E @ (w ⊙ x) ≈ S * X
    and then x = (E⁺ @ (S*X)) / w   (Moore–Penrose / LS, with optional Tikhonov).

    Parameters
    ----------
    t : (N,) array-like
        Sample times (seconds).
    freqs : (M,) array-like
        Frequencies (Hz) at which X was computed.
    X : (M,) array-like (complex)
        NUDFT spectrum values at `freqs` returned by the forward transform.
    scale : {"none","unitary","amplitude","power"}, default "amplitude"
        Must match what you used in the forward NUDFT.
    weights : (N,) array-like or None
        Same per-sample weights used in the forward. If None, uses ones.
    method : {"lstsq","tikhonov"}, default "lstsq"
        Solve with plain least squares, or Tikhonov (ridge) regularization.
    lam : float, default 0.0
        Tikhonov strength; only used if method="tikhonov".
    rcond : float or None
        Passed to np.linalg.lstsq.

    Returns
    -------
    x_hat : (N,) complex ndarray
        Reconstructed time samples at `t`.

    Notes
    -----
    • Exact recovery needs a well-conditioned E (typically M ≥ N, diverse freqs).
    • If `weights` contains zeros, inversion is impossible at those indices.
    • For noisy X, Tikhonov (lam>0) can improve stability.
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    freqs = np.asarray(freqs, dtype=np.float64).ravel()
    X = np.asarray(X).ravel().astype(np.complex128)

    N = t.size
    M = freqs.size
    if X.size != M:
        raise ValueError("X must have length equal to len(freqs)")

    # Match forward scaling
    if scale == "none":
        S = 1.0
    elif scale == "unitary":
        S = np.sqrt(N)
    elif scale == "amplitude":
        S = float(N)
    elif scale == "power":
        S = float(N**2)
    else:
        raise ValueError("scale must be one of {'none','unitary','amplitude','power'}")

    # Default weights (must match forward)
    if weights is None:
        w = np.ones(N, dtype=np.complex128)
    else:
        w = np.asarray(weights).ravel().astype(np.complex128)
        if w.size != N:
            raise ValueError("weights must have same length as t")
        if np.any(w == 0):
            raise ValueError("weights contain zeros; cannot invert.")

    # Build E (M×N): E[m,n] = exp(-j 2π f_m t_n)
    E = np.exp(-1j * 2.0 * np.pi * np.outer(freqs, t))

    # Solve for y = w ⊙ x  from  E y ≈ S * X
    b = S * X
    if method == "tikhonov" and lam > 0.0:
        # (EᴴE + lam I) y = Eᴴ b
        A = E.conj().T @ E
        y = np.linalg.solve(A + lam * np.eye(N, dtype=np.complex128), E.conj().T @ b)
    elif method == "lstsq":
        y, *_ = np.linalg.lstsq(E, b, rcond=rcond)
    else:
        raise ValueError("Unknown method. Use 'lstsq' or 'tikhonov'.")

    x_hat = y / w
    return x_hat

def sparse_ifft(signal, indices, N=64, axis=0):
    """
    Perform IFFT on a sparse signal where values are provided only at specific indices.
    
    Parameters:
    - signal: numpy array (num_indices, ...) containing the non-zero values (complex).
    - indices: numpy array (num_indices,) of integer indices (0 to N-1) where the values go.
    - N: int, the full length of the array for IFFT (default: 64).
    - axis: int, the axis along which to perform the IFFT (default: 0, for subcarriers x time).
    
    Returns:
    - result: numpy array of shape (N, ...) after IFFT.
    """
    # Validate inputs
    if signal.shape[axis] != len(indices):
        raise ValueError("Signal shape along axis must match the number of indices.")
    
    # Reconstruct full array with zeros
    shape = list(signal.shape)
    shape[axis] = N
    full = np.zeros(shape, dtype=complex)
    
    # Place signal values at specified indices
    np.put_along_axis(full, np.expand_dims(indices, axis=tuple(i for i in range(full.ndim) if i != axis)), signal, axis=axis)
    
    # Perform IFFT
    result = np.fft.ifft(full, n=N, axis=axis)
    
    return result

def sparse_fft(signal, indices, N=64, axis=0):
    """
    Perform FFT on a sparse signal where values are provided only at specific indices.
    
    Parameters:
    - signal: numpy array (num_indices, ...) containing the non-zero values (complex).
    - indices: numpy array (num_indices,) of integer indices (0 to N-1) where the values go.
    - N: int, the full length of the array for FFT (default: 64).
    - axis: int, the axis along which to perform the FFT (default: 0, for time x ...).
    
    Returns:
    - result: numpy array of shape (N, ...) after FFT.
    """
    # Validate inputs
    if signal.shape[axis] != len(indices):
        raise ValueError("Signal shape along axis must match the number of indices.")
    
    # Reconstruct full array with zeros
    shape = list(signal.shape)
    shape[axis] = N
    full = np.zeros(shape, dtype=complex)
    
    # Place signal values at specified indices
    np.put_along_axis(full, np.expand_dims(indices, axis=tuple(i for i in range(full.ndim) if i != axis)), signal, axis=axis)
    
    # Perform FFT
    result = np.fft.fft(full, n=N, axis=axis)
    
    return result

def decompose_csi(H, P, BW, fc, fn, N=64, window_length=11, beta_kaiser=8.0, plot_delay_profile=False, t0_index=None):
    """
    Decompose the CSI matrix into P components corresponding to different delays,
    and sort them based on the power in the Doppler profile for each component.
    """

    half_window = window_length // 2
    num_time_samples, N = H.shape
    N = 64 #64
    delta_tau = 1 / BW  # Delay resolution
    # delay_axis = np.fft.fftshift(np.arange(N) * delta_tau)  # Delays in seconds
    delta_f = BW / N
    # delay_axis = np.fft.fftshift(np.fft.fftfreq(N, d=delta_f))
    delay_axis = np.arange(N) / BW  # Delays in seconds
    # delay_axis_ns = delay_axis  # Convert delays to nanoseconds

    # delay_axis = delay_axis[np.array([i for i in range(64) if i not in [0,1,2,3,4,5,32,64,63,62,61,60,59]])]
    delay_axis = delay_axis[np.array([i for i in range(64)])]
    # delay_axis = delay_axis[np.array([i for i in range(256) if i not in [0, 1, 2, 3, 4, 5, 127, 128, 129, 250, 251, 252, 253, 254, 255]])]

    
    # N = 52#52
    # Select a fixed time t0 to analyze (e.g., middle of the time samples)
    if t0_index == None:
        t0_index = num_time_samples // 2  # Index corresponding to t0
    H_t0 = H[t0_index, :]  # CSI at time t0 across subcarriers

    # Step 1: Perform IFFT to get delay domain representation
    kept = np.array([i for i in range(64) if i not in [0,1,2,3,4,5,63,62,61,60,59,58,32]])
    # H_delay = np.fft.ifft(H_t0, n=N) * np.exp(2*np.pi*1j * fc * delay_axis)
    H_delay = sparse_ifft(H_t0, indices=kept, N=64) * np.exp(2*np.pi*1j * fc * delay_axis)

    # for ii in range(len(H_delay)):
    #     if np.angle(H_delay[ii])< 0:
    #         H_delay[ii] = H_delay[ii] * np.exp(1j * np.pi)
    # Step 2: Identify indices of the largest P components in the delay domain
    H_delay_abs = np.abs(H_delay)
    peak_indices = np.argsort(H_delay_abs)[::-1][:P]  # Indices of the largest P peaks

    estimated_delays = delay_axis[peak_indices]
    estimated_gains = H_delay[peak_indices]

    # Plot the delay profile using stem plot if requested
    if plot_delay_profile:
        delay_axis_ns = delay_axis  # Convert delays to nanoseconds
        plt.figure(figsize=(10, 4))
        plt.stem(delay_axis_ns, H_delay_abs / np.max(H_delay_abs))
        
        plt.plot(delay_axis_ns[peak_indices], H_delay_abs[peak_indices]/np.max(H_delay_abs), 'rx', label='Selected Peaks')
        window = np.zeros_like(H_t0)
        k_i = peak_indices[0]
        start_idx = max(k_i - half_window, 0)
        end_idx = min(k_i + half_window + 1, N)
        kaiser_window = kaiser(end_idx - start_idx, beta_kaiser)
        window[start_idx:end_idx] = kaiser_window
        window *= H_delay_abs[peak_indices[0]]
        # plt.plot(delay_axis_ns, window)
        plt.xlabel('Delay (ns)')
        plt.ylabel('Magnitude')
        # plt.ylim([-10,1500])
        plt.title('Delay Domain Response at {}'.format(t0_index))
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Plot the phase of the delay profile
        H_delay_phase = np.angle(H_delay)
        plt.figure(figsize=(10, 4))
        plt.stem(delay_axis_ns, H_delay_phase)
        plt.xlabel('Delay (ns)')
        plt.ylabel('Phase (radians)')
        plt.title('Delay Domain Phase Response at {}'.format(t0_index))
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(delay_axis_ns, np.unwrap(np.abs(H_delay_phase)))
        
        # X = [[-1.1], [0.2], [101.1], [0.3]]
        
        # clf = LocalOutlierFactor(n_neighbors=20)
        # outliers = clf.fit_predict(np.unwrap(H_delay_phase).reshape(-1,1))
        # print(outliers)
        svr_lin = LinearRegression()
        svr_lin.fit(np.array(delay_axis_ns).reshape(-1,1)[5:-5], np.unwrap(np.abs(H_delay_phase))[5:-5])
        H_delay_phase_fit = svr_lin.predict(np.array(delay_axis_ns).reshape(-1,1))
        H_delay_phase_new = np.exp(1j * (np.unwrap(H_delay_phase) - (H_delay_phase_fit) * np.sign(H_delay_phase)))
        plt.plot(delay_axis_ns,H_delay_phase_fit )
        plt.xlabel('Delay (ns)')
        plt.ylabel('Phase (radians) unwrap')
        plt.title('Delay Domain Phase Response at {}'.format(t0_index))
        plt.grid(True)
        plt.show()
        
        plt.figure(figsize=(10, 4))
        plt.stem(delay_axis_ns, np.angle(H_delay_phase_new))
        plt.xlabel('Delay (ns)')
        plt.ylabel('Phase (radians) _ fixed')
        plt.title('Delay Domain Phase Response at {}'.format(t0_index))
        plt.grid(True)
        plt.show()
        
    # Initialize list to store components
    H_components = [np.zeros((H.shape[0],64), dtype=np.complex64) for _ in range(P)]

        

    for idx in range(num_time_samples):
        H_ti = H[idx, :]
        # Step 1: Perform IFFT to get delay domain representation
        # H_delay_ti = np.fft.ifft(H_ti, n=N)
        H_delay_ti = sparse_ifft(H_ti, indices=kept, N=64)
        # print(H_delay_ti.shape)
        H_delay_phase = np.angle(H_delay_ti)
        svr_lin = LinearRegression()
        svr_lin.fit(np.array(delay_axis).reshape(-1,1)[5:-5], np.unwrap(np.abs(H_delay_phase))[5:-5])
        H_delay_phase_fit = svr_lin.predict(np.array(delay_axis).reshape(-1,1))
        H_delay_phase_new = np.exp(1j * (np.unwrap(H_delay_phase) - (H_delay_phase_fit) * np.sign(H_delay_phase)))
        
        H_delay_abs = np.abs(H_delay_ti)
        # H_delay_ti = H_delay_abs * H_delay_phase_new
        # peak_indices = np.argsort(H_delay_abs)[::-1][:P]  # Indices of the largest P peaks
        # print(peak_indices)
        estimated_delays = delay_axis[peak_indices]
        # Step 2: Extract each component using Kaiser window
        
        for i in range(P):
            # window = np.zeros_like(H_delay_ti)
            # k_i = peak_indices[i]
            # start_idx = max(k_i - half_window, 0)
            # end_idx = min(k_i + half_window + 1, N)
            # kaiser_window = kaiser(end_idx - start_idx, beta_kaiser)
            # window[start_idx:end_idx] = kaiser_window
            H_delay_i_ti = np.zeros(H_delay_ti.shape, dtype=np.complex128)
            H_delay_i_ti[i] = H_delay_ti[i]
            # print(H_delay_i_ti.shape)
            # tau_i = estimated_delays[i]
            # phase_shift = np.exp(1j * 2 * np.pi * (fc + fn) * tau_i)
            # print(np.angle(phase_shift))
            # print(np.mean(np.angle(H_delay_i_ti)))
            # if np.angle(H_delay_i_ti[i])< 0:
            #     H_delay_i_ti[i] = np.conjugate(H_delay_i_ti[i])
            H_n_i_ti_compensated = H_delay_i_ti * 1
            H_delay_i_ti = H_n_i_ti_compensated
            # Transform back to frequency domain
            H_n_i_ti = np.fft.fft(H_delay_i_ti, n=N)
            # Store the component
            H_components[i][idx, :] = H_n_i_ti

    # Compute Doppler power for each component
    doppler_powers = []
    for i in range(P):
        # Select a subcarrier (e.g., middle subcarrier)
        subcarrier_idx = N // 2
        component_time_series = H_components[i][:, subcarrier_idx]
        # Compute Doppler profile (FFT over time)
        doppler_profile = np.fft.fft(component_time_series)
        # Compute power (sum of squared magnitudes)
        power = np.sum(np.abs(doppler_profile)**2)
        doppler_powers.append(power)

    # Sort components based on Doppler power
    doppler_powers = np.array(doppler_powers)
    sorted_indices = np.argsort(doppler_powers)[::-1]  # Indices of components sorted by Doppler power

    # Sort components and corresponding delays and gains
    # H_components_sorted = [H_components[i] for i in sorted_indices]
    H_components_sorted = H_components
    estimated_delays_sorted = estimated_delays[sorted_indices]
    estimated_gains_sorted = estimated_gains[sorted_indices]

    # return H_components_sorted, estimated_delays_sorted, doppler_powers[sorted_indices]
    return H_components_sorted, delay_axis, H_delay_abs

from scipy.interpolate import interp1d

def resample_signal(signal: np.ndarray, target_samples: int, kind: str = 'linear') -> np.ndarray:
    """
    Resamples and interpolates a given 1D signal to have a specified number of samples.
    
    Parameters:
    signal (np.ndarray): The input 1D signal.
    target_samples (int): The desired number of samples in the output signal.
    kind (str): Type of interpolation. Options: 'linear', 'nearest', 'quadratic', 'cubic', etc.
                Default is 'linear'.
    
    Returns:
    np.ndarray: The resampled signal with `target_samples` points.
    """
    if len(signal) == 0:
        raise ValueError("Input signal is empty.")
    
    original_indices = np.linspace(0, 1, len(signal))
    target_indices = np.linspace(0, 1, target_samples)
    
    interpolator = interp1d(original_indices, signal, kind=kind, fill_value="extrapolate")
    resampled_signal = interpolator(target_indices)
    
    return resampled_signal
from scipy.optimize import linear_sum_assignment
import numpy as np
from scipy.optimize import linear_sum_assignment

def find_min_path(matrix):
    """
    Finds the minimum path in a 2D matrix, visiting each row and column exactly once.
    
    Parameters:
    matrix (list of list of int): 2D matrix representing the costs.

    Returns:
    min_cost (int): The minimum path cost.
    path (list of tuple): The path coordinates that result in the minimum cost.
    distances (list of int): The distance value for each pair in the path.
    """
    # Convert the input matrix to a numpy array if it's not already
    cost_matrix = np.array(matrix)
    
    # Use linear_sum_assignment to solve the assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Calculate the minimum path cost
    min_cost = cost_matrix[row_indices, col_indices].sum()
    
    # Construct the path based on row and column indices
    path = list(zip(row_indices, col_indices))
    
    # Get the distances (costs) for each pair in the path
    distances = cost_matrix[row_indices, col_indices].tolist()
    
    return min_cost, path, distances
import numpy as np
import numpy as np
import numpy as np
import numpy as np

import torch

# def estimate_velocity_from_radial(v_r, T, N, lambda_reg=0.1, gamma=0.01, sigma_x2=1.0, sigma_y2=1.0, sigma_z2=1.0, 
#                                   tolerance=1e-5, max_iter=1000, lr=0.01, huber_threshold=1.0, temporal_smoothing=0.1, 
#                                   independence_penalty=0.1, smoothness_penalty=0.1, exclusivity_penalty=0.1,
#                                   magnitude_independence_penalty=0.1, distance_ratios=None, camera_power=None):
#     """
#     Estimate the 3D velocity vector from radial velocities observed by multiple cameras,
#     with constraints for mean-zero velocity, robustness against noise and outliers, independence between velocity
#     components, temporal smoothness, mutual exclusivity across axes, independence of magnitudes, distance ratio
#     constraints on the camera positions, and robustness to noise levels via camera power weights using PyTorch.
    
#     Parameters:
#     - v_r (np.ndarray): Observed radial velocities, shape (T, N), where T is the number of time steps and N is the number of cameras.
#     - T (int): Number of time steps.
#     - N (int): Number of cameras.
#     - lambda_reg (float): Regularization parameter for temporal smoothness of direction vectors.
#     - gamma (float): Regularization parameter for independence of velocity components.
#     - sigma_x2 (float): Variance of the x-component of the velocity.
#     - sigma_y2 (float): Variance of the y-component of the velocity.
#     - sigma_z2 (float): Variance of the z-component of the velocity.
#     - tolerance (float): Convergence tolerance for the optimization.
#     - max_iter (int): Maximum number of iterations for the optimization.
#     - lr (float): Learning rate for the optimizer.
#     - huber_threshold (float): Threshold for Huber loss; controls sensitivity to outliers.
#     - temporal_smoothing (float): Smoothing factor for temporal averaging of velocity estimates.
#     - independence_penalty (float): Regularization parameter for penalizing correlation between velocity components.
#     - smoothness_penalty (float): Regularization parameter for temporal smoothness of velocity components.
#     - exclusivity_penalty (float): Regularization parameter for penalizing simultaneous high values across multiple axes.
#     - magnitude_independence_penalty (float): Regularization parameter for penalizing correlated magnitudes between axes.
#     - distance_ratios (np.ndarray or None): Relative distance ratios for the cameras, shape (N,). If None, this constraint is ignored.
#     - camera_power (np.ndarray or None): Power levels for each camera, shape (N,). If None, equal power is assumed for all.
    
#     Returns:
#     - v (torch.Tensor): Estimated 3D velocity vectors over time, shape (T, 3).
#     - r_hat (torch.Tensor): Estimated direction vectors for each camera, shape (N, 3).
#     - weights (torch.Tensor): Final weights assigned to each camera, shape (N,).
#     """
    
#     # Convert v_r and distance_ratios to PyTorch tensors
#     v_r = torch.tensor(v_r, dtype=torch.float32)
    
#     # Initialize the 3D velocity vector v and the direction vectors r_hat randomly with requires_grad=True
#     v = torch.randn((T, 3), requires_grad=True)  # Initialize 3D velocity vector at each time step
#     r_hat = torch.randn((N, 3), requires_grad=True)  # Initialize direction vectors randomly

#     # Normalize distance_ratios if provided
#     if distance_ratios is not None:
#         distance_ratios = torch.tensor(distance_ratios, dtype=torch.float32)
#         distance_ratios = distance_ratios / torch.norm(distance_ratios)
    
#     # If camera_power is provided, calculate initial weights; otherwise, assume equal weights
#     if camera_power is not None:
#         camera_power = torch.tensor(camera_power, dtype=torch.float32)
#         weights = camera_power
#         weights = weights / torch.max(weights)  # Normalize weights
#     else:
#         weights = torch.ones(N, dtype=torch.float32)

#     # Define the optimizer
#     optimizer = torch.optim.Adam([v, r_hat], lr=lr)

#     # Huber loss function
#     def huber_loss(residual, delta):
#         return torch.where(torch.abs(residual) <= delta, 
#                            0.5 * residual ** 2, 
#                            delta * (torch.abs(residual) - 0.5 * delta))

#     # Optimization loop
#     prev_loss = float('inf')
#     for iter in range(max_iter):
#         optimizer.zero_grad()

#         # Main loss function (reconstruction error with weights)
#         weighted_diff = (v_r - torch.matmul(v, r_hat.T)) * weights
#         loss = huber_loss(weighted_diff, huber_threshold).sum()

#         # Mean-zero constraint on v (enforce zero mean for each component)
#         mean_zero_loss = torch.sum(v.mean(dim=0) ** 2)
#         loss += lambda_reg * mean_zero_loss

#         # Temporal smoothing constraint on v
#         smoothness_loss = torch.sum((v[1:] - v[:-1]) ** 2)
#         loss += smoothness_penalty * smoothness_loss

#         # Independence penalty to minimize correlation between velocity components
#         independence_loss = torch.sum(v[:, 0] * v[:, 1] + v[:, 0] * v[:, 2] + v[:, 1] * v[:, 2])
#         loss += independence_penalty * independence_loss

#         # Exclusivity constraint to encourage only one axis to be high at any time
#         exclusivity_loss = torch.sum(v[:, 0] * v[:, 1] * v[:, 2])
#         loss += exclusivity_penalty * exclusivity_loss

#         # Magnitude independence constraint
#         magnitude_independence_loss = torch.sum(torch.abs(v[:, 0]) * torch.abs(v[:, 1]) + 
#                                                 torch.abs(v[:, 0]) * torch.abs(v[:, 2]) + 
#                                                 torch.abs(v[:, 1]) * torch.abs(v[:, 2]))
#         loss += magnitude_independence_penalty * magnitude_independence_loss

#         # Apply distance ratios to r_hat
#         if distance_ratios is not None:
#             current_magnitudes = torch.norm(r_hat, dim=1)
#             distance_ratio_loss = torch.sum((current_magnitudes - distance_ratios) ** 2)
#             loss += lambda_reg * distance_ratio_loss

#         # Backpropagation and optimization step
#         loss.backward()
#         optimizer.step()

#         # Renormalize r_hat to ensure it remains a unit vector
#         with torch.no_grad():
#             r_hat.div_(torch.norm(r_hat, dim=1, keepdim=True))

#         # Check for convergence based on loss change
#         # if abs(prev_loss - loss.item()) < tolerance:
#         #     print(f"Converged after {iter+1} iterations with loss {loss.item()}")
#         #     break
#         prev_loss = loss.item()

#         # Optional: Print loss every 100 iterations for tracking progress
#         if iter % 100 == 0:
#             print(f"Iteration {iter}, Loss: {loss.item()}")

#     return v.detach(), r_hat.detach(), weights

import torch

# def estimate_velocity_from_radial(v_r, T, N, lambda_reg=0.1, gamma=0.01, sigma_x2=1.0, sigma_y2=1.0, sigma_z2=1.0, 
#                                   tolerance=1e-5, max_iter=1000, lr=0.01, huber_threshold=1.0, temporal_smoothing=0.1, 
#                                   independence_penalty=0.1, smoothness_penalty=0.1, exclusivity_penalty=0.1,
#                                   magnitude_independence_penalty=0.1, distance_ratios=None, camera_power=None):
#     """
#     Estimate the 3D velocity vector from radial velocities observed by multiple cameras,
#     with constraints for mean-zero velocity, robustness against noise and outliers, independence between velocity
#     components, temporal smoothness, mutual exclusivity across axes, independence of magnitudes, distance ratio
#     constraints on the camera positions, and robustness to noise levels via camera power weights using PyTorch.
    
#     Parameters:
#     - v_r (np.ndarray): Observed radial velocities, shape (T, N), where T is the number of time steps and N is the number of cameras.
#     - T (int): Number of time steps.
#     - N (int): Number of cameras.
#     - lambda_reg (float): Regularization parameter for temporal smoothness of direction vectors.
#     - gamma (float): Regularization parameter for independence of velocity components.
#     - sigma_x2 (float): Variance of the x-component of the velocity.
#     - sigma_y2 (float): Variance of the y-component of the velocity.
#     - sigma_z2 (float): Variance of the z-component of the velocity.
#     - tolerance (float): Convergence tolerance for the optimization.
#     - max_iter (int): Maximum number of iterations for the optimization.
#     - lr (float): Learning rate for the optimizer.
#     - huber_threshold (float): Threshold for Huber loss; controls sensitivity to outliers.
#     - temporal_smoothing (float): Smoothing factor for temporal averaging of velocity estimates.
#     - independence_penalty (float): Regularization parameter for penalizing correlation between velocity components.
#     - smoothness_penalty (float): Regularization parameter for temporal smoothness of velocity components.
#     - exclusivity_penalty (float): Regularization parameter for penalizing simultaneous high values across multiple axes.
#     - magnitude_independence_penalty (float): Regularization parameter for penalizing correlated magnitudes between axes.
#     - distance_ratios (np.ndarray or None): Relative distance ratios for the cameras, shape (N,). If None, this constraint is ignored.
#     - camera_power (np.ndarray or None): Power levels for each camera, shape (N,). If None, equal power is assumed for all.
    
#     Returns:
#     - v (torch.Tensor): Estimated 3D velocity vectors over time, shape (T, 3).
#     - r_hat (torch.Tensor): Estimated direction vectors for each camera, shape (N, 3).
#     - weights (torch.Tensor): Final weights assigned to each camera, shape (N,).
#     """
    
#     # Convert v_r and distance_ratios to PyTorch tensors
#     v_r = torch.tensor(v_r, dtype=torch.float32)
    
#     # Initialize the 3D velocity vector v and the direction vectors r_hat randomly with requires_grad=True
#     v = torch.randn((T, 3), requires_grad=True)  # Initialize 3D velocity vector at each time step
#     r_hat = torch.randn((N, 3), requires_grad=True)  # Initialize direction vectors randomly

#     # Normalize distance_ratios if provided
#     if distance_ratios is not None:
#         distance_ratios = torch.tensor(distance_ratios, dtype=torch.float32)

#     # If camera_power is provided, use it directly as weights; otherwise, assume equal weights
#     if camera_power is not None:
#         camera_power = torch.tensor(camera_power, dtype=torch.float32)
#         weights = camera_power / torch.max(camera_power)  # Normalize weights to be between 0 and 1
#     else:
#         weights = torch.ones(N, dtype=torch.float32)

#     # Define the optimizer
#     optimizer = torch.optim.Adam([v, r_hat], lr=lr)

#     # Huber loss function
#     def huber_loss(residual, delta):
#         return torch.where(torch.abs(residual) <= delta, 
#                             0.5 * residual ** 2, 
#                             delta * (torch.abs(residual) - 0.5 * delta))

#     # Optimization loop
#     prev_loss = float('inf')
#     for iter in range(max_iter):
#         optimizer.zero_grad()

#         # Main loss function (reconstruction error with weights)
#         weighted_diff = (v_r - torch.matmul(v, r_hat.T)) * weights
#         loss = huber_loss(weighted_diff, huber_threshold).sum()

#         # Mean-zero constraint on v (enforce zero mean for each component)
#         mean_zero_loss = torch.sum(v.mean(dim=0) ** 2)
#         loss += lambda_reg * mean_zero_loss

#         # Temporal smoothing constraint on v
#         smoothness_loss = torch.sum((v[1:] - v[:-1]) ** 2)
#         loss += smoothness_penalty * smoothness_loss

#         # Independence penalty to minimize correlation between velocity components
#         independence_loss = torch.sum(v[:, 0] * v[:, 1] + v[:, 0] * v[:, 2] + v[:, 1] * v[:, 2])
#         loss += independence_penalty * independence_loss

#         # Exclusivity constraint to encourage only one axis to be high at any time
#         exclusivity_loss = torch.sum(v[:, 0] * v[:, 1] * v[:, 2])
#         loss += exclusivity_penalty * exclusivity_loss

#         # Magnitude independence constraint
#         magnitude_independence_loss = torch.sum(torch.abs(v[:, 0]) * torch.abs(v[:, 1]) + 
#                                                 torch.abs(v[:, 0]) * torch.abs(v[:, 2]) + 
#                                                 torch.abs(v[:, 1]) * torch.abs(v[:, 2]))
#         loss += magnitude_independence_penalty * magnitude_independence_loss

#         # Apply distance ratios to r_hat
#         if distance_ratios is not None:
#             # Calculate the current magnitudes of r_hat
#             current_magnitudes = torch.norm(r_hat, dim=1)
#             # Calculate the pairwise ratios between current magnitudes and compare to distance_ratios
#             distance_ratio_loss = 0.0
#             for i in range(N):
#                 for j in range(i + 1, N):
#                     desired_ratio = distance_ratios[i] / distance_ratios[j]
#                     current_ratio = current_magnitudes[i] / current_magnitudes[j]
#                     distance_ratio_loss += (desired_ratio - current_ratio) ** 2
#             loss += lambda_reg * distance_ratio_loss

#         # Backpropagation and optimization step
#         loss.backward()
#         optimizer.step()

#         # Renormalize r_hat to ensure it remains a unit vector
#         with torch.no_grad():
#             r_hat.div_(torch.norm(r_hat, dim=1, keepdim=True))

#         # Check for convergence based on loss change
#         if abs(prev_loss - loss.item()) < tolerance:
#             print(f"Converged after {iter+1} iterations with loss {loss.item()}")
#             break
#         prev_loss = loss.item()

#         # Optional: Print loss every 100 iterations for tracking progress
#         if iter % 100 == 0:
#             print(f"Iteration {iter}, Loss: {loss.item()}")

#     return v.detach(), r_hat.detach(), weights


# def estimate_velocity_from_radial(v_r, T, N, lambda_reg=0.1, gamma=0.01, sigma_x2=1.0, sigma_y2=1.0, sigma_z2=1.0, 
#                                   tolerance=1e-3, max_iter=100, huber_threshold=1.0, temporal_smoothing=0.1, 
#                                   independence_penalty=0.1, smoothness_penalty=0.1, exclusivity_penalty=0.1,
#                                   magnitude_independence_penalty=0.1, distance_ratios=None, camera_power=None):
#     """
#     Estimate the 3D velocity vector from radial velocities observed by multiple cameras,
#     with constraints for mean-zero velocity, robustness against noise and outliers, independence between velocity
#     components, temporal smoothness, mutual exclusivity across axes, independence of magnitudes, distance ratio
#     constraints on the camera positions, and robustness to noise levels via camera power weights.
    
#     Parameters:
#     - v_r (np.ndarray): Observed radial velocities, shape (T, N), where T is the number of time steps and N is the number of cameras.
#     - T (int): Number of time steps.
#     - N (int): Number of cameras.
#     - lambda_reg (float): Regularization parameter for temporal smoothness of direction vectors.
#     - gamma (float): Regularization parameter for independence of velocity components.
#     - sigma_x2 (float): Variance of the x-component of the velocity.
#     - sigma_y2 (float): Variance of the y-component of the velocity.
#     - sigma_z2 (float): Variance of the z-component of the velocity.
#     - tolerance (float): Convergence tolerance for the optimization.
#     - max_iter (int): Maximum number of iterations for the alternating optimization.
#     - huber_threshold (float): Threshold for Huber loss; controls sensitivity to outliers.
#     - temporal_smoothing (float): Smoothing factor for temporal averaging of velocity estimates.
#     - independence_penalty (float): Regularization parameter for penalizing correlation between velocity components.
#     - smoothness_penalty (float): Regularization parameter for temporal smoothness of velocity components.
#     - exclusivity_penalty (float): Regularization parameter for penalizing simultaneous high values across multiple axes.
#     - magnitude_independence_penalty (float): Regularization parameter for penalizing correlated magnitudes between axes.
#     - distance_ratios (np.ndarray or None): Relative distance ratios for the cameras, shape (N,). If None, this constraint is ignored.
#     - camera_power (np.ndarray or None): Power levels for each camera, shape (N,). If None, equal power is assumed for all.
    
#     Returns:
#     - v (np.ndarray): Estimated 3D velocity vectors over time, shape (T, 3).
#     - r_hat (np.ndarray): Estimated direction vectors for each camera, shape (N, 3).
#     - weights (np.ndarray): Final weights assigned to each camera, shape (N,).
#     """
    
#     # Initialize the 3D velocity vector v and the direction vectors r_hat randomly
#     v = np.random.randn(T, 3)      # Initialize 3D velocity vector at each time step
#     r_hat = np.random.randn(N, 3)  # Initialize direction vectors randomly
    
#     # Normalize r_hat to be unit vectors initially
#     r_hat = r_hat / np.linalg.norm(r_hat, axis=1, keepdims=True)

#     # Initialize weights for each camera based on power if provided, otherwise set equal weights
#     if camera_power is not None:
#         # Compute weights inversely proportional to the camera power
#         weights = np.array(camera_power)
#         weights /= np.max(weights)  # Normalize weights to be between 0 and 1
#     else:
#         weights = np.ones(N)

#     # Normalize the distance_ratios if provided
#     if distance_ratios is not None:
#         distance_ratios = np.array(distance_ratios) / np.linalg.norm(distance_ratios)

#     # Function to apply Huber loss on residuals
#     def huber(residual, delta):
#         return np.where(np.abs(residual) <= delta, 
#                         0.5 * residual**2, 
#                         delta * (np.abs(residual) - 0.5 * delta))
    
#     v_old = np.copy(v)

#     # Alternating optimization
#     for iter in range(max_iter):
#         # Step 1: Optimize v(t) for each time step t, given fixed r_hat and weights
#         for t in range(T):
#             # Matrix A and vector b for weighted least squares
#             A = r_hat * weights[:, np.newaxis]  # Shape: (N, 3)
#             b = v_r[t] * weights  # Shape: (N,)
            
#             # Regularization matrix for independence of components
#             L = np.diag([gamma / sigma_x2, gamma / sigma_y2, gamma / sigma_z2])
            
#             # Solve weighted regularized least squares: (A^T A + L) v(t) = A^T b
#             AtA = A.T @ A + L
#             Atb = A.T @ b
#             v[t] = np.linalg.solve(AtA, Atb)
        
#         # Enforce mean-zero constraint on v
#         v_mean = np.mean(v, axis=0)
#         v -= v_mean  # Subtract mean from each component
        
#         # Temporal smoothing of velocities
#         if iter > 0:  # Apply smoothing only after the first iteration
#             v = (1 - temporal_smoothing) * v + temporal_smoothing * v_old
        
#         # Independence regularization
#         # Penalize correlations between velocity components
#         for t in range(T):
#             v[t, 0] -= independence_penalty * (v[t, 1] + v[t, 2])
#             v[t, 1] -= independence_penalty * (v[t, 0] + v[t, 2])
#             v[t, 2] -= independence_penalty * (v[t, 0] + v[t, 1])
        
#         # Temporal smoothness regularization
#         for t in range(1, T - 1):
#             v[t] += smoothness_penalty * (v[t-1] - 2 * v[t] + v[t+1])
        
#         # Mutual exclusivity regularization
#         for t in range(T):
#             v[t, 0] -= exclusivity_penalty * (v[t, 1] * v[t, 2])
#             v[t, 1] -= exclusivity_penalty * (v[t, 0] * v[t, 2])
#             v[t, 2] -= exclusivity_penalty * (v[t, 0] * v[t, 1])
        
#         # Magnitude independence regularization
#         for t in range(T):
#             v[t, 0] -= magnitude_independence_penalty * (abs(v[t, 1]) * abs(v[t, 2]))
#             v[t, 1] -= magnitude_independence_penalty * (abs(v[t, 0]) * abs(v[t, 2]))
#             v[t, 2] -= magnitude_independence_penalty * (abs(v[t, 0]) * abs(v[t, 1]))
        
#         # Step 2: Optimize r_hat for each camera, given fixed v(t) and distance ratios
#         for i in range(N):
#             # Solve constrained optimization for each r_hat[i] with weights
#             b = v_r[:, i] * weights[i]
#             A = v  # Shape: (T, 3)
            
#             # Solve least squares for r_hat[i] without constraints initially
#             r_hat_i, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
#             # Apply distance ratio constraint if provided
#             if distance_ratios is not None:
#                 target_magnitude = distance_ratios[i]
#                 current_magnitude = np.linalg.norm(r_hat_i)
#                 if current_magnitude > 0:
#                     r_hat_i = (target_magnitude / current_magnitude) * r_hat_i  # Adjust magnitude to match ratio
            
#             # Ensure the direction vector remains on the unit sphere
#             r_hat[i] = r_hat_i / np.linalg.norm(r_hat_i)
        
#         # Step 3: Update weights based on Huber loss and camera power
#         residuals = np.array([v_r[t] - (v[t] @ r_hat.T) for t in range(T)])  # Shape: (T, N)
#         residuals_mean = np.median(np.abs(residuals), axis=0)  # Median absolute deviation for each camera
#         power_adjusted_weights = 1 / (1 + huber(residuals_mean, huber_threshold))  # Update weights inversely to residuals

#         # Combine with camera power weights
#         if camera_power is not None:
#             power_adjusted_weights *= weights

#         # Normalize weights to avoid scaling issues
#         weights = power_adjusted_weights / np.max(power_adjusted_weights)

#         # Step 4: Check for convergence by comparing v with v_old
#         delta_v = np.linalg.norm(v - v_old) / np.linalg.norm(v_old)
#         if iter % 100==0:
#             print("Delta V: ", delta_v)
#         if delta_v < tolerance:
#             print(f"Converged after {iter+1} iterations")
#             break
#         v_old = v.copy()
    
#     return v, r_hat, weights


# import torch
# import numpy as np
# import matplotlib.pyplot as plt

# def estimate_velocity_from_radial(v_r, T, N, room_dims, lambda_reg=0.1, gamma=0.01, sigma_x2=1.0, sigma_y2=1.0, sigma_z2=1.0, 
#                                   tolerance=1e-5, max_iter=1000, lr=0.01, huber_threshold=1.0, temporal_smoothing=0.1, 
#                                   independence_penalty=0.1, smoothness_penalty=0.1, exclusivity_penalty=0.1,
#                                   magnitude_independence_penalty=0.1, distance_ratios=None, camera_power=None):
#     """
#     Estimate the 3D velocity vector from radial velocities observed by multiple cameras,
#     with constraints for mean-zero velocity, robustness against noise and outliers, independence between velocity
#     components, temporal smoothness, mutual exclusivity across axes, independence of magnitudes, distance ratio
#     constraints on the camera positions, and robustness to noise levels via camera power weights using PyTorch.
    
#     Parameters:
#     - v_r (np.ndarray): Observed radial velocities, shape (T, N).
#     - T (int): Number of time steps.
#     - N (int): Number of cameras.
#     - room_dims (tuple): Dimensions of the room as (width, length, height).
#     - [other parameters as before]
    
#     Returns:
#     - v (torch.Tensor): Estimated 3D velocity vectors over time, shape (T, 3).
#     - r_hat (torch.Tensor): Estimated direction vectors for each camera, shape (N, 3).
#     - weights (torch.Tensor): Final weights assigned to each camera, shape (N,).
#     """
    
#     # Convert v_r and distance_ratios to PyTorch tensors
#     v_r = torch.tensor(v_r, dtype=torch.float32)
#     width, length, height = room_dims
    
#     # Initialize the 3D velocity vector v and the direction vectors r_hat
#     v = torch.nn.Parameter(torch.randn((T, 3)))  # Leaf tensor with requires_grad=True
#     r_hat_init = torch.rand((N, 3)) * torch.tensor([width, length, height])  # Initial positions within room
#     r_hat = torch.nn.Parameter(r_hat_init.clone())  # Leaf tensor with requires_grad=True
    
#     # Normalize distance_ratios if provided
#     if distance_ratios is not None:
#         distance_ratios = torch.tensor(distance_ratios, dtype=torch.float32)
    
#     # If camera_power is provided, use it directly as weights; otherwise, assume equal weights
#     if camera_power is not None:
#         camera_power = torch.tensor(camera_power, dtype=torch.float32)
#         weights = camera_power / torch.max(camera_power)  # Normalize weights to be between 0 and 1
#     else:
#         weights = torch.ones(N, dtype=torch.float32)
    
#     # Define the optimizer
#     optimizer = torch.optim.Adam([v, r_hat], lr=lr)
    
#     # Huber loss function
#     def huber_loss(residual, delta):
#         return torch.where(torch.abs(residual) <= delta, 
#                            0.5 * residual ** 2, 
#                            delta * (torch.abs(residual) - 0.5 * delta))
    
#     # Optimization loop
#     prev_loss = float('inf')
#     losses = []
#     for iter in range(max_iter):
#         optimizer.zero_grad()
    
#         # Main loss function (reconstruction error with weights)
#         weighted_diff = (v_r - torch.matmul(v, r_hat.T)) * weights
#         loss = huber_loss(weighted_diff, huber_threshold).sum()
    
#         # Mean-zero constraint on v (enforce zero mean for each component)
#         mean_zero_loss = torch.sum(v.mean(dim=0) ** 2)
#         loss += lambda_reg * mean_zero_loss
    
#         # Temporal smoothing constraint on v
#         smoothness_loss = torch.sum((v[1:] - v[:-1]) ** 2)
#         loss += smoothness_penalty * smoothness_loss
    
#         # Independence penalty to minimize correlation between velocity components
#         independence_loss = torch.sum(v[:, 0] * v[:, 1] + v[:, 0] * v[:, 2] + v[:, 1] * v[:, 2])
#         loss += independence_penalty * independence_loss
    
#         # Exclusivity constraint to encourage only one axis to be high at any time
#         exclusivity_loss = torch.sum(v[:, 0] * v[:, 1] * v[:, 2])
#         loss += exclusivity_penalty * exclusivity_loss
    
#         # Magnitude independence constraint
#         magnitude_independence_loss = torch.sum(torch.abs(v[:, 0]) * torch.abs(v[:, 1]) + 
#                                                 torch.abs(v[:, 0]) * torch.abs(v[:, 2]) + 
#                                                 torch.abs(v[:, 1]) * torch.abs(v[:, 2]))
#         loss += magnitude_independence_penalty * magnitude_independence_loss
    
#         # Covariance penalty for true independence of magnitudes
#         abs_v = torch.abs(v)
#         cov_xy = torch.mean((abs_v[:, 0] - abs_v[:, 0].mean()) * (abs_v[:, 1] - abs_v[:, 1].mean()))
#         cov_xz = torch.mean((abs_v[:, 0] - abs_v[:, 0].mean()) * (abs_v[:, 2] - abs_v[:, 2].mean()))
#         cov_yz = torch.mean((abs_v[:, 1] - abs_v[:, 1].mean()) * (abs_v[:, 2] - abs_v[:, 2].mean()))
#         covariance_loss = cov_xy ** 2 + cov_xz ** 2 + cov_yz ** 2
#         loss += independence_penalty * covariance_loss
    
#         # Distance ratio constraint on r_hat magnitudes
#         if distance_ratios is not None:
#             current_magnitudes = torch.norm(r_hat, dim=1)
#             distance_ratio_loss = 0.0
#             for i in range(N):
#                 for j in range(i + 1, N):
#                     desired_ratio = distance_ratios[i] / distance_ratios[j]
#                     current_ratio = current_magnitudes[i] / current_magnitudes[j]
#                     distance_ratio_loss += (desired_ratio - current_ratio) ** 2
#             loss += lambda_reg * distance_ratio_loss
    
#         # Room bounds constraint to keep r_hat within the room dimensions
#         bounds_penalty = torch.sum(torch.relu(r_hat[:, 0] - width) ** 2 + 
#                                    torch.relu(-r_hat[:, 0]) ** 2 +
#                                    torch.relu(r_hat[:, 1] - length) ** 2 + 
#                                    torch.relu(-r_hat[:, 1]) ** 2 +
#                                    torch.relu(r_hat[:, 2] - height) ** 2 + 
#                                    torch.relu(-r_hat[:, 2]) ** 2)
#         loss += lambda_reg * bounds_penalty
    
#         # Backpropagation and optimization step
#         loss.backward()
#         optimizer.step()
        
#         # Ensure r_hat remains within room bounds
#         with torch.no_grad():
#             r_hat[:, 0].clamp_(0, width)
#             r_hat[:, 1].clamp_(0, length)
#             r_hat[:, 2].clamp_(0, height)
        
#         # Save loss
#         losses.append(loss.item())
        
#         # Check for convergence based on loss change
#         if abs(prev_loss - loss.item()) < tolerance:
#             print(f"Converged after {iter+1} iterations with loss {loss.item()}")
#             break
#         prev_loss = loss.item()
    
#         # Optional: Print loss every 100 iterations for tracking progress
#         if iter % 100 == 0:
#             print(f"Iteration {iter}, Loss: {loss.item()}")
    
#     # Visualization
#     # Plot loss over iterations
#     plt.figure()
#     plt.plot(losses)
#     plt.title('Loss over iterations')
#     plt.xlabel('Iteration')
#     plt.ylabel('Loss')
#     plt.show()
    
#     # Plot velocities over time
#     v_np = v.detach().numpy()
#     plt.figure()
#     plt.plot(v_np[:, 0], label='v_x')
#     plt.plot(v_np[:, 1], label='v_y')
#     plt.plot(v_np[:, 2], label='v_z')
#     plt.title('Estimated Velocities over Time')
#     plt.xlabel('Time step')
#     plt.ylabel('Velocity')
#     plt.legend()
#     plt.show()
    
#     # Plot camera positions in 3D
#     r_hat_np = r_hat.detach().numpy()
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(r_hat_np[:, 0], r_hat_np[:, 1], r_hat_np[:, 2], c='r', marker='o')
#     ax.set_xlim(0, width)
#     ax.set_ylim(0, length)
#     ax.set_zlim(0, height)
#     ax.set_title('Estimated Camera Positions')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.show()
    
#     return v.detach(), r_hat.detach(), weights


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def animate_point(vel, save_path=None):
    """
    Animate a point moving through 3D space over time.

    Parameters:
    x (array-like): X-coordinates of the point over time.
    y (array-like): Y-coordinates of the point over time.
    z (array-like): Z-coordinates of the point over time.
    save_path (str, optional): Path to save the animation as a file (e.g., 'animation.mp4' or 'animation.gif').
    """
    x = np.cumsum(vel[:,0])
    y = np.cumsum(vel[:,1])
    z = np.cumsum(vel[:,2])
    displacement = np.concatenate([x,y,z],0)
    # Create a new figure for the animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set the limits of the axes based on the data ranges
    ax.set_xlim(np.min(displacement), np.max(displacement))
    ax.set_ylim(np.min(displacement), np.max(displacement))
    ax.set_zlim(np.min(displacement), np.max(displacement))

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Initialize the point object
    point, = ax.plot([], [], [], 'o', markersize=8, color='red')

    def init():
        # Initialize the point at the first data point
        point.set_data(x[0], y[0])
        point.set_3d_properties(z[0])
        return point,

    def update(frame):
        # Update the point's position
        point.set_data([x[frame]], [y[frame]])
        point.set_3d_properties([z[frame]])
        return point,

    # Create the animation object
    ani = FuncAnimation(fig, update, frames=len(x), init_func=init, blit=True, interval = 10)

    # Save animation if a path is provided
    if save_path:
        ani.save(save_path, fps=30, extra_args=['-vcodec', 'libx264'] if save_path.endswith('.mp4') else [])

    # Display the animation
    plt.show()
    
import numpy as np
import numpy as np
from sklearn.decomposition import FastICA
import numpy as np
from sklearn.decomposition import FastICA

# def estimate_velocity_from_radial(v_r, T, N, lambda_reg=0.1, gamma=0.01,
#                                   sigma_x2=1.0, sigma_y2=1.0, sigma_z2=1.0,
#                                   tolerance=1e-3, max_iter=100, huber_threshold=1.0,
#                                   temporal_smoothing=0.1, independence_penalty=0.1,
#                                   smoothness_penalty=0.1, exclusivity_penalty=0.1,
#                                   magnitude_independence_penalty=0.1, distance_ratios=None,
#                                   camera_power=None):
#     """
#     Estimate the 3D velocity vector from radial velocities observed by multiple cameras,
#     applying ICA at each iteration to make the resulting velocity components independent.
#     Rescale the final velocity vectors to minimize the loss.

#     Parameters:
#     - v_r (np.ndarray): Observed radial velocities, shape (T, N).
#     - T (int): Number of time steps.
#     - N (int): Number of cameras.
#     - Other parameters are as previously defined.

#     Returns:
#     - v (np.ndarray): Estimated 3D velocity vectors over time, shape (T, 3).
#     - r_hat (np.ndarray): Estimated direction vectors for each camera, shape (N, 3).
#     - weights (np.ndarray): Final weights assigned to each camera, shape (N,).
#     """
    
#     # Initialize the 3D velocity vector v and the direction vectors r_hat randomly
#     v = np.random.randn(T, 3)      # Initialize 3D velocity vectors at each time step
#     r_hat = np.random.randn(N, 3)  # Initialize direction vectors randomly

#     # Normalize r_hat to be unit vectors initially
#     r_hat = r_hat / np.linalg.norm(r_hat, axis=1, keepdims=True)

#     # Initialize weights for each camera based on power if provided
#     if camera_power is not None:
#         # Compute weights inversely proportional to the camera power
#         weights = np.array(camera_power)
#         weights /= np.max(weights)  # Normalize weights to be between 0 and 1
#     else:
#         weights = np.ones(N)

#     # Normalize the distance_ratios if provided
#     if distance_ratios is not None:
#         distance_ratios = np.array(distance_ratios) / np.linalg.norm(distance_ratios)

#     # Function to apply Huber loss on residuals
#     def huber(residual, delta):
#         return np.where(np.abs(residual) <= delta,
#                         0.5 * residual**2,
#                         delta * (np.abs(residual) - 0.5 * delta))

#     v_old = np.copy(v)

#     # Alternating optimization
#     for iter in range(max_iter):
#         # Step 1: Optimize v(t) for each time step t, given fixed r_hat and weights
#         for t in range(T):
#             # Matrix A and vector b for weighted least squares
#             A = r_hat * weights[:, np.newaxis]  # Shape: (N, 3)
#             b = v_r[t] * weights  # Shape: (N,)

#             # Regularization matrix for independence of components
#             L = np.diag([gamma / sigma_x2, gamma / sigma_y2, gamma / sigma_z2])

#             # Solve weighted regularized least squares: (A^T A + L) v(t) = A^T b
#             AtA = A.T @ A + L
#             Atb = A.T @ b
#             v[t] = np.linalg.solve(AtA, Atb)

#         # Enforce mean-zero constraint on v
#         v_mean = np.mean(v, axis=0)
#         v -= v_mean  # Subtract mean from each component

#         # Apply ICA to make velocity components independent
#         ica = FastICA(n_components=3, max_iter=1000, tol=1e-4)
#         v_ica = ica.fit_transform(v)

#         # Rescale the components to minimize loss
#         # We solve for scaling factors s that minimize || v_r - (v * s) @ r_hat^T ||
#         s = np.ones(3)
#         for i in range(3):
#             numerator = 0
#             denominator = 0
#             for t in range(T):
#                 predicted = v_ica[t, i] * r_hat[:, i]
#                 numerator += np.dot(weights, v_r[t] * predicted)
#                 denominator += np.dot(weights, predicted**2)
#             s[i] = numerator / (denominator + 1e-8)  # Add small epsilon to avoid division by zero

#         v = v_ica * s  # Rescaled velocities

#         # Update r_hat to maintain consistency with the transformed v
#         mixing_matrix = ica.mixing_ * s  # Apply scaling to mixing matrix
#         r_hat = r_hat @ mixing_matrix

#         # Normalize r_hat to be unit vectors
#         r_hat = r_hat / np.linalg.norm(r_hat, axis=1, keepdims=True)

#         # Temporal smoothing of velocities
#         if iter > 0:
#             v = (1 - temporal_smoothing) * v + temporal_smoothing * v_old

#         # Independence regularization is handled by ICA, so we can omit it

#         # Temporal smoothness regularization
#         for t in range(1, T - 1):
#             v[t] += smoothness_penalty * (v[t-1] - 2 * v[t] + v[t+1])

#         # Mutual exclusivity regularization
#         for t in range(T):
#             v[t, 0] -= exclusivity_penalty * (v[t, 1] * v[t, 2])
#             v[t, 1] -= exclusivity_penalty * (v[t, 0] * v[t, 2])
#             v[t, 2] -= exclusivity_penalty * (v[t, 0] * v[t, 1])

#         # Magnitude independence regularization
#         for t in range(T):
#             v[t, 0] -= magnitude_independence_penalty * (abs(v[t, 1]) * abs(v[t, 2]))
#             v[t, 1] -= magnitude_independence_penalty * (abs(v[t, 0]) * abs(v[t, 2]))
#             v[t, 2] -= magnitude_independence_penalty * (abs(v[t, 0]) * abs(v[t, 1]))

#         # Step 2: Optimize r_hat for each camera, given fixed v(t) and distance ratios
#         for i in range(N):
#             # Solve constrained optimization for each r_hat[i] with weights
#             b = v_r[:, i] * weights[i]
#             A = v  # Shape: (T, 3)

#             # Solve least squares for r_hat[i] without constraints initially
#             r_hat_i, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

#             # Apply distance ratio constraint if provided
#             if distance_ratios is not None:
#                 target_magnitude = distance_ratios[i]
#                 current_magnitude = np.linalg.norm(r_hat_i)
#                 if current_magnitude > 0:
#                     r_hat_i = (target_magnitude / current_magnitude) * r_hat_i  # Adjust magnitude to match ratio

#             # Ensure the direction vector remains on the unit sphere
#             r_hat[i] = r_hat_i / np.linalg.norm(r_hat_i)

#         # Step 3: Update weights based on Huber loss and camera power
#         residuals = np.array([v_r[t] - (v[t] @ r_hat.T) for t in range(T)])  # Shape: (T, N)
#         residuals_mean = np.median(np.abs(residuals), axis=0)  # Median absolute deviation for each camera
#         power_adjusted_weights = 1 / (1 + huber(residuals_mean, huber_threshold))  # Update weights inversely to residuals

#         # Combine with camera power weights
#         if camera_power is not None:
#             power_adjusted_weights *= weights

#         # Normalize weights to avoid scaling issues
#         weights = power_adjusted_weights / np.max(power_adjusted_weights)

#         # Step 4: Check for convergence by comparing v with v_old
#         delta_v = np.linalg.norm(v - v_old) / (np.linalg.norm(v_old) + 1e-8)  # Added epsilon to prevent division by zero
#         if delta_v < tolerance:
#             print(f"Converged after {iter+1} iterations")
#             break
#         v_old = v.copy()

#     return v, r_hat, weights

import numpy as np
from sklearn.decomposition import FastICA
from scipy.linalg import orthogonal_procrustes
import timeit

# # #Best
# def estimate_velocity_from_radial(v_r, T, N, lambda_reg=0.1, gamma=0.01,
#                                   sigma_x2=1.0, sigma_y2=1.0, sigma_z2=1.0,
#                                   tolerance=1e-3, max_iter=100, huber_threshold=1.0,
#                                   temporal_smoothing=0.1, independence_penalty=0.1,
#                                   smoothness_penalty=0.1, exclusivity_penalty=0.1,
#                                   magnitude_independence_penalty=0.1,symmetry_penalty=0.1, distance_ratios=None,
#                                   camera_power=None, fixed_first_camera_index=0):
#     """
#     Estimate the 3D velocity vector from radial velocities observed by multiple cameras,
#     applying ICA and fixing the rotational ambiguity to make the solution unique.

#     Parameters:
#     - v_r (np.ndarray): Observed radial velocities, shape (T, N).
#     - T (int): Number of time steps.
#     - N (int): Number of cameras.
#     - Other parameters are as previously defined.

#     Returns:
#     - v (np.ndarray): Estimated 3D velocity vectors over time, shape (T, 3).
#     - r_hat (np.ndarray): Estimated direction vectors for each camera, shape (N, 3).
#     - weights (np.ndarray): Final weights assigned to each camera, shape (N,).
#     """

#     # Initialize the 3D velocity vector v and the direction vectors r_hat randomly
#     # np.random.seed(42)  # For reproducibility
#     v = np.random.randn(T, 3)      # Initialize 3D velocity vectors at each time step
#     r_hat = np.random.randn(N, 3)  # Initialize direction vectors randomly

#     # Normalize r_hat to be unit vectors initially
#     r_hat = r_hat / np.linalg.norm(r_hat, axis=1, keepdims=True)
    
#     r_hat[fixed_first_camera_index] = np.array([1.0, 0.0, 0.0]) / np.linalg.norm(np.array([1.0, 0.0, 0.0]))

#     # Initialize weights for each camera based on power if provided
#     if camera_power is not None:
#         # Compute weights inversely proportional to the camera power
#         weights = np.array(camera_power)
#         weights /= np.max(weights)  # Normalize weights to be between 0 and 1
#     else:
#         weights = np.ones(N)

#     # Normalize the distance_ratios if provided
#     if distance_ratios is not None:
#         distance_ratios = np.array(distance_ratios) / np.linalg.norm(distance_ratios)

#     # Function to apply Huber loss on residuals
#     def huber(residual, delta):
#         return np.where(np.abs(residual) <= delta,
#                         0.5 * residual**2,
#                         delta * (np.abs(residual) - 0.5 * delta))

#     v_old = np.copy(v)

#     # Define target basis (standard basis vectors)
#     target_basis = np.eye(3)
#     min_loss = np.inf
#     v_min = None
#     r_hat_min = None
#     weights_min = None
#     loss_all = []
#     # Alternating optimization
#     for iter in (pbar := tqdm(range(max_iter))):
#         # Step 1: Optimize v(t) for each time step t, given fixed r_hat and weights
#         for t in range(T):
#             # Matrix A and vector b for weighted least squares
#             A = r_hat * weights[:, np.newaxis]  # Shape: (N, 3)
#             b = v_r[t] * weights  # Shape: (N,)

#             # Regularization matrix for independence of components
#             L = np.diag([gamma / sigma_x2, gamma / sigma_y2, gamma / sigma_z2])

#             # Solve weighted regularized least squares: (A^T A + L) v(t) = A^T b
#             AtA = A.T @ A + L
#             Atb = A.T @ b
#             start = timeit.default_timer()
#             v[t] =  scipy.linalg.lstsq(AtA, Atb)[0]
#             stop = timeit.default_timer()
#             # print("lstsq", start - stop)
#             v[t] = np.nan_to_num(v[t])

#         # Enforce mean-zero constraint on v
#         v_mean = np.mean(v, axis=0)
#         v -= v_mean  # Subtract mean from each component

#         # Apply ICA to make velocity components independent
#         start = timeit.default_timer()
#         ica = FastICA(n_components=3, max_iter=200, tol=1e-4, random_state=iter)
#         try:
#             v_ica = ica.fit_transform(v)
#         except:
#             v_ica = v
#         stop = timeit.default_timer()
#         # print("ICA", start - stop)
#         # Rescale the components to minimize loss (as before)
#         s = np.ones(3)
#         for i in range(3):
#             numerator = 0
#             denominator = 0
#             for t in range(T):
#                 predicted = v_ica[t, i] * r_hat[:, i]
#                 numerator += np.dot(weights, v_r[t] * predicted)
#                 denominator += np.dot(weights, predicted**2)
#             s[i] = numerator / (denominator + 1e-8)  # Add small epsilon to avoid division by zero

#         v_rescaled = v_ica * s  # Rescaled velocities

#         # Select first three time steps to construct matrices for Procrustes analysis
#         num_samples = 3  # Number of samples to use (must be <= T)
#         v_samples = v_rescaled[:num_samples, :]  # Shape: (3, 3)
#         target_samples = target_basis[:num_samples, :]  # Shape: (3, 3)
#         start = timeit.default_timer()
#         # Compute rotation matrix using Procrustes analysis to align with target basis
#         R, _ = orthogonal_procrustes(v_samples, target_samples)
#         stop = timeit.default_timer()
#         # print("orthogonal_procrustes", start - stop)
#         # Rotate velocities to align with target basis
#         v_aligned = v_rescaled @ R
#         # v_aligned = v_rescaled

#         # Rotate r_hat to maintain consistency
#         r_hat_aligned = r_hat @ R
#         # r_hat_aligned = r_hat

#         # Normalize r_hat to be unit vectors
#         r_hat_aligned = r_hat_aligned / (np.linalg.norm(r_hat_aligned, axis=1, keepdims=True) + 1e-8)

#         # Update v and r_hat
#         v = v_aligned
#         r_hat = r_hat_aligned

#         # Temporal smoothing of velocities
#         if iter > 0:
#             v = (1 - temporal_smoothing) * v + temporal_smoothing * v_old

#         # Temporal smoothness regularization
#         for t in range(1, T - 1):
#             v[t] += smoothness_penalty * (v[t-1] - 2 * v[t] + v[t+1])

#         # Mutual exclusivity regularization
#         for t in range(T):
#             v[t, 0] -= exclusivity_penalty * (v[t, 1] * v[t, 2])
#             v[t, 1] -= exclusivity_penalty * (v[t, 0] * v[t, 2])
#             v[t, 2] -= exclusivity_penalty * (v[t, 0] * v[t, 1])

#         # Magnitude independence regularization
#         for t in range(T):
#             v[t, 0] -= magnitude_independence_penalty * (abs(v[t, 1]) * abs(v[t, 2]))
#             v[t, 1] -= magnitude_independence_penalty * (abs(v[t, 0]) * abs(v[t, 2]))
#             v[t, 2] -= magnitude_independence_penalty * (abs(v[t, 0]) * abs(v[t, 1]))


#         # Symmetry constraint regularization
#         for t in range(T):
#             t_sym = T - 1 - t  # Symmetric time index
#             if t <= t_sym:
#                 v_diff = np.abs(v[t]) - np.abs(v[t_sym])
#                 v[t] -= symmetry_penalty * v_diff
#                 v[t_sym] -= symmetry_penalty * (-v_diff)
                
#         # Step 2: Optimize r_hat for each camera, given fixed v(t) and distance ratios
#         for i in range(0, N):
#             # if i==fixed_first_camera_index:
#             #     continue
#             # Solve constrained optimization for each r_hat[i] with weights
#             b = v_r[:, i] * weights[i]
#             A = v  # Shape: (T, 3)

#             # Solve least squares for r_hat[i] without constraints initially
#             r_hat_i, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

#             # Apply distance ratio constraint if provided
#             if distance_ratios is not None:
#                 target_magnitude = distance_ratios[i]
#                 current_magnitude = np.linalg.norm(r_hat_i)
#                 if current_magnitude > 0:
#                     r_hat_i = (target_magnitude / current_magnitude) * r_hat_i  # Adjust magnitude to match ratio

#             # Ensure the direction vector remains on the unit sphere
#             r_hat[i] = r_hat_i / np.linalg.norm(r_hat_i)

#         # Step 3: Update weights based on Huber loss and camera power
#         residuals = np.array([v_r[t] - (v[t] @ r_hat.T) for t in range(T)])  # Shape: (T, N)
#         residuals_squared = residuals**2
#         residuals_mean = np.median(np.abs(residuals), axis=0)  # Median absolute deviation for each camera
#         power_adjusted_weights = 1 / (1 + huber(residuals_mean, huber_threshold))  # Update weights inversely to residuals

#         # Combine with camera power weights
#         if camera_power is not None:
#             power_adjusted_weights *= weights

#         # Normalize weights to avoid scaling issues
#         weights = power_adjusted_weights / np.max(power_adjusted_weights)

#         # Step 4: Compute fit loss and check for convergence
#         fit_loss = np.sum(residuals_squared * weights) ** 0.5
#         # if iter % 100 == 0 or iter == 1:
#             # pbar.set_description(f"Iteration {iter}: Fit Loss = {fit_loss:.4f}")
#             # print(f"Iteration {iter}: Fit Loss = {fit_loss:.4f}")
#         pbar.set_description(f"Iteration {iter}: Fit Loss = {fit_loss:.4f}")
#         loss_all.append(fit_loss)
#         # Keep track of the minimum loss and corresponding variables
#         if fit_loss < min_loss:
#             min_loss = fit_loss
#             v_min = v.copy()
#             r_hat_min = r_hat.copy()
#             weights_min = weights.copy()


#         # Step 4: Check for convergence by comparing v with v_old
#         delta_v = np.linalg.norm(v - v_old) / (np.linalg.norm(v_old) + 1e-8)  # Added epsilon to prevent division by zero
#         if delta_v < tolerance:
#             print(f"Converged after {iter+1} iterations")
#             break
#         v_old = v.copy()

#     top_cameras_indices = np.argsort(weights_min)[-10:]

#     # Extract data for the top cameras
#     top_cameras_weights = weights_min[top_cameras_indices]
#     top_cameras_directions = r_hat_min[top_cameras_indices]
#     top_cameras_v_r = v_r[:, top_cameras_indices]

#     # Plot the estimated velocities over time
#     plt.figure(figsize=(12, 6))
#     plt.plot(v_min[:, 0], label='v_x')
#     plt.plot(v_min[:, 1], label='v_y')
#     plt.plot(v_min[:, 2], label='v_z')
#     plt.title('Estimated 3D Velocities over Time')
#     plt.xlabel('Time Steps')
#     plt.ylabel('Velocity')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     # Plot the observed radial velocities for the top ten cameras
#     plt.figure(figsize=(12, 6))
#     for i in range(10):
#         idx = top_cameras_indices[i]
#         plt.plot(v_r[:, idx], label=f'Camera {idx} (Weight: {top_cameras_weights[i]:.2f})')
#     plt.title('Observed Radial Velocities for Top 10 Cameras')
#     plt.xlabel('Time Steps')
#     plt.ylabel('Radial Velocity')
#     plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05))
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     # Plot the direction vectors for the top ten cameras with colors based on weights and show camera indices
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # Normalize weights for color mapping
#     normalized_weights = (top_cameras_weights - top_cameras_weights.min()) / (top_cameras_weights.max() - top_cameras_weights.min() + 1e-8)
#     colors = cm.viridis(normalized_weights)

#     # Plot the direction vectors
#     origin = np.zeros((10, 3))
#     for i in range(10):
#         ax.quiver(origin[i, 0], origin[i, 1], origin[i, 2],
#                   top_cameras_directions[i, 0], top_cameras_directions[i, 1], top_cameras_directions[i, 2],
#                   length=1.0, normalize=True, color=colors[i], linewidth=2)
#         # Annotate with camera index
#         end_point = top_cameras_directions[i]  # Since origin is zero, end point is the direction vector
#         ax.text(end_point[0], end_point[1], end_point[2],
#                 f'{(top_cameras_indices[i]//(52*3))+1}', color='black', fontsize=10)

#     # Set up the color bar
#     mappable = cm.ScalarMappable(cmap='viridis')
#     mappable.set_array(top_cameras_weights)
#     cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
#     cbar.set_label('Camera Weight', rotation=270, labelpad=15)

#     ax.set_title('Direction Vectors of Top 10 Cameras Colored by Weight and Labeled by Index')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.view_init(elev=20., azim=30)
#     plt.xlim([-1.2, 1.2])
#     plt.ylim([-1.2, 1.2])
#     ax.set_zlim(-1.2, 1.2)
#     plt.tight_layout()
#     plt.show()
    
#     plt.figure()
#     plt.plot(loss_all)
#     plt.ylabel('Fit Loss')
#     plt.xlabel('Iteration')
#     plt.grid()
#     plt.tight_layout()
#     plt.show()
#     return v_min, r_hat_min, weights_min, loss_all




import numpy as np
from sklearn.decomposition import FastICA
from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting


import numpy as np
from sklearn.decomposition import FastICA
from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from tqdm import tqdm  # Progress bar
import numpy as np
from sklearn.decomposition import FastICA
from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm


import numpy as np
from sklearn.decomposition import FastICA
from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import scipy.linalg
import warnings

import numpy as np
from sklearn.decomposition import FastICA
from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import scipy
from sklearn.decomposition import KernelPCA
from scipy.optimize import nnls
from scipy.optimize import lsq_linear

def estimate_velocity_from_radial(
        v_r, T, N, lambda_reg=0.1, gamma=0.01,
        sigma_x2=1.0, sigma_y2=1.0, sigma_z2=1.0,
        tolerance=1e-3, max_iter=1000, huber_threshold=1.0,
        temporal_smoothing=0.1, independence_penalty=0.1,
        smoothness_penalty=0.1, exclusivity_penalty=0.1,
        magnitude_independence_penalty=0.1, symmetry_penalty=0.1,
        distance_ratios=None, camera_power=None, fixed_camera_indices=[0, 1, 2], 
        subset_fraction=0.5, weight_learning_rate=0.1, num_rotations=27 * (12**3), v_init = None, r_hat_init = None, visualization=False):
    """
    Estimate the 3D velocity vector from radial velocities observed by multiple cameras,
    applying ICA and fixing the rotational ambiguity to make the solution unique.
    
    In each iteration, randomly selects a subset of cameras (e.g., 50%) for optimization,
    making the process faster and more robust to outlier cameras. Fit loss is calculated
    using residuals from all cameras.
    
    Parameters:
    - v_r (np.ndarray): Observed radial velocities, shape (T, N).
    - T (int): Number of time steps.
    - N (int): Number of cameras.
    - lambda_reg (float): Regularization parameter.
    - gamma (float): Regularization parameter for ICA.
    - sigma_x2, sigma_y2, sigma_z2 (float): Variance parameters for regularization.
    - tolerance (float): Convergence tolerance.
    - max_iter (int): Maximum number of iterations.
    - huber_threshold (float): Threshold for Huber loss.
    - temporal_smoothing (float): Temporal smoothing factor.
    - independence_penalty (float): Penalty for independence regularization.
    - smoothness_penalty (float): Penalty for temporal smoothness.
    - exclusivity_penalty (float): Penalty for mutual exclusivity.
    - magnitude_independence_penalty (float): Penalty for magnitude independence.
    - symmetry_penalty (float): Penalty for symmetry constraints.
    - distance_ratios (list or np.ndarray): Distance ratios for each camera.
    - camera_power (list or np.ndarray): Initial camera power levels.
    - fixed_first_camera_index (int): Index of the camera to always include in optimization.
    - subset_fraction (float): Fraction of cameras to select in each iteration (e.g., 0.5 for 50%).
    - weight_learning_rate (float): Learning rate for updating weights (0 < weight_learning_rate <= 1).
    
    Returns:
    - v_best (np.ndarray): Estimated 3D velocity vectors over time corresponding to minimum loss, shape (T, 3).
    - r_hat_best (np.ndarray): Estimated direction vectors for each camera corresponding to minimum loss, shape (N, 3).
    - weights_best (np.ndarray): Final weights assigned to each camera corresponding to minimum loss, shape (N,).
    - loss_all (list): List of fit loss values over iterations.
    """
    # ==========================
    # Helper Functions
    # ==========================
    
    def safe_normalize(vec, epsilon=1e-8):
        norm = np.linalg.norm(vec)
        if norm < epsilon:
            return np.ones_like(vec) / np.sqrt(len(vec))  # Default unit vector
        return vec / norm

    def validate_input(v_r):
        if np.isnan(v_r).any() or np.isinf(v_r).any():
            raise ValueError("Input radial velocities contain NaN or infinite values.")
    
    def impute_nan(v):
        nan_indices = np.isnan(v)
        if np.any(nan_indices):
            # Replace NaN with column mean
            col_mean = np.nanmean(v, axis=0)
            inds = np.where(nan_indices)
            v[nan_indices] = np.take(col_mean, inds[1])
        return v

    # ==========================
    # Input Validation
    # ==========================
    
    validate_input(v_r)
    
    # ==========================
    # Initialization
    # ==========================
    
    # Initialize the 3D velocity vectors and direction vectors randomly
    # np.random.seed(42)  # For reproducibility    
    v = np.random.randn(T, 3)      # Initialize 3D velocity vectors at each time step
    if v_init is not None:
        v = v_init
        
    r_hat = np.random.randn(N, 3)  # Initialize direction vectors randomly
    # r_hat[:,0] = 0
    if r_hat_init is not None:
        r_hat = r_hat_init    
    # Normalize r_hat to be unit vectors initially
    r_hat = safe_normalize(r_hat, epsilon=1e-8)
    # fixed_camera_locations = [[0.7, 0.2, 0.1], [0.4, 0.4 , 0.2], [0.2, 0.7, 0.1]]
    # Fix the direction vector of the first camera
    # print("fixed_camera_indices: ", fixed_camera_indices)
    # for fixed_first_camera_index_i, fixed_first_camera_index in enumerate(fixed_camera_indices):
    #     r_hat[fixed_first_camera_index] = safe_normalize(np.array(fixed_camera_locations[fixed_first_camera_index_i]))
    
    # Initialize weights for each camera based on power if provided
    if camera_power is not None:
        weights = np.array(camera_power)
        weights /= np.max(weights)  # Normalize weights to be between 0 and 1
    else:
        weights = np.ones(N)
    
    # Normalize the distance_ratios if provided
    if distance_ratios is not None:
        distance_ratios = np.array(distance_ratios) / np.linalg.norm(distance_ratios)
    
    # Function to apply Huber loss on residuals
    def huber(residual, delta):
        return np.where(np.abs(residual) <= delta,
                        0.5 * residual**2,
                        delta * (np.abs(residual) - 0.5 * delta))

    def generate_orderly_rotations(num_rotations):
        """
        Generate rotation matrices in an orderly fashion using Euler angles.

        Parameters:
        - num_rotations (int): Total number of rotations to generate.

        Returns:
        - rotations (list): List of rotation matrices.
        """
        # Determine the number of steps for each angle
        steps = int(np.ceil(num_rotations ** (1/3)))
        angles = np.linspace(0, 2 * np.pi, steps, endpoint=False)  # Exclude the endpoint to avoid duplicate rotations

        rotations = []
        for alpha in angles:  # Rotation around z-axis (yaw)
            for beta in angles:  # Rotation around y-axis (pitch)
                for gamma in angles:  # Rotation around x-axis (roll)
                    # Compute the rotation matrix using the Euler angles
                    Rz = np.array([
                        [np.cos(alpha), -np.sin(alpha), 0],
                        [np.sin(alpha),  np.cos(alpha), 0],
                        [0,              0,             1]
                    ])
                    Ry = np.array([
                        [ np.cos(beta), 0, np.sin(beta)],
                        [0,             1, 0           ],
                        [-np.sin(beta), 0, np.cos(beta)]
                    ])
                    Rx = np.array([
                        [1, 0,              0             ],
                        [0, np.cos(gamma), -np.sin(gamma)],
                        [0, np.sin(gamma),  np.cos(gamma)]
                    ])
                    R = Rz @ Ry @ Rx  # Combined rotation matrix
                    rotations.append(R)
                    if len(rotations) >= num_rotations:
                        return rotations
        return rotations
    # Initialize previous velocity for convergence check
    v_old = np.copy(v)
    
    # Define target basis (standard basis vectors)
    target_basis = np.eye(3)
    
    # Variables to track the minimum loss and corresponding variables
    min_loss = np.inf
    v_best = None
    r_hat_best = None
    weights_best = None
    loss_all = []
    
    # ==========================
    # Optimization Loop
    # ==========================
    # camera_grid = generate_nonsymmetric_sphere_grid_2d(100, 1.0, seed=42)
    for iter in (pbar := tqdm(range(max_iter))):
        # Step 1: Randomly select a subset of cameras (50%) for optimization
        subset_size = max(1, int(subset_fraction * N))
        selected_indices = np.random.choice(N, size=subset_size, replace=False)
        # Ensure the fixed camera is always included
        for fixed_first_camera_index_i, fixed_first_camera_index in enumerate(fixed_camera_indices):
            if fixed_first_camera_index not in selected_indices:
                selected_indices[fixed_first_camera_index_i] = fixed_first_camera_index  # Replace the first selected index with the fixed camera
        
        # Extract selected subset data
        selected_r_hat = r_hat[selected_indices]  # Shape: (subset_size, 3)
        selected_weights = weights[selected_indices]  # Shape: (subset_size,)
        selected_v_r = v_r[:, selected_indices]  # Shape: (T, subset_size)
        
        # Step 2: Optimize v(t) for each time step t using selected subset
        for t in range(T):
            A = selected_r_hat * selected_weights[:, np.newaxis]  # Shape: (subset_size, 3)
            b = selected_v_r[t] * selected_weights  # Shape: (subset_size,)
            
            # Regularization matrix for independence of components
            L = np.diag([gamma / sigma_x2, gamma / sigma_y2, gamma / sigma_z2])
            
            # Solve weighted regularized least squares: (A^T A + L) v(t) = A^T b
            AtA = A.T @ A + L
            Atb = A.T @ b
            try:
                v[t] = np.linalg.solve(AtA, Atb)
            except np.linalg.LinAlgError:
                # If singular matrix, use least squares solution
                v[t], _, _, _ = np.linalg.lstsq(AtA, Atb, rcond=None)
            
            # Handle potential NaN values
            v[t] = np.nan_to_num(v[t], nan=0.0, posinf=0.0, neginf=0.0)
           
        # for ddd in range(3):
        #     if np.sum(v[0:T//2, ddd]) < 0:                
        #         v[0:T//2,ddd] = v[0:T//2,ddd] * -1
        #     if np.sum(v[T//2:, ddd]) > 0:                
        #         v[T//2:,ddd] = -1 * v[T//2:,ddd]
        # Enforce mean-zero constraint on v
        v_mean = np.mean(v, axis=0)
        v -= v_mean  # Subtract mean from each component
        
        # # Step 3: Apply ICA to make velocity components independent
        # try:
        #     ica = KernelPCA(n_components=3, random_state=iter)
        #     v_ica = ica.fit_transform(v)
        # except Exception as e:
        #     print(f"ICA failed at iteration {iter+1}: {e}")
        v_ica = v  # Fallback to raw velocities if ICA fails
        
        # Rescale the components to minimize loss
        s = np.ones(3)
        # for i in range(3):
        #     numerator = 0
        #     denominator = 0
        #     for t in range(T):
        #         predicted = v_ica[t, i] * r_hat[:, i]
        #         numerator += np.dot(weights, v_r[t] * predicted)
        #         denominator += np.dot(weights, predicted**2)
        #     s[i] = numerator / (denominator + 1e-8)  # Prevent division by zero
        
        v_rescaled = v_ica * s  # Rescaled velocities
        
        # Step 4: Apply Orthogonal Procrustes to align with target basis
        num_samples = min(3, T)  # Ensure at least 3 samples for Procrustes
        v_samples = v_rescaled[:num_samples, :]  # Shape: (num_samples, 3)
        target_samples = target_basis[:num_samples, :]  # Shape: (num_samples, 3)
        # try:
        #     R, _ = orthogonal_procrustes(v_samples, target_samples)
        # except ValueError as e:
        #     print(f"Procrustes analysis failed at iteration {iter+1}: {e}")
        #     R = np.eye(3)  # Fallback to identity matrix if Procrustes fails
        # R = np.eye(3)
        # Rotate velocities and direction vectors
        R = np.eye(3)
        v_aligned = v_rescaled @ R
        r_hat_aligned = r_hat @ R
        
        # Normalize r_hat to be unit vectors
        r_hat_aligned = r_hat_aligned / (np.linalg.norm(r_hat_aligned, axis=1, keepdims=True) + 1e-8)
        
        # Update v and r_hat
        v = v_aligned
        r_hat = r_hat_aligned
        
        # for fixed_first_camera_index_i, fixed_first_camera_index in enumerate(fixed_camera_indices):
        #     r_hat[fixed_first_camera_index] = safe_normalize(np.array(fixed_camera_locations[fixed_first_camera_index_i]))
        
        # # Step 5: Apply Regularizations on v
        # # Temporal smoothing
        # if iter > 0:
        #     v = (1 - temporal_smoothing) * v + temporal_smoothing * v_old
        
        # # Temporal smoothness regularization
        # for t in range(1, T - 1):
        #     v[t] += smoothness_penalty * (v[t-1] - 2 * v[t] + v[t+1])
        
        # # Mutual exclusivity regularization
        # for t in range(T):
        #     v[t, 0] -= exclusivity_penalty * (v[t, 1] * v[t, 2])
        #     v[t, 1] -= exclusivity_penalty * (v[t, 0] * v[t, 2])
        #     v[t, 2] -= exclusivity_penalty * (v[t, 0] * v[t, 1])
        
        # # Magnitude independence regularization
        # for t in range(T):
        #     v[t, 0] -= magnitude_independence_penalty * (abs(v[t, 1]) * abs(v[t, 2]))
        #     v[t, 1] -= magnitude_independence_penalty * (abs(v[t, 0]) * abs(v[t, 2]))
        #     v[t, 2] -= magnitude_independence_penalty * (abs(v[t, 0]) * abs(v[t, 1]))
        
        # # Symmetry constraint regularization
        # for t in range(T):
        #     t_sym = T - 1 - t  # Symmetric time index
        #     if t <= t_sym:
        #         v_diff = np.abs(v[t]) - np.abs(v[t_sym])
        #         v[t] -= symmetry_penalty * v_diff
        #         v[t_sym] += symmetry_penalty * v_diff  # Adjust symmetrically
        
        # Step 6: Optimize r_hat for each camera
        for i in range(N):
            # Skip the fixed first camera if desired
            if i in fixed_camera_indices:
                continue
            
            b = v_r[:, i] * weights[i]
            A = v  # Shape: (T, 3)
            
            # Solve least squares for r_hat[i]
            try:
                r_hat_i, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                # r_hat_i, residual = nnls(A, b)
                # res = lsq_linear(A, b, bounds=(0, np.inf))
                # r_hat_i = res.x
                # r_hat_i, residual = solve_constrained_least_squares(A, b)
                
                # res = solve_least_squares_with_angle_constraints(A, b)
                # r_hat_i = res.x
                # camera_grid_losses = []
                # for cam_pos in camera_grid:
                #     r_ = cam_pos
                #     l_cam = np.sum((A @ r_ - b)**2)
                #     camera_grid_losses.append(l_cam)
                # r_hat_i_0 = camera_grid[np.argmin(camera_grid_losses)]
                # r_hat_i, residual = solve_constrained_least_squares(A, b, r_hat_i_0)
                # print(np.min(camera_grid_losses))

            except np.linalg.LinAlgError:
                r_hat_i = r_hat[i]  # Fallback to previous value if lstsq fails
            
            # Apply distance ratio constraint if provided
            if distance_ratios is not None:
                target_magnitude = distance_ratios[i]
                current_magnitude = np.linalg.norm(r_hat_i)
                if current_magnitude > 1e-8:
                    r_hat_i = (target_magnitude / current_magnitude) * r_hat_i  # Adjust magnitude
                else:
                    r_hat_i = np.array([1.0, 0.0, 0.0])  # Default direction if magnitude is too small
            
            # Ensure the direction vector remains on the unit sphere            
            r_hat[i] = safe_normalize(r_hat_i, epsilon=1e-8)
        
        # r_hat = np.abs(r_hat)
        # Step 7: Calculate Residuals and Fit Loss on All Cameras
        residuals = v_r - (v @ r_hat.T)  # Shape: (T, N)
        residuals_squared = residuals**2

        # Calculate Huber loss
        if huber_threshold is not None:
            loss = np.sum(huber(residuals, huber_threshold) * weights)
        else:
            loss = np.sum((residuals_squared * weights)) ** 0.5
        
        pbar.set_description(f"Iteration {iter}: Fit Loss = {loss:.4f}")
        
        loss_all.append(loss)
        
        # # Step 8: Update Weights Based on Residuals from All Cameras
        # residuals_mean = np.median(np.abs(residuals), axis=0)  # Median absolute deviation for each camera
        # power_adjusted_weights = 1 / (1 + huber(residuals_mean, huber_threshold))  # Inversely related to residuals
        
        # # Combine with camera power weights
        # if camera_power is not None:
        #     power_adjusted_weights *= weights  # Incorporate initial camera power
        
        # # Normalize weights to avoid scaling issues
        # weights = power_adjusted_weights / (np.max(power_adjusted_weights) + 1e-8)
        
        # Step 9: Track the Best Estimates
        if loss < min_loss:
            min_loss = loss
            v_best = v.copy()
            r_hat_best = r_hat.copy()
            weights_best = weights.copy()
        
        # Step 10: Check for Convergence
        delta_v = np.linalg.norm(v - v_old) / (np.linalg.norm(v_old) + 1e-8)
        if delta_v < tolerance:
            print(f"Converged after {iter+1} iterations")
            break
        v_old = v.copy()
    # ==========================
    # Sample Rotations and Generate Possible Solutions
    # ==========================

    if v_best is not None:
        rotations = generate_orderly_rotations(num_rotations)
        rotated_velocities_list = []
        rotations_list = []
        # for R in rotations:
        #     v_rotated = v_best @ R.T  # Apply rotation to velocities
        #     rotated_velocities_list.append(v_rotated)
        #     rotations_list.append(R)
    else:
        print("No valid estimates to generate possible velocities.")
        rotated_velocities_list = []
        rotations_list = []


    # ==========================
    # Visualization of Results
    # ==========================
    if visualization:
        if v_best is not None and r_hat_best is not None and weights_best is not None:
            # Identify the indices of the top ten cameras by weight
            top_cameras_indices = np.argsort(weights_best)[-10:]
        
            # Extract data for the top cameras
            top_cameras_weights = weights_best[top_cameras_indices]
            top_cameras_directions = r_hat_best[top_cameras_indices]
            top_cameras_v_r = v_r[:, top_cameras_indices]
        
            # Plot the estimated velocities over time
            plt.figure(figsize=(12, 6))
            plt.plot(v_best[:, 0], label='v_x')
            plt.plot(v_best[:, 1], label='v_y')
            plt.plot(v_best[:, 2], label='v_z')
            plt.title('Estimated 3D Velocities over Time')
            plt.xlabel('Time Steps')
            plt.ylabel('Velocity')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        
            # Plot the observed radial velocities for the top ten cameras
            plt.figure(figsize=(12, 6))
            for i in range(10):
                idx = top_cameras_indices[i]
                plt.plot(v_r[:, idx], label=f'Camera {idx} (Weight: {top_cameras_weights[i]:.2f})')
            plt.title('Observed Radial Velocities for Top 10 Cameras')
            plt.xlabel('Time Steps')
            plt.ylabel('Radial Velocity')
            plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05))
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        
            # Plot the direction vectors for the top ten cameras with colors based on weights and show camera indices
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
            # Normalize weights for color mapping
            normalized_weights = (top_cameras_weights - top_cameras_weights.min()) / (top_cameras_weights.max() - top_cameras_weights.min() + 1e-8)
            colors = cm.viridis(normalized_weights)
        
            # Plot the direction vectors
            origin = np.zeros((10, 3))
            for i in range(10):
                ax.quiver(origin[i, 0], origin[i, 1], origin[i, 2],
                          top_cameras_directions[i, 0], top_cameras_directions[i, 1], top_cameras_directions[i, 2],
                          length=1.0, normalize=True, color=colors[i], linewidth=2)
                # Annotate with camera index
                end_point = top_cameras_directions[i]
                ax.text(end_point[0], end_point[1], end_point[2],
                        f'{top_cameras_indices[i] // (52 * 3) + 1}', color='black', fontsize=10)
        
            # Set up the color bar
            mappable = cm.ScalarMappable(cmap='viridis')
            mappable.set_array(top_cameras_weights)
            cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
            cbar.set_label('Camera Weight', rotation=270, labelpad=15)
        
            ax.set_title('Direction Vectors of Top 10 Cameras Colored by Weight and Labeled by Index')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=20., azim=30)
            plt.xlim([-1.2, 1.2])
            plt.ylim([-1.2, 1.2])
            ax.set_zlim(-1.2, 1.2)
            plt.tight_layout()
            plt.show()
            
            # Plot the fit loss over iterations
            plt.figure(figsize=(10, 5))
            plt.plot(loss_all, marker='o')
            plt.title('Fit Loss Over Iterations')
            plt.xlabel('Iteration')
            plt.ylabel('Fit Loss')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print("No valid estimates to visualize.")
    
    return v_best, r_hat_best, weights_best, loss_all


import numpy as np
from tqdm import tqdm
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.transform import Rotation as R
from matplotlib import cm

# def estimate_velocity_from_radial(
#         v_r, T, N, camera_groups,
#         lambda_reg=0.1, gamma=0.01,
#         sigma_x2=1.0, sigma_y2=1.0, sigma_z2=1.0,
#         tolerance=1e-3, max_iter=1000, huber_threshold=1.0,
#         temporal_smoothing=0.1, independence_penalty=0.1,
#         smoothness_penalty=0.1, exclusivity_penalty=0.1,
#         magnitude_independence_penalty=0.1, symmetry_penalty=0.1,
#         distance_ratios=None, camera_power=None, 
#         # For rotational fixing:
#         fixed_camera_indices=[0, 1],
#         fixed_camera_orientations=None,
#         # Subset/weights params:
#         subset_fraction=0.5, weight_learning_rate=0.1, 
#         # For enumerating sample rotations
#         num_rotations=27*(12**3), 
#         # Initial values:
#         v_init=None, r_hat_init=None):
#     """
#     Estimate the 3D velocity vectors from radial velocities observed by multiple camera groups,
#     ensuring that the y-component of each group's velocity vector is the same for each time step.
#     The function also fixes the shape mismatch issue by constructing predicted radial velocities 
#     in shape (T, N) from the group-based 3D velocities.

#     Parameters:
#     -----------
#     - v_r: Observed radial velocities, shape (T, N).
#     - T: Number of time steps.
#     - N: Number of cameras.
#     - camera_groups (list of lists): Each sublist contains camera indices for one group; total 5 groups.
#     - ... (regularization, smoothing, outlier-related params) ...
#     - fixed_camera_indices: Indices of cameras to fix orientation, removing rotational ambiguity.
#     - fixed_camera_orientations: The fixed directions for those cameras, shape (M, 3).
#     - subset_fraction: Fraction of cameras to select in each iteration (e.g., 0.5).
#     - weight_learning_rate: Learning rate for updating weights (0 < weight_learning_rate <= 1).
#     - num_rotations: Number of rotation samples for enumerating solutions if needed.
#     - v_init: Initial velocity array, shape (T, 5, 3).
#     - r_hat_init: Initial camera direction array, shape (N, 3).

#     Returns:
#     --------
#     - v_best: Estimated group velocity vectors over time, shape (T, 5, 3).
#     - r_hat_best: Estimated camera direction vectors, shape (N, 3).
#     - weights_best: Final weights assigned to each camera, shape (N,).
#     - loss_all: List of fit loss values over iterations.
#     - rotated_velocities_list: List of rotated velocity solutions (if enumerating).
#     """
    
#     # ==========================
#     # 1. Helper Functions
#     # ==========================

#     def safe_normalize(vec, epsilon=1e-8):
#         norm = np.linalg.norm(vec)
#         if norm < epsilon:
#             return np.ones_like(vec) / np.sqrt(len(vec))  # Default fallback unit vector
#         return vec / norm

#     def validate_input(v_r):
#         if np.isnan(v_r).any() or np.isinf(v_r).any():
#             raise ValueError("Input radial velocities contain NaN or infinite values.")
    
#     def huber(residual, delta):
#         return np.where(
#             np.abs(residual) <= delta,
#             0.5 * residual**2,
#             delta*(np.abs(residual) - 0.5 * delta)
#         )

#     def generate_orderly_rotations(num_rot):
#         """Generate a set of rotation matrices in an orderly fashion using Euler angles."""
#         steps = int(np.ceil(num_rot ** (1/3)))
#         angles = np.linspace(0, 2*np.pi, steps, endpoint=False)
#         rotations = []
#         for alpha in angles:
#             for beta in angles:
#                 for gamma in angles:
#                     Rz = np.array([
#                         [np.cos(alpha), -np.sin(alpha), 0],
#                         [np.sin(alpha),  np.cos(alpha), 0],
#                         [0,              0,             1]
#                     ])
#                     Ry = np.array([
#                         [ np.cos(beta), 0, np.sin(beta)],
#                         [0,             1, 0           ],
#                         [-np.sin(beta), 0, np.cos(beta)]
#                     ])
#                     Rx = np.array([
#                         [1, 0,               0],
#                         [0, np.cos(gamma),  -np.sin(gamma)],
#                         [0, np.sin(gamma),   np.cos(gamma)]
#                     ])
#                     R_combined = Rz @ Ry @ Rx
#                     rotations.append(R_combined)
#                     if len(rotations) >= num_rot:
#                         return rotations
#         return rotations

#     # ==========================
#     # 2. Validation & Initialization
#     # ==========================

#     validate_input(v_r)

#     # G = number of groups
#     G = len(camera_groups)

#     # Initialize velocities: shape (T, G, 3)
#     if v_init is not None:
#         v = v_init
#     else:
#         v = np.random.randn(T, G, 3)

#     # Initialize directions: shape (N, 3)
#     if r_hat_init is not None:
#         r_hat = r_hat_init
#     else:
#         r_hat = np.random.randn(N, 3)
#     # Normalize camera directions
#     for i in range(N):
#         r_hat[i] = safe_normalize(r_hat[i])

#     # If we fix certain cameras, enforce their orientations
#     if (fixed_camera_indices is not None) and (fixed_camera_orientations is not None):
#         for idx, cam_idx in enumerate(fixed_camera_indices):
#             r_hat[cam_idx] = safe_normalize(fixed_camera_orientations[idx])

#     # Initialize weights
#     if camera_power is not None:
#         weights = np.array(camera_power, dtype=float)
#         weights /= (np.max(weights) + 1e-8)
#     else:
#         weights = np.ones(N, dtype=float)

#     if distance_ratios is not None:
#         distance_ratios = np.array(distance_ratios) / np.linalg.norm(distance_ratios)

#     # For tracking best solutions
#     min_loss = np.inf
#     v_best = None
#     r_hat_best = None
#     weights_best = None
#     loss_all = []

#     # Prepare "old" velocities for temporal smoothing
#     v_old = np.copy(v)

#     # For enumerating rotation solutions if needed
#     rotations_list = []
#     rotated_velocities_list = []

#     # ==========================
#     # 3. Optimization Loop
#     # ==========================
    
#     for iteration in (pbar := tqdm(range(max_iter), desc="Optimizing")):
#         # Step A: Randomly select subset of cameras
#         subset_size = max(1, int(subset_fraction * N))
#         selected_indices = np.random.choice(N, size=subset_size, replace=False)

#         # Ensure at least one camera from each group is in the subset 
#         # (optional logic depending on your preference)
#         for group in camera_groups:
#             if not any(cam in selected_indices for cam in group):
#                 selected_indices[0] = np.random.choice(group)

#         # Step B: Optimize each group's velocity vector
#         for t in range(T):
#             for g, group in enumerate(camera_groups):
#                 # Weighted direction subset
#                 group_r_hat = r_hat[group] * weights[group, np.newaxis]  # shape: (group_size, 3)
#                 group_v_r = v_r[t, group] * weights[group]                # shape: (group_size,)

#                 # Regularization for the velocity components
#                 L = np.diag([gamma/sigma_x2, gamma/sigma_y2, gamma/sigma_z2])

#                 # Weighted least squares
#                 A = group_r_hat  # shape (group_size, 3)
#                 b = group_v_r    # shape (group_size,)
#                 AtA = A.T @ A + L
#                 Atb = A.T @ b
#                 try:
#                     v_est = np.linalg.solve(AtA, Atb)
#                 except np.linalg.LinAlgError:
#                     v_est, _, _, _ = np.linalg.lstsq(AtA, Atb, rcond=None)

#                 v_est = np.nan_to_num(v_est, nan=0.0, posinf=0.0, neginf=0.0)

#                 # Store x and z; y to be fixed below
#                 v[t, g, 0] = v_est[0]
#                 v[t, g, 2] = v_est[2]
#                 # We'll unify the y-values later.

#         # Step C: Enforce shared y-component across all groups for each time step
#         for t in range(T):
#             # Average y-value across groups
#             shared_y = np.mean(v[t, :, 1])
#             v[t, :, 1] = shared_y

#         # Step D: Mean-zero constraint across groups
#         # shape of v_mean = (T, 1, 3)
#         v_mean = np.mean(v, axis=1, keepdims=True)
#         v -= v_mean  # shape (T, G, 3)

#         # Step E: Regularizations on v

#         # E.1 Temporal Smoothing
#         if iteration > 0:
#             v = (1 - temporal_smoothing)*v + temporal_smoothing*v_old
        
#         # E.2 Temporal smoothness
#         for t in range(1, T-1):
#             v[t] += smoothness_penalty * (v[t-1] - 2*v[t] + v[t+1])

#         # E.3 Mutual exclusivity
#         for t in range(T):
#             for g in range(G):
#                 vx, vy, vz = v[t, g]
#                 v[t, g, 0] -= exclusivity_penalty * (vy * vz)
#                 v[t, g, 1] -= exclusivity_penalty * (vx * vz)
#                 v[t, g, 2] -= exclusivity_penalty * (vx * vy)
        
#         # E.4 Magnitude independence
#         for t in range(T):
#             for g in range(G):
#                 vx, vy, vz = v[t, g]
#                 v[t, g, 0] -= magnitude_independence_penalty * (abs(vy)*abs(vz))
#                 v[t, g, 1] -= magnitude_independence_penalty * (abs(vx)*abs(vz))
#                 v[t, g, 2] -= magnitude_independence_penalty * (abs(vx)*abs(vy))
        
#         # E.5 Symmetry constraint
#         for t in range(T):
#             t_sym = T-1-t
#             if t <= t_sym:
#                 v_diff = np.abs(v[t]) - np.abs(v[t_sym])
#                 v[t] -= symmetry_penalty * v_diff
#                 v[t_sym] += symmetry_penalty * v_diff

#         # Step F: Optimize r_hat for each group
#         # Each camera i belongs to exactly one group
#         for g, group in enumerate(camera_groups):
#             for i in group:
#                 if i in fixed_camera_indices:
#                     # Skip cameras with fixed orientations
#                     continue
#                 b = v_r[:, i] * weights[i]    # shape (T,)
#                 A = v[:, g, :]               # shape (T, 3)
                
#                 try:
#                     r_est, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
#                 except np.linalg.LinAlgError:
#                     r_est = r_hat[i]
                
#                 # Distance ratio if any
#                 if distance_ratios is not None:
#                     target_magnitude = distance_ratios[i]
#                     cmag = np.linalg.norm(r_est)
#                     if cmag > 1e-8:
#                         r_est = (target_magnitude / cmag)*r_est
#                     else:
#                         r_est = np.array([1.0, 0.0, 0.0])
                
#                 r_hat[i] = safe_normalize(r_est)

#         # Step G: Construct predicted radial velocities (T, N) from group velocities
#         # so we can compute residuals and shape (T, N) - no shape mismatch
#         v_r_pred = np.zeros((T, N))
#         for g, group in enumerate(camera_groups):
#             for i in group:
#                 for t in range(T):
#                     v_r_pred[t, i] = np.dot(v[t, g], r_hat[i])
        
#         residuals = v_r - v_r_pred  # both (T, N)

#         # Step H: Calculate Huber loss
#         residuals_abs = np.abs(residuals)
#         if huber_threshold is not None:
#             # Weighted Huber loss
#             # We can do a per-camera approach to incorporate weights
#             loss = 0.0
#             for i in range(N):
#                 # shape (T,) for camera i
#                 r_i = residuals[:, i]
#                 huber_i = huber(r_i, huber_threshold)
#                 loss += np.sum(huber_i * weights[i])
#         else:
#             # Weighted MSE
#             loss = 0.0
#             for i in range(N):
#                 r_i = residuals[:, i]**2
#                 loss += np.sum(r_i * weights[i])

#         pbar.set_description(f"Iteration {iteration}: Fit Loss = {loss:.4f}")
#         loss_all.append(loss)

#         # Step I: Update Weights Based on Residuals
#         # We can use median absolute deviation logic
#         residuals_median = np.median(residuals_abs, axis=0)  # shape (N,)
#         power_adjusted_weights = 1 / (1 + residuals_median)
#         if camera_power is not None:
#             power_adjusted_weights *= weights
        
#         # Normalize
#         weights = power_adjusted_weights / (np.max(power_adjusted_weights) + 1e-8)

#         # Step J: Track Best Solutions
#         if loss < min_loss:
#             min_loss = loss
#             v_best = np.copy(v)
#             r_hat_best = np.copy(r_hat)
#             weights_best = np.copy(weights)

#         # Step K: Check Convergence
#         delta_v = np.linalg.norm(v - v_old) / (np.linalg.norm(v_old) + 1e-8)
#         if delta_v < tolerance:
#             print(f"Converged after {iteration+1} iterations")
#             break

#         # Update old velocities
#         v_old = np.copy(v)

#     # ==========================
#     # 4. Post-Processing
#     # ==========================

#     # Optionally generate rotations
#     rotated_velocities_list = []
#     if v_best is not None:
#         rotations = generate_orderly_rotations(num_rotations)
#         for R_mat in rotations:
#             # Apply rotation
#             v_rotated = v_best @ R_mat.T  # shape (T, G, 3)
#             rotated_velocities_list.append(v_rotated)

#     # ==========================
#     # 5. Visualization
#     # (Adjust or remove if you handle plotting outside)
#     # ==========================
#     if v_best is not None and r_hat_best is not None and weights_best is not None:
#         # 5.1 Plot velocities for each group
#         fig_vel, axes_vel = plt.subplots(G, 1, figsize=(15, 3*G), sharex=True)
#         if G == 1:
#             axes_vel = [axes_vel]
#         for g in range(G):
#             axes_vel[g].plot(v_best[:, g, 0], label='v_x')
#             axes_vel[g].plot(v_best[:, g, 1], label='v_y')
#             axes_vel[g].plot(v_best[:, g, 2], label='v_z')
#             axes_vel[g].set_title(f'Group {g+1} Velocity Over Time')
#             axes_vel[g].legend()
#             axes_vel[g].grid(True)
#         axes_vel[-1].set_xlabel('Time Steps')
#         plt.tight_layout()
#         plt.show()

#         # 5.2 Plot 3D velocity trajectories for each group
#         fig_3d = plt.figure(figsize=(5*G, 5))
#         for g in range(G):
#             ax = fig_3d.add_subplot(1, G, g+1, projection='3d')
#             vx = v_best[:, g, 0]
#             vy = v_best[:, g, 1]
#             vz = v_best[:, g, 2]
#             ax.plot(vx, vy, vz, label=f'Group {g+1}')
#             ax.set_title(f'3D Velocity Trajectory (Group {g+1})')
#             ax.set_xlabel('v_x')
#             ax.set_ylabel('v_y')
#             ax.set_zlabel('v_z')
#             ax.legend()
#             ax.grid(True)
#         plt.tight_layout()
#         plt.show()

#         # 5.3 Plot fit loss
#         plt.figure()
#         plt.plot(loss_all, marker='o')
#         plt.title('Fit Loss Over Iterations')
#         plt.xlabel('Iteration')
#         plt.ylabel('Fit Loss')
#         plt.grid(True)
#         plt.show()

#     else:
#         print("No valid estimates to visualize.")

#     return v_best, r_hat_best, weights_best, loss_all, rotated_velocities_list

# def estimate_velocity_from_radial(
#         v_r, T, N,
#         lambda_reg=0.1, gamma=0.01,
#         sigma_x2=1.0, sigma_y2=1.0, sigma_z2=1.0,
#         tolerance=1e-3, max_iter=1000, huber_threshold=1.0,
#         temporal_smoothing=0.1, independence_penalty=0.1,
#         smoothness_penalty=0.1, exclusivity_penalty=0.1,
#         magnitude_independence_penalty=0.1, symmetry_penalty=0.1,
#         distance_ratios=None, camera_power=None,
#         fixed_camera_indices=None, fixed_camera_orientations=None,
#         subset_fraction=0.5, weight_learning_rate=0.1,
#         group_indices=None, target_angles=None,
#         angle_penalty_weight=1.0):
#     """
#     Estimate the 3D velocity vector from radial velocities observed by multiple cameras,
#     applying constraints on the angles between the average direction vectors of specified groups.

#     Parameters:
#     - v_r (np.ndarray): Observed radial velocities, shape (T, N).
#     - T (int): Number of time steps.
#     - N (int): Number of cameras.
#     - lambda_reg (float): Regularization parameter.
#     - gamma (float): Regularization parameter for ICA.
#     - sigma_x2, sigma_y2, sigma_z2 (float): Variance parameters for regularization.
#     - tolerance (float): Convergence tolerance.
#     - max_iter (int): Maximum number of iterations.
#     - huber_threshold (float): Threshold for Huber loss.
#     - temporal_smoothing (float): Temporal smoothing factor.
#     - independence_penalty (float): Penalty for independence regularization.
#     - smoothness_penalty (float): Penalty for temporal smoothness.
#     - exclusivity_penalty (float): Penalty for mutual exclusivity.
#     - magnitude_independence_penalty (float): Penalty for magnitude independence.
#     - symmetry_penalty (float): Penalty for symmetry constraints.
#     - distance_ratios (list or np.ndarray): Distance ratios for each camera.
#     - camera_power (list or np.ndarray): Initial camera power levels.
#     - fixed_camera_indices (list): Indices of cameras to fix (optional).
#     - fixed_camera_orientations (np.ndarray): Orientations of fixed cameras, shape (M, 3).
#     - subset_fraction (float): Fraction of cameras to select in each iteration.
#     - weight_learning_rate (float): Learning rate for updating weights.
#     - group_indices (list of lists): List containing indices of cameras for each group.
#     - target_angles (list of floats): Desired angles in radians between the average vectors of the groups.
#       Should be a list of three angles: [angle between group 1 and 2, group 1 and 3, group 2 and 3].
#     - angle_penalty_weight (float): Weight of the angle penalty term in the loss function.

#     Returns:
#     - v_best (np.ndarray): Estimated 3D velocity vectors over time, shape (T, 3).
#     - r_hat_best (np.ndarray): Estimated direction vectors for each camera, shape (N, 3).
#     - weights_best (np.ndarray): Final weights assigned to each camera, shape (N,).
#     - loss_all (list): List of fit loss values over iterations.
#     """
#     # ==========================
#     # Helper Functions
#     # ==========================

#     def safe_normalize(vec, epsilon=1e-8):
#         norm = np.linalg.norm(vec)
#         if norm < epsilon:
#             return np.ones_like(vec) / np.sqrt(len(vec))  # Default unit vector
#         return vec / norm

#     def validate_input(v_r):
#         if np.isnan(v_r).any() or np.isinf(v_r).any():
#             raise ValueError("Input radial velocities contain NaN or infinite values.")

#     def huber(residual, delta):
#         return np.where(np.abs(residual) <= delta,
#                         0.5 * residual**2,
#                         delta * (np.abs(residual) - 0.5 * delta))

#     # ==========================
#     # Input Validation
#     # ==========================

#     validate_input(v_r)

#     if group_indices is None or len(group_indices) != 3:
#         raise ValueError("group_indices must be provided and contain three lists of camera indices.")
#     if target_angles is None or len(target_angles) != 3:
#         raise ValueError("target_angles must be provided and contain three angles in radians.")

#     # ==========================
#     # Initialization
#     # ==========================

#     np.random.seed(42)  # For reproducibility
#     v = np.random.randn(T, 3)      # Initialize 3D velocity vectors at each time step
#     r_hat = np.random.randn(N, 3)  # Initialize direction vectors randomly

#     # Normalize r_hat to be unit vectors initially
#     r_hat = np.array([safe_normalize(vec) for vec in r_hat])

#     # Fix the direction vectors of the fixed cameras
#     if fixed_camera_indices is not None and fixed_camera_orientations is not None:
#         for idx, cam_idx in enumerate(fixed_camera_indices):
#             r_hat[cam_idx] = safe_normalize(fixed_camera_orientations[idx])

#     # Initialize weights for each camera based on power if provided
#     if camera_power is not None:
#         weights = np.array(camera_power)
#         weights /= np.max(weights)  # Normalize weights to be between 0 and 1
#     else:
#         weights = np.ones(N)

#     # Normalize the distance_ratios if provided
#     if distance_ratios is not None:
#         distance_ratios = np.array(distance_ratios) / np.linalg.norm(distance_ratios)

#     # Initialize previous velocity for convergence check
#     v_old = np.copy(v)

#     # Variables to track the minimum loss and corresponding variables
#     min_loss = np.inf
#     v_best = None
#     r_hat_best = None
#     weights_best = None
#     loss_all = []

#     # ==========================
#     # Optimization Loop
#     # ==========================

#     for iter in (pbar := tqdm(range(max_iter))):
#         # Step 1: Randomly select a subset of cameras
#         subset_size = max(1, int(subset_fraction * N))
#         selected_indices = np.random.choice(N, size=subset_size, replace=False)

#         # Ensure the fixed cameras are always included
#         if fixed_camera_indices is not None:
#             for cam_idx in fixed_camera_indices:
#                 if cam_idx not in selected_indices:
#                     selected_indices[0] = cam_idx  # Replace the first selected index with the fixed camera

#         # Extract selected subset data
#         selected_r_hat = r_hat[selected_indices]  # Shape: (subset_size, 3)
#         selected_weights = weights[selected_indices]  # Shape: (subset_size,)
#         selected_v_r = v_r[:, selected_indices]  # Shape: (T, subset_size)

#         # Step 2: Optimize v(t) for each time step t using selected subset
#         for t in range(T):
#             A = selected_r_hat * selected_weights[:, np.newaxis]  # Shape: (subset_size, 3)
#             b = selected_v_r[t] * selected_weights  # Shape: (subset_size,)

#             # Regularization matrix for independence of components
#             L = np.diag([gamma / sigma_x2, gamma / sigma_y2, gamma / sigma_z2])

#             # Solve weighted regularized least squares: (A^T A + L) v(t) = A^T b
#             AtA = A.T @ A + L
#             Atb = A.T @ b
#             try:
#                 v[t] = np.linalg.solve(AtA, Atb)
#             except np.linalg.LinAlgError:
#                 # If singular matrix, use least squares solution
#                 v[t], _, _, _ = np.linalg.lstsq(AtA, Atb, rcond=None)

#             # Handle potential NaN values
#             v[t] = np.nan_to_num(v[t], nan=0.0, posinf=0.0, neginf=0.0)

#         # Enforce mean-zero constraint on v
#         v_mean = np.mean(v, axis=0)
#         v -= v_mean  # Subtract mean from each component

#         # Step 3: Apply ICA to make velocity components independent
#         try:
#             ica = FastICA(n_components=3, random_state=iter)
#             v_ica = ica.fit_transform(v)
#         except Exception as e:
#             print(f"ICA failed at iteration {iter+1}: {e}")
#             v_ica = v  # Fallback to raw velocities if ICA fails

#         # Rescale the components to minimize loss
#         s = np.ones(3)
#         for i in range(3):
#             numerator = 0
#             denominator = 0
#             for t in range(T):
#                 predicted = v_ica[t, i] * r_hat[:, i]
#                 numerator += np.dot(weights, v_r[t] * predicted)
#                 denominator += np.dot(weights, predicted**2)
#             s[i] = numerator / (denominator + 1e-8)  # Prevent division by zero

#         v_rescaled = v_ica * s  # Rescaled velocities

#         # Update v
#         v = v_rescaled

#         # Step 4: Apply Regularizations on v

#         # Temporal smoothing
#         if iter > 0:
#             v = (1 - temporal_smoothing) * v + temporal_smoothing * v_old

#         # Temporal smoothness regularization
#         v_diff = np.diff(v, axis=0)
#         v[:-1] -= smoothness_penalty * v_diff
#         v[1:] -= smoothness_penalty * (-v_diff)

#         # Mutual exclusivity regularization
#         v -= exclusivity_penalty * (v * v**2)

#         # Magnitude independence regularization
#         v -= magnitude_independence_penalty * (np.abs(v) * np.abs(v))

#         # Symmetry constraint regularization
#         for t in range(T // 2):
#             t_sym = T - 1 - t  # Symmetric time index
#             v_mean = (v[t] + v[t_sym]) / 2
#             v[t] = v_mean + symmetry_penalty * (v[t] - v_mean)
#             v[t_sym] = v_mean + symmetry_penalty * (v[t_sym] - v_mean)

#         # Step 5: Optimize r_hat for each camera
#         for i in range(N):
#             # Skip fixed cameras
#             if fixed_camera_indices is not None and i in fixed_camera_indices:
#                 continue

#             b = v_r[:, i] * weights[i]
#             A = v  # Shape: (T, 3)

#             # Solve least squares for r_hat[i]
#             try:
#                 r_hat_i, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
#             except np.linalg.LinAlgError:
#                 r_hat_i = r_hat[i]  # Fallback to previous value if lstsq fails

#             # Apply distance ratio constraint if provided
#             if distance_ratios is not None:
#                 target_magnitude = distance_ratios[i]
#                 current_magnitude = np.linalg.norm(r_hat_i)
#                 if current_magnitude > 1e-8:
#                     r_hat_i = (target_magnitude / current_magnitude) * r_hat_i  # Adjust magnitude
#                 else:
#                     r_hat_i = np.array([1.0, 0.0, 0.0])  # Default direction if magnitude is too small

#             # Ensure the direction vector remains on the unit sphere
#             r_hat[i] = safe_normalize(r_hat_i, epsilon=1e-8)

#         # Step 6: Compute Angle Penalty and Add to Loss
#         # Compute average vectors of the groups
#         r_avg_1 = np.mean(r_hat[group_indices[0]], axis=0)
#         r_avg_2 = np.mean(r_hat[group_indices[1]], axis=0)
#         r_avg_3 = np.mean(r_hat[group_indices[2]], axis=0)

#         # Normalize average vectors
#         r_avg_1 = safe_normalize(r_avg_1)
#         r_avg_2 = safe_normalize(r_avg_2)
#         r_avg_3 = safe_normalize(r_avg_3)

#         # Compute angles between average vectors
#         cos_theta_12 = np.dot(r_avg_1, r_avg_2)
#         cos_theta_12 = np.clip(cos_theta_12, -1.0, 1.0)
#         theta_12 = np.arccos(cos_theta_12)

#         cos_theta_13 = np.dot(r_avg_1, r_avg_3)
#         cos_theta_13 = np.clip(cos_theta_13, -1.0, 1.0)
#         theta_13 = np.arccos(cos_theta_13)

#         cos_theta_23 = np.dot(r_avg_2, r_avg_3)
#         cos_theta_23 = np.clip(cos_theta_23, -1.0, 1.0)
#         theta_23 = np.arccos(cos_theta_23)

#         # Compute angle differences
#         angle_diff_12 = theta_12 - target_angles[0]
#         angle_diff_13 = theta_13 - target_angles[1]
#         angle_diff_23 = theta_23 - target_angles[2]

#         # Compute angle penalty
#         angle_penalty = angle_diff_12**2 + angle_diff_13**2 + angle_diff_23**2

#         # Step 7: Calculate Residuals and Fit Loss on All Cameras
#         residuals = v_r - (v @ r_hat.T)  # Shape: (T, N)

#         # Calculate Huber loss
#         if huber_threshold is not None:
#             fit_loss = np.sum(huber(residuals, huber_threshold) * weights)
#         else:
#             fit_loss = np.sum((residuals**2 * weights))

#         # Total loss
#         loss = fit_loss + angle_penalty_weight * angle_penalty

#         pbar.set_description(f"Iter {iter}: Loss={loss:.4f}, Fit Loss={fit_loss:.4f}, Angle Penalty={angle_penalty:.4f}")

#         loss_all.append(loss)

#         # Step 8: Update Weights Based on Residuals
#         residuals_abs = np.abs(residuals)
#         residuals_median = np.median(residuals_abs, axis=0)
#         weights_update = 1 / (1 + residuals_median)
#         weights = (1 - weight_learning_rate) * weights + weight_learning_rate * weights_update
#         weights = weights / (np.max(weights) + 1e-8)  # Normalize weights

#         # Step 9: Track the Best Estimates
#         if loss < min_loss:
#             min_loss = loss
#             v_best = v.copy()
#             r_hat_best = r_hat.copy()
#             weights_best = weights.copy()

#         # Step 10: Check for Convergence
#         delta_v = np.linalg.norm(v - v_old) / (np.linalg.norm(v_old) + 1e-8)
#         if delta_v < tolerance:
#             print(f"Converged after {iter+1} iterations")
#             break
#         v_old = v.copy()

#     # ==========================
#     # Return the Results
#     # ==========================

#     return v_best, r_hat_best, weights_best, loss_all


# import numpy as np
# from tqdm import tqdm
# from sklearn.decomposition import FastICA
# import matplotlib.pyplot as plt

# def estimate_velocity_from_radial(
#         v_r, T, N,
#         lambda_reg=0.1, gamma=0.01,
#         sigma_x2=1.0, sigma_y2=1.0, sigma_z2=1.0,
#         tolerance=1e-3, max_iter=1000, huber_threshold=1.0,
#         temporal_smoothing=0.1, independence_penalty=0.1,
#         smoothness_penalty=0.1, exclusivity_penalty=0.1,
#         magnitude_independence_penalty=0.1, symmetry_penalty=0.1,
#         distance_ratios=None, camera_power=None,
#         fixed_camera_indices=None, fixed_camera_orientations=None,
#         subset_fraction=0.5, weight_learning_rate=0.1,
#         group_indices=None, target_angles=None,
#         angle_penalty_weight=1.0):
#     """
#     Estimate the 3D velocity vector from radial velocities observed by multiple cameras,
#     applying constraints on the angles between the average direction vectors of specified groups.

#     Parameters:
#     - v_r (np.ndarray): Observed radial velocities, shape (T, N).
#     - T (int): Number of time steps.
#     - N (int): Number of cameras.
#     - ... [other parameters remain the same] ...
#     """

#     # ==========================
#     # Helper Functions
#     # ==========================

#     def safe_normalize(vec, epsilon=1e-8):
#         norm = np.linalg.norm(vec)
#         if norm < epsilon:
#             return np.ones_like(vec) / np.sqrt(len(vec))  # Default unit vector
#         return vec / norm

#     def validate_input(v_r):
#         if np.isnan(v_r).any() or np.isinf(v_r).any():
#             raise ValueError("Input radial velocities contain NaN or infinite values.")

#     def huber(residual, delta):
#         return np.where(np.abs(residual) <= delta,
#                         0.5 * residual**2,
#                         delta * (np.abs(residual) - 0.5 * delta))

#     # ==========================
#     # Input Validation
#     # ==========================

#     validate_input(v_r)

#     if group_indices is None or len(group_indices) != 3:
#         raise ValueError("group_indices must be provided and contain three lists of camera indices.")
#     if target_angles is None or len(target_angles) != 3:
#         raise ValueError("target_angles must be provided and contain three angles in radians.")

#     # ==========================
#     # Initialization
#     # ==========================

#     np.random.seed(42)  # For reproducibility
#     v = np.random.randn(T, 3)      # Initialize 3D velocity vectors at each time step
#     r_hat = np.random.randn(N, 3)  # Initialize direction vectors randomly

#     # Normalize r_hat to be unit vectors initially
#     r_hat = np.array([safe_normalize(vec) for vec in r_hat])

#     # Fix the direction vectors of the fixed cameras
#     if fixed_camera_indices is not None and fixed_camera_orientations is not None:
#         for idx, cam_idx in enumerate(fixed_camera_indices):
#             r_hat[cam_idx] = safe_normalize(fixed_camera_orientations[idx])

#     # Initialize weights for each camera based on power if provided
#     if camera_power is not None:
#         weights = np.array(camera_power)
#         weights /= np.max(weights)  # Normalize weights to be between 0 and 1
#     else:
#         weights = np.ones(N)

#     # Normalize the distance_ratios if provided
#     if distance_ratios is not None:
#         distance_ratios = np.array(distance_ratios) / np.linalg.norm(distance_ratios)

#     # Initialize previous velocity for convergence check
#     v_old = np.copy(v)

#     # Variables to track the minimum loss and corresponding variables
#     min_loss = np.inf
#     v_best = None
#     r_hat_best = None
#     weights_best = None
#     loss_all = []

#     # ==========================
#     # Optimization Loop
#     # ==========================

#     for iter in (pbar := tqdm(range(max_iter))):
#         # Step 1: Randomly select a subset of cameras
#         subset_size = max(1, int(subset_fraction * N))
#         selected_indices = np.random.choice(N, size=subset_size, replace=False)

#         # Ensure the fixed cameras are always included
#         if fixed_camera_indices is not None:
#             for cam_idx in fixed_camera_indices:
#                 if cam_idx not in selected_indices:
#                     selected_indices[0] = cam_idx  # Replace the first selected index with the fixed camera

#         # Extract selected subset data
#         selected_r_hat = r_hat[selected_indices]  # Shape: (subset_size, 3)
#         selected_weights = weights[selected_indices]  # Shape: (subset_size,)
#         selected_v_r = v_r[:, selected_indices]  # Shape: (T, subset_size)

#         # Step 2: Optimize v(t) for each time step t using selected subset
#         for t in range(T):
#             A = selected_r_hat * selected_weights[:, np.newaxis]  # Shape: (subset_size, 3)
#             b = selected_v_r[t] * selected_weights  # Shape: (subset_size,)

#             # Regularization matrix for independence of components
#             L = np.diag([gamma / sigma_x2, gamma / sigma_y2, gamma / sigma_z2])

#             # Solve weighted regularized least squares: (A^T A + L) v(t) = A^T b
#             AtA = A.T @ A + L
#             Atb = A.T @ b
#             try:
#                 v[t] = np.linalg.solve(AtA, Atb)
#             except np.linalg.LinAlgError:
#                 # If singular matrix, use least squares solution
#                 v[t], _, _, _ = np.linalg.lstsq(AtA, Atb, rcond=None)

#             # Handle potential NaN values
#             v[t] = np.nan_to_num(v[t], nan=0.0, posinf=0.0, neginf=0.0)

#         # Enforce mean-zero constraint on v
#         v_mean = np.mean(v, axis=0)
#         v -= v_mean  # Subtract mean from each component

#         # Step 3: Apply ICA to make velocity components independent
#         try:
#             ica = FastICA(n_components=3, random_state=iter)
#             v_ica = ica.fit_transform(v)
#         except Exception as e:
#             print(f"ICA failed at iteration {iter+1}: {e}")
#             v_ica = v  # Fallback to raw velocities if ICA fails

#         # Rescale the components to minimize loss
#         s = np.ones(3)
#         for i in range(3):
#             numerator = 0
#             denominator = 0
#             for t in range(T):
#                 predicted = v_ica[t, i] * r_hat[:, i]
#                 numerator += np.dot(weights, v_r[t] * predicted)
#                 denominator += np.dot(weights, predicted**2)
#             s[i] = numerator / (denominator + 1e-8)  # Prevent division by zero

#         v_rescaled = v_ica * s  # Rescaled velocities

#         # Update v
#         v = v_rescaled

#         # Step 4: Apply Regularizations on v

#         # Temporal smoothing
#         if iter > 0:
#             v = (1 - temporal_smoothing) * v + temporal_smoothing * v_old

#         # Temporal smoothness regularization
#         v_diff = np.diff(v, axis=0)
#         v[:-1] -= smoothness_penalty * v_diff
#         v[1:] -= smoothness_penalty * (-v_diff)

#         # Mutual exclusivity regularization
#         v -= exclusivity_penalty * (v * v**2)

#         # Magnitude independence regularization
#         v -= magnitude_independence_penalty * (np.abs(v) * np.abs(v))

#         # Symmetry constraint regularization
#         for t in range(T // 2):
#             t_sym = T - 1 - t  # Symmetric time index
#             v_mean = (v[t] + v[t_sym]) / 2
#             v[t] = v_mean + symmetry_penalty * (v[t] - v_mean)
#             v[t_sym] = v_mean + symmetry_penalty * (v[t_sym] - v_mean)

#         # Step 5: Optimize r_hat for each camera
#         for i in range(N):
#             # Skip fixed cameras
#             if fixed_camera_indices is not None and i in fixed_camera_indices:
#                 continue

#             b = v_r[:, i] * weights[i]
#             A = v  # Shape: (T, 3)

#             # Solve least squares for r_hat[i]
#             try:
#                 r_hat_i, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
#             except np.linalg.LinAlgError:
#                 r_hat_i = r_hat[i]  # Fallback to previous value if lstsq fails

#             # Apply distance ratio constraint if provided
#             if distance_ratios is not None:
#                 target_magnitude = distance_ratios[i]
#                 current_magnitude = np.linalg.norm(r_hat_i)
#                 if current_magnitude > 1e-8:
#                     r_hat_i = (target_magnitude / current_magnitude) * r_hat_i  # Adjust magnitude
#                 else:
#                     r_hat_i = np.array([1.0, 0.0, 0.0])  # Default direction if magnitude is too small

#             # Ensure the direction vector remains on the unit sphere
#             r_hat[i] = safe_normalize(r_hat_i, epsilon=1e-8)

#         # Step 6: Compute Angle Penalty and Its Gradient
#         # Compute average vectors of the groups
#         r_avg = []
#         for group in group_indices:
#             avg = np.mean(r_hat[group], axis=0)
#             r_avg.append(safe_normalize(avg))
#         r_avg_1, r_avg_2, r_avg_3 = r_avg

#         # Compute angles between average vectors
#         cos_theta_12 = np.dot(r_avg_1, r_avg_2)
#         theta_12 = np.arccos(np.clip(cos_theta_12, -1.0, 1.0))

#         cos_theta_13 = np.dot(r_avg_1, r_avg_3)
#         theta_13 = np.arccos(np.clip(cos_theta_13, -1.0, 1.0))

#         cos_theta_23 = np.dot(r_avg_2, r_avg_3)
#         theta_23 = np.arccos(np.clip(cos_theta_23, -1.0, 1.0))

#         # Compute angle differences
#         angle_diff_12 = theta_12 - target_angles[0]
#         angle_diff_13 = theta_13 - target_angles[1]
#         angle_diff_23 = theta_23 - target_angles[2]

#         # Compute angle penalty
#         angle_penalty = angle_diff_12**2 + angle_diff_13**2 + angle_diff_23**2

#         # Compute gradients of the angle penalty with respect to r_hat
#         grad_r_hat = np.zeros_like(r_hat)

#         # Helper function to compute gradient for a pair of groups
#         def compute_angle_gradient(r_avg_k, r_avg_l, angle_diff, group_k, group_l):
#             # Derivative of arccos(dot_product) with respect to r_avg_k
#             cos_theta = np.dot(r_avg_k, r_avg_l)
#             sin_theta = np.sqrt(1 - cos_theta**2) + 1e-8  # Add epsilon to avoid division by zero
#             dtheta_dr_avg_k = - (1 / sin_theta) * r_avg_l

#             # Distribute gradient to each camera in group_k
#             N_k = len(group_k)
#             for idx in group_k:
#                 grad_r_hat[idx] += (2 * angle_diff * dtheta_dr_avg_k) / N_k

#             # Similarly for group_l
#             dtheta_dr_avg_l = - (1 / sin_theta) * r_avg_k
#             N_l = len(group_l)
#             for idx in group_l:
#                 grad_r_hat[idx] += (2 * angle_diff * dtheta_dr_avg_l) / N_l

#         # Compute gradients for each pair
#         compute_angle_gradient(r_avg_1, r_avg_2, angle_diff_12, group_indices[0], group_indices[1])
#         compute_angle_gradient(r_avg_1, r_avg_3, angle_diff_13, group_indices[0], group_indices[2])
#         compute_angle_gradient(r_avg_2, r_avg_3, angle_diff_23, group_indices[1], group_indices[2])

#         # Update r_hat using the gradient of the angle penalty
#         for i in range(N):
#             if fixed_camera_indices is not None and i in fixed_camera_indices:
#                 continue  # Skip fixed cameras

#             # Update r_hat with gradient descent step
#             r_hat[i] -= angle_penalty_weight * grad_r_hat[i]

#             # Re-normalize r_hat[i]
#             r_hat[i] = safe_normalize(r_hat[i])

#         # Step 7: Calculate Residuals and Fit Loss on All Cameras
#         residuals = v_r - (v @ r_hat.T)  # Shape: (T, N)

#         # Calculate Huber loss
#         if huber_threshold is not None:
#             fit_loss = np.sum(huber(residuals, huber_threshold) * weights)
#         else:
#             fit_loss = np.sum((residuals**2 * weights))

#         # Total loss
#         loss = fit_loss + angle_penalty_weight * angle_penalty

#         pbar.set_description(f"Iter {iter}: Loss={loss:.4f}, Fit Loss={fit_loss:.4f}, Angle Penalty={angle_penalty:.4f}")

#         loss_all.append(loss)

#         # Step 8: Update Weights Based on Residuals
#         residuals_abs = np.abs(residuals)
#         residuals_median = np.median(residuals_abs, axis=0)
#         weights_update = 1 / (1 + residuals_median)
#         weights = (1 - weight_learning_rate) * weights + weight_learning_rate * weights_update
#         weights = weights / (np.max(weights) + 1e-8)  # Normalize weights

#         # Step 9: Track the Best Estimates
#         if loss < min_loss:
#             min_loss = loss
#             v_best = v.copy()
#             r_hat_best = r_hat.copy()
#             weights_best = weights.copy()

#         # Step 10: Check for Convergence
#         delta_v = np.linalg.norm(v - v_old) / (np.linalg.norm(v_old) + 1e-8)
#         if delta_v < tolerance:
#             print(f"Converged after {iter+1} iterations")
#             break
#         v_old = v.copy()

#     # ==========================
#     # Return the Results
#     # ==========================

#     return v_best, r_hat_best, weights_best, loss_all

import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def create_3d_surface_from_boundary(x, y, z):
    """
    Create and plot a 3D surface with the boundary defined by the given x, y, z points.
    
    Parameters:
        x (list or array): List of x coordinates (N elements)
        y (list or array): List of y coordinates (N elements)
        z (list or array): List of z coordinates (N elements)
    """
    # Ensure x, y, z are NumPy arrays
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    
    # Combine x, y into a single 2D array for triangulation
    points2D = np.column_stack((x, y))
    
    # Perform Delaunay triangulation on the 2D projection of the points
    tri = Delaunay(points2D)
    
    # Plot the 3D surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    ax.plot_trisurf(x, y, z, triangles=tri.simplices, cmap='viridis', edgecolor='k')
    
    # Highlight the boundary
    ax.plot(x, y, z, color='red', marker='o', linestyle='-', linewidth=2)
    
    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Surface with Defined Boundary')
    
    plt.show()

# import numpy as np
# from tqdm import tqdm
# from sklearn.decomposition import FastICA
# from scipy.linalg import orthogonal_procrustes
# import matplotlib.pyplot as plt

# def estimate_velocity_from_radial(
#         v_r, T, N, lambda_reg=0.1, gamma=0.01,
#         sigma_x2=1.0, sigma_y2=1.0, sigma_z2=1.0,
#         tolerance=1e-3, max_iter=1000, huber_threshold=1.0,
#         temporal_smoothing=0.1, independence_penalty=0.1,
#         smoothness_penalty=0.1, exclusivity_penalty=0.1,
#         magnitude_independence_penalty=0.1, symmetry_penalty=0.1,
#         distance_ratios=None, camera_power=None, fixed_first_camera_index=0,
#         subset_fraction=0.5, weight_learning_rate=0.1,
#         num_rotations=100):
#     """
#     Estimate the 3D velocity vector from radial velocities observed by multiple cameras,
#     and generate multiple possible solutions due to rotational ambiguities by sampling rotations.
    
#     Parameters:
#     - v_r (np.ndarray): Observed radial velocities, shape (T, N).
#     - T (int): Number of time steps.
#     - N (int): Number of cameras.
#     - lambda_reg (float): Regularization parameter.
#     - gamma (float): Regularization parameter for ICA.
#     - sigma_x2, sigma_y2, sigma_z2 (float): Variance parameters for regularization.
#     - tolerance (float): Convergence tolerance.
#     - max_iter (int): Maximum number of iterations.
#     - huber_threshold (float): Threshold for Huber loss.
#     - temporal_smoothing (float): Temporal smoothing factor.
#     - independence_penalty (float): Penalty for independence regularization.
#     - smoothness_penalty (float): Penalty for temporal smoothness.
#     - exclusivity_penalty (float): Penalty for mutual exclusivity.
#     - magnitude_independence_penalty (float): Penalty for magnitude independence.
#     - symmetry_penalty (float): Penalty for symmetry constraints.
#     - distance_ratios (list or np.ndarray): Distance ratios for each camera.
#     - camera_power (list or np.ndarray): Initial camera power levels.
#     - fixed_first_camera_index (int): Index of the camera to always include in optimization.
#     - subset_fraction (float): Fraction of cameras to select in each iteration (e.g., 0.5 for 50%).
#     - weight_learning_rate (float): Learning rate for updating weights (0 < weight_learning_rate <= 1).
#     - num_rotations (int): Number of rotations to sample from SO(3).
    
#     Returns:
#     - rotated_velocities_list (list): List of rotated velocities, each of shape (T, 3).
#     - rotations_list (list): Corresponding list of rotation matrices applied.
#     - loss_all (list): List of fit loss values over iterations.
#     """
#     # ==========================
#     # Helper Functions
#     # ==========================
    
#     def safe_normalize(vec, epsilon=1e-8):
#         norm = np.linalg.norm(vec)
#         if norm < epsilon:
#             return np.ones_like(vec) / np.sqrt(len(vec))  # Default unit vector
#         return vec / norm

#     def validate_input(v_r):
#         if np.isnan(v_r).any() or np.isinf(v_r).any():
#             raise ValueError("Input radial velocities contain NaN or infinite values.")

#     def huber(residual, delta):
#         return np.where(np.abs(residual) <= delta,
#                         0.5 * residual**2,
#                         delta * (np.abs(residual) - 0.5 * delta))

#     def sample_random_rotation():
#         """Generate a random rotation matrix uniformly sampled from SO(3)."""
#         # Random unit quaternion
#         u1, u2, u3 = np.random.uniform(0, 1, 3)
#         q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
#         q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
#         q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
#         q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
#         # Quaternion to rotation matrix
#         R = np.array([
#             [1 - 2*(q3**2 + q4**2),     2*(q2*q3 - q1*q4),     2*(q2*q4 + q1*q3)],
#             [    2*(q2*q3 + q1*q4), 1 - 2*(q2**2 + q4**2),     2*(q3*q4 - q1*q2)],
#             [    2*(q2*q4 - q1*q3),     2*(q3*q4 + q1*q2), 1 - 2*(q2**2 + q3**2)]
#         ])
#         return R

#     # ==========================
#     # Input Validation
#     # ==========================

#     validate_input(v_r)

#     # ==========================
#     # Initialization
#     # ==========================

#     np.random.seed(42)  # For reproducibility
#     v = np.random.randn(T, 3)      # Initialize 3D velocity vectors at each time step
#     r_hat = np.random.randn(N, 3)  # Initialize direction vectors randomly

#     # Normalize r_hat to be unit vectors initially
#     r_hat = np.array([safe_normalize(vec) for vec in r_hat])

#     # Fix the direction vector of the first camera
#     r_hat[fixed_first_camera_index] = safe_normalize(np.array([1.0, 0.0, 0.0]))

#     # Initialize weights for each camera based on power if provided
#     if camera_power is not None:
#         weights = np.array(camera_power)
#         weights /= np.max(weights)  # Normalize weights to be between 0 and 1
#     else:
#         weights = np.ones(N)

#     # Normalize the distance_ratios if provided
#     if distance_ratios is not None:
#         distance_ratios = np.array(distance_ratios) / np.linalg.norm(distance_ratios)

#     # Initialize previous velocity for convergence check
#     v_old = np.copy(v)

#     # Variables to track the minimum loss and corresponding variables
#     min_loss = np.inf
#     v_best = None
#     r_hat_best = None
#     weights_best = None
#     loss_all = []

#     # ==========================
#     # Optimization Loop
#     # ==========================

#     for iter in (pbar := tqdm(range(max_iter))):
#         # Step 1: Randomly select a subset of cameras
#         subset_size = max(1, int(subset_fraction * N))
#         selected_indices = np.random.choice(N, size=subset_size, replace=False)
#         # Ensure the fixed camera is always included
#         if fixed_first_camera_index not in selected_indices:
#             selected_indices[0] = fixed_first_camera_index  # Replace the first selected index with the fixed camera

#         # Extract selected subset data
#         selected_r_hat = r_hat[selected_indices]  # Shape: (subset_size, 3)
#         selected_weights = weights[selected_indices]  # Shape: (subset_size,)
#         selected_v_r = v_r[:, selected_indices]  # Shape: (T, subset_size)

#         # Step 2: Optimize v(t) for each time step t using selected subset
#         for t in range(T):
#             A = selected_r_hat * selected_weights[:, np.newaxis]  # Shape: (subset_size, 3)
#             b = selected_v_r[t] * selected_weights  # Shape: (subset_size,)

#             # Regularization matrix for independence of components
#             L = np.diag([gamma / sigma_x2, gamma / sigma_y2, gamma / sigma_z2])

#             # Solve weighted regularized least squares: (A^T A + L) v(t) = A^T b
#             AtA = A.T @ A + L
#             Atb = A.T @ b
#             try:
#                 v[t] = np.linalg.solve(AtA, Atb)
#             except np.linalg.LinAlgError:
#                 # If singular matrix, use least squares solution
#                 v[t], _, _, _ = np.linalg.lstsq(AtA, Atb, rcond=None)

#             # Handle potential NaN values
#             v[t] = np.nan_to_num(v[t], nan=0.0, posinf=0.0, neginf=0.0)

#         # Enforce mean-zero constraint on v
#         v_mean = np.mean(v, axis=0)
#         v -= v_mean  # Subtract mean from each component

#         # Step 3: Apply ICA to make velocity components independent
#         try:
#             ica = FastICA(n_components=3, random_state=iter)
#             v_ica = ica.fit_transform(v)
#         except Exception as e:
#             print(f"ICA failed at iteration {iter+1}: {e}")
#             v_ica = v  # Fallback to raw velocities if ICA fails

#         # Rescale the components to minimize loss
#         s = np.ones(3)
#         for i in range(3):
#             numerator = 0
#             denominator = 0
#             for t in range(T):
#                 predicted = v_ica[t, i] * r_hat[:, i]
#                 numerator += np.dot(weights, v_r[t] * predicted)
#                 denominator += np.dot(weights, predicted**2)
#             s[i] = numerator / (denominator + 1e-8)  # Prevent division by zero

#         v_rescaled = v_ica * s  # Rescaled velocities

#         # Update v and r_hat
#         v = v_rescaled
#         # r_hat remains unchanged

#         # Step 4: Regularizations (if necessary)
#         # ... [Apply regularizations based on your requirements] ...

#         # Step 5: Calculate Residuals and Fit Loss on All Cameras
#         residuals = v_r - (v @ r_hat.T)  # Shape: (T, N)

#         # Calculate Huber loss
#         if huber_threshold is not None:
#             loss = np.sum(huber(residuals, huber_threshold) * weights)
#         else:
#             loss = np.sum((residuals**2 * weights))

#         pbar.set_description(f"Iteration {iter}: Fit Loss = {loss:.4f}")
#         loss_all.append(loss)

#         # Step 6: Update Weights Based on Residuals (optional)
#         # ... [Update weights if necessary] ...

#         # Step 7: Track the Best Estimates
#         if loss < min_loss:
#             min_loss = loss
#             v_best = v.copy()
#             r_hat_best = r_hat.copy()
#             weights_best = weights.copy()

#         # Step 8: Check for Convergence
#         delta_v = np.linalg.norm(v - v_old) / (np.linalg.norm(v_old) + 1e-8)
#         if delta_v < tolerance:
#             print(f"Converged after {iter+1} iterations")
#             break
#         v_old = v.copy()

#     # ==========================
#     # Sample Rotations and Generate Possible Solutions
#     # ==========================

#     if v_best is not None:
#         rotated_velocities_list = []
#         rotations_list = []
#         for _ in range(num_rotations):
#             R = sample_random_rotation()
#             v_rotated = v_best @ R.T  # Apply rotation to velocities
#             rotated_velocities_list.append(v_rotated)
#             rotations_list.append(R)
#     else:
#         print("No valid estimates to generate possible velocities.")
#         rotated_velocities_list = []
#         rotations_list = []

#     # ==========================
#     # Return the Results
#     # ==========================

#     return rotated_velocities_list, rotations_list, loss_all

def angle_between_vectors(v1, v2):
    """
    Calculate the angle (in degrees) between two vectors.
    
    Parameters:
        v1 (list or numpy array): First vector.
        v2 (list or numpy array): Second vector.
    
    Returns:
        float: The angle between the two vectors in degrees.
    """
    # Convert vectors to numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    # Compute the dot product and magnitudes of the vectors
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    # Calculate the cosine of the angle
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    
    # Clip the value to avoid numerical errors outside the valid range of arccos
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Compute the angle in radians and convert to degrees
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg
# def estimate_velocity_from_radial(v_r, T, N, lambda_reg=0.1, gamma=0.01,
#                                   sigma_x2=1.0, sigma_y2=1.0, sigma_z2=1.0,
#                                   tolerance=1e-3, max_iter=1000, huber_threshold=1.0,
#                                   temporal_smoothing=0.1, independence_penalty=0.1,
#                                   smoothness_penalty=0.1, exclusivity_penalty=0.1,
#                                   magnitude_independence_penalty=0.1, distance_ratios=None,
#                                   camera_power=None, weight_learning_rate=0.05,
#                                   align_each_iteration=True):
#     """
#     Estimate the 3D velocity vector from radial velocities observed by multiple cameras,
#     applying ICA and fixing the rotational ambiguity to make the solution unique.
#     Prints fit loss every 100 iterations and returns the velocities, weights, and directions
#     corresponding to the minimum loss. Includes visualization as part of the function.
#     This version updates weights during optimization, allowing cameras to be ignored or considered
#     regardless of initial weights. It also includes an option to apply alignment in every iteration
#     or only at the end.

#     Parameters:
#     - v_r (np.ndarray): Observed radial velocities, shape (T, N).
#     - T (int): Number of time steps.
#     - N (int): Number of cameras.
#     - weight_learning_rate (float): Learning rate for updating weights (0 < weight_learning_rate <= 1).
#     - align_each_iteration (bool): If True, apply alignment at every iteration; if False, only at the end.
#     - Other parameters are as previously defined.

#     Returns:
#     - v_best (np.ndarray): Estimated 3D velocity vectors over time corresponding to minimum loss, shape (T, 3).
#     - r_hat_best (np.ndarray): Estimated direction vectors for each camera corresponding to minimum loss, shape (N, 3).
#     - weights_best (np.ndarray): Final weights assigned to each camera corresponding to minimum loss, shape (N,).
#     """
#     # Initialize the 3D velocity vector v and the direction vectors r_hat randomly
#     np.random.seed(42)  # For reproducibility
#     v = np.random.randn(T, 3)      # Initialize 3D velocity vectors at each time step
#     r_hat = np.random.randn(N, 3)  # Initialize direction vectors randomly

#     # Normalize r_hat to be unit vectors initially
#     r_hat = r_hat / np.linalg.norm(r_hat, axis=1, keepdims=True)

#     # Initialize weights for each camera based on power if provided
#     if camera_power is not None:
#         # Compute weights inversely proportional to the camera power
#         weights = np.array(camera_power)
#         weights /= np.max(weights)  # Normalize weights to be between 0 and 1
#     else:
#         weights = np.ones(N)

#     # Normalize the distance_ratios if provided
#     if distance_ratios is not None:
#         distance_ratios = np.array(distance_ratios) / np.linalg.norm(distance_ratios)

#     # Initialize EMA for residuals
#     ema_residuals_mse = np.zeros(N)

#     # Function to apply Huber loss on residuals
#     def huber(residual, delta):
#         return np.where(np.abs(residual) <= delta,
#                         0.5 * residual**2,
#                         delta * (np.abs(residual) - 0.5 * delta))

#     v_old = np.copy(v)

#     # Variables to track the minimum loss and corresponding variables
#     min_loss = np.inf
#     v_best = None
#     r_hat_best = None
#     weights_best = None

#     # Define target basis (standard basis vectors)
#     target_basis = np.eye(3)

#     # Parameters for weight updates
#     min_weight = 0.1  # Minimum allowable weight
#     alpha = 0.1       # Smoothing factor for EMA
#     lambda_reg_weight = 0.01  # Regularization strength for weights

#     # Alternating optimization
#     for iter in range(max_iter):
#         # Step 1: Optimize v(t) for each time step t, given fixed r_hat and weights
#         for t in range(T):
#             # Matrix A and vector b for weighted least squares
#             A = r_hat * weights[:, np.newaxis]  # Shape: (N, 3)
#             b = v_r[t] * weights  # Shape: (N,)

#             # Regularization matrix for independence of components
#             L = np.diag([gamma / sigma_x2, gamma / sigma_y2, gamma / sigma_z2])

#             # Solve weighted regularized least squares: (A^T A + L) v(t) = A^T b
#             AtA = A.T @ A + L
#             Atb = A.T @ b
#             v[t] = np.linalg.solve(AtA, Atb)

#         # Enforce mean-zero constraint on v
#         v_mean = np.mean(v, axis=0)
#         v -= v_mean  # Subtract mean from each component

#         # Apply ICA to make velocity components independent
#         ica = FastICA(n_components=3, max_iter=200, tol=1e-4, random_state=iter)
#         v_ica = ica.fit_transform(v)

#         # Rescale the components to minimize loss
#         s = np.ones(3)
#         for i in range(3):
#             numerator = 0
#             denominator = 0
#             for t in range(T):
#                 predicted = v_ica[t, i] * r_hat[:, i]
#                 numerator += np.dot(weights, v_r[t] * predicted)
#                 denominator += np.dot(weights, predicted**2)
#             s[i] = numerator / (denominator + 1e-8)  # Add small epsilon to avoid division by zero

#         v_rescaled = v_ica * s  # Rescaled velocities

#         # Alignment step
#         if align_each_iteration:
#             # Select first three time steps to construct matrices for Procrustes analysis
#             num_samples = 3  # Number of samples to use (must be <= T)
#             v_samples = v_rescaled[:num_samples, :]  # Shape: (3, 3)
#             target_samples = target_basis[:num_samples, :]  # Shape: (3, 3)

#             # Compute rotation matrix using Procrustes analysis to align with target basis
#             R, _ = orthogonal_procrustes(v_samples, target_samples)

#             # Rotate velocities to align with target basis
#             v_aligned = v_rescaled @ R

#             # Rotate r_hat to maintain consistency
#             r_hat_aligned = r_hat @ R

#             # Normalize r_hat to be unit vectors
#             r_hat_aligned = r_hat_aligned / np.linalg.norm(r_hat_aligned, axis=1, keepdims=True)

#             # Update v and r_hat
#             v = v_aligned
#             r_hat = r_hat_aligned
#         else:
#             # If not aligning each iteration, keep v and r_hat as is
#             v = v_rescaled

#         # Temporal smoothing of velocities
#         if iter > 0:
#             v = (1 - temporal_smoothing) * v + temporal_smoothing * v_old

#         # Temporal smoothness regularization
#         for t in range(1, T - 1):
#             v[t] += smoothness_penalty * (v[t-1] - 2 * v[t] + v[t+1])

#         # Mutual exclusivity regularization
#         for t in range(T):
#             v[t, 0] -= exclusivity_penalty * (v[t, 1] * v[t, 2])
#             v[t, 1] -= exclusivity_penalty * (v[t, 0] * v[t, 2])
#             v[t, 2] -= exclusivity_penalty * (v[t, 0] * v[t, 1])

#         # Magnitude independence regularization
#         for t in range(T):
#             v[t, 0] -= magnitude_independence_penalty * (abs(v[t, 1]) * abs(v[t, 2]))
#             v[t, 1] -= magnitude_independence_penalty * (abs(v[t, 0]) * abs(v[t, 2]))
#             v[t, 2] -= magnitude_independence_penalty * (abs(v[t, 0]) * abs(v[t, 1]))

#         # Step 2: Optimize r_hat for each camera, given fixed v(t) and distance ratios
#         for i in range(N):
#             # Solve constrained optimization for each r_hat[i]
#             b = v_r[:, i]
#             A = v  # Shape: (T, 3)

#             # Solve least squares for r_hat[i]
#             r_hat_i, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

#             # Apply distance ratio constraint if provided
#             if distance_ratios is not None:
#                 target_magnitude = distance_ratios[i]
#                 current_magnitude = np.linalg.norm(r_hat_i)
#                 if current_magnitude > 0:
#                     r_hat_i = (target_magnitude / current_magnitude) * r_hat_i  # Adjust magnitude to match ratio

#             # Ensure the direction vector remains on the unit sphere
#             r_hat[i] = r_hat_i / np.linalg.norm(r_hat_i)

#         # Step 3: Update weights based on residuals and learning rate
#         residuals = np.array([v_r[t] - (v[t] @ r_hat.T) for t in range(T)])  # Shape: (T, N)

#         # Compute robust errors using Huber loss for each camera
#         robust_errors = np.mean(huber(residuals, huber_threshold), axis=0)

#         # Update EMA of robust errors
#         ema_residuals_mse = alpha * robust_errors + (1 - alpha) * ema_residuals_mse

#         # Normalize EMA of robust errors to [0, 1]
#         robust_errors_norm = (ema_residuals_mse - np.min(ema_residuals_mse)) / (np.max(ema_residuals_mse) - np.min(ema_residuals_mse) + 1e-8)

#         # Update weights inversely proportional to robust errors
#         weights_update = 1 - robust_errors_norm

#         # Apply learning rate
#         weights = (1 - weight_learning_rate) * weights + weight_learning_rate * weights_update

#         # Apply weight regularization
#         mean_weight = np.mean(weights)
#         weights -= lambda_reg_weight * (weights - mean_weight)

#         # Ensure weights are between min_weight and 1
#         weights = np.clip(weights, min_weight, 1)

#         # Step 4: Compute fit loss
#         total_residuals = residuals.flatten()
#         if huber_threshold is not None:
#             total_loss = np.sum(huber(total_residuals, huber_threshold))
#         else:
#             total_loss = np.sum(total_residuals**2)

#         # Every 100 iterations, print the fit loss
#         if (iter + 1) % 100 == 0 or iter == max_iter - 1:
#             print(f"Iteration {iter+1}, Fit Loss: {total_loss}")

#         # If current loss is less than minimum loss, update minimum loss and store variables
#         if total_loss < min_loss:
#             min_loss = total_loss
#             v_best = v.copy()
#             r_hat_best = r_hat.copy()
#             weights_best = weights.copy()

#         # Step 5: Check for convergence by comparing v with v_old
#         delta_v = np.linalg.norm(v - v_old) / (np.linalg.norm(v_old) + 1e-8)  # Added epsilon to prevent division by zero
#         if delta_v < tolerance:
#             print(f"Converged after {iter+1} iterations")
#             break
#         v_old = v.copy()

#     # Apply alignment at the end if not applied during iterations
#     if not align_each_iteration:
#         # Apply alignment using the best velocities
#         v_rescaled = v_best
#         # Select first three time steps to construct matrices for Procrustes analysis
#         num_samples = 3  # Number of samples to use (must be <= T)
#         v_samples = v_rescaled[:num_samples, :]  # Shape: (3, 3)
#         target_samples = target_basis[:num_samples, :]  # Shape: (3, 3)

#         # Compute rotation matrix using Procrustes analysis to align with target basis
#         R, _ = orthogonal_procrustes(v_samples, target_samples)

#         # Rotate velocities to align with target basis
#         v_aligned = v_rescaled @ R

#         # Rotate r_hat to maintain consistency
#         r_hat_aligned = r_hat_best @ R

#         # Normalize r_hat to be unit vectors
#         r_hat_aligned = r_hat_aligned / np.linalg.norm(r_hat_aligned, axis=1, keepdims=True)

#         # Update the best estimates
#         v_best = v_aligned
#         r_hat_best = r_hat_aligned

#     # Visualization of results using the best estimates
#     # Visualize the results for the top ten cameras with the highest weights

#     # Identify the indices of the top ten cameras by weight
#     top_cameras_indices = np.argsort(weights_best)[-10:]

#     # Extract data for the top cameras
#     top_cameras_weights = weights_best[top_cameras_indices]
#     top_cameras_directions = r_hat_best[top_cameras_indices]
#     top_cameras_v_r = v_r[:, top_cameras_indices]

#     # Plot the estimated velocities over time
#     plt.figure(figsize=(12, 6))
#     plt.plot(v_best[:, 0], label='v_x')
#     plt.plot(v_best[:, 1], label='v_y')
#     plt.plot(v_best[:, 2], label='v_z')
#     plt.title('Estimated 3D Velocities over Time')
#     plt.xlabel('Time Steps')
#     plt.ylabel('Velocity')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     # Plot the observed radial velocities for the top ten cameras
#     plt.figure(figsize=(12, 6))
#     for i in range(10):
#         idx = top_cameras_indices[i]
#         plt.plot(v_r[:, idx], label=f'Camera {idx} (Weight: {top_cameras_weights[i]:.2f})')
#     plt.title('Observed Radial Velocities for Top 10 Cameras')
#     plt.xlabel('Time Steps')
#     plt.ylabel('Radial Velocity')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     # Plot the direction vectors for the top ten cameras with colors based on weights and show camera indices
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # Normalize weights for color mapping
#     normalized_weights = (top_cameras_weights - top_cameras_weights.min()) / (top_cameras_weights.max() - top_cameras_weights.min() + 1e-8)
#     colors = cm.viridis(normalized_weights)

#     # Plot the direction vectors
#     origin = np.zeros((10, 3))
#     for i in range(10):
#         ax.quiver(origin[i, 0], origin[i, 1], origin[i, 2],
#                   top_cameras_directions[i, 0], top_cameras_directions[i, 1], top_cameras_directions[i, 2],
#                   length=1.0, normalize=True, color=colors[i], linewidth=2)
#         # Annotate with camera index
#         end_point = top_cameras_directions[i]  # Since origin is zero, end point is the direction vector
#         ax.text(end_point[0], end_point[1], end_point[2],
#                 f'{top_cameras_indices[i]}', color='black', fontsize=10)

#     # Set up the color bar
#     mappable = cm.ScalarMappable(cmap='viridis')
#     mappable.set_array(top_cameras_weights)
#     cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
#     cbar.set_label('Camera Weight', rotation=270, labelpad=15)

#     ax.set_title('Direction Vectors of Top 10 Cameras Colored by Weight and Labeled by Index')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.view_init(elev=20., azim=30)
#     plt.show()

#     return v_best, r_hat_best, weights_best

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D


def minimal_surface(x, y, z, resolution=100, show_plot=True):
    """
    Create and display a minimal surface bounded by the given x, y, z boundary points.

    Parameters:
    -----------
    x : list or np.array
        x-coordinates of boundary points
    y : list or np.array
        y-coordinates of boundary points
    z : list or np.array
        z-coordinates of boundary points
    resolution : int
        Resolution of the grid for surface interpolation
    show_plot : bool
        Whether to display the resulting 3D plot

    Returns:
    --------
    fig, ax : matplotlib Figure and Axes3D
        The figure and 3D axes containing the minimal surface plot.
    """
    # Ensure that x, y, z have the same length
    if not (len(x) == len(y) == len(z)):
        raise ValueError("x, y, z must have the same length.")

    # Create a grid in the XY plane
    xi = np.linspace(min(x), max(x), resolution)
    yi = np.linspace(min(y), max(y), resolution)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate Z values over the grid
    zi = griddata((x, y), z, (xi, yi), method='cubic')

    # Create a minimal surface approximation
    zi = np.nan_to_num(zi)  # Handle NaN values gracefully

    # Plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='k', alpha=0.7)

    # Plot the boundary points
    # ax.scatter(x, y, z, color='red', s=50, label='Boundary Points')

    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    # Labels and Legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    if show_plot:
        plt.show()

    return fig, ax

import matplotlib.pyplot as plt
import numpy as np

def plot_3d_scatter_colored_by_time(
    x, 
    y, 
    z, 
    time, 
    colormap='viridis', 
    marker_size=50, 
    alpha=0.8, 
    title='3D Scatter Colored by Time'
):
    """
    Plots a 3D scatter where each point's color depends on the given 'time' array.
    
    Parameters
    ----------
    x : array-like
        x-coordinates of the data points.
    y : array-like
        y-coordinates of the data points.
    z : array-like
        z-coordinates of the data points.
    time : array-like
        Values used to color each point (e.g., sample indices or time steps).
    colormap : str, optional
        Name of a matplotlib colormap (default: 'viridis').
    marker_size : float, optional
        Size of the scatter markers (default: 50).
    alpha : float, optional
        Transparency of the markers (default: 0.8).
    title : str, optional
        Title of the plot (default: '3D Scatter Colored by Time').
    """
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a scatter plot, coloring by "time"
    sc = ax.scatter(
        x, 
        y, 
        z, 
        c=time,            # map color to 'time'
        cmap=colormap,     # set the colormap
        s=marker_size, 
        alpha=alpha
    )
    
    min_val = np.min([np.min(x), np.min(y), np.min(z)])
    max_val = np.max([np.max(x), np.max(y), np.max(z)])
    
    # -------------------------------------------------------------------------
    # 2) Set the axis limits to the global min/max so all axes have same range
    # -------------------------------------------------------------------------
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_zlim(min_val, max_val)


    # Add color bar to explain the color-to-time mapping
    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Time / Sample Index')
    
    # Label axes and add a title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    plt.show()

import numpy as np
from scipy.optimize import minimize

def solve_constrained_least_squares(A, b, r0=None):
    """
    Solve min ||A r - b||^2 subject to:
        1) r_x, r_y, r_z >= 0
        2) angle w.r.t. x-axis in [0, pi/4], i.e. r_x^2 >= r_y^2 + r_z^2
        3) angle w.r.t. y-axis in [pi/4, pi/2], i.e. r_y^2 <= r_x^2 + r_z^2

    Parameters
    ----------
    A : np.ndarray, shape (T, 3)
        Design matrix (or data matrix).
    b : np.ndarray, shape (T,)
        Target vector/values.
    r0 : array-like of length 3, optional
        Initial guess for (r_x, r_y, r_z). If None, defaults to [1,1,1].

    Returns
    -------
    r_opt : np.ndarray, shape (3,)
        The best-fit vector (r_x, r_y, r_z) satisfying the constraints.
    res : OptimizeResult
        The full result object from `scipy.optimize.minimize`.

    Notes
    -----
    - This uses the SLSQP (Sequential Least SQuares Programming) method,
      which can handle bound constraints (r_x, r_y, r_z >= 0) and nonlinear
      inequality constraints.
    - The constraints are formulated as:
        C1(r) = r_x^2 - r_y^2 - r_z^2 >= 0
        C2(r) = r_x^2 + r_z^2 - r_y^2 >= 0
      plus the bound constraints r_x >= 0, r_y >= 0, r_z >= 0.
    - Make sure your problem is scaled appropriately, as nonlinear solvers
      can be sensitive to the scale of variables and data.
    """

    # --------------------------
    # Objective function
    # --------------------------
    def objective(r, A, b):
        """
        r is a 3-element vector [r_x, r_y, r_z].
        Returns the sum of squares of residuals: ||A r - b||^2
        """
        return np.sum((A @ r - b)**2)

    # --------------------------
    # Nonlinear constraints
    # --------------------------
    def angle_constraint_x(r):
        """
        r_x^2 >= r_y^2 + r_z^2
        => r_x^2 - (r_y^2 + r_z^2) >= 0
        """
        r_x, r_y, r_z = r
        return r_x**2 - (r_y**2 + r_z**2)

    def angle_constraint_y(r):
        """
        r_y^2 <= r_x^2 + r_z^2
        => (r_x^2 + r_z^2) - r_y^2 >= 0
        """
        r_x, r_y, r_z = r
        return (r_x**2 + r_z**2) - r_y**2

    # If no initial guess provided, pick something
    if r0 is None:
        r0 = np.array([np.random.rand(), np.random.rand(), np.random.rand()])

    # Bounds for each component: r_x, r_y, r_z >= 0
    bounds = [(-10, 10), (-10, 10), (-10, 10)]

    # Build a list of 'ineq' constraints for SLSQP
    constraints = [
        {'type': 'ineq', 'fun': angle_constraint_x},  # r_x^2 - r_y^2 - r_z^2 >= 0
        {'type': 'ineq', 'fun': angle_constraint_y},  # r_x^2 + r_z^2 - r_y^2 >= 0
    ]

    # Call the solver
    res = minimize(
        fun=objective,
        x0=r0,
        args=(A, b),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}  # example solver settings
    )

    # Extract the solution
    r_opt = res.x  # This should satisfy the constraints if the solver converged

    return r_opt, res

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, Bounds

# def solve_least_squares_with_angle_constraints(A, b, x0=None, method='SLSQP'):
#     """
#     Solve the least-squares problem:  minimize || A * r - b ||^2,
#     where r = [x, y, z], subject to:
#         1) x >= 0, y >= 0, z >= 0
#         2) angle_x = arccos(x / ||r||) in [0, pi/4]
#         3) angle_y = arccos(y / ||r||) in [3pi/8, pi/2]
    
#     Parameters
#     ----------
#     A : ndarray of shape (T, 3)
#         The matrix in the least-squares problem.
#     b : ndarray of shape (T,)
#         The target vector in the least-squares problem.
#     x0 : ndarray of shape (3,), optional
#         Initial guess for (x, y, z). Must be non-negative or you may get an
#         infeasible start. Defaults to [1, 0.1, 0.1] if not provided.
#     method : str, optional
#         The optimization method. Common choices:
#           - 'trust-constr' (default)
#           - 'SLSQP'
    
#     Returns
#     -------
#     result : OptimizeResult
#         The result object returned by `scipy.optimize.minimize`. The
#         solution is in `result.x`.
#     """

#     # ---------------
#     # 1) Define the objective (least squares)
#     # ---------------
#     def objective(r):
#         # r is [x, y, z]
#         residual = A.dot(r) - b
#         return np.sum(residual**2)

#     # ---------------
#     # 2) Define the nonlinear constraints for angles
#     #    We want: angle_x in [0, pi/4] and angle_y in [3pi/8, pi/2]
#     # ---------------
#     def angle_constraints(r):
#         x, y, z = r
#         norm_r = np.sqrt(x*x + y*y + z*z) + 1e-15  # avoid division by zero
#         theta_x = np.arccos(x / norm_r)  # angle wrt x-axis
#         theta_y = np.arccos(y / norm_r)  # angle wrt y-axis
#         # Return [theta_x, theta_y]
#         return np.array([theta_x, theta_y])

#     # Bounds for these angles:
#     angle_lb = np.array([0.0,       2.0*np.pi/8.0])
#     angle_ub = np.array([np.pi/4.0, np.pi/2.0     ])

#     # Create the NonlinearConstraint
#     # angle_constraints(r) ∈ [angle_lb, angle_ub]
#     nlc = NonlinearConstraint(angle_constraints, angle_lb, angle_ub)

#     # ---------------
#     # 3) Positivity bounds: x >= 0, y >= 0, z >= 0
#     # ---------------
#     positivity_bounds = Bounds(lb=[0.0, 0.0, 0.0],
#                                ub=[np.inf, np.inf, np.inf])

#     # ---------------
#     # 4) Initial guess
#     # ---------------
#     if x0 is None:
#         x0 = np.array([1.0, 0.1, 0.1])
#     # Make sure x0 is feasible (non-negative, angles in the desired range)
#     # If not, the solver may fail or return an infeasible result.

#     # ---------------
#     # 5) Call the optimizer
#     # ---------------
#     result = minimize(
#         fun=objective,
#         x0=x0,
#         method=method,
#         bounds=positivity_bounds,
#         constraints=[nlc],
#         options={'verbose': 2},  # Adjust verbosity as needed
#     )

#     return result


# # Example usage (uncomment to run):
# # if __name__ == "__main__":
# #     # Suppose you have A of shape (T, 3) and b of shape (T,)
# #     T = 5
# #     A = np.random.rand(T, 3)
# #     b = np.random.rand(T)
# #
# #     # Solve:
# #     res = solve_least_squares_with_angle_constraints(A, b, x0=np.array([1.0, 0.2, 0.5]))
# #     print("Optimal r:", res.x)
# #     print("Objective value:", res.fun)
# #     print("Success:", res.success)
# #     print("Message:", res.message)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def random_points_on_sphere_first_octant(num_points, radius=1.0, seed=None):
    """
    Generate 'num_points' random points uniformly on the sphere of given radius,
    restricted to x>0, y>0, z>0 (the first octant).
    """
    if seed is not None:
        np.random.seed(seed)
        
    points = []
    for _ in range(num_points):
        # Method: Generate a random direction, then filter out
        # until x>0, y>0, z>0
        while True:
            # Sample a random 3D point from Normal(0,1)
            v = np.random.randn(3)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-12:
                continue
            v = v / norm_v  # normalize to unit length
            # if v[0] > 0 and v[1] > 0 and v[2] > 0:
            if True:
                # Scale to the desired radius, add to list, break
                points.append(radius * v)
                break

    return np.array(points)


def repulsion_relaxation(points, iterations=100, step_size=0.05):
    """
    Simple iterative "push-apart":
      - For every pair of points, compute a repulsion vector if they are too close.
      - Accumulate repulsion updates and apply them each iteration.
      - Renormalize each point onto the sphere to ensure they stay on surface.
    
    Parameters
    ----------
    points : (N,3) array, each row is [x,y,z] on the sphere.
    iterations : int, how many times to iterate.
    step_size : float, how strong the push is each iteration.
    
    Returns
    -------
    points : (N,3) array, the final positions on the sphere.
    """
    N = len(points)
    for _ in range(iterations):
        # We'll accumulate displacement vectors in disp
        disp = np.zeros_like(points)
        
        # Repulsion among all pairs i<j
        for i in range(N):
            for j in range(i+1, N):
                # Vector from points[i] to points[j]
                diff = points[j] - points[i]
                dist = np.linalg.norm(diff)
                if dist < 1e-12:
                    continue
                
                # Repulsion with 1/dist^2 or simpler 1/dist
                # We'll do a mild 1/dist repulsion
                push_strength = 1.0 / dist
                
                # The direction is diff / dist
                direction = diff / dist
                
                # Each point gets pushed away from the other
                disp[i] -= push_strength * direction
                disp[j] += push_strength * direction
        
        # Apply the displacements with a small step_size
        points += step_size * disp
        
        # Renormalize each point onto the sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms  # preserve radius=1
        
        # Keep only the portion in the first octant
        # If a point drifts out, reflect or clamp it back
        # (Alternatively, just zero it out, but let's reflect to keep variety)
        # points = np.abs(points)  # move to x>=0, y>=0, z>=0
        # Sometimes all-positive might fail if the repulsion pushes a coordinate
        # slightly negative. "abs" ensures x>0,y>0,z>0, but can re-introduce small symmetry.
        # A more advanced approach might discard or re-sample out-of-octant points.

    return points


def generate_nonsymmetric_sphere_grid_2d(num_points=50, sphere_radius=1.0, seed=None):
    """
    Generate a 2D "grid" of points on the quarter of a sphere (x>0, y>0, z>0).
    1) Randomly sample points on the sphere cap (first octant)
    2) Apply a "repulsion" phase to spread them out
    """
    # 1) Randomly generate points
    pts = random_points_on_sphere_first_octant(num_points, radius=sphere_radius, seed=seed)
    
    # 2) Repulsion-based relaxation
    pts_final = repulsion_relaxation(pts, iterations=10000, step_size=0.02)
    
    return pts_final

import matplotlib.pyplot as plt

def compute_observed_velocities_for_groups_scalar(
    velocity_3xN: np.ndarray,
    camera_groups: list,
    visualize: bool = False,
    fig_adr = None,
) -> np.ndarray:
    """
    Computes the 'average observed velocity' (scalar projection) for multiple camera groups.
    
    Parameters:
    -----------
    velocity_3xN : np.ndarray
        A 3 x N array where N is the number of time points; each column is a 3D velocity vector.
    
    camera_groups : list of dict
        A list of group definitions. Each element of this list is a dict that must contain:
          - "locations": An M_g x 3 array representing M_g camera direction vectors.
          - "weights": A 1D array of length M_g representing the weights for these cameras.
    
    visualize : bool, optional
        If True, plots the observed scalar velocities for each group over time.
    
    Returns:
    --------
    group_observed_velocities : np.ndarray
        A G x N array where G is the number of camera groups. Each element [g, t] represents the scalar
        velocity observed by group g at time point t.
    """
    # --------------------------
    # 1) Validate inputs
    # --------------------------
    
    # Check velocity_3xN dimensions
    if not isinstance(velocity_3xN, np.ndarray):
        raise TypeError("velocity_3xN must be a NumPy array.")
    if velocity_3xN.ndim != 2 or velocity_3xN.shape[0] != 3:
        raise ValueError("velocity_3xN must be a 2D NumPy array with shape (3, N).")
    
    N = velocity_3xN.shape[1]
    G = len(camera_groups)
    
    # Initialize the output array
    group_observed_velocities = np.zeros((G, N), dtype=np.float64)
    
    # Iterate over each camera group
    for g_idx, group in enumerate(camera_groups):
        # Extract camera locations and weights
        try:
            camera_locations = group["locations"]  # Shape: (M_g, 3)
            camera_weights = group["weights"]      # Shape: (M_g,)
        except KeyError as e:
            raise KeyError(f"Group {g_idx}: Missing key {e} in camera group definition.")
        
        # Validate camera_locations and camera_weights
        if not isinstance(camera_locations, np.ndarray):
            raise TypeError(f"Group {g_idx}: 'locations' must be a NumPy array.")
        if camera_locations.ndim != 2 or camera_locations.shape[1] != 3:
            raise ValueError(f"Group {g_idx}: 'locations' must have shape (M_g, 3).")
        
        M_g = camera_locations.shape[0]
        
        if not isinstance(camera_weights, np.ndarray):
            raise TypeError(f"Group {g_idx}: 'weights' must be a NumPy array.")
        if camera_weights.ndim != 1 or camera_weights.shape[0] != M_g:
            print(camera_weights.shape)
            print(camera_weights.ndim)
            raise ValueError(f"Group {g_idx}: 'weights' must be a 1D array of length M_g.")
        
        # --------------------------
        # 2) Compute the weighted average direction
        # --------------------------
        
        # Compute weighted sum of camera directions: shape (3,)
        weighted_sum = np.dot(camera_weights, camera_locations)  # (3,)
        
        # Compute norm of the weighted sum
        norm_weighted_sum = np.linalg.norm(weighted_sum)
        
        if norm_weighted_sum < 1e-12:
            # Handle near-zero norm: return zero observed velocities for this group
            print(f"Warning: Weighted average direction for group {g_idx} is near zero. Returning zeros for this group.")
            observed_velocity = np.zeros(N, dtype=np.float64)
        else:
            # Normalize to get average direction
            avg_direction = weighted_sum / norm_weighted_sum  # (3,)
            
            # --------------------------
            # 3) Compute scalar projections for all time points
            # --------------------------
            
            # Vectorized dot product: (3, N) dot (3,) -> (N,)
            alpha = np.dot(avg_direction, velocity_3xN)  # (N,)
            
            # Store the scalar observed velocities
            observed_velocity = alpha  # (N,)
        
        # Assign to the output array
        group_observed_velocities[g_idx, :] = observed_velocity
    
    # --------------------------
    # 4) Visualization
    # --------------------------
    
    if visualize or fig_adr:
        time_points = np.arange(1, N + 1)
        fig = plt.figure(figsize=(12 * 4, 6 * G))
        
        for g_idx in range(G):
            plt.subplot(G//3, 3, g_idx + 1)
            plt.plot(time_points, group_observed_velocities[g_idx, :])
            plt.title(f'Group {g_idx + 1} Observed Scalar Velocity Over Time')
            plt.xlabel('Time Point')
            plt.ylabel('Observed Velocity (Scalar)')
            plt.grid(True)
        
        plt.tight_layout()
        if fig_adr:
            plt.savefig(fig_adr, dpi=300)
        if visualize:
            plt.show()
        else:
            plt.close(fig)
    
    return group_observed_velocities

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_3d_scatter_with_time(x, y, z, time):
    """
    Plots a 3D scatter plot of points with x, y, z coordinates and colors them based on time.
    
    Parameters:
        x (list or np.array): X-coordinates of points.
        y (list or np.array): Y-coordinates of points.
        z (list or np.array): Z-coordinates of points.
        time (list or np.array): Time values corresponding to each point (used for coloring).
    """
    # Ensure all inputs are numpy arrays for consistency
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    time = np.array(time)
    
    # Check that all inputs have the same length
    if not (len(x) == len(y) == len(z) == len(time)):
        raise ValueError("All input arrays must have the same length.")
    
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot with colormap
    scatter = ax.scatter(x, y, z, c=time, cmap='viridis', marker='o')
    
    # Add a color bar to represent time
    colorbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    colorbar.set_label('Time')
    
    # Label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scatter Plot with Time-based Coloring')
    
    # Show the plot
    plt.show()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

def animate_3d_scatter_with_time(x, y, z, time, interval=100, save_as_gif=False, gif_filename='animation.gif'):
    """
    Creates an animated 3D scatter plot where points are added sequentially based on time.
    
    Parameters:
        x (list or np.array): X-coordinates of points.
        y (list or np.array): Y-coordinates of points.
        z (list or np.array): Z-coordinates of points.
        time (list or np.array): Time values corresponding to each point (used for animation order).
        interval (int): Time interval between frames in milliseconds.
        save_as_gif (bool): Whether to save the animation as a GIF file.
        gif_filename (str): Filename for the saved GIF (if enabled).
    """
    # Ensure all inputs are numpy arrays
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    time = np.array(time)
    
    # Check that all inputs have the same length
    if not (len(x) == len(y) == len(z) == len(time)):
        raise ValueError("All input arrays must have the same length.")
    
    # # Sort data based on time
    # sorted_indices = np.argsort(time)
    # x, y, z, time = x[sorted_indices], y[sorted_indices], z[sorted_indices], time[sorted_indices]
    
    # Create the figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize the scatter plot
    scatter = ax.scatter([], [], [], c=[], cmap='viridis', marker='o')
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Animated 3D Scatter Plot with Time-based Coloring')
    
    # Set axis limits
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.set_zlim(min(z), max(z))
    
    # Add a color bar
    colorbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    colorbar.set_label('Time')
    
    # Animation initialization function
    def init():
        scatter._offsets3d = ([], [], [])
        scatter.set_array(np.array([]))
        return scatter,
    
    # Animation update function
    def update(frame):
        scatter._offsets3d = (x[:frame+1], y[:frame+1], z[:frame+1])
        scatter.set_array(time[:frame+1])
        return scatter,
    
    # Create the animation
    ani = FuncAnimation(
        fig, update, frames=len(time), init_func=init, blit=False, interval=interval
    )
    
    if save_as_gif:
        ani.save(gif_filename, writer=PillowWriter(fps=10))
        print(f"Animation saved as {gif_filename}")
    else:
        plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import interp1d

def plot_3d_scatter_with_time_interpolated(x, y, z, time, upsample_factor=10):
    """
    Plots a 3D scatter plot of points with x, y, z coordinates and colors them based on time,
    with interpolation and upsampling for smoother visualization. Ensures equal axis scaling.
    
    Parameters:
        x (list or np.array): X-coordinates of points.
        y (list or np.array): Y-coordinates of points.
        z (list or np.array): Z-coordinates of points.
        time (list or np.array): Time values corresponding to each point (used for coloring).
        upsample_factor (int): Factor by which to increase the number of points via interpolation.
    """
    # Ensure all inputs are numpy arrays for consistency
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    time = np.array(time)
    
    # Check that all inputs have the same length
    if not (len(x) == len(y) == len(z) == len(time)):
        raise ValueError("All input arrays must have the same length.")
    
    # Remove duplicate time points and sort by time
    unique_time, unique_indices = np.unique(time, return_index=True)
    x = x[unique_indices]
    y = y[unique_indices]
    z = z[unique_indices]
    time = unique_time
    
    # Create a uniform time array for interpolation
    time_dense = np.linspace(time.min(), time.max(), len(time) * upsample_factor)
    
    # Perform interpolation for x, y, z
    interp_x = interp1d(time, x, kind='cubic')
    interp_y = interp1d(time, y, kind='cubic')
    interp_z = interp1d(time, z, kind='cubic')
    
    x_dense = interp_x(time_dense)
    y_dense = interp_y(time_dense)
    z_dense = interp_z(time_dense)
    
    # Determine global limits for equal scaling
    min_limit = min(np.min(x_dense), np.min(y_dense), np.min(z_dense))
    max_limit = max(np.max(x_dense), np.max(y_dense), np.max(z_dense))
    
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot with colormap
    scatter = ax.scatter(x_dense, y_dense, z_dense, c=time_dense, cmap='viridis', marker='o', s=5)
    
    # Add a color bar to represent time
    colorbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    colorbar.set_label('Time')
    
    # Label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scatter Plot with Time-based Coloring (Interpolated)')
    
    # Set equal axis limits
    ax.set_xlim([min_limit, max_limit])
    ax.set_ylim([min_limit, max_limit])
    ax.set_zlim([min_limit, max_limit])
    
    # Ensure aspect ratio is equal
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for x, y, z axes
    
    # Show the plot
    plt.show()

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------------------------------------------
# Inner Solver: Estimate v(t) & r_hat for a given subset of cameras
# -------------------------------------------------------------------
def estimate_velocity_and_directions_for_subset(
    v_r,
    subset_mask,
    lambda_reg=0.1,
    gamma=0.01,
    sigma_x2=1.0,
    sigma_y2=1.0,
    sigma_z2=1.0,
    huber_threshold=1.0,
    max_iter=5,
    random_seed=42
):
    """
    Given:
      v_r: shape (T, N), radial velocities
      subset_mask: shape (N,), boolean array indicating which cameras are used
    Solve for:
      - r_hat[i] for i in subset
      - v(t) for each time step t
    Return the solution (v, r_hat_full, loss)

    We do a small iterative approach:
      1) Initialize r_hat for cameras in subset randomly.
      2) Repeat for `max_iter`:
         a) Solve v(t) in closed form (weighted LS) for t=1..T
         b) Solve r_hat[i] in closed form for i in subset
      3) Compute final residual and Huber/MSE loss.

    Cameras not in the subset get r_hat=0. We only compute loss over the subset.
    """
    np.random.seed(random_seed)

    T, N = v_r.shape
    # Indices of cameras in subset
    subset_indices = np.where(subset_mask)[0]
    num_subset = len(subset_indices)

    if num_subset == 0:
        # No cameras selected -> degenerate
        # Return zero velocity & zero directions, large loss to discourage empty subsets
        v_est = np.zeros((T, 3))
        r_hat_full = np.zeros((N, 3))
        return v_est, r_hat_full, 1e10

    # Initialize directions randomly for the subset
    r_hat_sub = np.random.randn(num_subset, 3)
    for i in range(num_subset):
        norm_i = np.linalg.norm(r_hat_sub[i])
        r_hat_sub[i] = r_hat_sub[i]/(norm_i+1e-8)

    # We'll store the final direction estimate in a full-size array, with zeros for out-of-subset
    r_hat_full = np.zeros((N, 3))

    # If you want to include camera-based weighting:
    weights = np.ones(num_subset)

    # Helper: Huber loss
    def huber_loss(res, delta):
        abs_res = np.abs(res)
        in_quadratic = abs_res <= delta
        return np.where(
            in_quadratic,
            0.5 * res**2,
            delta*(abs_res - 0.5*delta)
        )

    # Regularization diag
    L = np.diag([gamma/sigma_x2, gamma/sigma_y2, gamma/sigma_z2])

    # Iterative refinement
    v_est = np.zeros((T,3))

    for _ in range(max_iter):
        # (a) Solve v(t) for each time step with current r_hat_sub
        for t_idx in range(T):
            # A shape: (num_subset, 3)
            # b shape: (num_subset,)
            A = r_hat_sub * weights[:,None]
            b = v_r[t_idx, subset_indices] * weights

            AtA = A.T @ A + L
            Atb = A.T @ b
            # solve
            try:
                v_est[t_idx] = np.linalg.solve(AtA, Atb)
            except np.linalg.LinAlgError:
                v_est[t_idx], _, _, _ = np.linalg.lstsq(AtA, Atb, rcond=None)

        # (b) Solve r_hat[i] for each camera in subset
        for sidx, cam_idx in enumerate(subset_indices):
            # We want to solve for r_hat[cam_idx]
            # Minimizing sum_t [ v_r(t, cam_idx) - v_est[t].dot(r_hat) ]^2
            A = v_est  # shape (T, 3)
            b = v_r[:, cam_idx]

            # Solve least squares
            try:
                rh, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                rh = r_hat_sub[sidx]  # fallback
            # Normalize
            norm_rh = np.linalg.norm(rh)
            if norm_rh < 1e-8:
                rh = np.array([1.0, 0.0, 0.0])
            else:
                rh /= norm_rh

            r_hat_sub[sidx] = rh

    # Store final subset directions in the full array
    for sidx, cam_idx in enumerate(subset_indices):
        r_hat_full[cam_idx] = r_hat_sub[sidx]

    # Compute final loss on just the subset
    # residual = v_r - (v_est @ r_hat_full.T)  # shape (T, N)
    # We'll zero out columns not in the subset so they don't contribute to loss
    residual_sub = v_r[:, subset_indices] - (v_est @ r_hat_sub.T)  # shape (T, num_subset)

    if huber_threshold is not None:
        loss = np.sum(huber_loss(residual_sub, huber_threshold))
    else:
        loss = np.sum(residual_sub**2)

    return v_est, r_hat_full, loss

# # -------------------------------------------------------------------
# # Outer GA: Evolve a subset of cameras to remove outliers
# # -------------------------------------------------------------------
# def genetic_algorithm_camera_subset(
#     v_r,
#     subset_fraction=0.5,
#     population_size=20,
#     generations=30,
#     crossover_rate=0.7,
#     mutation_rate=0.1,
#     elitism=1,
#     random_seed=42,
#     # Inner solver hyperparameters
#     lambda_reg=0.1,
#     gamma=0.01,
#     sigma_x2=1.0,
#     sigma_y2=1.0,
#     sigma_z2=1.0,
#     huber_threshold=1.0,
#     max_inner_iter=5,
#     visualization=False
# ):
#     """
#     An outer Genetic Algorithm to pick a subset of cameras (ratio = subset_fraction) that
#     yields the best velocity+direction fit. This helps exclude outlier cameras.

#     GA Chromosome: A boolean mask of shape (N,) with exactly round(subset_fraction*N) cameras = True.

#     For each chromosome (subset):
#       - We run `estimate_velocity_and_directions_for_subset` to get a fit loss.
#       - That loss is the GA fitness (lower is better).

#     Parameters
#     ----------
#     v_r : (T, N) array
#         Radial velocities over time T from N cameras.
#     subset_fraction : float
#         Fraction of cameras to keep in each subset (e.g. 0.5 means keep 50% of cameras).
#     population_size : int
#         Size of GA population.
#     generations : int
#         Number of GA generations.
#     crossover_rate : float
#         Probability of crossover in GA.
#     mutation_rate : float
#         Probability of mutation in GA.
#     elitism : int
#         Number of top individuals carried over directly each generation.
#     random_seed : int
#         Random seed for reproducibility.
#     visualization : bool
#         If True, plots the best loss over generations.

#     Returns
#     -------
#     best_v : (T, 3) array
#         Velocity for each time step from the best subset solution.
#     best_r_hat : (N, 3) array
#         Direction vectors for all cameras (0 for excluded cameras).
#     best_subset_mask : (N,) boolean
#         The best camera subset found.
#     best_loss : float
#         The final best loss.
#     loss_history : list of float
#         The best loss found at each generation.
#     """
#     np.random.seed(random_seed)
#     T, N = v_r.shape

#     # number of cameras to keep
#     subset_size = int(round(subset_fraction * N))
#     if subset_size < 1:
#         subset_size = 1
#     if subset_size > N:
#         subset_size = N

#     # -----------------------------------------
#     # GA Utility Functions
#     # -----------------------------------------
#     def create_random_mask():
#         """
#         Create a boolean mask of length N with exactly subset_size True.
#         """
#         mask = np.zeros(N, dtype=bool)
#         true_indices = np.random.choice(N, size=subset_size, replace=False)
#         mask[true_indices] = True
#         return mask

#     def initialize_population(pop_size):
#         """Initialize pop_size random subsets."""
#         return np.array([create_random_mask() for _ in range(pop_size)])

#     def evaluate_population(pop):
#         """Compute the loss for each subset in pop."""
#         losses = np.zeros(len(pop))
#         for i, mask in enumerate(pop):
#             _, _, loss_i = estimate_velocity_and_directions_for_subset(
#                 v_r,
#                 subset_mask=mask,
#                 lambda_reg=lambda_reg,
#                 gamma=gamma,
#                 sigma_x2=sigma_x2,
#                 sigma_y2=sigma_y2,
#                 sigma_z2=sigma_z2,
#                 huber_threshold=huber_threshold,
#                 max_iter=max_inner_iter,
#                 random_seed=random_seed + i  # Different seed for diversity
#             )
#             losses[i] = loss_i
#         return losses

#     def tournament_selection(pop, losses, k=3):
#         """
#         Randomly pick k individuals, return the one with the lowest loss (best).
#         """
#         idx_choices = np.random.choice(len(pop), size=k, replace=False)
#         best_idx = idx_choices[np.argmin(losses[idx_choices])]
#         return pop[best_idx]

#     def single_point_crossover(parent1, parent2):
#         """
#         Single-point crossover on boolean arrays, preserving the *count* of True.
#         """
#         N = len(parent1)
#         cut = np.random.randint(1, N)
#         child1 = np.concatenate([parent1[:cut], parent2[cut:]]).copy()
#         child2 = np.concatenate([parent2[:cut], parent1[cut:]]).copy()

#         # fix cardinality
#         child1 = fix_cardinality(child1)
#         child2 = fix_cardinality(child2)
#         return child1, child2

#     def fix_cardinality(mask):
#         """
#         Ensure the mask has exactly subset_size True.
#         If too many True, switch some to False; if too few, switch some to True.
#         """
#         current_true = np.sum(mask)
#         if current_true > subset_size:
#             # randomly pick which True to set to False
#             to_flip = current_true - subset_size
#             true_indices = np.where(mask)[0]
#             flip_indices = np.random.choice(true_indices, size=to_flip, replace=False)
#             mask[flip_indices] = False
#         elif current_true < subset_size:
#             # randomly pick which False to set to True
#             to_flip = subset_size - current_true
#             false_indices = np.where(~mask)[0]
#             flip_indices = np.random.choice(false_indices, size=to_flip, replace=False)
#             mask[flip_indices] = True
#         return mask

#     def mutate_mask(mask):
#         """
#         With probability mutation_rate, we flip bits.
#         We keep the total number of True = subset_size.
#         We'll do exactly 1 flip from True->False and 1 flip from False->True if we mutate.
#         """
#         if np.random.rand() < mutation_rate:
#             # pick one True to flip off, pick one False to flip on
#             true_indices = np.where(mask)[0]
#             false_indices = np.where(~mask)[0]
#             if len(true_indices) > 0 and len(false_indices) > 0:
#                 idx_off = np.random.choice(true_indices)
#                 idx_on  = np.random.choice(false_indices)
#                 mask[idx_off] = False
#                 mask[idx_on]  = True
#         return mask

#     # -----------------------------------------
#     # GA Main Loop
#     # -----------------------------------------
#     population = initialize_population(population_size)
#     losses = evaluate_population(population)
#     best_idx = np.argmin(losses)
#     best_loss = losses[best_idx]
#     best_subset_mask = population[best_idx].copy()

#     # Recompute the best v, r_hat
#     best_v, best_r_hat, _ = estimate_velocity_and_directions_for_subset(
#         v_r,
#         best_subset_mask,
#         lambda_reg=lambda_reg,
#         gamma=gamma,
#         sigma_x2=sigma_x2,
#         sigma_y2=sigma_y2,
#         sigma_z2=sigma_z2,
#         huber_threshold=huber_threshold,
#         max_iter=max_inner_iter,
#         random_seed=random_seed
#     )

#     loss_history = [best_loss]

#     for gen in tqdm(range(generations), desc="Subset-GA"):
#         new_population = []
#         # Sort by ascending loss
#         sorted_idx = np.argsort(losses)

#         # Elitism: keep top 'elitism' individuals
#         for e in range(elitism):
#             idx_e = sorted_idx[e]
#             new_population.append(population[idx_e].copy())

#         # Fill the rest with crossover + mutation
#         while len(new_population) < population_size:
#             # select parents
#             p1 = tournament_selection(population, losses, k=3).copy()
#             p2 = tournament_selection(population, losses, k=3).copy()

#             if np.random.rand() < crossover_rate:
#                 c1, c2 = single_point_crossover(p1, p2)
#             else:
#                 c1, c2 = p1.copy(), p2.copy()

#             c1 = mutate_mask(c1)
#             c2 = mutate_mask(c2)

#             new_population.append(c1)
#             if len(new_population) < population_size:
#                 new_population.append(c2)

#         population = np.array(new_population)
#         # Evaluate
#         losses = evaluate_population(population)

#         # Track best
#         gen_best_idx = np.argmin(losses)
#         gen_best_loss = losses[gen_best_idx]
#         if gen_best_loss < best_loss:
#             best_loss = gen_best_loss
#             best_subset_mask = population[gen_best_idx].copy()
#             best_v, best_r_hat, _ = estimate_velocity_and_directions_for_subset(
#                 v_r,
#                 best_subset_mask,
#                 lambda_reg=lambda_reg,
#                 gamma=gamma,
#                 sigma_x2=sigma_x2,
#                 sigma_y2=sigma_y2,
#                 sigma_z2=sigma_z2,
#                 huber_threshold=huber_threshold,
#                 max_iter=max_inner_iter,
#                 random_seed=random_seed + gen  # Diverse seeds
#             )

#         loss_history.append(best_loss)

#     if visualization:
#         # 1) Plot loss over generations
#         plt.figure(figsize=(8, 5))
#         plt.plot(loss_history, 'b-o')
#         plt.title("GA Convergence (Best Loss by Generation)")
#         plt.xlabel("Generation")
#         plt.ylabel("Loss")
#         plt.grid(True)
#         plt.show()

#         # 2) Plot the final velocity estimate vs time
#         plt.figure(figsize=(10, 5))
#         plt.plot(best_v[:, 0], label="v_x")
#         plt.plot(best_v[:, 1], label="v_y")
#         plt.plot(best_v[:, 2], label="v_z")
#         plt.title("Estimated Velocity Components over Time (GA Best)")
#         plt.xlabel("Time Step")
#         plt.ylabel("Velocity")
#         plt.legend()
#         plt.grid(True)
#         plt.show()

#         # 3) Show distribution of final camera directions
#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(111, projection='3d')
#         ax.set_title("Camera Directions (Best Subset)")

#         # Identify the indices of the selected cameras
#         selected_indices = np.where(best_subset_mask)[0]
#         excluded_indices = np.where(~best_subset_mask)[0]

#         # Plot the direction vectors for selected cameras
#         for i in selected_indices:
#             ax.quiver(
#                 0, 0, 0,
#                 best_r_hat[i, 0], best_r_hat[i, 1], best_r_hat[i, 2],
#                 length=1.0, normalize=True, color='blue', linewidth=1.5, arrow_length_ratio=0.1
#             )
#             ax.text(
#                 best_r_hat[i, 0], best_r_hat[i, 1], best_r_hat[i, 2],
#                 f"Cam {i}", color='blue', fontsize=9
#             )

#         # Optionally, plot excluded cameras in a different color
#         for i in excluded_indices:
#             ax.quiver(
#                 0, 0, 0,
#                 best_r_hat[i, 0], best_r_hat[i, 1], best_r_hat[i, 2],
#                 length=1.0, normalize=True, color='red', linewidth=1.0, arrow_length_ratio=0.1
#             )
#             ax.text(
#                 best_r_hat[i, 0], best_r_hat[i, 1], best_r_hat[i, 2],
#                 f"Cam {i}", color='red', fontsize=9
#             )

#         # Set axes limits
#         ax.set_xlim([-1.2, 1.2])
#         ax.set_ylim([-1.2, 1.2])
#         ax.set_zlim([-1.2, 1.2])
#         ax.set_xlabel("X")
#         ax.set_ylabel("Y")
#         ax.set_zlabel("Z")
#         ax.view_init(elev=20., azim=30)
#         plt.tight_layout()
#         plt.show()

#         # 4) Plot the observed radial velocities for the top ten cameras by selection
#         if len(selected_indices) >= 10:
#             top_cameras_indices = selected_indices[:10]
#         else:
#             top_cameras_indices = selected_indices

#         plt.figure(figsize=(12, 6))
#         for i in top_cameras_indices:
#             plt.plot(v_r[:, i], label=f'Camera {i}')
#         plt.title('Observed Radial Velocities for Selected Cameras')
#         plt.xlabel('Time Steps')
#         plt.ylabel('Radial Velocity')
#         plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05))
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()

#     return best_v, best_r_hat, best_subset_mask, best_loss, loss_history

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def estimate_velocity_and_directions_for_subset(
    v_r,
    subset_mask,
    lambda_reg=0.1,
    gamma=0.01,
    sigma_x2=1.0,
    sigma_y2=1.0,
    sigma_z2=1.0,
    huber_threshold=1.0,
    max_iter=5,
    random_seed=42
):
    """
    Given:
      v_r: shape (T, N), radial velocities
      subset_mask: shape (N,), boolean array indicating which cameras are used

    Solve for:
      - r_hat[i] for i in subset (now representing positions inside/on sphere)
      - v(t) for each time step t

    Returns:
      v_est        : shape (T,3), velocity at each time step
      r_hat_full   : shape (N,3), camera positions for all cameras
                     (clamped to radius <=1 if norm>1)
      loss         : scalar, final Huber or MSE loss on the subset

    We do a small iterative approach:
      1) Initialize r_hat for cameras in subset randomly (within radius 1).
      2) Repeat for `max_iter`:
         a) Solve v(t) in closed form (weighted LS) for t=1..T
         b) Solve r_hat[i] in closed form for i in subset (and clamp norm <=1)
      3) Compute final residual and Huber/MSE loss over selected cameras.
    """
    np.random.seed(random_seed)

    T, N = v_r.shape
    subset_indices = np.where(subset_mask)[0]
    num_subset = len(subset_indices)

    if num_subset == 0:
        # No cameras selected -> degenerate
        v_est = np.zeros((T, 3))
        r_hat_full = np.zeros((N, 3))
        return v_est, r_hat_full, 1e10  # Large loss to discourage empty subsets

    # ---------------------------------------------------------
    # 1) Initialize random camera positions (within radius <= 1)
    # ---------------------------------------------------------
    r_hat_sub = np.random.uniform(-1.0, 1.0, size=(num_subset, 3))
    for i in range(num_subset):
        norm_i = np.linalg.norm(r_hat_sub[i])
        if norm_i > 1.0:
            r_hat_sub[i] /= norm_i  # clamp to unit sphere boundary

    r_hat_full = np.zeros((N, 3))

    # We'll assume uniform weights for these cameras
    weights = np.ones(num_subset)

    # Huber loss helper
    def huber_loss(res, delta):
        abs_res = np.abs(res)
        in_quadratic = abs_res <= delta
        return np.where(
            in_quadratic,
            0.5 * res**2,
            delta * (abs_res - 0.5*delta)
        )

    # Regularization diag
    L = np.diag([gamma/sigma_x2, gamma/sigma_y2, gamma/sigma_z2])

    # v_est shape: (T,3)
    v_est = np.zeros((T,3))

    # ---------------------------------------------------------
    # 2) Iterative refinement
    # ---------------------------------------------------------
    for _ in range(max_iter):
        # (a) Solve v(t) with current r_hat_sub
        for t_idx in range(T):
            A = r_hat_sub * weights[:,None]  # shape (num_subset, 3)
            b = v_r[t_idx, subset_indices] * weights

            AtA = A.T @ A + L
            Atb = A.T @ b
            try:
                v_est[t_idx] = np.linalg.solve(AtA, Atb)
            except np.linalg.LinAlgError:
                # fallback if solve fails
                v_est[t_idx], _, _, _ = np.linalg.lstsq(AtA, Atb, rcond=None)

        # (b) Solve r_hat[i] for each camera in subset
        for sidx, cam_idx in enumerate(subset_indices):
            A = v_est  # shape (T, 3)
            b = v_r[:, cam_idx]

            try:
                rh, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                rh = r_hat_sub[sidx]  # fallback

            # Clamp camera position to radius <= 1
            norm_rh = np.linalg.norm(rh)
            if norm_rh > 1.0:
                rh /= norm_rh

            r_hat_sub[sidx] = rh

    # ---------------------------------------------------------
    # 3) Compute final loss on selected cameras
    # ---------------------------------------------------------
    for sidx, cam_idx in enumerate(subset_indices):
        r_hat_full[cam_idx] = r_hat_sub[sidx]

    residual_sub = v_r[:, subset_indices] - (v_est @ r_hat_sub.T)  # (T, num_subset)

    if huber_threshold is not None:
        loss = np.sum(huber_loss(residual_sub, huber_threshold))
    else:
        loss = np.sum(residual_sub**2)

    return v_est, r_hat_full, loss

def genetic_algorithm_camera_subset(
    v_r,
    subset_fraction=0.5,
    population_size=20,
    generations=30,
    crossover_rate=0.7,
    mutation_rate=0.1,
    elitism=1,
    random_seed=42,
    # Inner solver hyperparameters
    lambda_reg=0.1,
    gamma=0.01,
    sigma_x2=1.0,
    sigma_y2=1.0,
    sigma_z2=1.0,
    huber_threshold=1.0,
    max_inner_iter=5,
    visualization=False
):
    """
    Outer GA that evolves a subset of cameras to remove outliers (ratio = subset_fraction).
    Each chromosome is a boolean mask of shape (N,) with exactly round(subset_fraction*N) True.

    For each mask:
      - Run `estimate_velocity_and_directions_for_subset` to get v(t), r_hat, and loss.
      - That loss is the GA fitness (lower=better).

    Returns:
      best_v            : (T,3), best velocity estimate
      best_r_hat        : (N,3), best camera positions (within radius=1)
      best_subset_mask  : (N,), boolean subset
      best_loss         : float, best loss achieved
      loss_history      : list of float, best loss each generation
    """
    np.random.seed(random_seed)
    T, N = v_r.shape

    # number of cameras to keep
    subset_size = int(round(subset_fraction * N))
    if subset_size < 1: subset_size = 1
    if subset_size > N: subset_size = N

    # --- GA Utility Functions ---
    def create_random_mask():
        """Create a boolean mask of length N with exactly subset_size True."""
        mask = np.zeros(N, dtype=bool)
        true_indices = np.random.choice(N, size=subset_size, replace=False)
        mask[true_indices] = True
        return mask

    def initialize_population(pop_size):
        return np.array([create_random_mask() for _ in range(pop_size)])

    def evaluate_population(pop):
        """Compute the loss for each subset in pop."""
        losses = np.zeros(len(pop))
        for i, mask in enumerate(pop):
            _, _, loss_i = estimate_velocity_and_directions_for_subset(
                v_r,
                subset_mask=mask,
                lambda_reg=lambda_reg,
                gamma=gamma,
                sigma_x2=sigma_x2,
                sigma_y2=sigma_y2,
                sigma_z2=sigma_z2,
                huber_threshold=huber_threshold,
                max_iter=max_inner_iter,
                random_seed=random_seed + i
            )
            losses[i] = loss_i
        return losses

    def tournament_selection(pop, losses, k=3):
        """Randomly pick k individuals, return the one with the lowest loss (best)."""
        idx_choices = np.random.choice(len(pop), size=k, replace=False)
        best_idx = idx_choices[np.argmin(losses[idx_choices])]
        return pop[best_idx]

    def single_point_crossover(parent1, parent2):
        """Single-point crossover on boolean arrays, preserving count of True."""
        cut = np.random.randint(1, N)
        child1 = np.concatenate([parent1[:cut], parent2[cut:]]).copy()
        child2 = np.concatenate([parent2[:cut], parent1[cut:]]).copy()
        child1 = fix_cardinality(child1)
        child2 = fix_cardinality(child2)
        return child1, child2

    def fix_cardinality(mask):
        """Ensure the mask has exactly subset_size True."""
        current_true = np.sum(mask)
        if current_true > subset_size:
            to_flip = current_true - subset_size
            true_indices = np.where(mask)[0]
            flip_indices = np.random.choice(true_indices, size=to_flip, replace=False)
            mask[flip_indices] = False
        elif current_true < subset_size:
            to_flip = subset_size - current_true
            false_indices = np.where(~mask)[0]
            flip_indices = np.random.choice(false_indices, size=to_flip, replace=False)
            mask[flip_indices] = True
        return mask

    def mutate_mask(mask):
        """
        With probability mutation_rate, swap one True->False and one False->True,
        preserving the total count of True bits.
        """
        if np.random.rand() < mutation_rate:
            true_indices = np.where(mask)[0]
            false_indices = np.where(~mask)[0]
            if len(true_indices) > 0 and len(false_indices) > 0:
                idx_off = np.random.choice(true_indices)
                idx_on  = np.random.choice(false_indices)
                mask[idx_off] = False
                mask[idx_on]  = True
        return mask

    # --- GA Main Loop ---
    population = initialize_population(population_size)
    losses = evaluate_population(population)
    best_idx = np.argmin(losses)
    best_loss = losses[best_idx]
    best_subset_mask = population[best_idx].copy()

    # Recompute best solution
    best_v, best_r_hat, _ = estimate_velocity_and_directions_for_subset(
        v_r,
        best_subset_mask,
        lambda_reg=lambda_reg,
        gamma=gamma,
        sigma_x2=sigma_x2,
        sigma_y2=sigma_y2,
        sigma_z2=sigma_z2,
        huber_threshold=huber_threshold,
        max_iter=max_inner_iter,
        random_seed=random_seed
    )

    loss_history = [best_loss]

    for gen in tqdm(range(generations), desc="Subset-GA"):
        new_population = []
        sorted_idx = np.argsort(losses)

        # Elitism
        for e in range(elitism):
            idx_e = sorted_idx[e]
            new_population.append(population[idx_e].copy())

        # Crossover + Mutation
        while len(new_population) < population_size:
            p1 = tournament_selection(population, losses, k=3).copy()
            p2 = tournament_selection(population, losses, k=3).copy()

            if np.random.rand() < crossover_rate:
                c1, c2 = single_point_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = mutate_mask(c1)
            c2 = mutate_mask(c2)
            new_population.append(c1)
            if len(new_population) < population_size:
                new_population.append(c2)

        population = np.array(new_population)
        # Evaluate
        losses = evaluate_population(population)

        # Track best
        gen_best_idx = np.argmin(losses)
        gen_best_loss = losses[gen_best_idx]
        if gen_best_loss < best_loss:
            best_loss = gen_best_loss
            best_subset_mask = population[gen_best_idx].copy()
            best_v, best_r_hat, _ = estimate_velocity_and_directions_for_subset(
                v_r,
                best_subset_mask,
                lambda_reg=lambda_reg,
                gamma=gamma,
                sigma_x2=sigma_x2,
                sigma_y2=sigma_y2,
                sigma_z2=sigma_z2,
                huber_threshold=huber_threshold,
                max_iter=max_inner_iter,
                random_seed=random_seed + gen
            )

        loss_history.append(best_loss)

    # --- Visualization ---
    if visualization:
        # 1) Plot loss over generations
        plt.figure(figsize=(8, 5))
        plt.plot(loss_history, 'b-o')
        plt.title("GA Convergence (Best Loss by Generation)")
        plt.xlabel("Generation")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

        # 2) Plot the final velocity estimate vs time
        plt.figure(figsize=(10, 5))
        plt.plot(best_v[:, 0], label="v_x")
        plt.plot(best_v[:, 1], label="v_y")
        plt.plot(best_v[:, 2], label="v_z")
        plt.title("Estimated Velocity Components over Time (GA Best)")
        plt.xlabel("Time Step")
        plt.ylabel("Velocity")
        plt.legend()
        plt.grid(True)
        plt.show()

        # 3) Show distribution of final camera positions
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Camera Positions (Best Subset)")

        selected_indices = np.where(best_subset_mask)[0]
        excluded_indices = np.where(~best_subset_mask)[0]

        # Plot selected cameras in blue
        for i in selected_indices:
            ax.quiver(
                0, 0, 0,
                best_r_hat[i, 0], best_r_hat[i, 1], best_r_hat[i, 2],
                length=1.0, normalize=False, color='blue', linewidth=1.5, arrow_length_ratio=0.1
            )
            ax.text(
                best_r_hat[i, 0], best_r_hat[i, 1], best_r_hat[i, 2],
                f"Cam {i}", color='blue', fontsize=9
            )

        # Plot excluded cameras in red
        for i in excluded_indices:
            ax.quiver(
                0, 0, 0,
                best_r_hat[i, 0], best_r_hat[i, 1], best_r_hat[i, 2],
                length=1.0, normalize=False, color='red', linewidth=1.0, arrow_length_ratio=0.1
            )
            ax.text(
                best_r_hat[i, 0], best_r_hat[i, 1], best_r_hat[i, 2],
                f"Cam {i}", color='red', fontsize=9
            )

        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=20., azim=30)
        plt.tight_layout()
        plt.show()

        # 4) Plot the observed radial velocities for a few selected cameras
        if len(selected_indices) >= 10:
            top_cameras_indices = selected_indices[:10]
        else:
            top_cameras_indices = selected_indices

        if len(top_cameras_indices) > 0:
            plt.figure(figsize=(12, 6))
            for i in top_cameras_indices:
                plt.plot(v_r[:, i], label=f'Camera {i}')
            plt.title('Observed Radial Velocities for Selected Cameras')
            plt.xlabel('Time Steps')
            plt.ylabel('Radial Velocity')
            plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05))
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    return best_v, best_r_hat, best_subset_mask, best_loss, loss_history

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from matplotlib import cm
from typing import Tuple

def generate_rotation_matrices(N: int) -> np.ndarray:
    """
    Generate N random rotation matrices.
    """
    angles = np.random.uniform(0, 2*np.pi, size=(N, 3))  # Euler angles: alpha, beta, gamma
    rotation_matrices = []
    for alpha, beta, gamma in angles:
        # Rotation matrix around X-axis
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(alpha), -np.sin(alpha)],
                       [0, np.sin(alpha), np.cos(alpha)]])
        
        # Rotation matrix around Y-axis
        Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                       [0, 1, 0],
                       [-np.sin(beta), 0, np.cos(beta)]])
        
        # Rotation matrix around Z-axis
        Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                       [np.sin(gamma), np.cos(gamma), 0],
                       [0, 0, 1]])
        
        # Combined rotation matrix
        R = Rz @ Ry @ Rx
        rotation_matrices.append(R)
    return np.array(rotation_matrices)  # Shape: (N, 3, 3)

def rotate_vector(data: np.ndarray, rotation_matrices: np.ndarray) -> np.ndarray:
    """
    Apply each rotation matrix to the data.
    
    Parameters:
    - data: np.ndarray of shape (3, T)
    - rotation_matrices: np.ndarray of shape (N, 3, 3)
    
    Returns:
    - rotated_data: np.ndarray of shape (N, 3, T)
    """
    N = rotation_matrices.shape[0]
    rotated_data = np.empty((N, 3, data.shape[1]))
    for i in range(N):
        rotated_data[i] = rotation_matrices[i] @ data
    return rotated_data

def visualize_rotations(
    rotated_data: np.ndarray,
    N: int,
    original_data: np.ndarray = None,
    equal_axis: bool = True,
    colormap: str = 'viridis'
):
    """
    Visualize the rotated data in a grid of 3D plots with the same axis limits and color based on time.
    
    Parameters:
    - rotated_data: np.ndarray of shape (N, 3, T)
    - N: number of rotations
    - original_data: np.ndarray of shape (3, T), optional original data to plot
    - equal_axis: bool, if True all subplots will have the same axis limits
    - colormap: str, name of the matplotlib colormap to use
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from matplotlib.colors import Normalize
    from matplotlib import cm

    # Determine grid size
    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / cols))
    
    # -- Compute global axis limits across all data --
    # Collect all x, y, z points from rotated_data
    all_points = rotated_data.reshape(-1, rotated_data.shape[-1])  # shape: (3N, T)
    
    xs = all_points[0::3, :].flatten()  # x-coordinates from all rotations
    ys = all_points[1::3, :].flatten()  # y-coordinates from all rotations
    zs = all_points[2::3, :].flatten()  # z-coordinates from all rotations

    # If original data is provided, include those points too
    if original_data is not None:
        xs = np.concatenate([xs, original_data[0, :]])
        ys = np.concatenate([ys, original_data[1, :]])
        zs = np.concatenate([zs, original_data[2, :]])

    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    z_min, z_max = np.min(zs), np.max(zs)

    # Optional: Add a margin for better visuals
    margin = 0.1 * max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_min, x_max = x_min - margin, x_max + margin
    y_min, y_max = y_min - margin, y_max + margin
    z_min, z_max = z_min - margin, z_max + margin

    # Prepare color mapping based on time
    N_segments = rotated_data.shape[2] - 1  # Number of line segments
    norm = Normalize(vmin=0, vmax=rotated_data.shape[2]-1)
    cmap = cm.get_cmap(colormap)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = mappable.to_rgba(np.arange(rotated_data.shape[2]))

    # Create segments for the original data (used for all rotations)
    # Shape: (T-1, 2, 3)
    T = rotated_data.shape[2]
    segments = np.stack([original_data.T[:-1], original_data.T[1:]], axis=1) if original_data is not None else None

    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    
    for i in range(N):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        
        # Prepare segments for the current rotated data
        points = rotated_data[i].T  # Shape: (T, 3)
        segments = np.stack([points[:-1], points[1:]], axis=1)  # Shape: (T-1, 2, 3)
        
        # Create a Line3DCollection
        lc = Line3DCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(np.arange(T))
        lc.set_linewidth(2)
        lc.set_color(colors[:-1])  # Assign colors to each segment
        
        ax.add_collection(lc)
        
        # Auto scale to the data
        if equal_axis:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
        else:
            ax.auto_scale_xyz(points[:,0], points[:,1], points[:,2])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(True)
        
        # Optionally, add a color bar to the first subplot
        if i == 0:
            mappable.set_array(np.arange(T))
            cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, pad=0.1)
            cbar.set_label('Time')
    
    # Optionally, plot the original data in the first subplot
    if original_data is not None:
        ax_orig = fig.add_subplot(rows, cols, 1, projection='3d')
        points_orig = original_data.T  # Shape: (T, 3)
        segments_orig = np.stack([points_orig[:-1], points_orig[1:]], axis=1)
        
        lc_orig = Line3DCollection(segments_orig, cmap=cmap, norm=norm)
        lc_orig.set_array(np.arange(T))
        lc_orig.set_linewidth(2)
        lc_orig.set_color(colors[:-1])
        
        ax_orig.add_collection(lc_orig)
        
        if equal_axis:
            ax_orig.set_xlim(x_min, x_max)
            ax_orig.set_ylim(y_min, y_max)
            ax_orig.set_zlim(z_min, z_max)
        else:
            ax_orig.auto_scale_xyz(points_orig[:,0], points_orig[:,1], points_orig[:,2])
        
        ax_orig.set_xlabel('X')
        ax_orig.set_ylabel('Y')
        ax_orig.set_zlabel('Z')
        ax_orig.set_title('Original View')
        ax_orig.grid(True)
        
        # Re-add color bar if it wasn't added
        if N > 0:
            mappable.set_array(np.arange(T))
            cbar = fig.colorbar(mappable, ax=ax_orig, shrink=0.5, aspect=10, pad=0.1)
            cbar.set_label('Time')
    
    plt.tight_layout()
    plt.show()

def generate_rotated_views(data: np.ndarray, N: int, colormap: str = 'viridis') -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a 3D vector (3 x T), generate N rotated versions and visualize them with color based on time.
    
    Parameters:
    - data: np.ndarray of shape (3, T)
    - N: number of rotated views to generate
    - colormap: str, name of the matplotlib colormap to use
    
    Returns:
    - rotated_data: np.ndarray of shape (N, 3, T)
    - rotation_matrices: np.ndarray of shape (N, 3, 3)
    """
    rotation_matrices = generate_rotation_matrices(N)
    rotated_data = rotate_vector(data, rotation_matrices)
    visualize_rotations(rotated_data, N, original_data=data, equal_axis=True, colormap=colormap)
    return rotated_data, rotation_matrices




import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------------
# GA + LSQ Hybrid Implementation
# -------------------------------
def estimate_velocity_from_radial_ga(
    v_r,
    T,
    N,
    # Original hyperparameters
    lambda_reg=0.1,
    gamma=0.01,
    sigma_x2=1.0,
    sigma_y2=1.0,
    sigma_z2=1.0,
    huber_threshold=1.0,
    # GA hyperparameters
    population_size=50,
    generations=100,
    crossover_rate=0.7,
    mutation_rate=0.1,
    mutation_scale=0.05,
    elitism=1,
    # Others
    weights_init=None,
    fixed_camera_indices=(),
    random_seed=42,
    visualization=False
):
    """
    Use a Genetic Algorithm (GA) to estimate:
      - The 3D velocity vector v(t) for each time step t.
      - The direction vectors r_hat(i) for each camera i.

    GA Representation: 
      Each GA chromosome encodes only the camera direction vectors [r_hat(1), ..., r_hat(N)].
      For each chromosome, we do a least-squares solve for v(t) across time steps,
      then compute the fit loss as the GA fitness function.

    Parameters
    ----------
    v_r : np.ndarray
        Observed radial velocities of shape (T, N).
    T : int
        Number of time steps.
    N : int
        Number of cameras.
    lambda_reg, gamma : float
        Regularization parameters.
    sigma_x2, sigma_y2, sigma_z2 : float
        Variances used in regularization for velocity.
    huber_threshold : float
        Threshold for Huber loss.
    population_size : int
        GA population size.
    generations : int
        Number of GA generations.
    crossover_rate : float
        Probability of crossover between two parents.
    mutation_rate : float
        Probability of mutation of a child.
    mutation_scale : float
        Scale factor for random perturbations in mutation.
    elitism : int
        Number of top individuals preserved from one generation to the next (elitism).
    weights_init : np.ndarray or None
        Optional weighting for cameras (shape (N,)). If None, uses ones.
    fixed_camera_indices : tuple or list
        Indices of cameras whose directions are fixed (not evolved by GA).
    random_seed : int
        Random seed for reproducibility.
    visualization : bool
        If True, plot final results.

    Returns
    -------
    v_best : np.ndarray
        Best estimated 3D velocity vectors over time, shape (T, 3).
    r_hat_best : np.ndarray
        Best estimated direction vectors for each camera, shape (N, 3).
    loss_best : float
        Best (lowest) loss achieved by the GA.
    loss_history : list
        Loss of the best chromosome in each generation.
    """
    np.random.seed(random_seed)

    # --------------------------------------
    # Helper Functions
    # --------------------------------------
    def safe_normalize(vec, epsilon=1e-8):
        norm = np.linalg.norm(vec)
        if norm < epsilon:
            # default direction if norm is too small
            return np.array([1.0, 0.0, 0.0])
        return vec / norm

    def huber(residual, delta):
        """Huber loss for an array of residuals."""
        abs_res = np.abs(residual)
        quad_mask = abs_res <= delta
        # piecewise definition
        loss = np.where(
            quad_mask,
            0.5 * residual**2,
            delta * (abs_res - 0.5 * delta)
        )
        return loss

    def compute_loss_and_velocity(chromosome_r_hat):
        """
        Given a set of camera directions chromosome_r_hat of shape (N, 3),
        compute v(t) for t=1..T via weighted regularized least squares,
        then compute the total Huber loss.
        """
        # Expand camera directions to shape (N, 3)
        # (We already have it in that shape, just rename for clarity)
        r_hat_candidate = chromosome_r_hat

        # Solve for v(t) using LSQ for each time step
        v_est = np.zeros((T, 3))
        # Weighted version of A^T A + regularization
        if weights_init is None:
            w = np.ones(N)
        else:
            w = weights_init

        # Precompute diagonal for velocity regularization
        L = np.diag([gamma / sigma_x2, gamma / sigma_y2, gamma / sigma_z2])

        for t_idx in range(T):
            b = v_r[t_idx] * w  # shape (N,)
            A = r_hat_candidate * w[:, None]  # shape (N, 3)

            AtA = A.T @ A + L
            Atb = A.T @ b
            try:
                v_est[t_idx] = np.linalg.solve(AtA, Atb)
            except np.linalg.LinAlgError:
                # fallback if solve fails
                v_est[t_idx], _, _, _ = np.linalg.lstsq(AtA, Atb, rcond=None)

        # Compute residual & Huber loss
        residual = v_r - (v_est @ r_hat_candidate.T)  # shape (T, N)
        if huber_threshold is not None:
            loss = np.sum(huber(residual, huber_threshold))
        else:
            loss = np.sum(residual**2)

        return loss, v_est

    # --------------------------------------
    # GA Utility Functions
    # --------------------------------------
    def initialize_population(pop_size):
        """
        Create random directions for each camera: shape (pop_size, N, 3).
        For cameras in fixed_camera_indices, we set them to a default or random but keep them frozen in mutation.
        """
        pop = []
        for _ in range(pop_size):
            # random directions
            r_candidate = np.random.randn(N, 3)
            for i in range(N):
                r_candidate[i] = safe_normalize(r_candidate[i])
            pop.append(r_candidate)
        return np.array(pop)  # shape (pop_size, N, 3)

    def evaluate_population(pop):
        """Compute loss for each individual in pop."""
        losses = np.zeros(len(pop))
        for idx, r_hat_candidate in enumerate(pop):
            loss, _ = compute_loss_and_velocity(r_hat_candidate)
            losses[idx] = loss
        return losses

    def tournament_selection(pop, losses, k=3):
        """
        Tournament selection of one individual.
        Pick k random individuals, return best (lowest loss).
        """
        candidates = np.random.choice(len(pop), size=k, replace=False)
        best_idx = candidates[np.argmin(losses[candidates])]
        return pop[best_idx]

    def crossover(parent1, parent2):
        """
        Single-point or uniform crossover for r_hat arrays of shape (N, 3).
        Here we do uniform crossover with probability 0.5 for each gene.
        """
        child1 = parent1.copy()
        child2 = parent2.copy()
        for i in range(N):
            if np.random.rand() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        return child1, child2

    def mutate(individual):
        """
        Randomly perturb directions. Then re-normalize each camera direction.
        We skip mutation on fixed_camera_indices.
        """
        for i in range(N):
            if i in fixed_camera_indices:
                continue
            if np.random.rand() < mutation_rate:
                # random Gaussian perturbation
                individual[i] += mutation_scale * np.random.randn(3)
                individual[i] = safe_normalize(individual[i])
        return individual

    # --------------------------------------
    # GA Main Loop
    # --------------------------------------
    population = initialize_population(population_size)
    loss_history = []
    best_loss = np.inf
    best_r_hat = None
    best_v = None

    for gen in tqdm(range(generations), desc="GA Generations"):
        # Evaluate
        losses = evaluate_population(population)
        # Track best
        gen_best_idx = np.argmin(losses)
        gen_best_loss = losses[gen_best_idx]
        if gen_best_loss < best_loss:
            best_loss = gen_best_loss
            best_r_hat = population[gen_best_idx].copy()
            # Re-compute velocity for the best chromosome
            _, best_v_candidate = compute_loss_and_velocity(best_r_hat)
            best_v = best_v_candidate.copy()

        loss_history.append(best_loss)

        # Elitism: select top individuals
        sorted_idx = np.argsort(losses)
        next_population = [population[i].copy() for i in sorted_idx[:elitism]]

        # Generate new offspring
        while len(next_population) < population_size:
            # selection
            parent1 = tournament_selection(population, losses, k=3)
            parent2 = tournament_selection(population, losses, k=3)
            # crossover
            if np.random.rand() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            # mutation
            child1 = mutate(child1)
            child2 = mutate(child2)
            # add to next population
            next_population.append(child1)
            if len(next_population) < population_size:
                next_population.append(child2)

        population = np.array(next_population)

    # After GA ends, compute final best solution
    if best_r_hat is None:
        # Fallback if something went wrong
        best_r_hat = population[0]
        best_loss, best_v = compute_loss_and_velocity(best_r_hat)

    # --------------------------------------
    # Visualization
    # --------------------------------------
    if visualization:
        # 1) Plot loss over generations
        plt.figure(figsize=(8, 5))
        plt.plot(loss_history, 'b-o')
        plt.title("GA Convergence (Best Loss by Generation)")
        plt.xlabel("Generation")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

        # 2) Plot the final velocity estimate vs time
        plt.figure(figsize=(10, 5))
        plt.plot(best_v[:, 0], label="v_x")
        plt.plot(best_v[:, 1], label="v_y")
        plt.plot(best_v[:, 2], label="v_z")
        plt.title("Estimated Velocity Components over Time (GA Best)")
        plt.xlabel("Time Step")
        plt.ylabel("Velocity")
        plt.legend()
        plt.grid(True)
        plt.show()

        # 3) Show distribution of final camera directions
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Camera Directions (Best Chromosome)")
        origin = np.zeros((N, 3))
        for i in range(N):
            ax.quiver(
                origin[i, 0], origin[i, 1], origin[i, 2],
                best_r_hat[i, 0], best_r_hat[i, 1], best_r_hat[i, 2],
                length=1.0, normalize=True
            )
            ax.text(
                best_r_hat[i, 0], best_r_hat[i, 1], best_r_hat[i, 2],
                f"Cam {i}", color='red'
            )

        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

    return best_v, best_r_hat, best_loss, loss_history

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def estimate_velocity_from_radial_mlp(
    v_r,
    random_seed=5,
    visualization=True,
    num_epochs=200,
    batch_size=1,
    hidden_size=512,
    latent_size=256,
    learning_rate=1e-3,
    huber_delta=1.0,
    outlier_threshold_percentile=90
):
    """
    Estimate 3D velocities and camera directions from radial velocities using an MLP encoder-decoder.
    
    Parameters:
    - v_r (np.ndarray): Observed radial velocities, shape (T, N).
    - random_seed (int): Seed for reproducibility.
    - visualization (bool): If True, generate plots of the estimation process and results.
    - num_epochs (int): Number of training epochs for the MLP.
    - batch_size (int): Batch size for training (default is 1 since data is treated as a single sample).
    - hidden_size (int): Number of neurons in hidden layers of the MLP.
    - latent_size (int): Size of the latent representation in the encoder.
    - learning_rate (float): Learning rate for the optimizer.
    - huber_delta (float): Delta parameter for the Huber loss function.
    - outlier_threshold_percentile (float): Percentile to determine outlier cameras based on residuals.
    
    Returns:
    - estimated_velocities (np.ndarray): Estimated 3D velocity vectors over time, shape (T, 3).
    - estimated_directions (np.ndarray): Estimated camera direction vectors, shape (N, 3).
    - best_loss (float): Final loss value after training.
    - loss_history (list of float): Loss values recorded at each epoch.
    """
    
    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    
    # Check input dimensions
    if v_r.ndim != 2:
        raise ValueError("v_r must be a 2D numpy array with shape (T, N)")
    
    T, N = v_r.shape  # Number of time steps and cameras
    
    # Define the Dataset
    class RadialVelocityDataset(Dataset):
        def __init__(self, v_r):
            """
            Dataset for radial velocities.
            
            Parameters:
            - v_r (np.ndarray): Observed radial velocities of shape (T, N).
            """
            self.v_r = v_r.astype(np.float32)
        
        def __len__(self):
            return 1  # Single sample
        
        def __getitem__(self, idx):
            return torch.from_numpy(self.v_r)  # Shape: (T, N)
    
    # Define the MLP Encoder-Decoder Model
    class VelocityCameraEstimator(nn.Module):
        def __init__(self, T, N, hidden_size=512, latent_size=256):
            """
            Encoder-Decoder MLP for estimating 3D velocities and camera directions.
            
            Parameters:
            - T (int): Number of time steps.
            - N (int): Number of cameras.
            - hidden_size (int): Number of neurons in hidden layers.
            - latent_size (int): Size of the latent representation.
            """
            super(VelocityCameraEstimator, self).__init__()
            self.T = T
            self.N = N
            self.input_size = T * N
            self.latent_size = latent_size
            
            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(self.input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, latent_size),
                nn.ReLU()
            )
            
            # Decoder for velocities
            self.decoder_v = nn.Sequential(
                nn.Linear(latent_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, T * 3)  # Output velocities for each time step
            )
            
            # Decoder for camera directions
            self.decoder_r = nn.Sequential(
                nn.Linear(latent_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, N * 3),  # Output directions for each camera
                nn.Tanh()  # To bound the directions before normalization
            )
        
        def forward(self, x):
            batch_size = x.size(0)
            x_flat = x.reshape(batch_size, -1)  # Flatten to (batch_size, T*N)
            latent = self.encoder(x_flat)       # (batch_size, latent_size)
            
            v_flat = self.decoder_v(latent)     # (batch_size, T*3)
            v_pred = v_flat.reshape(batch_size, self.T, 3)  # (batch_size, T, 3)
            
            r_flat = self.decoder_r(latent)     # (batch_size, N*3)
            r_pred = r_flat.reshape(batch_size, self.N, 3)  # (batch_size, N, 3)
            r_pred = nn.functional.normalize(r_pred, p=2, dim=2)  # Normalize directions
            
            return v_pred, r_pred
    
    # Define the Huber Loss Function
    def huber_loss(residual, delta=1.0):
        """
        Compute the Huber loss.
        
        Parameters:
        - residual (torch.Tensor): Residuals between predicted and observed radial velocities.
        - delta (float): Threshold parameter for Huber loss.
        
        Returns:
        - loss (torch.Tensor): Computed Huber loss.
        """
        abs_res = torch.abs(residual)
        in_quadratic = abs_res <= delta
        quadratic = torch.where(in_quadratic, 0.5 * residual**2, torch.zeros_like(residual))
        linear = torch.where(~in_quadratic, delta * (abs_res - 0.5 * delta), torch.zeros_like(residual))
        return torch.sum(quadratic + linear)
    
    # Prepare the Dataset and DataLoader
    dataset = RadialVelocityDataset(v_r)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the Model
    model = VelocityCameraEstimator(T=T, N=N, hidden_size=hidden_size, latent_size=latent_size)
    
    # Define the Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    loss_history = []
    best_loss = float('inf')
    best_v = None
    best_r_hat = None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            v_r_input = batch.to(device)  # (batch_size, T, N)
            
            optimizer.zero_grad()
            v_pred, r_pred = model(v_r_input)  # (batch_size, T, 3), (batch_size, N, 3)
            
            # Compute predicted radial velocities
            # v_pred: (batch_size, T, 3), r_pred: (batch_size, N, 3)
            # Need to compute dot product for each t and i
            # Expand dimensions to compute pairwise dot products
            v_pred_exp = v_pred.unsqueeze(2)  # (batch_size, T, 1, 3)
            r_pred_exp = r_pred.unsqueeze(1)  # (batch_size, 1, N, 3)
            radial_pred = torch.sum(v_pred_exp * r_pred_exp, dim=3)  # (batch_size, T, N)
            
            # Compute residuals
            residual = radial_pred - v_r_input  # (batch_size, T, N)
            
            # Compute Huber loss
            loss = huber_loss(residual, delta=huber_delta)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Average loss for the epoch
        epoch_loss /= len(dataloader)
        loss_history.append(epoch_loss)
        
        # Track the best loss and corresponding estimates
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_v = v_pred.detach().cpu().numpy().reshape(T, 3)
            best_r_hat = r_pred.detach().cpu().numpy().reshape(N, 3)
    
    # After Training: Identify Outliers Based on Residuals
    model.eval()
    with torch.no_grad():
        v_pred, r_pred = model(torch.from_numpy(v_r).float().unsqueeze(0).to(device))  # (1, T, 3), (1, N, 3)
        v_pred = v_pred.cpu().numpy().reshape(T, 3)
        r_pred = r_pred.cpu().numpy().reshape(N, 3)
        radial_pred = v_pred @ r_pred.T  # (T, N)
        residuals = np.abs(radial_pred - v_r)  # Absolute residuals
    
    # Determine outlier cameras based on residuals
    threshold = np.percentile(np.max(residuals, axis=0), outlier_threshold_percentile)
    excluded_cams = np.where(np.max(residuals, axis=0) > threshold)[0].tolist()
    selected_cams = [i for i in range(N) if i not in excluded_cams]
    
    if visualization:
        # 1. Plot Loss Over Epochs
        plt.figure(figsize=(8, 5))
        plt.plot(loss_history, 'b-o')
        plt.title("Training Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Huber Loss")
        plt.grid(True)
        plt.show()
        
        # 2. Plot Estimated Velocity Components Over Time
        plt.figure(figsize=(12, 6))
        plt.plot(best_v[:, 0], label='Estimated v_x')
        plt.plot(best_v[:, 1], label='Estimated v_y')
        plt.plot(best_v[:, 2], label='Estimated v_z')
        plt.title('Estimated 3D Velocities Over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Velocity')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # 3. Plot Camera Directions in 3D Space
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Estimated Camera Directions")
        
        # Plot selected cameras
        for i in selected_cams:
            ax.quiver(
                0, 0, 0,
                best_r_hat[i, 0], best_r_hat[i, 1], best_r_hat[i, 2],
                length=1.0, normalize=True, color='blue', linewidth=1.5, arrow_length_ratio=0.1
            )
            ax.text(
                best_r_hat[i, 0], best_r_hat[i, 1], best_r_hat[i, 2],
                f"Cam {i}", color='blue', fontsize=9
            )
        
        # Plot excluded cameras
        for i in excluded_cams:
            ax.quiver(
                0, 0, 0,
                best_r_hat[i, 0], best_r_hat[i, 1], best_r_hat[i, 2],
                length=1.0, normalize=True, color='red', linewidth=1.0, arrow_length_ratio=0.1
            )
            ax.text(
                best_r_hat[i, 0], best_r_hat[i, 1], best_r_hat[i, 2],
                f"Cam {i}", color='red', fontsize=9
            )
        
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=20., azim=30)
        plt.tight_layout()
        plt.show()
        
        # 4. Plot Observed vs Predicted Radial Velocities for Selected Cameras
        plt.figure(figsize=(12, 6))
        for i in selected_cams[:10]:  # Plot up to 10 cameras
            plt.plot(radial_pred[:, i], label=f'Predicted Cam {i}')
            plt.plot(v_r[:, i], '--', label=f'Observed Cam {i}')
        plt.title('Observed vs Predicted Radial Velocities for Selected Cameras')
        plt.xlabel('Time Steps')
        plt.ylabel('Radial Velocity')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05))
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    # Return the results
    return best_v, best_r_hat, best_loss, loss_history

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

##############################################################################
# 1) Synthetic Data: Two-Bar System
##############################################################################
def generate_time_varying_2bar_data(
    T=5,
    N=12,
    random_seed=0,
    noise_std=0.02
):
    """
    Generate radial velocity data from a time-varying "two-bar" system 
    with a fixed joint M=0.
    Each camera has alpha_i in [0,2], no time dimension (the same alpha across time).
    """
    np.random.seed(random_seed)
    
    # Time-varying endpoints A(t), B(t); M(t)=0
    M_true = np.zeros((T,3))
    A_true = np.zeros((T,3))
    B_true = np.zeros((T,3))
    
    for t in range(T):
        # Some random motion for A(t) and B(t)
        A_true[t] = np.array([0.3*t, np.sin(0.3*t), 0.1*t]) + 0.1*np.random.randn(3)
        B_true[t] = 0 * np.array([0.6*t, np.cos(0.4*t), 0.2*t]) + 0.1*np.random.randn(3)
    
    # Cameras
    cameras_gt = 3.0*(np.random.rand(N,3) - 0.5)
    
    # alpha_i in [0,2], shape(N,)
    alpha_true = 2.0*np.random.rand(N)
    
    # Build radial velocities, shape (T,N)
    v_r_obs = np.zeros((T,N))
    for i in range(N):
        alpha_i = alpha_true[i]
        for t in range(T):
            if t < T-1:
                vA = A_true[t+1] - A_true[t]
                vB = B_true[t+1] - B_true[t]
            else:
                vA = A_true[t] - A_true[t-1]
                vB = B_true[t] - B_true[t-1]
            
            if alpha_i <=1:
                # A->0
                P_t = A_true[t] + alpha_i*(0 - A_true[t])
                vP  = vA + alpha_i*(0 - vA)
            else:
                alpha2 = alpha_i -1.0
                P_t = 0 + alpha2*(B_true[t] - 0)
                vP = alpha2*vB
            
            direction = P_t - cameras_gt[i]
            dist = np.linalg.norm(direction)
            if dist<1e-8:
                direction_hat = np.array([1.0,0,0])
            else:
                direction_hat = direction/dist
            
            radial_vel = np.dot(vP, direction_hat)
            radial_vel += noise_std*np.random.randn()
            v_r_obs[t,i] = radial_vel
    
    return v_r_obs, cameras_gt, A_true, B_true, alpha_true

##############################################################################
# 2) Dataset
##############################################################################
class TimeVaryingDataset(Dataset):
    def __init__(self, v_r_obs):
        self.v_r_obs = v_r_obs.astype(np.float32)
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        return self.v_r_obs  # shape (T,N)

##############################################################################
# 3) Helpers: Spherical -> Cartesian, Huber Loss
##############################################################################
def spherical_to_cartesian(theta, phi):
    """
    Convert (theta,phi) in [0,pi],[0,2pi] => (x,y,z) on unit sphere.
    """
    x = torch.sin(theta)*torch.cos(phi)
    y = torch.sin(theta)*torch.sin(phi)
    z = torch.cos(theta)
    return x,y,z

def huber_loss(residual, delta=1.0):
    abs_x = torch.abs(residual)
    quadratic = torch.where(abs_x<delta, 0.5*residual**2, torch.zeros_like(residual))
    linear = torch.where(abs_x>=delta, delta*(abs_x - 0.5*delta), torch.zeros_like(residual))
    return torch.sum(quadratic+linear)

##############################################################################
# 4) Single-Point Model on Sphere
##############################################################################
class SinglePointSphereModel(nn.Module):
    """
    p(t) => 3T
    camera angles => 2N
    radius => 1
    total => 3T + 2N +1
    We'll compute center-of-mass of p(t), place cameras on that sphere.
    """
    def __init__(self, T, N, hidden_size=256):
        super().__init__()
        self.T = T
        self.N = N
        input_size = T*N
        output_size = 3*self.T + 2*self.N + 1
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        bsz= x.shape[0]  #1
        x_flat= x.view(bsz, -1)
        h= self.encoder(x_flat)
        out= self.decoder(h) # (1, 3T+2N+1)
        
        idxP_end= 3*self.T
        idxAngles_end= idxP_end + 2*self.N
        idxRadius_end= idxAngles_end +1
        
        p_flat= out[:, :idxP_end]          # (1,3T)
        angles_flat= out[:, idxP_end: idxAngles_end]  # (1,2N)
        radius_= out[:, idxAngles_end: idxRadius_end] # (1,)
        
        p= p_flat.view(bsz, self.T, 3)
        angles= angles_flat.view(bsz, self.N,2)
        radius= torch.abs(radius_)+ 0.01
        return p, angles, radius

def reconstruct_single_point_sphere(p, angles, radius):
    device= p.device
    bsz= p.shape[0] #1
    T= p.shape[1]
    # center-of-mass => average p(t)
    cm= torch.mean(p, dim=1, keepdim=True) # (1,1,3)
    
    # angles => shape(1,N,2)
    # radius => shape(1,)
    rad_b= radius.view(1,1)
    theta= angles[:,:,0] # (1,N)
    phi=   angles[:,:,1]
    x,y,z= spherical_to_cartesian(theta,phi)
    cams_dir= torch.stack([x,y,z], dim=2) # (1,N,3)
    cams= cm + rad_b.unsqueeze(2)*cams_dir  # (1,N,3)
    
    # single-point radial velocity
    radial_pred_list= []
    for t in range(T):
        if t< T-1:
            v_t= p[:,t+1,:] - p[:,t,:]
        else:
            if T>1:
                v_t= p[:,t,:] - p[:,t-1,:]
            else:
                v_t= torch.zeros_like(p[:,0,:])
        p_t= p[:,t,:].unsqueeze(1) # (1,1,3)
        direction= p_t- cams       # (1,N,3)
        dist= torch.norm(direction, dim=2, keepdim=True)+1e-8
        dir_hat= direction/dist
        v_exp= v_t.unsqueeze(1) # (1,1,3)
        r_t= torch.sum(v_exp*dir_hat, dim=2) #(1,N)
        radial_pred_list.append(r_t)
    radial_pred= torch.stack(radial_pred_list, dim=1) # (1,T,N)
    return radial_pred

def train_single_point_sphere(
    v_r_obs, T,N,
    hidden_size=256,
    lr=1e-3,
    epochs=1000,
    huber_delta=0.1,
    random_seed=0,
    device='cpu'
):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    ds= TimeVaryingDataset(v_r_obs)
    dl= DataLoader(ds, batch_size=1, shuffle=False)
    model= SinglePointSphereModel(T,N,hidden_size).to(device)
    optimizer= torch.optim.Adam(model.parameters(), lr=lr)
    
    best_loss= float('inf')
    best_params= None
    loss_history=[]
    
    for epoch in tqdm(range(epochs), desc="TrainSinglePtSphere", leave=False):
        for batch in dl:
            batch= batch.to(device)
            optimizer.zero_grad()
            p, ang, rad= model(batch)
            radial_pred= reconstruct_single_point_sphere(p, ang, rad)
            residual= radial_pred- batch
            loss= huber_loss(residual, huber_delta)
            loss.backward()
            optimizer.step()
            
            curr_loss= loss.item()
            loss_history.append(curr_loss)
            if curr_loss< best_loss:
                best_loss= curr_loss
                p_np= p.detach().cpu().numpy()[0]
                ang_np= ang.detach().cpu().numpy()[0]
                rad_np= rad.detach().cpu().numpy()[0]
                best_params= (p_np, ang_np, rad_np)
    return best_params, best_loss, loss_history

##############################################################################
# 5) Single-Bar with Endpoint=0, alpha_i not time-varying, cameras on sphere
##############################################################################
class SingleBarSphereModel(nn.Module):
    """
    B(t) => 3T
    alpha => N in [0,1]
    camera angles => 2N
    radius => 1
    total => 3T + N + 2N +1
    """
    def __init__(self, T,N, hidden_size=256):
        super().__init__()
        self.T= T
        self.N= N
        input_size= T*N
        output_size= 3*self.T + self.N + 2*self.N +1
        
        self.encoder= nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.decoder= nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        bsz= x.shape[0]
        x_flat= x.view(bsz,-1)
        h= self.encoder(x_flat)
        out= self.decoder(h)
        
        idxB_end= 3*self.T
        idxAlpha_end= idxB_end + self.N
        idxAngles_end= idxAlpha_end + 2*self.N
        idxRadius_end= idxAngles_end +1
        
        B_flat= out[:, :idxB_end]       # (1,3T)
        alpha_flat= out[:, idxB_end:idxAlpha_end] # (1,N)
        angles_flat= out[:, idxAlpha_end: idxAngles_end]
        radius_    = out[:, idxAngles_end: idxRadius_end]
        
        B= B_flat.view(bsz, self.T,3)
        alpha_= torch.sigmoid(alpha_flat) # (1,N)
        angles= angles_flat.view(bsz, self.N,2)
        radius= torch.abs(radius_)+ 0.01
        return B, alpha_, angles, radius

def reconstruct_single_bar_sphere(B, alpha_, angles, radius):
    """
    One endpoint=0 => bar from 0-> B(t).
    alpha in [0,1], shape(1,N).
    cameras on sphere => we compute center-of-mass ~ average( B(t)/2 ), etc.
    """
    device= B.device
    T= B.shape[1]
    midpoint= 0.5*B
    cm= torch.mean(midpoint, dim=1, keepdim=True) # (1,1,3)
    
    # build cameras from angles, radius
    theta= angles[:,:,0]
    phi  = angles[:,:,1]
    x,y,z= spherical_to_cartesian(theta, phi)
    cams_dir= torch.stack([x,y,z], dim=2) # (1,N,3)
    rad_b= radius.view(1,1)
    cams= cm + rad_b.unsqueeze(2)*cams_dir  # (1,N,3)
    
    radial_pred_list= []
    for t in range(T):
        if t< T-1:
            vB= B[:,t+1,:] - B[:,t,:]
        else:
            if T>1:
                vB= B[:,t,:] - B[:,t-1,:]
            else:
                vB= torch.zeros_like(B[:,0,:])
        
        alpha_exp= alpha_.unsqueeze(2) # (1,N,1)
        B_t= B[:,t,:].unsqueeze(1)     # (1,1,3)
        P_ti= alpha_exp*B_t           # (1,N,3)
        vP= alpha_exp*(vB.unsqueeze(1)) # (1,N,3)
        
        direction= P_ti - cams
        dist= torch.norm(direction, dim=2, keepdim=True)+1e-8
        dir_hat= direction/dist
        
        rad_t= torch.sum(vP*dir_hat, dim=2)
        radial_pred_list.append(rad_t)
    radial_pred= torch.stack(radial_pred_list, dim=1)
    return radial_pred

def train_single_bar_sphere(
    v_r_obs, T,N,
    hidden_size=256,
    lr=1e-3,
    epochs=1000,
    huber_delta=0.1,
    random_seed=0,
    device='cpu'
):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    ds= TimeVaryingDataset(v_r_obs)
    dl= DataLoader(ds, batch_size=1, shuffle=False)
    model= SingleBarSphereModel(T,N, hidden_size).to(device)
    optimizer= torch.optim.Adam(model.parameters(), lr=lr)
    
    best_loss= float('inf')
    best_params=None
    loss_history=[]
    for epoch in tqdm(range(epochs), desc="TrainBarSphere", leave=False):
        for batch in dl:
            batch= batch.to(device)
            optimizer.zero_grad()
            B_, alpha_, ang, rad= model(batch)
            radial_pred= reconstruct_single_bar_sphere(B_, alpha_, ang, rad)
            residual= radial_pred- batch
            loss= huber_loss(residual, huber_delta)
            loss.backward()
            optimizer.step()
            
            curr_loss= loss.item()
            loss_history.append(curr_loss)
            if curr_loss< best_loss:
                best_loss= curr_loss
                B_np= B_.detach().cpu().numpy()[0]
                alpha_np= alpha_.detach().cpu().numpy()[0]
                angles_np= ang.detach().cpu().numpy()[0]
                rad_np= rad.detach().cpu().numpy()[0]
                best_params= (B_np, alpha_np, angles_np, rad_np)
    return best_params, best_loss, loss_history

##############################################################################
# 6) Two-Bar with M=0, alpha_i in [0,2], cameras on sphere
##############################################################################
class TwoBarSphereModel(nn.Module):
    """
    A(t), B(t) => 3T +3T=6T
    alpha => N in [0,2]
    camera angles => 2N
    radius =>1
    => total=6T + N +2N +1=6T+3N+1
    """
    def __init__(self, T,N, hidden_size=256):
        super().__init__()
        self.T=T
        self.N=N
        input_size= T*N
        output_size= 6*self.T + self.N + 2*self.N +1
        
        self.encoder= nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.decoder= nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        bsz= x.shape[0]
        x_flat= x.view(bsz,-1)
        h= self.encoder(x_flat)
        out= self.decoder(h) # shape(1, 6T+ 3N+1)
        
        idxA_end= 3*self.T
        idxB_end= idxA_end +3*self.T
        idxAlpha_end= idxB_end + self.N
        idxAngles_end= idxAlpha_end + 2*self.N
        idxRadius_end= idxAngles_end +1
        
        A_flat= out[:, :idxA_end]
        B_flat= out[:, idxA_end: idxB_end]
        alpha_flat= out[:, idxB_end: idxAlpha_end]
        angles_flat= out[:, idxAlpha_end: idxAngles_end]
        radius_   = out[:, idxAngles_end: idxRadius_end]
        
        A_= A_flat.view(bsz, self.T,3)
        B_= B_flat.view(bsz, self.T,3)
        alpha_= 2.0*torch.sigmoid(alpha_flat) # (1,N) in [0,2]
        
        angles= angles_flat.view(bsz, self.N,2)
        radius= torch.abs(radius_)+0.01
        return A_, B_, alpha_, angles, radius

def reconstruct_two_bar_sphere(A_, B_, alpha_, angles, radius):
    """
    M=0 => bar #1 is A->0, bar #2 is 0->B.
    alpha in [0,2], shape(1,N).
    cameras on sphere => center_of_mass = average( (A(t)+B(t))/2 ), etc.
    """
    device= A_.device
    T= A_.shape[1]
    
    # center_of_mass => average( (A(t)+B(t))/2 ), shape(1,1,3)
    half_sum= 0.5*(A_ + B_)
    cm= torch.mean(half_sum, dim=1, keepdim=True)
    
    # build cameras
    theta= angles[:,:,0]
    phi=   angles[:,:,1]
    x,y,z= spherical_to_cartesian(theta,phi)
    cams_dir= torch.stack([x,y,z], dim=2)
    rad_b= radius.view(1,1)
    cams= cm + rad_b.unsqueeze(2)*cams_dir #(1,N,3)
    
    # piecewise radial velocity
    radial_pred_list=[]
    for t in range(T):
        if t< T-1:
            vA= A_[:,t+1,:]- A_[:,t,:]
            vB= B_[:,t+1,:]- B_[:,t,:]
        else:
            if T>1:
                vA= A_[:,t,:]- A_[:,t-1,:]
                vB= B_[:,t,:]- B_[:,t-1,:]
            else:
                vA= torch.zeros_like(A_[:,0,:])
                vB= torch.zeros_like(B_[:,0,:])
        
        alpha_exp= alpha_.unsqueeze(2)
        mask1= (alpha_exp<=1.0).float()
        mask2= (alpha_exp> 1.0).float()
        
        alpha_bar1= torch.clamp(alpha_exp, 0,1)
        alpha_bar2= torch.clamp(alpha_exp-1.0,0,1)
        
        A_t= A_[:,t,:].unsqueeze(1)
        B_t= B_[:,t,:].unsqueeze(1)
        
        P_bar1= A_t + alpha_bar1*(0 -A_t)  # A->0
        P_bar2= alpha_bar2* B_t           # 0->B
        P_ti= mask1*P_bar1+ mask2*P_bar2
        
        vA_exp= vA.unsqueeze(1)
        vB_exp= vB.unsqueeze(1)
        vP_bar1= vA_exp + alpha_bar1*(0- vA_exp)
        vP_bar2= alpha_bar2*vB_exp
        vP= mask1*vP_bar1+ mask2*vP_bar2
        
        direction= P_ti- cams
        dist= torch.norm(direction, dim=2, keepdim=True)+1e-8
        dir_hat= direction/dist
        
        rad_t= torch.sum(vP*dir_hat, dim=2)
        radial_pred_list.append(rad_t)
    
    radial_pred= torch.stack(radial_pred_list, dim=1)
    return radial_pred

def train_two_bar_sphere(
    v_r_obs, T,N,
    hidden_size=256,
    lr=1e-3,
    epochs=1000,
    huber_delta=0.1,
    random_seed=0,
    device='cpu'
):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    ds= TimeVaryingDataset(v_r_obs)
    dl= DataLoader(ds,batch_size=1, shuffle=False)
    model= TwoBarSphereModel(T,N, hidden_size).to(device)
    optimizer= torch.optim.Adam(model.parameters(), lr=lr)
    
    best_loss= float('inf')
    best_params=None
    loss_history=[]
    for epoch in tqdm(range(epochs), desc="TrainTwoBarSphere", leave=False):
        for batch in dl:
            batch= batch.to(device)
            optimizer.zero_grad()
            A_, B_, alpha_, angles, radius= model(batch)
            radial_pred= reconstruct_two_bar_sphere(A_, B_, alpha_, angles, radius)
            residual= radial_pred- batch
            loss= huber_loss(residual, huber_delta)
            loss.backward()
            optimizer.step()
            
            curr_loss= loss.item()
            loss_history.append(curr_loss)
            if curr_loss< best_loss:
                best_loss= curr_loss
                A_np= A_.detach().cpu().numpy()[0]
                B_np= B_.detach().cpu().numpy()[0]
                alpha_np= alpha_.detach().cpu().numpy()[0]
                angles_np= angles.detach().cpu().numpy()[0]
                radius_np= radius.detach().cpu().numpy()[0]
                best_params= (A_np,B_np, alpha_np, angles_np, radius_np)
    return best_params, best_loss, loss_history

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import pi
def visualize_cameras(cameras, show_labels=True):
    """
    Visualize camera positions in 3D.
    
    Parameters
    ----------
    cameras : np.ndarray
        Array of shape (N, 3) indicating camera positions in 3D.
    show_labels : bool, optional
        If True, label each camera with its index in the 3D plot.
    """
    if not isinstance(cameras, np.ndarray):
        cameras = np.array(cameras, dtype=float)
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter-plot the camera positions
    ax.scatter(cameras[:, 0],
               cameras[:, 1],
               cameras[:, 2],
               c='blue',
               s=50,
               alpha=0.8,
               label='Cameras')
    
    if show_labels:
        # Label each camera with its index
        for i in range(len(cameras)):
            x, y, z = cameras[i]
            ax.text(x, y, z, f"Cam {i}", color='black')
    
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3D Camera Positions")
    ax.legend()
    ax.set_box_aspect((1,1,1))  # Make the plot aspect ratio equal in x,y,z
    plt.tight_layout()
    plt.show()
from scipy.signal import welch
def estimate_velocity_phase_derivative(alpha, Fs, fc, c=3e8):
    """
    Estimate Doppler shift by unwrapping the phase of alpha(t) and fitting a slope.
    Then map to velocity = (c/fc) * fD.

    WARNING: This works reliably only if alpha(t) has a strong stable tone
             (i.e., large K-factor). Otherwise, the phase is random and the result is noisy.
    """
    t = np.arange(len(alpha)) / Fs
    phase = np.unwrap(np.angle(alpha))

    # Linear fit of phase(t) => slope = 2*pi*fD
    p = np.polyfit(t, phase, 1)
    slope = p[0]
    fD_est = slope / (2*pi)
    v_est = (c / fc) * fD_est

    return v_est


def estimate_velocity_correlation(alpha, Fs, fc, c=3e8, max_lag_fraction=0.1):
    """
    Estimate Doppler shift by a naive autocorrelation approach.
    - We compute the complex autocorrelation of alpha(t).
    - Then look for the lag where the magnitude drops below some fraction (e.g., 0.5).
    - We approximate fD via known Bessel J0 crossing if Rayleigh-like.

    Returns a positive velocity magnitude (cannot sign it).
    """
    N = len(alpha)
    max_lag = int(N * max_lag_fraction)

    # Full autocorrelation (complex)
    r_full = np.correlate(alpha, alpha.conjugate(), mode='full')
    # r_full has length 2N-1. Zero-lag is at index (N-1).
    mid_idx = N - 1
    r = r_full[mid_idx - max_lag : mid_idx + max_lag + 1]
    lags = np.arange(-max_lag, max_lag + 1)
    tau = lags / Fs

    # Normalize so that r(0) = 1
    r_norm = np.abs(r) / np.abs(r[max_lag])

    # Find first crossing below 0.5 after zero-lag
    half_idx = np.where(r_norm < 0.5)[0]
    if len(half_idx) < 1:
        # If it never drops below 0.5 => presumably strong LoS => let's return near zero
        return 0.0
    # The first crossing after zero-lag
    # (the zero-lag is at index 'max_lag' in r_norm)
    crossing_idx = half_idx[0] if half_idx[0] > 0 else half_idx[1]
    tau_half = np.abs(tau[crossing_idx])

    # For J0(x)=0.5 => x ~ 1.2 => 2*pi * fD * tau_half = 1.2 => fD = 1.2 / (2*pi * tau_half)
    x_half = 1.2
    fD_est = x_half / (2*pi*tau_half) if tau_half != 0 else 0.0

    v_est = (c / fc)*fD_est
    return v_est

def estimate_velocity_psd(alpha, Fs, fc, Nfft=2**14, c=3e8, ratio_tol=1e-2):
    """
    Estimate the signed Doppler velocity from the complex envelope `alpha(t)`.

    The FFT is converted to a two-sided power-spectral density (PSD).
    If several bins have PSD values within `ratio_tol` of the global maximum,
    the *highest-index* bin is selected.

    Parameters
    ----------
    alpha : 1-D complex np.ndarray
        Complex baseband (or passband-shifted) signal samples.
    Fs : float
        Sampling rate [Hz].
    fc : float
        Carrier frequency of the illuminating signal [Hz].
    c  : float, optional
        Propagation speed (default = 3 × 10⁸ m/s).
    ratio_tol : float, optional
        Relative tolerance expressed as a fraction of the peak PSD
        (e.g. 1e-3 keeps bins ≥ 99.9 % of the maximum). Default 1e-3.

    Returns
    -------
    v_est : float
        Estimated Doppler velocity (signed) [m/s].
    """
    
    X     = np.fft.fftshift(np.fft.fft(alpha, n=Nfft))
    freqs = np.fft.fftshift(np.fft.fftfreq(Nfft, d=1/Fs))
    PSD   = np.abs(X)**2

    max_val   = PSD.max()
    tie_mask  = PSD >= max_val * (1 - ratio_tol)
    tie_idxs  = np.flatnonzero(tie_mask)

    peak_idx  = tie_idxs[-1]          # highest-index peak
    peak_freq = freqs[peak_idx]

    v_est = (c / fc) * peak_freq
    return v_est

# def estimate_velocity_psd(alpha, Fs, fc, c=3e8):
#     """
#     Estimate Doppler shift (and sign) by taking the FFT of the *complex* alpha(t),
#     computing the two-sided PSD, and finding the frequency where the PSD is maximum.

#     This yields a *signed* velocity: negative freq => negative velocity, etc.
#     """
#     Nfft = 2**15  # or another length
#     # FFT of the complex signal
#     X = np.fft.fft(alpha, n=Nfft)
#     X = np.fft.fftshift(X)
#     freqs = np.fft.fftshift(np.fft.fftfreq(Nfft, 1/Fs))  # from -Fs/2..+Fs/2
#     PSD = np.abs(X)**2

#     # Find the frequency index with the maximum PSD
#     # plt.figure()
#     # plt.plot(PSD)    
#     peak_idx = np.argmax(PSD)
#     peak_freq = freqs[peak_idx]  # can be negative or positive

#     # This is our estimated Doppler shift in Hz
#     fD_est = peak_freq

#     # Convert to velocity (signed)
#     v_est = (c / fc) * fD_est
#     return v_est


def estimate_velocity_phase_derivative(alpha, Fs, fc, c=3e8):
    """
    Estimate Doppler shift by unwrapping the phase of alpha(t) and fitting a slope.
    Then map to velocity = (c/fc) * fD.

    WARNING: This works reliably only if alpha(t) has a strong stable tone
             (i.e., large K-factor). Otherwise, the phase is random and the result is noisy.
    """
    t = np.arange(len(alpha)) / Fs
    phase = np.unwrap(np.angle(alpha))

    # Linear fit of phase(t) => slope = 2*pi*fD
    p = np.polyfit(t, phase, 1)
    slope = p[0]
    fD_est = slope / (2*pi)
    v_est = (c / fc) * fD_est

    return v_est

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0, i1
from scipy.optimize import minimize

# =================== Global Helper Functions ===================

def s_sigma_from_K(K):
    """
    Given a Rician K-factor (linear scale) with total power=1,
    compute (s, sigma) such that:
        s^2 + 2*sigma^2 = 1  and
        K = s^2 / (2 * sigma^2).
    """
    if K < 0:
        # Negative K is not physical
        return np.nan, np.nan
    sigma_sq = 1/(2*(K+1))
    s_sq = 2*K*sigma_sq
    if sigma_sq <= 0 or s_sq < 0:
        return np.nan, np.nan
    return np.sqrt(s_sq), np.sqrt(sigma_sq)

def rician_pdf(r, s, sigma):
    """
    Rician PDF for amplitude r >=0, given parameters (s, sigma).
    PDF(R) = (r / sigma^2)*exp(-(r^2 + s^2)/(2*sigma^2))*I0(r*s/sigma^2).
    """
    # Avoid negative or zero sigma
    if sigma <= 0 or s < 0:
        return np.zeros_like(r)

    rs = r*s/(sigma**2)
    vals = (r/(sigma**2)) * np.exp(-(r**2 + s**2)/(2*sigma**2)) * i0(rs)
    # clip small negatives that can arise from numeric round-off
    return np.maximum(vals, 0.0)

def generate_rician_csi(num_samples=10000, K=5.0, seed=42):
    """
    Generates a single-subcarrier Rician-fading CSI signal (complex)
    with total power=1 and K-factor=K.
    LoS is constant in time (no Doppler in this example).
    """
    np.random.seed(seed)
    s, sigma = s_sigma_from_K(K)

    if np.isnan(s) or np.isnan(sigma):
        # If K is invalid, fallback to Rayleigh (K=0)
        s, sigma = s_sigma_from_K(0.0)

    phi_0 = 2*np.pi*np.random.rand()
    los_component = s * np.exp(1j * phi_0)

    real_part = sigma * np.random.randn(num_samples)
    imag_part = sigma * np.random.randn(num_samples)
    scatter_component = real_part + 1j*imag_part

    csi = los_component + scatter_component
    return csi

# =================== K Estimation Methods ===================

def estimate_k_moments(r):
    """
    Estimate Rician K-factor from amplitude samples r using
    a simplified Method of Moments + numeric search.

    Returns: K_est (float) or np.nan if it fails.
    """
    # Check data: if nearly no variation or all zeros, K is indefinite.
    mean_r = np.mean(r)
    std_r  = np.std(r)
    if std_r < 1e-12:
        # Data is basically constant => can't reliably fit
        # If that constant is >0, it might imply huge K or near pure LoS.
        # We'll just return a safe fallback
        return 0.0 if mean_r < 1e-12 else 30.0

    r_m2 = np.mean(r**2)

    # Quick check if r_m2 < small => fallback
    if r_m2 < 1e-12:
        return 0.0

    Ks = np.linspace(0, 60, 601)
    best_K = np.nan
    best_err = 1e12

    for K_trial in Ks:
        s, sigma = s_sigma_from_K(K_trial)
        if np.isnan(s) or np.isnan(sigma):
            continue

        x = (s**2)/(2 * sigma**2)
        # Theoretical mean amplitude
        from scipy.special import i0, i1
        I0_x = i0(x)
        I1_x = i1(x)
        term = (1 + s**2/sigma**2)*I0_x + (s**2/sigma**2)*I1_x
        E_R_theory = sigma * np.sqrt(np.pi/2) * np.exp(-x) * term

        # Theoretical second moment => 1 by design
        # Compare to actual mean, second moment
        err_mean = abs(E_R_theory - mean_r)
        err_m2   = abs(r_m2 - 1.0)
        err = err_mean + err_m2

        if err < best_err:
            best_err = err
            best_K   = K_trial

    return best_K

def neg_log_likelihood(params, r):
    """
    Negative log-likelihood for Rician amplitude samples.
    params: [s, sigma]
    r: amplitude array
    """
    s, sigma = params
    if s <= 0 or sigma <= 0:
        return 1e15  # penalty
    pdf_vals = rician_pdf(r, s, sigma)
    pdf_vals[pdf_vals < 1e-15] = 1e-15  # avoid log(0)
    return -np.sum(np.log(pdf_vals))

def estimate_k_mle(r, init_K=None):
    """
    Estimate K by MLE, fallback on method-of-moments for initial guess
    if init_K not provided or invalid.
    Returns: float or np.nan if fails.
    """
    # Basic data checks
    if len(r) < 2:
        return np.nan
    if np.allclose(r, 0, atol=1e-12):
        # no variation => fallback to K=0
        return 0.0

    # If not given an initial guess, use MoM
    if init_K is None:
        init_K = estimate_k_moments(r)
        if np.isnan(init_K):
            init_K = 0.1

    # Convert that K into (s, sigma)
    s0, sigma0 = s_sigma_from_K(init_K)
    if np.isnan(s0) or np.isnan(sigma0):
        s0, sigma0 = s_sigma_from_K(1.0)  # fallback

    # Minimize negative log-likelihood
    result = minimize(neg_log_likelihood, x0=[s0, sigma0], args=(r,),
                      method='L-BFGS-B', bounds=[(1e-12, None), (1e-12, None)])

    if not result.success:
        # fallback
        return np.nan

    s_hat, sigma_hat = result.x
    if np.isnan(s_hat) or np.isnan(sigma_hat):
        return np.nan

    K_est = (s_hat**2)/(2*sigma_hat**2)
    if K_est < 0:
        K_est = np.nan
    return K_est

def hist_curve_fit(params, r_hist, r_bins):
    """
    Objective: sum of squared errors between histogram data r_hist
    and the Rician PDF with params=[s, sigma].
    """
    s, sigma = params
    if s <= 0 or sigma <= 0:
        return 1e15
    bin_centers = 0.5*(r_bins[:-1] + r_bins[1:])
    model = rician_pdf(bin_centers, s, sigma)
    return np.sum((r_hist - model)**2)

def estimate_k_histfit(r, n_bins=60, init_K=None):
    """
    Estimate K by fitting the Rician PDF to amplitude histogram
    with a least-squares approach.
    Falls back if fails.
    """
    if len(r) < 2:
        return np.nan
    counts, bin_edges = np.histogram(r, bins=n_bins, density=True)

    if init_K is None:
        init_K = estimate_k_moments(r)
        if np.isnan(init_K):
            init_K = 0.1

    s0, sigma0 = s_sigma_from_K(init_K)
    if np.isnan(s0) or np.isnan(sigma0):
        s0, sigma0 = s_sigma_from_K(1.0)

    def objective(params):
        return hist_curve_fit(params, counts, bin_edges)

    from scipy.optimize import minimize
    result = minimize(objective, x0=[s0, sigma0], method='L-BFGS-B',
                      bounds=[(1e-12,None),(1e-12,None)])
    if not result.success:
        return np.nan

    s_hat, sigma_hat = result.x
    if np.isnan(s_hat) or np.isnan(sigma_hat):
        return np.nan

    K_est = (s_hat**2)/(2*sigma_hat**2)
    return K_est

# 2) Estimate K by each method, with robust fallback
def robust_est(fun, *args, fallback=0.0, **kwargs):
    """Helper to wrap any estimator, catch np.nan or fails, return fallback."""
    val = fun(*args, **kwargs)
    if val is None or np.isnan(val):
        return fallback
    return val

import numpy as np
from scipy.special import i0, i1
from scipy.optimize import minimize

def estimate_k_mle(r, init_K=None):
    """
    Estimate the Rician K-factor (linear scale) from amplitude samples r
    using Maximum Likelihood Estimation (MLE).
    
    Parameters
    ----------
    r : 1D np.ndarray
        Amplitude (magnitude) samples, assumed to follow a Rician distribution.
    init_K : float or None
        Optional initial guess for K. If None, a simple method-of-moments
        estimate will be used internally.
        
    Returns
    -------
    K_est : float
        Estimated Rician K-factor (>= 0). NaN if the estimation fails or data is degenerate.
    
    Example
    -------
    >>> r = np.abs(generate_rician_csi(10000, K=5.0))  # your amplitude data
    >>> K_hat = estimate_k_mle(r)
    >>> print("Estimated K =", K_hat)
    """
    
    # 1) Check for degenerate data
    if len(r) < 2 or np.allclose(r, 0, atol=1e-12):
        # Not enough data or all zero => no meaningful estimate
        return np.nan
    
    # 2) (Optional) quick method-of-moments guess for K, if init_K is None
    def method_of_moments_guess(r_):
        """
        Simple numeric search to guess K using method-of-moments.
        Returns float or np.nan if it fails.
        """
        mean_r = np.mean(r_)
        r_m2 = np.mean(r_**2)
        if r_m2 < 1e-12:
            return 0.0
        
        Ks = np.linspace(0, 30, 301)
        best_K = np.nan
        best_err = 1e12
        
        for K_trial in Ks:
            s_sq, sigma_sq = _k_to_sigsq(K_trial)
            if np.isnan(s_sq) or np.isnan(sigma_sq):
                continue
            
            s_ = np.sqrt(s_sq)
            sigma_ = np.sqrt(sigma_sq)
            
            x = s_sq/(2*sigma_sq)
            # Theoretical mean amplitude
            I0_x = i0(x)
            I1_x = i1(x)
            term = (1 + s_sq/sigma_sq)*I0_x + (s_sq/sigma_sq)*I1_x
            E_R_theory = sigma_ * np.sqrt(np.pi/2) * np.exp(-x) * term
            
            # second moment is 1 by design if total power=1
            err_mean = abs(E_R_theory - mean_r)
            err_m2   = abs(r_m2 - 1.0)
            err = err_mean + err_m2
            
            if err < best_err:
                best_err = err
                best_K   = K_trial
        
        return best_K
    
    def _k_to_sigsq(K_val):
        """
        Convert K to (s^2, sigma^2) assuming total power=1.
        K = s^2 / (2*sigma^2),  s^2 + 2 sigma^2 = 1.
        """
        if K_val < 0:
            return (np.nan, np.nan)
        sigma_sq = 1.0 / (2.0*(K_val + 1.0))
        s_sq     = 2.0*K_val*sigma_sq
        if sigma_sq <= 0 or s_sq < 0:
            return (np.nan, np.nan)
        return (s_sq, sigma_sq)
    
    # If init_K not provided, use method-of-moments
    if init_K is None:
        init_K = method_of_moments_guess(r)
        if np.isnan(init_K):
            init_K = 1.0  # fallback
    
    # 3) Define Rician PDF
    def rician_pdf(r_vals, s, sigma):
        # Avoid negative or zero sigma
        if sigma <= 0.0:
            return np.zeros_like(r_vals)
        # Rician formula: pdf(r) = (r/sigma^2)*exp(-(r^2 + s^2)/(2*sigma^2))*I0(r*s/sigma^2)
        rs = r_vals * s / (sigma**2)
        pdf_ = (r_vals/(sigma**2)) * np.exp(-(r_vals**2 + s**2)/(2.0*sigma**2)) * i0(rs)
        return np.maximum(pdf_, 1e-30)  # clip to avoid log(0)
    
    # 4) Negative Log-likelihood
    def neg_log_likelihood(params):
        s_, sigma_ = params
        if s_ <= 0 or sigma_ <= 0:
            return 1e15  # penalty
        pdf_vals = rician_pdf(r, s_, sigma_)
        return -np.sum(np.log(pdf_vals))
    
    # 5) Convert initial K => (s0, sigma0)
    s0_sq, sigma0_sq = _k_to_sigsq(init_K)
    if np.isnan(s0_sq) or np.isnan(sigma0_sq):
        s0, sigma0 = 1.0, 1.0  # fallback
    else:
        s0, sigma0 = np.sqrt(s0_sq), np.sqrt(sigma0_sq)
    
    # 6) Run the optimizer
    result = minimize(
        neg_log_likelihood,
        x0=[s0, sigma0],
        method='L-BFGS-B',
        bounds=[(1e-12, None), (1e-12, None)]  # keep s, sigma > 0
    )
    
    if not result.success or np.isnan(result.fun):
        # Optimization failed or got nonsense
        return np.nan
    
    s_hat, sigma_hat = result.x
    # 7) Convert (s_hat, sigma_hat) => K
    if s_hat <= 0 or sigma_hat <= 0:
        return np.nan
    K_est = (s_hat**2) / (2.0*(sigma_hat**2))
    if K_est < 0:
        K_est = np.nan
    
    return K_est

