#!/usr/bin/env python
# coding: utf-8

# ## Advanced CSI Pre-processing + Advanced MUSIC

# In[3]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

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
from scipy.optimize import minimize
from utils_f2D import *
from scipy import signal
from sklearn.decomposition import FastICA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD

from dtaidistance import dtw
import matplotlib
# matplotlib.use('Agg')

import numpy as np

def calculate_complex_snr(signal):
    """
    Calculate SNR of a complex signal using first and last 10% as noise segments.
    
    Parameters:
    signal (array-like): Complex-valued signal (numpy array or list)
    
    Returns:
    float: SNR in decibels (dB)
    """
    # Convert to numpy array if not already
    signal = np.asarray(signal)
    
    # Check if signal is complex
    if not np.iscomplexobj(signal):
        raise ValueError("Input signal must be complex-valued")
    
    # Calculate segment lengths
    total_length = len(signal)
    noise_segment_length = int(0.15 * total_length)
    
    # Extract noise segments (first and last 10%)
    noise_segment = np.concatenate([
        signal[:noise_segment_length],
        signal[-noise_segment_length:]
    ])
    
    # Calculate noise power (average of squared magnitudes)
    noise_segment = np.unwrap(np.angle(noise_segment))
    noise_segment = noise_segment - np.mean(noise_segment)
    signal = np.unwrap(np.angle(signal))
    signal = signal - np.mean(signal)
    
    noise_power = np.mean(np.abs(noise_segment))**2
    
    # Calculate signal power (entire signal)
    signal_power = np.mean(np.abs(signal)[noise_segment_length:-noise_segment_length])**2
    
    # Calculate pure signal power (subtract noise power)
    pure_signal_power = signal_power - noise_power
    
    # Handle case where noise power >= signal power
    if pure_signal_power <= 0:
        return -np.inf  # SNR is negative infinity (no signal present)
    
    # Calculate SNR in linear scale
    snr_linear = pure_signal_power / noise_power
    
    # Convert to decibels
    snr_db = 10 * np.log10(snr_linear)
    
    
    
    return snr_db



def get_video_speed(X_video_trj, names_, AP_, subject_id, action_id, window_num, padding=0):
    trj = X_video_trj[(AP_, names_[subject_id].lower(), action_id)][window_num]
    rh_vis = (trj['right hand']['visibility'] > 0.2)
    rh_vis = moving_average(rh_vis, 4)[:-1]
    vel_video_x_rh = np.diff(trj['right hand']['x']) * rh_vis
    vel_video_y_rh = np.diff(trj['right hand']['y']) * rh_vis
    vel_video_z_rh = np.diff(trj['right hand']['z']) * rh_vis
    
    ra_vis = (trj['right arm']['visibility'] > 0.2)
    ra_vis = moving_average(ra_vis, 4)[:-1]
    vel_video_x_ra = np.diff(trj['right arm']['x']) * ra_vis
    vel_video_y_ra = np.diff(trj['right arm']['y']) * ra_vis
    vel_video_z_ra = np.diff(trj['right arm']['z']) * ra_vis
    
    lh_vis = (trj['left hand']['visibility'] > 0.2)
    lh_vis = moving_average(lh_vis, 4)[:-1]
    vel_video_x_lh = np.diff(trj['left hand']['x']) * lh_vis
    vel_video_y_lh = np.diff(trj['left hand']['y']) * lh_vis
    vel_video_z_lh = np.diff(trj['left hand']['z']) * lh_vis
    
    la_vis = (trj['left arm']['visibility'] > 0.2)
    la_vis = moving_average(la_vis, 4)[:-1]
    vel_video_x_la = np.diff(trj['left arm']['x']) * la_vis
    vel_video_y_la = np.diff(trj['left arm']['y']) * la_vis
    vel_video_z_la = np.diff(trj['left arm']['z']) * la_vis
    
    h_vis = (trj['head']['visibility'] > 0.2)
    h_vis = moving_average(h_vis, 4)[:-1]  
    vel_video_x_h = np.diff(trj['head']['x']) * h_vis
    vel_video_y_h = np.diff(trj['head']['y']) * h_vis
    vel_video_z_h = np.diff(trj['head']['z']) * h_vis

    # vel_video_x = np.diff(X_video_trj[('Navid', 1)][14,:,2]) * (X_video_trj[('Navid', 1)][14,:,6] >=0.3)[:-1]
    # vel_video_y = np.diff(X_video_trj[('Navid', 1)][14,:,3]) * (X_video_trj[('Navid', 1)][14,:,7] >=0.3)[:-1]
    # vel_video = -1*((20 + vel_video_x) ** 2 + (20 + vel_video_y) ** 2) ** 0.5

    # video_all_vects = np.array([vel_video_x_rh, vel_video_y_rh, vel_video_z_rh, vel_video_x_ra, vel_video_y_ra, vel_video_z_ra, vel_video_x_lh, vel_video_y_lh, vel_video_z_lh, vel_video_x_la
    #                             , vel_video_y_la, vel_video_z_la, vel_video_x_h, vel_video_y_h, vel_video_z_h])
    video_all_vects = np.array([vel_video_x_rh, vel_video_y_rh, vel_video_z_rh, vel_video_x_ra, vel_video_y_ra, vel_video_z_ra, vel_video_x_h, vel_video_y_h, vel_video_z_h])
    # video_all_vects = np.array([vel_video_x_rh, vel_video_y_rh, vel_video_z_rh])
    video_all_vects_padded = []
    for i in range(len(video_all_vects)):
        video_all_vects_padded.append(add_average_padding(video_all_vects[i], padding))
    
    video_all_vects = np.array(video_all_vects_padded)    
    return video_all_vects


def get_delay(CSI_vector, vel_video):
    
    CSI_vel = add_average_padding(CSI_vector, 100)
    video_vel = add_average_padding(vel_video, 100)

    CSI_vel = (CSI_vel - CSI_vel.mean()) / CSI_vel.std()
    
    video_vel = (video_vel - video_vel.mean()) / video_vel.std()
    
    corrs = []
    scores = []
    padded_a = add_average_padding(1*moving_average(video_vel,32), 0)
    padded_b = add_average_padding(1*moving_average(CSI_vel,32)[:-1], 0)
    for c in range(-200, 200):
      
        time_series_a = circular_shift(padded_a, c)
        time_series_b = padded_b
        corr = np.corrcoef(time_series_a, time_series_b)[0][1]
        # corr = mean_squared_error(time_series_a, time_series_b)
        corrs.append(corr)
        # distance, paths = dtw.warping_paths(time_series_a, time_series_b, use_c=False)
        # best_path = dtw.best_path(paths)
        # similarity_score = distance / len(best_path)
        # scores.append(0)
    
    corr = np.max(np.abs(corrs))
    corr_index = np.argmax(np.abs(corrs)) - 200
    sign = np.sign(corrs[200 + corr_index])
    return corr, corr_index, sign


# plt.figure();plt.plot(corrs_all)
def snr_periodogram_median(x: np.ndarray):
    """PSD-median blind SNR estimator via periodogram."""
    N = len(x)
    X = np.fft.fft(x)
    Pxx = np.abs(X) ** 2 / N
    Pn = np.median(Pxx)
    P_total = np.mean(np.abs(x) ** 2)
    Ps = P_total - Pn
    return 10 * np.log10(Ps / Pn)

def get_aligned_signals(sig1, sig2, corr_index, sign, padding=100):
    sig1_padded = add_average_padding(sig1, padding)
    sig2_padded = add_average_padding(sig2, padding)
    y1 = sign * moving_average(sig1_padded,1)       # First signal
    y2 = circular_shift(moving_average(sig2_padded,1),corr_index)   # Second signal with different amplitude
    y1 = y1[padding:-1 * padding]
    y2 = y2[padding:-1 * padding]

    return y1, y2

def get_CSI_Video_Aligned(VEL_CSI_ALL, X_video_trj, names_, ALL_APs, subject_id, action_id, window_num, do_plot = False):
    outputs = {}    
    for AP_ in ALL_APs:
        ref = 2
        video_all_vects = get_video_speed(X_video_trj, names_, AP_, subject_id, action_id, window_num)
        vel_video_x_rh, vel_video_y_rh = video_all_vects[0], video_all_vects[1]
        CSI_vector = VEL_CSI_ALL[AP_]
        vel_video_rh = 1 *0.5 * (np.abs(ref + 3*vel_video_x_rh) + np.abs(ref + vel_video_y_rh))
        corr, corr_index, sign = get_delay(CSI_vector, vel_video_rh)
        CSI_vector_aligned, vel_video_rh_aligned = get_aligned_signals(CSI_vector, vel_video_rh, corr_index, sign)      
        output = {}
        output['CSI'] = CSI_vector_aligned
        output['video'] = vel_video_rh_aligned
        output['delay'] = corr_index * 0.0
        output['sign'] = sign
        outputs[AP_] = output
        x = np.linspace(0, 10, 500)
        if do_plot:
            # Create the figure and the first y-axis
            w_num = 435
            y = Video_ALL[4][w_num][1]
            x = Video_ALL[4][w_num][0]
            v_vid = ((x+40) ** 2 + (y+40) **2) ** 0.5
            v_vid = v_vid - v_vid.mean()
            V_Video_opt = v_vid
            CSI_vector_aligned =Doppler_ALL[4][w_num] /5
            vel_video_rh_aligned=  V_Video_opt
            fig, ax1 = plt.subplots()
            import matplotlib
            xx = np.linspace(0,5, 500)
            # Plot the first signal on the first y-axis
            ax1.plot(xx, moving_average(CSI_vector_aligned, 64), 'g-', label='Signal 1 (sin)')
            ax1.set_xlabel('Time (sec)')
            ax1.set_ylabel('CSI Doppler Velocity (m/sec)', color='g')
            ax1.spines['left'].set_color('g')
            ax1.tick_params(axis='y', colors='g')
            # Create the second y-axis and plot the second signal
            ax2 = ax1.twinx()
            ax2.plot(xx, moving_average(vel_video_rh_aligned, 16), 'b-', label='Signal 2 (0.1*cos)')
            ax2.set_ylabel('Video Velocity (pixels/sec)', color='b')
            ax2.spines['right'].set_color('b')
            ax2.spines['left'].set_color('g')
            ax2.tick_params(axis='y', colors='b')

            # Show the plot with both signals
            # fig.tight_layout()  # Adjust layout to prevent overlap
            # title = "{} {} - AP {} | Corr Coef. {}".format(names_[subject_id], Gestures[action_id],AP_, np.round(corr*100,1))
            title = "Subject 2, Gesture: Left-right"
            plt.title(title)
            fig.canvas.manager.set_window_title(title)
            plt.grid()
            plt.savefig('leftright.png', dpi=300)
            plt.show()
    return outputs
            

def get_Video_processed(CSI_Video_Delays, X_video_trj, names_, ALL_APs, subject_id, action_id, window_num):
    outputs = {}
    for AP_ in ALL_APs:
        video_all_vects = get_video_speed(X_video_trj, names_, AP_, subject_id, action_id, window_num)
        video_all_vects_shifted = []
        for i in range(len(video_all_vects)):
            video_all_vects_p = add_average_padding(video_all_vects[i], 100)
            # video_all_vects_p_shifted = circular_shift(video_all_vects_p, CSI_Video_Delays[AP_]['delay'])
            # video_all_vects_p_shifted = video_all_vects_p_shifted[100:-100] * CSI_Video_Delays[AP_]['sign']
            video_all_vects_shifted.append(video_all_vects_p)
        video_all_vects = np.array(video_all_vects_shifted)    
        
        video_all_vects_smooth = []
        for i in range(len(video_all_vects)):
            video_all_vects_smooth.append(moving_average(video_all_vects[i], 8))
        video_all_vects = np.array(video_all_vects_smooth)    

        video_all_vects_resample = []
        for i in range(len(video_all_vects)):
            aaa = signal.resample(video_all_vects[i], 500)
            video_all_vects_resample.append(aaa)
        video_all_vects = np.array(video_all_vects_resample)   
        
        # video_all_vects_unit = []
        # for i in range(len(video_all_vects)):
        #     aaa = video_all_vects[i] - video_all_vects[i].mean()
        #     aaa = aaa / video_all_vects[i].std()
        #     video_all_vects_unit.append(aaa)
        # video_all_vects = np.array(video_all_vects_unit)   
        outputs[AP_] = video_all_vects
    return outputs

def normalize_CSI(CSI_signals, ALL_APs):
    CSI_signals_norm = {}
    for AP_ in ALL_APs:
        a = copy.copy(CSI_signals[AP_])
        a = a - a.mean()
        a = a / a.std()
        CSI_signals_norm[AP_] = a
    return CSI_signals_norm

def interp(ys, mul):
    # linear extrapolation for last (mul - 1) points
    ys = list(ys)
    ys.append(2*ys[-1] - ys[-2])
    # make interpolation function
    xs = np.arange(len(ys))
    fn = interp1d(xs, ys, kind="cubic")
    # call it on desired data points
    new_xs = np.arange(len(ys) - 1, step=1./mul)
    return fn(new_xs)

from sklearn.linear_model import Lasso

def sharp_phase_sanitization(x, fc=2.4e9, delta_f=312.5e3, T_sym=3.2e-6, P0=64, alpha=0.1):
    """
    SHARP phase sanitization for CSI matrix x of shape (N, T)
    
    Args:
        x : np.ndarray
            Complex CSI matrix of shape (N subcarriers, T time snapshots)
        fc : float
            Center frequency (Hz), default 5.32 GHz
        delta_f : float
            Subcarrier spacing (Hz), default for 802.11ac is 312.5 kHz
        T_sym : float
            OFDM symbol duration (s), default 3.2 us
        P0 : int
            Number of candidate paths in delay dictionary
        alpha : float
            Regularization parameter for Lasso (sparsity control)
    
    Returns:
        x_sanitized : np.ndarray
            Sanitized CSI matrix of same shape (N, T)
    """
    N, T = x.shape
    k_vals = np.arange(-N//2, N//2)

    # Delay grid (assume uniformly spaced delays)
    t_grid = np.linspace(0, T_sym, P0)

    # Build dictionary matrix T of shape (N, P0)
    Tk = np.exp(-1j * 2 * np.pi * np.outer(k_vals, t_grid) / T_sym)

    # Convert to real-valued matrix for Lasso
    Text = np.block([
        [Tk.real, -Tk.imag],
        [Tk.imag,  Tk.real]
    ])  # shape: (2N, 2P0)

    x_sanitized = np.zeros_like(x, dtype=complex)

    for t in range(T):
        h = x[:, t]
        h_ext = np.concatenate([h.real, h.imag])  # shape: (2N,)
        
        # Sparse recovery using Lasso
        lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
        lasso.fit(Text, h_ext)
        r_ext = lasso.coef_

        # Convert back to complex-valued vector r
        r = r_ext[:P0] + 1j * r_ext[P0:]

        # Identify strongest path
        p_star = np.argmax(np.abs(r))

        # Compute X_k vector
        X_k = Tk @ r  # shape: (N,)
        X_k_pstar = X_k[p_star]

        # Phase sanitization via conjugate of strongest path
        x_hat = X_k * np.conj(X_k_pstar)  # Hadamard product (element-wise)
        x_sanitized[:, t] = x_hat

    return x_sanitized 

def potts_baseline(y, beta):
    """
    Exact L0 jump-penalized baseline (Potts) using dynamic programming.

    Args:
        y (array-like): observed 1D signal length n
        beta (float): penalty per jump (larger => fewer jumps). Must be >= 0.
    Returns:
        x (np.ndarray): estimated piecewise-constant baseline
        cps (list[int]): change-point indices (1-based end indices of segments)
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if n == 0:
        return y.copy(), []

    # Prefix sums for fast SSE of any segment (i..j), inclusive, 1-based math
    Y = np.concatenate(([0.0], np.cumsum(y)))
    Y2 = np.concatenate(([0.0], np.cumsum(y**2)))

    def seg_cost(i, j):
        """0.5 * SSE of y[i:j] (1-based, inclusive) fitted by its mean."""
        m = j - i + 1
        s1 = Y[j] - Y[i-1]
        s2 = Y2[j] - Y2[i-1]
        # SSE = sum (y - mean)^2 = sum y^2 - (sum y)^2 / m
        sse = s2 - (s1*s1)/m
        return 0.5 * sse

    # DP arrays
    F = np.empty(n+1)          # F[t] = optimal cost up to t
    prev = np.empty(n+1, int)  # prev[t] = previous segment end index
    F[0] = -beta               # trick so first segment doesn't pay beta
    prev[0] = 0

    for t in range(1, n+1):
        # Evaluate all candidate start points s in [1..t]
        best_cost = np.inf
        best_s = 0
        # try splitting at s-1 | s..t is a segment
        for s in range(1, t+1):
            cost = F[s-1] + seg_cost(s, t) + beta
            if cost < best_cost:
                best_cost = cost
                best_s = s-1
        F[t] = best_cost
        prev[t] = best_s

    # Backtrack segments
    cps = []
    t = n
    while t > 0:
        s = prev[t] + 1
        cps.append(t)  # segment end
        t = prev[t]
    cps = list(reversed(cps))

    # Build baseline
    x = np.empty(n)
    start = 0
    for end in cps:
        seg = slice(start, end)
        x[seg] = np.mean(y[seg])
        start = end
    return x, cps

# In[4]:
from pathlib import Path
import glob
import numpy as np
import pickle

subjects = ['Ali_Arasteh', 'Alireza_Keshavarzian', 'Mohammad_Sheikholeslami', 'Navid_Hasanzadeh', 'Shiva_Akbari', 'Adnan_Hamida', 'Ahmed', 'Ali_Parchekani', 'Steven_Sun']
subject_idx = 0

session = 2
for sniffer_idx in [3,5,6,1,2,4]:
    subject_name = subjects[subject_idx]
    root_directory = "E:/SinglePerson/" + subject_name + "/" + "Session0" + str(session) + "/"
    
    
    files_list = list(glob.glob(root_directory + "/*/" + "*.pickle"))
    file_target = [f for f in files_list if "Sniffer_{}".format(sniffer_idx) in f][0]
    with open(Path(file_target), 'rb') as handle:
    # with open(Path(r"G:/Dataset/Jan 25/Navid_30-40_945716_e57dabaa52d7_20260125_152748_788/raw_data_PIDe57dabaa52d7_EXP945716/Navid_e57dabaa52d7_945716_20260125_153954_Sniffer_6.pickle"), 'rb') as handle:
        data, activities, info = pickle.load(handle)
    
    print(f"Loaded {len(data)} entries.")
    print(f"Activities: {activities}")
    print(f"Info: {info}")
    
    # In[9]:
    
    
    # names = ['Navid']
    
    # # names = ["subject5_rd"]
    # # names_ = ["Radomir"]
    
    # ALL_APs = [0,1,2,3,4,5]
    # Orien = 180
    # Gestures = {0: activities[0], 1: activities[1], 2: activities[2], 3: activities[3], 4: activities[4]}
    
    # In[10]:
    
    
    # In[12]:
    
    def compute_delay_profile(csi_reduced, bandwidth=20e6):
        """
        Compute the Power Delay Profile (PDP) from reduced CSI data.
        
        Parameters:
        - csi_reduced: numpy array of shape (52, num_times), complex-valued CSI after fftshift and removal of non-informative subcarriers.
        - bandwidth: float, channel bandwidth in Hz (default: 20e6 for 20 MHz).
        
        Returns:
        - pdp: numpy array of shape (64, num_times), the power delay profile.
        - delays: numpy array of shape (64,), delay values in seconds.
        """
        # Define all indices (0-63)
        all_indices = np.arange(64)
        
        # Removed indices (as specified)
        removed = np.array([0,1,2,3,4,5,32,59,60,61,62,63])
        
        # Kept indices
        kept = np.setdiff1d(all_indices, removed)
        
        # Reconstruct shifted array
        num_times = csi_reduced.shape[1]
        H_shifted = np.zeros((64, num_times), dtype=complex)
        H_shifted[kept, :] = csi_reduced
        
        # Restore to standard order for IFFT
        H = np.fft.ifftshift(H_shifted, axes=0)
        
        # Compute CIR via IFFT
        cir = np.fft.ifft(H, n=64, axis=0)
        
        # Compute PDP
        pdp = np.sum(np.abs(cir)**2,1)
        
        # Compute delay axis
        delta_t = 1 / bandwidth
        delays = np.arange(64) * delta_t
        
        return cir, pdp, delays
    
    import numpy as np
    
    def align_and_resample(signals, times, fs=100.0,
                           range_mode="intersection",
                           t_min=None, t_max=None,
                           extrapolate="edge",
                           pad_to=None,
                           interp_kind="linear"):
        """
        Align and resample multiple signals to a common regular grid (default 100 Hz),
        filling missing samples via interpolation. Optionally pad all outputs to a
        specific length with edge values, and return indices mapping original samples
        to the resampled grid.
    
        Parameters
        ----------
        signals : list of 1D array-like
            signals[i] are the samples for the i-th signal (same length as times[i]).
        times : list of 1D array-like
            times[i] are the time stamps (seconds) for signals[i]. They may be irregular
            with missing samples; function will sort them and drop duplicate timestamps.
        fs : float, default 100.0
            Target sampling rate in Hz. (dt = 1/fs)
        range_mode : {"intersection", "union"}, default "intersection"
            - "intersection": use the time range common to all inputs (avoids extrapolation).
            - "union": use the full span across all inputs (may require extrapolation).
        t_min, t_max : float or None
            Optional explicit start/end (seconds) for the output grid. If provided, they
            override `range_mode`. If only one is provided, the other is inferred from
            `range_mode`.
        extrapolate : {"edge", "nan"}, default "edge"
            Behavior outside each signal's time span (for interpolation step):
            - "edge": hold the nearest edge value (zero-order hold).
            - "nan": use NaN outside the known range.
        pad_to : int or None, default None
            If provided and the resampled length is less than this number, pad all signals
            symmetrically (equally on both sides, +/-1 sample if odd) using EDGE values.
            The returned time grid is extended accordingly using the same `fs`.
        interp_kind : {"linear", "nearest"}, default "linear"
            Interpolation type:
            - "linear": piecewise-linear interpolation (np.interp).
            - "nearest": nearest-neighbor interpolation.
    
        Returns
        -------
        Y : (N_signals, N_samples_out) ndarray
            Resampled (and possibly padded) signals aligned on the common grid.
        t_grid : (N_samples_out,) ndarray
            The common regular time grid (seconds).
        original_indices : list of 1D ndarrays
            original_indices[i][k] is the index into t_grid (and Y[i])
            corresponding (by nearest time index) to the k-th original sample time
            in times[i], after cleaning and sorting. If padding is applied, indices
            already account for the left padding offset.
    
            Note: if your timestamps lie exactly on the regular 1/fs grid, then
            Y[i, original_indices[i]] will exactly recover the original samples.
            If timestamps are slightly jittered, this recovers them approximately.
    
        Notes
        -----
        - Interpolation is linear or nearest between known samples.
        - Padding (if applied) always uses edge values, independent of `extrapolate`.
        """
        if len(signals) != len(times):
            raise ValueError("signals and times must have the same length (one time vector per signal).")
    
        if interp_kind not in {"linear", "nearest"}:
            raise ValueError("interp_kind must be 'linear' or 'nearest'.")
    
        # Clean and sort each (t, x) pair; drop non-finite and duplicate t
        cleaned = []
        t_mins, t_maxs = [], []
        for x, t in zip(signals, times):
            t = np.asarray(t, dtype=float)
            x = np.asarray(x, dtype=float)
            if t.shape != x.shape:
                raise ValueError("Each signals[i] and times[i] must have the same shape.")
            mask = np.isfinite(t) & np.isfinite(x)
            t = t[mask]
            x = x[mask]
            if t.size < 2:
                raise ValueError("Each signal needs at least two finite (time, value) points for interpolation.")
            # sort by time
            idx = np.argsort(t)
            t, x = t[idx], x[idx]
            # unique times (keep first occurrence)
            t_unique, first_idx = np.unique(t, return_index=True)
            x = x[first_idx]
            t = t_unique
            cleaned.append((t, x))
            t_mins.append(t[0])
            t_maxs.append(t[-1])
    
        dt = 1.0 / fs
    
        # Determine output range
        if t_min is None or t_max is None:
            if range_mode not in {"intersection", "union"}:
                raise ValueError("range_mode must be 'intersection' or 'union'.")
            if t_min is None:
                t_min = max(t_mins) if range_mode == "intersection" else min(t_mins)
            if t_max is None:
                t_max = min(t_maxs) if range_mode == "intersection" else max(t_maxs)
    
        if t_max <= t_min:
            raise ValueError("t_max must be greater than t_min after resolving range.")
    
        # Build regular time grid
        n_samples = int(np.floor((t_max - t_min) * fs)) + 1
        t_grid = t_min + np.arange(n_samples) * dt
    
        # Map original samples to nearest grid indices (before padding)
        original_indices = []
        for (t, x) in cleaned:
            idx = np.round((t - t_min) / dt).astype(int)
            idx = np.clip(idx, 0, n_samples - 1)
            original_indices.append(idx)
    
        # Interpolate each signal onto the grid
        Y_list = []
        for (t, x) in cleaned:
            if interp_kind == "linear":
                if extrapolate == "edge":
                    left, right = float(x[0]), float(x[-1])
                elif extrapolate == "nan":
                    left = right = np.nan
                else:
                    raise ValueError("extrapolate must be 'edge' or 'nan'.")
    
                y = np.interp(t_grid, t, x, left=left, right=right)
    
            else:  # nearest neighbor
                y = np.empty_like(t_grid, dtype=float)
                if extrapolate == "nan":
                    y.fill(np.nan)
                    valid = (t_grid >= t[0]) & (t_grid <= t[-1])
                elif extrapolate == "edge":
                    # all grid points treated as valid; nearest neighbor with clamping
                    valid = np.ones_like(t_grid, dtype=bool)
                else:
                    raise ValueError("extrapolate must be 'edge' or 'nan'.")
    
                # For valid points, find nearest neighbor in t
                idx_right = np.searchsorted(t, t_grid[valid], side='left')
                idx_right = np.clip(idx_right, 0, len(t) - 1)
                idx_left = np.clip(idx_right - 1, 0, len(t) - 1)
    
                t_valid = t_grid[valid]
                dist_left = np.abs(t_valid - t[idx_left])
                dist_right = np.abs(t_valid - t[idx_right])
    
                use_right = dist_right < dist_left
                idx_nearest = np.where(use_right, idx_right, idx_left)
    
                y[valid] = x[idx_nearest]
    
                if extrapolate == "edge":
                    # nearest with clipping already handles edges
                    pass
    
            Y_list.append(y)
    
        Y = np.vstack(Y_list)
    
        # Optional symmetric padding with EDGE values
        if pad_to is not None:
            if pad_to <= 0:
                raise ValueError("pad_to must be a positive integer.")
            if n_samples < pad_to:
                pad_needed = pad_to - n_samples
                pad_left = pad_needed // 2
                pad_right = pad_needed - pad_left
    
                # Extend time grid
                if pad_left > 0:
                    pre_times = t_grid[0] - np.arange(pad_left, 0, -1) * dt
                else:
                    pre_times = np.empty(0, dtype=float)
                if pad_right > 0:
                    post_times = t_grid[-1] + np.arange(1, pad_right + 1) * dt
                else:
                    post_times = np.empty(0, dtype=float)
                t_grid = np.concatenate((pre_times, t_grid, post_times), axis=0)
    
                # Pad signals with edge values (independent of `extrapolate`)
                Y = np.vstack([
                    np.pad(y, (pad_left, pad_right), mode="edge")
                    for y in Y
                ])
    
                # Shift original indices to account for left padding
                original_indices = [idx + pad_left for idx in original_indices]
    
        return Y, t_grid, original_indices
    
    def viterbi_2state_gaussian(obs, muA, muB, Sigma, p_stay=0.995):
        """
        obs: (T, 2) array of observations
        State 0 (A): obs ~ N(muA, Sigma)
        State 1 (B): obs ~ N(muB, Sigma)
        Returns: most likely state path, shape (T,)
        """
        T = obs.shape[0]
        p_change = 1 - p_stay
        A = np.array([[p_stay,  p_change],
                      [p_change, p_stay ]])
    
        # precompute stuff for Gaussian logpdf
        SigInv = np.linalg.inv(Sigma)
        sign, logdet = np.linalg.slogdet(Sigma)
        const = -0.5 * (2 * np.log(2 * np.pi) + logdet)
    
        def logpdf(x, mu):
            d = x - mu
            return const - 0.5 * d.T @ SigInv @ d
    
        logdelta = np.zeros((T, 2))
        psi = np.zeros((T, 2), dtype=int)
    
        # init: equal priors
        logdelta[0, 0] = logpdf(obs[0], muA)
        logdelta[0, 1] = logpdf(obs[0], muB)
    
        for t in range(1, T):
            for s in (0, 1):
                prev = logdelta[t-1] + np.log(A[:, s])
                best_prev = np.argmax(prev)
                psi[t, s] = best_prev
                logdelta[t, s] = prev[best_prev] + (logpdf(obs[t], muA) if s == 0 else logpdf(obs[t], muB))
    
        # backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(logdelta[-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
    
        return states
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import butter, filtfilt
    
    def plot_fft_energy_and_lowpass(
        x,
        fs,
        fc,
        order=4,
        nfft=None,
        window="hann",
        detrend=True,
        one_sided=True,
        db=True,
        eps=1e-12,
        bin_hz=5.0,
        visualize=True,
    ):
        """
        Given a signal x and sampling rate fs:
          1) Plot FFT magnitude and spectral energy (original).
          2) Apply zero-phase Butterworth low-pass via filtfilt with cutoff fc.
          3) Plot time-domain signals (original vs filtered).
          4) Plot FFT magnitude and spectral energy (filtered).
          5) Normalize spectra and return energy in fixed-width frequency bins.
    
        Parameters
        ----------
        x : array-like
            Input signal (1D).
        fs : float
            Sampling rate in Hz.
        fc : float
            Low-pass cutoff frequency in Hz (0 < fc < fs/2).
        order : int
            Butterworth filter order.
        nfft : int or None
            FFT length. If None, uses next power of 2 >= len(x).
        window : {"hann","hamming","blackman",None} or array-like
            Window applied before FFT.
        detrend : bool
            If True, remove mean before FFT/filtering.
        one_sided : bool
            If True, use one-sided spectrum for real signals.
        db : bool
            If True, plot magnitude in dB and energy in dB.
        eps : float
            Small constant to avoid log(0).
        bin_hz : float
            Bin width in Hz for returned binned energy.
        visualize : bool
            If True, show plots. If False, perform computations and return values only.
    
        Returns
        -------
        dict with keys:
            "t", "x", "x_filt", "f",
            "X", "X_filt",
            "mag_norm", "mag_norm_filt",
            "energy_norm", "energy_norm_filt",
            "energy_bins_hz" (list of dicts for original),
            "energy_bins_hz_filt" (list of dicts for filtered),
            "filter_ba"
        Notes
        -----
        - Spectra are normalized so that sum(|X|^2) = 1 (within the plotted spectrum).
        - "energy bins" are sums of normalized energy over frequency intervals of width bin_hz.
        """
        x = np.asarray(x).ravel()
        if x.size < 2:
            raise ValueError("Signal must have at least 2 samples.")
        if fs <= 0:
            raise ValueError("Sampling rate fs must be positive.")
        if not (0 < fc < fs / 2):
            raise ValueError("Cutoff fc must satisfy 0 < fc < fs/2.")
        if bin_hz <= 0:
            raise ValueError("bin_hz must be positive.")
    
        N = x.size
        t = np.arange(N) / fs
    
        if detrend:
            x = x - np.mean(x)
    
        # ----- Design + apply low-pass (zero-phase) -----
        wn = fc / (fs / 2.0)  # normalized cutoff
        b, a = butter(order, wn, btype="low", analog=False)
        x_filt = filtfilt(b, a, x)
    
        # ----- Window for FFT -----
        if window is None:
            w = np.ones(N)
            w_name = "rect"
        elif isinstance(window, str):
            win = window.lower()
            if win == "hann":
                w = np.hanning(N)
            elif win == "hamming":
                w = np.hamming(N)
            elif win == "blackman":
                w = np.blackman(N)
            else:
                raise ValueError(f"Unsupported window '{window}'. Use hann/hamming/blackman/None or an array.")
            w_name = win
        else:
            w = np.asarray(window).ravel()
            if w.size != N:
                raise ValueError("Custom window must have the same length as the signal.")
            w_name = "custom"
    
        # FFT length
        if nfft is None:
            nfft = 1 << int(np.ceil(np.log2(N)))
        if nfft < N:
            raise ValueError("nfft must be >= len(signal).")
    
        def _spectrum(x_in):
            xw = x_in * w
            if one_sided and np.isrealobj(xw):
                X = np.fft.rfft(xw, n=nfft)
                f = np.fft.rfftfreq(nfft, d=1.0 / fs)
            else:
                X = np.fft.fft(xw, n=nfft)
                f = np.fft.fftfreq(nfft, d=1.0 / fs)
                idx = np.argsort(f)
                f = f[idx]
                X = X[idx]
            mag = np.abs(X)
            energy = mag**2
            return f, X, mag, energy
    
        def _normalize_energy(energy):
            s = float(np.sum(energy))
            if s <= 0:
                return energy, 0.0
            return energy / s, s
    
        def _normalize_mag_from_energy(energy_norm):
            return np.sqrt(np.maximum(energy_norm, 0.0))
    
        def _bin_energy(f_axis, energy_norm, bin_width_hz):
            f_min = float(f_axis[0])
            f_max = float(f_axis[-1])
    
            start = 0.0 if f_min >= 0 else np.floor(f_min / bin_width_hz) * bin_width_hz
            end = np.ceil(f_max / bin_width_hz) * bin_width_hz
            edges = np.arange(start, end + bin_width_hz, bin_width_hz)
    
            bins = []
            for i in range(len(edges) - 1):
                lo, hi = float(edges[i]), float(edges[i + 1])
                mask = (f_axis >= lo) & (f_axis < hi) if i < len(edges) - 2 else (f_axis >= lo) & (f_axis <= hi)
                e = float(np.sum(energy_norm[mask]))
                bins.append({"f_lo_hz": lo, "f_hi_hz": hi, "energy": e})
            return bins
    
        f, X, mag, energy = _spectrum(x)
        f2, Xf, mag_f, energy_f = _spectrum(x_filt)
    
        if f2.shape != f.shape or np.max(np.abs(f2 - f)) > 0:
            f = f2
    
        # ----- Normalize spectra -----
        energy_norm, energy_sum = _normalize_energy(energy)
        energy_norm_f, energy_sum_f = _normalize_energy(energy_f)
    
        mag_norm = _normalize_mag_from_energy(energy_norm)
        mag_norm_f = _normalize_mag_from_energy(energy_norm_f)
    
        # dB scaling for plots (use normalized mags/energies)
        if db:
            mag_p = 20.0 * np.log10(mag_norm + eps)
            mag_f_p = 20.0 * np.log10(mag_norm_f + eps)
            energy_p = 10.0 * np.log10(energy_norm + eps)
            energy_f_p = 10.0 * np.log10(energy_norm_f + eps)
            mag_ylabel = "Normalized magnitude (dB)"
            energy_ylabel = "Normalized energy (dB)"
        else:
            mag_p, mag_f_p = mag_norm, mag_norm_f
            energy_p, energy_f_p = energy_norm, energy_norm_f
            mag_ylabel = "Normalized magnitude"
            energy_ylabel = "Normalized energy"
    
        # ----- Visualization (optional) -----
        if visualize:
            # Time-domain
            plt.figure()
            plt.plot(t, x, label="Original")
            plt.plot(t, x_filt, label=f"Low-pass (fc={fc} Hz, order={order})", linewidth=2)
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.title("Signal (Time Domain)")
            plt.grid(True)
            plt.legend()
    
            # Original spectrum
            plt.figure()
            plt.plot(f, mag_p)
            plt.axvline(fc, linestyle="--", label="Cutoff fc")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel(mag_ylabel)
            plt.title(f"Original FFT Magnitude (normalized, window={w_name}, nfft={nfft})")
            plt.grid(True)
            plt.legend()
    
            plt.figure()
            plt.plot(f, energy_p)
            plt.axvline(fc, linestyle="--", label="Cutoff fc")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel(energy_ylabel)
            plt.title(f"Original Spectral Energy (normalized, window={w_name}, nfft={nfft})")
            plt.grid(True)
            plt.legend()
    
            # Filtered spectrum
            plt.figure()
            plt.plot(f, mag_f_p)
            plt.axvline(fc, linestyle="--", label="Cutoff fc")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel(mag_ylabel)
            plt.title(f"Filtered FFT Magnitude (normalized, window={w_name}, nfft={nfft})")
            plt.grid(True)
            plt.legend()
    
            plt.figure()
            plt.plot(f, energy_f_p)
            plt.axvline(fc, linestyle="--", label="Cutoff fc")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel(energy_ylabel)
            plt.title(f"Filtered Spectral Energy (normalized, window={w_name}, nfft={nfft})")
            plt.grid(True)
            plt.legend()
    
            plt.show()
    
        # ----- Bin energies -----
        energy_bins = _bin_energy(f, energy_norm, bin_hz)
        energy_bins_f = _bin_energy(f, energy_norm_f, bin_hz)
    
        return {
            "t": t,
            "x": x,
            "x_filt": x_filt,
            "f": f,
            "X": X,
            "X_filt": Xf,
            "mag_norm": mag_norm,
            "mag_norm_filt": mag_norm_f,
            "energy_norm": energy_norm,
            "energy_norm_filt": energy_norm_f,
            "energy_bins_hz": energy_bins,
            "energy_bins_hz_filt": energy_bins_f,
            "filter_ba": (b, a),
            "energy_sum_raw": energy_sum,
            "energy_sum_raw_filt": energy_sum_f,
            "bin_hz": float(bin_hz),
        }
    
    
    def recover_bias_swaps_hmm(y1, y2, p_stay=0.995, sigma=0.15, visualize=True):
        """
        We assume:
          - channel 1 has waveform s1(t)
          - channel 2 has waveform s2(t)
          - BUT sometimes the *biases* of the two channels are swapped.
        So in state 0: y1 = s1 + b_high, y2 = s2 + b_low
           in state 1: y1 = s1 + b_low,  y2 = s2 + b_high
    
        We don't know s1, s2, b_low, b_high.
        We'll:
          1) estimate b_low, b_high from all data (like before),
          2) model emissions on the *demeaned* signals? No — simpler:
             since the waveforms can be different, we model
             (y1 - s1_est, y2 - s2_est) where s*_est are smoothed versions,
             OR we just directly model (y1, y2) but let sigma be larger.
        Here we'll keep it simple: model (y1, y2) directly.
        Then we correct the biases according to the inferred state.
        """
        y1 = np.asarray(y1).ravel()
        y2 = np.asarray(y2).ravel()
        T = len(y1)
        obs = np.stack([y1, y2], axis=1)
    
        # 1) Estimate two bias levels from all data pooled
        all_vals = np.concatenate([y1, y2])[:, None]
        # 1D k-means for 2 clusters
        c = np.array([all_vals.min(), all_vals.max()], float)
        for _ in range(30):
            d = np.abs(all_vals - c[None, :])
            lab = d.argmin(axis=1)
            new_c = []
            for k in (0, 1):
                if np.any(lab == k):
                    new_c.append(all_vals[lab == k].mean())
                else:
                    new_c.append(c[k])
            c = np.array(new_c)
        b_low, b_high = np.sort(c)
    
        # 2) Emission means for the two states
        # state 0: channel1 high, channel2 low
        muA = np.array([b_high, b_low])
        # state 1: channel1 low, channel2 high
        muB = np.array([b_low, b_high])
    
        # 3) Covariance
        Sigma = np.array([[sigma**2, 0.0],
                          [0.0,      sigma**2]])
    
        # 4) Viterbi to find when biases got swapped
        states = viterbi_2state_gaussian(obs, muA, muB, Sigma, p_stay=p_stay)
    
        # 5) Reconstruct signals so that
        #    channel 1 ALWAYS has high bias
        #    channel 2 ALWAYS has low bias
        # current bias on ch1 = b_high if state==0 else b_low
        current_bias_ch1 = np.where(states == 0, b_high, b_low)
        current_bias_ch2 = np.where(states == 0, b_low,  b_high)
    
        # remove current bias, add desired bias
        y1_rec = y1 - current_bias_ch1 + b_high
        y2_rec = y2 - current_bias_ch2 + b_low
    
        if visualize:
            t = np.arange(T)
            fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    
            axs[0].plot(t, y1, label='observed y1')
            axs[0].plot(t, y2, label='observed y2', alpha=0.7)
            axs[0].set_title("Observed signals (same waveforms, sometimes bias-swapped)")
            axs[0].legend()
    
            axs[1].plot(t, states, drawstyle='steps-post')
            axs[1].set_title("Inferred HMM state: 0 = (ch1 high, ch2 low), 1 = (ch1 low, ch2 high)")
            axs[1].set_ylim(-0.2, 1.2)
    
            axs[2].plot(t, y1_rec, label='recovered y1 (forced high bias)')
            axs[2].plot(t, y2_rec, label='recovered y2 (forced low bias)', alpha=0.7)
            axs[2].set_title("Bias-corrected signals")
            axs[2].legend()
    
            axs[3].hlines([b_low, b_high], 0, T-1, colors=['gray', 'black'], linestyles='dashed')
            axs[3].set_title(f"Estimated bias levels: low={b_low:.2f}, high={b_high:.2f}")
            axs[3].set_ylim(min(all_vals)-0.5, max(all_vals)+0.5)
    
            plt.tight_layout()
            plt.show()
    
        return y1_rec, y2_rec, states, (b_low, b_high)
    
    
    def CSI_phase_bias_correction(csi_modem, csi_pi, sample_data_time_modem, sample_data_time):
          z_pi = []
          z_modem = []
          z_sum = []
          z_sum_abs = []
          print("In: ", len(sample_data_time_modem), id(sample_data_time_modem))
    
          for i in range(52):
              z_sum_abs.append(np.median(np.abs(csi_pi[i,:])))
              x1 = (np.angle(csi_modem[i,:]).T ).T
              x2 = ((np.angle(csi_pi[i,:])).T ).T
              
              Y, t_grid, original_indices = align_and_resample([np.unwrap(x1),np.unwrap(x2)], [sample_data_time_modem, sample_data_time],interp_kind="linear", fs=100, pad_to=500)
              csi_aligned_pi = Y[1]
              csi_aligned_modem = Y[0]
              csi_aligned_sum = 0.5 * ((Y[0]) + (Y[1]))
              Y[0] = np.angle(np.exp(1j * Y[0]))
              Y[1] = np.angle(np.exp(1j * Y[1]))
              csi_aligned_sum = np.angle(np.exp(1j * csi_aligned_sum))
              csi_aligned_pi = Y[1]
              csi_aligned_modem = Y[0]
              
              y1_rec, y2_rec, states_est, (b_low_est, b_high_est) = recover_bias_swaps_hmm(csi_aligned_pi, csi_aligned_modem, visualize=False)
              # y1_rec = csi_aligned_pi * np.array(states_est) + csi_aligned_modem * (1-np.array(states_est))
              # y2_rec = csi_aligned_pi * (1-np.array(states_est)) + csi_aligned_modem * np.array(states_est)
    
              # print(states_est)
              y1_rec = hampel(y1_rec, window_size=16).filtered_data
              y2_rec = hampel(y2_rec, window_size=16).filtered_data
    
              y_modem = y2_rec[original_indices[0]]
              y_pi = y1_rec[original_indices[1]]
              y_modem = hampel(y_modem, window_size=16).filtered_data
              y_pi = hampel(y_pi, window_size=16).filtered_data
              csi_aligned_sum = csi_aligned_sum[original_indices[1]]
              csi_aligned_sum = hampel(csi_aligned_sum, window_size=16).filtered_data
              # csi_aligned_sum = moving_average(csi_aligned_sum, 8)
              z_pi.append(y_pi)
              z_sum.append(csi_aligned_sum)
              z_modem.append(y_modem)
              
          z_pi = np.abs(csi_pi) * np.exp(1j * np.array(z_pi))
          z_modem = np.abs(csi_modem) * np.exp(1j * np.array(z_modem))
          z_sum = (1 * np.exp(1j * np.array(z_sum)))
          for i in range(52):
              z_sum[i] = z_sum[i] * z_sum_abs[i]
          return z_sum, z_pi, z_modem
    
    import numpy as np
    
    def clean_asus_5ghz_csi(csi_raw):
        """
        Input:  csi_raw   shape (256, N_packets)   complex128 / complex64
        Output: cleaned   shape (242, N_packets)
                (null subcarriers removed, order preserved)
        """
        if csi_raw.shape[0] != 256:
            print("Warning: expected 256 subcarriers, got", csi_raw.shape[0])
            return csi_raw
    
        # ASUS/Intel 80 MHz 5GHz most common pattern
        # Keep: 3~63  and  65~191  (total 61 + 127 = 188? wait → actually different mapping)
    
        # Most popular & empirically confirmed mapping for many ASUS + Intel CSI Tool captures:
        valid_indices = list(range(3,  61))   + list(range(63,  127)) + \
                        list(range(129, 189)) + list(range(191, 253))
    
        #  → 58 + 64 + 60 + 62 = 244 ?  ← sometimes slightly different
    
        # Most accurate commonly used version (242 subcarriers):
        valid_indices = [
            *range(  3,  61),   # left part
            *range( 63, 127),   # right part 1
            *range(129, 189),   # right part 2
            *range(191, 253)    # right part 3
        ]
    
        # len(valid_indices) should be ≈ 242
        print(f"Selected {len(valid_indices)} subcarriers")
    
        cleaned = csi_raw[valid_indices, :]
    
        return cleaned
    
    
    # # Usage example:
    # csi = data[(names[subject_id], "Sniffer_" + str(AP+1), Env, str(window_num), activities[action_id])]['csi'][:,:,0,0].T
    # # .T because you want shape (subcarriers, packets)
    
    # csi_clean = clean_asus_5ghz_csi(csi)
    
    
    # ───────────────────────────────────────────────────────────────
    # Do you want centered (dc at index ≈121) like in many papers?
    # ───────────────────────────────────────────────────────────────
    
    def clean_and_fftshift_asus(csi_raw):
        csi_clean = clean_asus_5ghz_csi(csi_raw)
        # Most papers prefer negative → positive frequency order
        csi_shifted = np.fft.fftshift(csi_clean, axes=0)
        return csi_shifted
    
    
    # csi_final = clean_and_fftshift_asus(csi)
    
    import numpy as np
    from sklearn.linear_model import RANSACRegressor
    # Assume you have these helper functions already defined:
    # def P2R(mag, phase): return mag * np.exp(1j * phase)
    # def hampel(...)  # your outlier filter implementation
    
    def ant_processing_5ghz(x, use_ransac=True, hampel_window=7):
        """
        Phase sanitization adapted for 5 GHz CSI (Intel AX200/AX210 style, ~242 subcarriers)
        
        Parameters:
            x: complex ndarray, shape (N_subcarriers, N_packets)
            use_ransac: whether to use RANSAC instead of simple polyfit (recommended)
            hampel_window: increased for 5GHz (less aggressive outlier removal)
        """
        x = x.copy()           # important!
        N_sub, N_packets = x.shape
        
        # ─── First part: per-packet phase slope removal (CFO/SFO/PDD compensation) ───
        sub_idx = np.arange(N_sub)  # 0,1,2,...,N_sub-1
        
        for i in range(N_packets):
            mag   = np.abs(x[:, i])
            phase = np.unwrap(np.angle(x[:, i]))   # crucial: unwrap first!
            
            if use_ransac:
                ransac = RANSACRegressor(random_state=42, max_trials=100)
                ransac.fit(sub_idx.reshape(-1, 1), phase)
                trend = ransac.predict(sub_idx.reshape(-1, 1))
            else:
                # Fallback: simple linear fit (sometimes still used)
                coef = np.polyfit(sub_idx, phase, 1)
                trend = np.polyval(coef, sub_idx)
            
            phase_detrended = phase - trend
            
            x[:, i] = P2R(mag, phase_detrended)
        
        # ─── Second part: per-subcarrier time-domain cleaning ───
        # Usually much lighter on 5 GHz / 802.11ax because of higher time resolution & different noise
        for i in range(N_sub):
            phase = np.angle(x[i, :])
            mag   = np.abs(x[i, :])
            
            # Option A - Hampel outlier removal (recommended starting point)
            phase_clean = hampel(phase, window_size=hampel_window).filtered_data
            
            # Option B - more gentle smoothing (often better on 5GHz)
            # from scipy.signal import savgol_filter
            # phase_clean = savgol_filter(phase, window_length=9, polyorder=2)
            
            # Option C - almost no filtering → many recent ax pipelines do this
            # phase_clean = phase
            
            x[i, :] = P2R(mag, phase_clean)
        
        return x
    
    def method1(csi, delta_f=312.5e3, csd_table=None):
        """
        Method 1: Reverse Known CSD Table.
        Compensates for per-stream CSD phase ramp, then fits and subtracts a linear model per stream.
        
        Args:
        - csi: Complex CSI array (time, subcarriers, rx, tx).
        - delta_f: Subcarrier spacing in Hz (default: 312.5 kHz).
        - csd_table: List of CSD delays in seconds per stream (default: standard for 4 streams).
        
        Returns:
        - Cleaned phase array (same shape as input phase).
        """
        if csd_table is None:
            csd_table = [0, -400e-9, -200e-9, -600e-9]  # Standard ns to sec
        phase = np.angle(csi)
        unwrapped = np.unwrap(phase, axis=1)  # Unwrap along subcarriers
        time, subs, rx, txs = csi.shape
        sub_idx = np.arange(-subs//2, subs//2)  # Centered indexing
        cleaned_phase = np.zeros_like(unwrapped)
        for t in range(time):
            for r in range(rx):
                for s in range(txs):
                    tau = csd_table[s]
                    phi_csd = -2 * np.pi * delta_f * tau * sub_idx
                    compensated = unwrapped[t, :, r, s] - phi_csd
                    coeffs = np.polyfit(sub_idx, compensated, 1)
                    linear = np.polyval(coeffs, sub_idx)
                    cleaned_phase[t, :, r, s] = compensated - linear
        return cleaned_phase
    
    Doppler_ALL = {0: [], 1:[], 2:[], 3:[], 4:[], 5:[]}
    Video_ALL = {0: [], 1:[], 2:[], 3:[], 4:[], 5:[]}
    # Env = "demo1"
    from nerfs2 import estimate_velocity_from_radial_inside_sphere, estimate_velocity_from_radial_local_sampling, estimate_velocity_from_radial_gaussian_neighbours_fused, estimate_velocity_from_radial_gaussian_neighbours, estimate_velocity_from_radial,estimate_velocity_from_radial_v20,estimate_velocity_from_radial_old, estimate_velocity_from_radial_old_dtw
    
    Env =  'Setup_6AP_4UP_LOS_2_DOWN_NOLOS'
    activities = ["_".join(act.split(" ")) for act in activities]
    sorted_keys = sorted(data.keys(), key=lambda k: int(k[3]) )
    X_proj = []
    X_3D = []
    X_dopplers = []
    X_mag = []
    y = []
    for key in tqdm(sorted_keys):          
                # try:
                        abc_all_APs = []
                        power_all_APs = []
                        # matplotlib.use('TkAgg')
                        ants_ = []
                        ants_null= []
                        # CSI = copy.copy(data[(names[subject_id], "Sniffer_" + str(AP+1), Env, str(window_num), activities[action_id])]['csi'])
                        # key = sorted_keys[2]
                        CSI = copy.copy(data[key])['csi']
                        # weights, bf_csi = compute_activity_beamformers(CSI, normalize=True, per_subcarrier=False)
                        # CSI = bf_csi
                        # plt.figure();plt.plot(hampel(np.angle(CSI[:,17,2,2]/CSI[:,17,2,1]),8).filtered_data, 'r');plt.xlabel('Time (sample)');plt.ylabel('CSI Ratio Phase');plt.title("CSI Ratio [RX3,TX3]/[RX3,TX1] - " + key[1])
                        
                        # X_mag.append(np.abs(CSI))
    
                        dopplers = []
                        CSI_ratio_scores_all = []
                        for tx in ([(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]):
                            for rx in [0,1,2,3]:
                                subant = 0
                                CSI_ratio = [] 
                                CSI_ratio_scores = []
                                for subant in range(4):
                                    CSI_1 = CSI[:,:,rx,tx[0]]
                                    CSI_1_1 = CSI_1[:,64*subant:64*(1+subant)]
                                    CSI_1_1 = CSI_1_1[:,6:-6]
                                    CSI_1_1 = CSI_1_1[:,np.array([i for i in range(len(CSI_1_1.T)) if i not in [19]])]
                                    CSI_1_1 = CSI_1_1[:,np.array([i for i in range(len(CSI_1_1.T)) if i not in [46]])]
                
                                    CSI_2 = CSI[:,:,rx,tx[1]]
                                    CSI_2_1 = CSI_2[:,64*subant:64*(1+subant)]
                                    CSI_2_1 = CSI_2_1[:,6:-6]
                                    CSI_2_1 = CSI_2_1[:,np.array([i for i in range(len(CSI_2_1.T)) if i not in [19]])]
                                    CSI_2_1 = CSI_2_1[:,np.array([i for i in range(len(CSI_2_1.T)) if i not in [46]])]
                                    # ants_score = antenna_pair_score_vis([(np.angle(CSI_1_1[:,10:13]),np.angle(CSI_2_1[:,10:13]))])
                                    # CSI_ratio_scores.append(ants_score)
                                    
                                    CSI21 = CSI_2_1/(1e-6 + CSI_1_1)
                                    CSI_ratio.append(CSI21)
                                CSI_ratio = np.concatenate(CSI_ratio,1)
                                good_subcarriers = []
                                for iii in range(len(CSI_ratio.T)-1):
                                    c_sc = np.corrcoef(np.angle(CSI_ratio[:,iii]),np.angle(CSI_ratio[:,iii+1]))[0][1]
                                    good_subcarriers.append(c_sc) 
                                good_subcarriers = np.abs(good_subcarriers)
                                # good_subcarriers = good_subcarriers * (good_subcarriers>0.8)
                                # print(rx, tx, np.sum(good_subcarriers), np.sum(np.sort(good_subcarriers)[::-1]))
                                # good_subcarriers = np.argsort(good_subcarriers)[::-1]
                                good_subcarriers = np.where(np.abs(good_subcarriers)>0.6)[0]
                                # good_subcarriers = np.sort(good_subcarriers)
                                CSI_ratio = CSI_ratio[:,good_subcarriers[:]]
                                
                                for iii in range(np.shape(CSI_ratio)[1]):
                                    CSI_phase = np.angle(CSI_ratio[:,iii])
                                    CSI_phase = hampel(CSI_phase, 8).filtered_data
                                    # CSI_phase = plot_fft_energy_and_lowpass(CSI_phase,150,5, visualize= False)['X_filt']                             
                                    CSI_ratio[:,iii] = 1 * np.exp(1j * CSI_phase)
                                    
                                
                                # sample_data_bad_idx = np.array([ 0,   1,   2,   3,   4,   5,  25, 129, 231, 251, 252, 253, 254, 255])
                                v = Root_MUSIC_CSI(CSI_ratio.T[:,:])
                                v = np.array(v).reshape(-1)
                                # plt.figure();plt.plot(v)
                                # print("score: (rx, tx)", np.mean(CSI_ratio_scores), rx, tx)
                                # if np.mean(CSI_ratio_scores) < 0.7:
                                #     print("bad antennas (rx, tx)", rx, tx)
                                #     v = v * 0
                                # print(rx, tx, energy05)
                      
                                dopplers.append(v)
                                # CSI_ratio_scores_all.append(np.mean(CSI_ratio_scores))
                                
                        # plt.figure(); plt.plot(np.array(dopplers).reshape(24,-1)[:].T)
                        v_r_ = np.array(dopplers).reshape(24,-1).T
                        # from sklearn.decomposition import PCA
                        # transformer = PCA(n_components=3)
                        # X_transformed = transformer.fit_transform(v_r_)
                        v_r = v_r_
                        X_dopplers.append(v_r)
                        norms = []
                        for i in range(np.shape(v_r)[1]):
    
                            v_r[:,i] = v_r[:,i] - np.mean(v_r[:,i])
                            # v_r[:,i] = v_r[:,i] / (1e-8 + np.max(np.abs((v_r[:,i]))))
            
                            
                            # n=np.linalg.norm(v_r[:,i])
                            # norms.append(n)
                            # v_r[:,i] = v_r[:,i] / n if n>1e-12 else v_r[:,i] / (1e-5 + n)
     
                            
                        # for i in range(np.shape(v_r)[1]):
    
                        #     v_r[:,i] = v_r[:,i] - np.mean(v_r[:,i])
                            # v_r[:,i] = v_r[:,i] / (1e-12 + np.max(norms))
     
                        t = np.arange(v_r.shape[0])
                        # best_v, best_r, best_mask, best_loss, loss_hist, proj_images, a = \
                        #     estimate_velocity_from_radial_local_sampling(
                        #         v_r  + 0.00 * np.random.randn(np.shape(v_r)[0], np.shape(v_r)[1]) ,
                        #         subset_fraction   = 1.0,
                        #         outer_iterations  = 10,
                        #         mean_zero_velocity= False,
                        #         true_v            = None,
                        #         time_axis         = t,
                        #         camera_numbers    = list(range(v_r.shape[1])),
                        #         dtw_window        = 8,
                        #         max_clusters = 2,
                        #         use_support_dtw   = False,
                        #         visualise= False,
                        #         grid_res          = 6,
                        #     ) 
                            
                        best_v, best_r, best_mask, best_loss, loss_hist, proj_images, clusters_sig = \
                            estimate_velocity_from_radial_old_dtw(
                                v_r[:,:]  ,
                                subset_fraction   = 1.0,
                                outer_iterations  = 10,
                                mean_zero_velocity= False,
                                true_v            = None,
                                time_axis         = t,
                                camera_numbers    = list(range(v_r.shape[1])),
                                dtw_window        = 8,
                                use_support_dtw   = False,
                                visualise= False,
                                grid_res          = 6,
                                max_clusters = 2
    
                            ) 
                        proj_images = proj_images.reshape(-1,72)
                        X_proj.append(proj_images)
                        X_3D.append(best_v)
                        y.append((int(key[3])-1) // 10)
    for i in range(len(X_3D)):
        
        X_3D[i] = X_3D[i][150:750,:]
        X_proj[i] = X_proj[i][150:750,:]
        X_dopplers[i] = X_dopplers[i][150:750,:]
        
    for i in range(len(X_3D)):
        if len(X_3D[i])<600:
            X_3D[i] = X_3D[i-1]
            X_proj[i] = X_proj[i-1]
            X_dopplers[i] = X_dopplers[i-1]
    
    
    with open(file_target.split(".pickle")[0] + "_DoRFwithNoPreNormalization.pickle", 'wb') as handle:
        pickle.dump([np.array(X_3D), np.array(X_proj), np.array(y), np.array(X_dopplers), {"name": subjects[subject_idx], "session": session ,"sniffer": sniffer_idx}], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
