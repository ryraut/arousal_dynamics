import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from scipy.special import legendre

def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")
        
def scaler(data_in, scaler_in = None):

    if scaler_in is None:
        scaler_in = MinMaxScaler(feature_range=(-1, 1))
        scaler_in.fit(data_in)
        
    x = scaler_in.transform(data_in)
    x = StandardScaler(with_std=False).fit_transform(x)
    
    return x, scaler_in

def hankel_matrix(Xin, n_delay_coordinates, spacing=1):
    n_inputs, n_samples = Xin.shape

    X = np.zeros((n_inputs * (n_delay_coordinates), n_samples - spacing*(n_delay_coordinates-1)))
    for i in range(n_delay_coordinates):
        idxs = np.arange(spacing*i, spacing*(i+1) + n_samples - spacing*n_delay_coordinates)
        X[i*n_inputs:(i+1)*n_inputs] = Xin[:, idxs]
    return X

def legendre_polys(n, stackmax):
    domain = np.linspace(-1,1,stackmax)
    polys = np.zeros((len(domain), n))
    for i in range(n):
        polys[:, i] = legendre(i)(domain)
    return polys

def find_opt_lag(signal1, signal2, lim):
    
    # Find the lag that optimizes cross-correlation
    cross_corr = np.correlate(signal1.reshape(-1), signal2.reshape(-1), mode='full')
    lags = np.arange(-len(signal1) + 1, len(signal1))
    cross_corr[np.abs(lags)>lim] = 0
    opt_lag = np.argmax(cross_corr) - (len(signal1) - 1)
    
    return opt_lag,cross_corr,lags
