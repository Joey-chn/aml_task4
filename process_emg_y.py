
import numpy as np
import pandas as pd

def feature_extraction_emg(emg, is_testset = False):
    #calculate variance & signal power of emg
    X_new = []
    mav_bf = 0
    for sig in emg :
        n = sig.size
        variance = np.variance(sig)
        mav = np.sum(np.abs(sig))/n
        mav_l = 2*np.sum(np.abs(sig[:int(n/2)]))/n
        mav_h = 2*np.sum(np.abs(sig[int(n/2):]))/n
        mavs_l = mav_l - mav_bf
        mavs_r = mav_h - mav_l
        mav_bf = mav_h
        third_moment = 0
        log = 0
        wave = 0
        for i in range(0, n) :
            third_moment += np.abs(sig[i]**3)
            log += np.log(np.abs(i))
            if i > 0 :
                wave += abs(sig[i] - sig[i-1])
        log = np.exp(log/n)
        third_moment/=n
        wave/=n 
        X_new.append([variance, third_moment, log, wave, mav, mavs_l, mavs_r])

    X_new = np.array(X_new)
    print('shape of emg features array: ', X_new.shape)
    return X_new

def feature_extraction_y(y_test) :
    
