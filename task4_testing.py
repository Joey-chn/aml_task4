import numpy as np
import biosppy.signals.eeg as eeg
import pandas as pd


def test_eeg(eeg1, eeg2):
    # testing plot
    x_sample = np.concatenate((eeg1, eeg2), axis = 0)
    # transpose to put the signals into column
    x_sample = np.transpose(x_sample)
    # print theta
    # print(x_sample)

    # x_sample = np.concatenate((x_sample, x_sample), axis=0)
    signal_processed = eeg.eeg(signal=x_sample, sampling_rate=128, show=False)
    # # theta
    theta = signal_processed[3]
    alow = signal_processed[4]
    ahigh = signal_processed[5]
    beta = signal_processed[6]
    gamma = signal_processed[7]
    features = np.concatenate((theta, alow, ahigh, beta, gamma), axis=0).ravel()
    print(features.shape)

def read_from_file(eeg1, eeg2, nrows = 10):
    # read from files
    x_train_eeg1 = pd.read_csv(eeg1, index_col='Id', nrows = nrows).to_numpy()
    x_train_eeg2 = pd.read_csv(eeg2, index_col='Id', nrows = nrows).to_numpy()
    return x_train_eeg1, x_train_eeg2


if __name__ == '__main__':
    train_part = read_from_file("train_eeg1.csv", "train_eeg2.csv", 4)
    eeg1s = train_part[0]
    eeg2s = train_part[1]
    # for mat in zip(eeg1s, eeg2s):
    #     print(mat)
    #     print(mat[0])
    #     break
    eeg1 = eeg1s[3, :].reshape(1, -1)
    eeg2 = eeg1s[3, :].reshape(1, -1)
    # print(eeg1.shape) # size 1 * 512
    test_eeg(eeg1, eeg2)