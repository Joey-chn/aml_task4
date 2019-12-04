import numpy as np
import biosppy.signals.eeg as eeg
import pandas as pd


def test_eeg(eeg1, eeg2):
    # testing plot
    # remove nan value in nparray
    x_sample = np.concatenate((eeg1, eeg2), axis = 0)
    x_sample = np.asarray(x_sample)
    # transpose to put the signals into column
    x_sample = np.transpose(x_sample)
    print(x_sample)

    # x_sample = np.concatenate((x_sample, x_sample), axis=0)
    signal_processed = eeg.eeg(signal=x_sample, sampling_rate=128, show=True)


def read_from_file(eeg1, eeg2, nrows = 10):
    # read from files
    x_train_eeg1 = pd.read_csv(eeg1, index_col='Id', nrows = nrows).to_numpy()
    x_train_eeg2 = pd.read_csv(eeg2, index_col='Id', nrows = nrows).to_numpy()
    return x_train_eeg1, x_train_eeg2


if __name__ == '__main__':
    train_part = read_from_file("train_eeg1.csv", "train_eeg2.csv", 5)
    eeg1s = train_part[0]
    eeg2s = train_part[1]

    eeg1 = eeg1s[1, :].reshape(1, -1)
    eeg2 = eeg1s[1, :].reshape(1, -1)
    print(eeg1.shape) # size 1 * 512
    test_eeg(eeg1, eeg2)