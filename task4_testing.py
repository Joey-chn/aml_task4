import numpy as np
import biosppy.signals.eeg as eeg
from biosppy.signals import tools
import pandas as pd
import neurokit as nk
from sklearn.externals import joblib
import pyhrv as hr

def test_eeg(eeg1, eeg2):
    # testing plot
    # remove nan value in nparray
    x_sample = np.concatenate((eeg1, eeg2), axis = 0)
    x_sample = np.asarray(x_sample)
    print(x_sample)

    # x_sample = np.concatenate((x_sample, x_sample), axis=0)
    signal_processed = eeg.eeg(signal=x_sample, sampling_rate=128, show=True)


def read_from_file(eeg1, eeg2, rows_to_keep):
    # read from files
    x_train_eeg1 = pd.read_csv(eeg1, index_col='id', nrows = 10).to_numpy()
    x_train_eeg2 = pd.read_csv(eeg2, index_col='id', nrows = 10).to_numpy()
    return x_train_eeg1, x_train_eeg2


if __name__ == '__main__':
    train_part = read_from_file("train_eeg1.csv", "train_eeg2.csv")
    eeg1 = train_part[0]
    eeg2 = train_part[1]

    test_eeg(eeg1, eeg2)