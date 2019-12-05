import numpy as np
import biosppy.signals.eeg as eeg
from biosppy.signals import tools
import pandas as pd
import neurokit as nk
from sklearn.externals import joblib
from pyhht import emd, plot_imfs
from matplotlib import pyplot as plt

def test_eeg(eeg1, eeg2):
    # testing plot
    # remove nan value in nparray
    x_sample = np.concatenate((eeg1, eeg2), axis = 0)
    x_sample = np.asarray(x_sample)
    print(x_sample)

    # x_sample = np.concatenate((x_sample, x_sample), axis=0)
    signal_processed = eeg.eeg(signal=x_sample, sampling_rate=128, show=True)

def HHT(emg_sig) :
    decomposer = emd.EMD(emg_sig)
    return decomposer.decompose()
    
def show_components(emg_sig, y):
    count = 0
    for sig in emg_sig :
        plt.figure()
        plt.ion()
        plot_imfs(sig,HHT(sig))
        plt.savefig("plots/Epoch_" + str(count)+ "_Class_"  + str(y[count])+ ".png")
        plt.close()
        count += 1


def read_from_file(eeg1, eeg2, emg, y, rows = 50):
    # read from files
    x_train_emg = pd.read_csv(emg, index_col='Id', nrows = rows).to_numpy()
    x_train_eeg1 = pd.read_csv(eeg1, index_col='Id', nrows = rows).to_numpy()
    x_train_eeg2 = pd.read_csv(eeg2, index_col='Id', nrows = rows).to_numpy()
    y_train = pd.read_csv(y, index_col='Id', nrows = rows).to_numpy()
    return x_train_eeg1, x_train_eeg2, x_train_emg, y_train


if __name__ == '__main__':
    train_part = read_from_file("train_eeg1.csv", "train_eeg2.csv", "train_emg.csv", "train_labels.csv")
    eeg1 = train_part[0]
    eeg2 = train_part[1]

    show_components(train_part[2], train_part[3]) 

    test_eeg(eeg1, eeg2)
