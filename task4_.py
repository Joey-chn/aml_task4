import biosppy.signals.eeg as eeg
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.metrics import f1_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import pyhrv as hr


def read_from_file(X_train_file, X_predict_file,  y_train_file = None, is_testing = False):
    y_train = []
    if is_testing:
        # read from files
        x_train = pd.read_csv(X_train_file, index_col='id', nrows = 30).to_numpy()
        x_predict = pd.read_csv(X_predict_file, nrows = 30).to_numpy()
    else:
        x_train = pd.read_csv(X_train_file, index_col='id').to_numpy()
        y_train = pd.read_csv(y_train_file, index_col='id').to_numpy()
        if y_train_file:
            x_predict = pd.read_csv(X_predict_file).to_numpy()
    return x_train, x_predict, y_train



def feature_extraction_eeg(eeg1, eeg2, is_testset = False):
    # remove nan value in nparray
    X_new = []
    count = 0
    for sig_mat in zip(eeg1, eeg2):
        # read signal pairs in the matrix
        elem1 = sig_mat[0]
        elem2 = sig_mat[1]
        x_sample = np.concatenate((elem1, elem2), axis=0)
        x_sample = np.transpose(x_sample)

        # feature construction
        signal_processed = eeg.eeg(signal=x_sample, sampling_rate=128, show=True)
        theta = signal_processed[3]
        alow = signal_processed[4]
        ahigh = signal_processed[5]
        beta = signal_processed[6]
        gamma = signal_processed[7]

        features = np.concatenate((theta, alow, ahigh, beta, gamma), axix = 0).ravel() # put it into one dimension array
        X_new.append(features)
    X_new = np.array(X_new)
    print(X_new.shape)
    return X_new


def processed_to_csv(X_train, flag = 'train'):
    X = np.asarray(X_train)
    if flag == 'test':
        np.savetxt('X_test_temMed.csv', X)
    else:
        np.savetxt('X_train_temMed.csv', X)


def result_to_csv(predict_y, sample_file):
    # write the result to the CSV file
    sample_file = pd.read_csv(sample_file)
    id = sample_file['id'].to_numpy().reshape(-1, 1)
    result = np.concatenate((id, predict_y.reshape(-1, 1)), axis=1)
    result = pd.DataFrame(result, columns=['id', 'y'])
    result.to_csv('predict_y.csv', index=False)


def standarlization(train_x, test_x):
    # standarlization
    scalar = StandardScaler()
    train_x = scalar.fit_transform(train_x.astype('float64'))
    test_x = scalar.transform(test_x.astype('float64'))
    return train_x, test_x


def svmClassifier(train_x, train_y, test_x):
    train_y = train_y.ravel()
    classifier = SVC(class_weight='balanced', gamma=0.005, C=20)  # c the penalty term for misclassification
    # make balanced_accuracy_scorer
    score_func = make_scorer(f1_score, average='micro') # additional param for f1_score
    # cross validation
    scores = cross_val_score(classifier, train_x, train_y, cv=5, scoring=score_func)
    print(scores)
    # learn on all data
    classifier.fit(train_x, train_y)
    y_predict_test = classifier.predict(test_x)
    return y_predict_test


def grid_search(train_x, train_y, test_x):
    parameters = {'C': [ 10, 20, 25, 30], 'gamma': [0.001, 0.005, 0.01]}
    svcClassifier = SVC(kernel='rbf', class_weight='balanced')
    score_func = make_scorer(f1_score, average='micro')
    gs = GridSearchCV(svcClassifier, parameters, cv=5, scoring=score_func)
    gs.fit(train_x, train_y)
    print(gs.cv_results_)
    print(gs.best_params_)
    print(gs.best_score_)
    y_predict_test = gs.predict(test_x)
    return y_predict_test


def adaBoostClassifier(train_x, train_y, test_x):
    train_y = train_y.ravel()
    classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=None), n_estimators=60, learning_rate=0.8)
    # make balanced_accuracy_scorer
    score_func = make_scorer(f1_score, average='micro')  # additional param for f1_score
    # cross validation
    scores = cross_val_score(classifier, train_x, train_y, cv=5, scoring=score_func)
    print(scores)
    # learn on all data
    classifier.fit(train_x, train_y)
    y_predict_test = classifier.predict(test_x)
    return y_predict_test


if __name__ == '__main__':
    is_start = True
    is_testing = False
    # read data from files
    if is_start:
        # read eeg1
        eeg1s = read_from_file("train_eeg1.csv", "test_eeg1.csv", "train_labels.csv")
        eeg2s = read_from_file("train_eeg2.csv", "test_eeg.csv")
        emgs  = read_from_file("train_emg.csv", "test_emg.csv")

        # get different files
        train_eeg1 = eeg1s[0]
        train_eeg2 = eeg2s[0]
        train_emg = emgs[0]

        test_eeg1 = eeg1s[1]
        test_eeg2 = eeg2s[1]
        test_emg = emgs[1]

        # feature extraction
        train_features_eeg = feature_extraction_eeg(train_eeg1, train_eeg2)
        test_features_eeg =  feature_extraction_eeg(test_eeg1, test_eeg2)

        train_features = []
        test_features = []

        # standarlization
        x_std = standarlization(train_features, test_features)
        x_train_std = x_std[0]
        x_test_std = x_std[1]

        # write processed data to csv
        processed_to_csv(x_train_std)
        processed_to_csv(x_test_std, flag = 'test')

    if not is_start:
        x_train_std =  pd.read_csv('X_train_temMed.csv', delimiter=' ', index_col=False, header = None).to_numpy()
        x_test_std = pd.read_csv('X_test_temMed.csv', delimiter=' ', index_col=False, header=None).to_numpy()
        y_train = pd.read_csv('y_train.csv', index_col='id').to_numpy()
        # print(x_train_std[[10, 14, 17, 18]][:, -2:])
    # prediction
    # y_predict = grid_search(x_train_std, y_train, x_test_std)
    y_predict = svmClassifier(x_train_std, y_train, x_test_std)
    # neural net
    # y_predict = neurNet_classifier(x_train_std, y_train, x_test_std)
    # Adaboost classifier
    # y_predict = adaBoostClassifier(x_train_std, y_train, x_test_std)
    # grid search
    result_to_csv(y_predict, 'sample.csv')
