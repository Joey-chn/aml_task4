import biosppy.signals.eeg as eeg
import numpy as np
import pandas as pd

import biosppy.signals.emg as emg

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import biosppy.signals.tools as bt
from sklearn.model_selection import GridSearchCV


def read_from_file(X_train_file, X_predict_file, is_testing=False, test_rows = 60):
    if is_testing:
        # read from files
        x_train = pd.read_csv(X_train_file, index_col='Id', nrows=test_rows).to_numpy()
        x_predict = pd.read_csv(X_predict_file, index_col='Id', nrows=test_rows).to_numpy()

    else:
        x_train = pd.read_csv(X_train_file, index_col='Id').to_numpy()
        x_predict = pd.read_csv(X_predict_file, index_col='Id').to_numpy()

    return x_train, x_predict


# return a list of statistics
def transform(wave_name):
    mean = np.mean(wave_name)
    var = np.var(wave_name)
    high = np.amax(wave_name)
    low = np.amin(wave_name)
    stats = np.array([mean, var, high, low])
    return stats


def feature_extraction(eeg1, eeg2, emg):
    # remove nan value in nparray
    x_new = []
    for sig_mat in zip(eeg1, eeg2, emg):
        # read signal pairs in the matrix
        elem1 = sig_mat[0].reshape(1, -1)
        elem2 = sig_mat[1].reshape(1, -1)
        emg = sig_mat[2]

        eegs = np.concatenate((elem1, elem2), axis=0)
        eegs = np.transpose(eegs)

        # eeg feature construction
        signal_processed = eeg.eeg(signal=eegs, sampling_rate=128, show=False)
        # theta = signal_processed[3]
        # alow = signal_processed[4]
        # ahigh = signal_processed[5]
        # beta = signal_processed[6]
        # gamma = signal_processed[7]

        sig_trans_eeg1 = bt.analytic_signal(elem1)
        sig_trans_eeg2 = bt.analytic_signal(elem2)

        features = np.array([])
        # add the stats os theta ... gamma
        for idx in range(3, 8):
            wave_type = signal_processed[idx]
            features = np.append(features, [transform(wave_type[:, 0]), transform(wave_type[:, 1])])

        # add the amplitude from the Hilbert transform
        np.append(features, [transform(sig_trans_eeg1[1]), transform(sig_trans_eeg2[1])])

        # emg feature construction
        sig_trans_emg = bt.analytic_signal(emg)
        # pulse

        np.append(features, transform(sig_trans_emg))

        x_new.append(features)
    x_new = np.array(x_new)
    print("features", x_new.shape)
    return x_new


def processed_to_csv(X_train, flag='train', catogery = 'all_'):
    X = np.asarray(X_train)
    if flag == 'test':
        np.savetxt(copa + catogery +'X_test_temMed.csv', X)
    else:
        np.savetxt(copa + catogery + 'X_train_temMed.csv', X)
    print("wrote")


def result_to_csv(predict_y, sample_file, is_testing = False, nrows = 1200):
    # write the result to the CSV file
    predict_y = np.array(predict_y)
    if is_testing:
        sample_file = pd.read_csv(sample_file, nrows= nrows)
    else:
        sample_file = pd.read_csv(sample_file)
    id = sample_file['Id'].to_numpy().reshape(-1, 1)
    result = np.concatenate((id, predict_y.reshape(-1, 1)), axis=1)
    result = pd.DataFrame(result, columns=['Id', 'y'])
    result.to_csv(copa + 'predict_y.csv', index=False)


def standarlization(train_x, test_x):
    # standarlization
    scalar = StandardScaler()
    train_x = scalar.fit_transform(train_x.astype('float64'))
    test_x = scalar.transform(test_x.astype('float64'))
    return train_x, test_x


def svmClassifier(train_x, train_y, test_x, g, c):
    xy = np.concatenate((train_x, train_y), axis = 1)
    print(test_x.shape)
    np.random.shuffle(xy)
    train_x = xy[:, :-1]
    train_y = xy[:, -1].ravel()

    classifier = SVC(class_weight='balanced', gamma=g, C=c)  # c the penalty term for misclassification
    # make balanced_accuracy_scorer
    score_func = make_scorer(balanced_accuracy_score)  # additional param for f1_score
    # cross validation
    scores = cross_val_score(classifier, train_x, train_y, cv=5, scoring=score_func)
    print(scores)
    # learn on all data
    classifier.fit(train_x, train_y)
    y_predict_test = classifier.predict(test_x)
    return y_predict_test


def grid_search(train_x, train_y, test_x):
    train_y = train_y.ravel()
    parameters = {'C': [10, 20, 25, 30], 'gamma': [0.001, 0.005, 0.01]}
    svcClassifier = SVC(kernel='rbf', class_weight='balanced')
    score_func = make_scorer(balanced_accuracy_score)
    gs = GridSearchCV(svcClassifier, parameters, cv=5, scoring=score_func)
    gs.fit(train_x, train_y)
    print(gs.cv_results_)
    print(gs.best_params_)
    print(gs.best_score_)
    y_predict_test = gs.predict(test_x)
    return y_predict_test


def adaBoostClassifier(train_x, train_y, test_x):
    train_y = train_y.ravel()
    classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=None), n_estimators=60,
                                    learning_rate=0.8)
    # make balanced_accuracy_scorer
    score_func = make_scorer(balanced_accuracy_score)  # additional param for f1_score
    # cross validation
    scores = cross_val_score(classifier, train_x, train_y, cv=5, scoring=score_func)
    print(scores)
    # learn on all data
    classifier.fit(train_x, train_y)
    y_predict_test = classifier.predict(test_x)
    return y_predict_test


if __name__ == '__main__':
    is_start = True
    is_testing = True
    is_colab = False
    test_rows = 1200
    copa = ''
    if is_colab:
        copa = '/content/drive/My Drive/aml_task4/'
    # read data from files
    if is_start:
        # read
        eeg1s = read_from_file(copa + "train_eeg1.csv", copa + "test_eeg1.csv", is_testing=is_testing, test_rows = test_rows)
        eeg2s = read_from_file(copa + "train_eeg2.csv", copa + "test_eeg2.csv", is_testing=is_testing, test_rows = test_rows)
        emgs = read_from_file(copa + "train_emg.csv", copa + "test_emg.csv", is_testing=is_testing, test_rows = test_rows)


        if is_testing:
            y_train = pd.read_csv(copa + "train_labels.csv", index_col='Id', nrows=test_rows).to_numpy()
        else:
            y_train = pd.read_csv(copa + "train_labels.csv", index_col='Id').to_numpy()
        # get different files
        train_eeg1 = eeg1s[0]
        train_eeg2 = eeg2s[0]
        train_emg = emgs[0]

        test_eeg1 = eeg1s[1]
        test_eeg2 = eeg2s[1]
        test_emg = emgs[1]

        # feature extraction
        train_features = feature_extraction(train_eeg1, train_eeg2, train_emg)
        test_features = feature_extraction(test_eeg1, test_eeg2, test_emg)
        print(train_features.shape, test_features.shape)

        # standarlization
        x_std = standarlization(train_features, test_features)
        x_train_std = x_std[0]
        x_test_std = x_std[1]

        # write processed data to csv
        processed_to_csv(x_train_std)
        processed_to_csv(x_test_std, flag='test')

        #===== data based on different subjects =======
        # train data
        size_all = y_train.shape[0]
        idx = 0
        sub_num = 0
        subjects_X = {}
        subjects_y = {}
        interval = size_all//3 - 1
        size_part = size_all//3
        while idx < size_all:
            train_eeg1_part = train_eeg1[idx:idx + interval, :]
            train_eeg2_part = train_eeg2[idx:idx + interval, :]
            train_emg_part = train_eeg1[idx:idx + interval, :]

            labels_part = y_train[idx : idx + interval, :]
            sub_X = np.concatenate((train_eeg1_part,train_eeg2_part, train_emg_part), axis = 1)

            subjects_X[sub_num] = sub_X
            subjects_y[sub_num] = labels_part
            # print("subject_y:", subjects_y)
            sub_num += 1
            idx += size_part
            # write it to csv
            processed_to_csv(sub_X, catogery='s{}'.format(sub_num))
        # test data
        test_all = np.concatenate((test_eeg1, test_eeg2, test_emg), axis = 1)
        processed_to_csv(test_all, catogery='sub_', flag = 'test')
    if not is_start:
        x_train_std = pd.read_csv(copa + 'all_X_train_temMed.csv', delimiter=' ', index_col=False, header=None).to_numpy()
        x_test_std = pd.read_csv(copa + 'all_X_test_temMed.csv', delimiter=' ', index_col=False, header=None).to_numpy()

    # prediction
    predictions = {}

    # y_predict = grid_search(x_train_std, y_train, x_test_std)
    y_predict = svmClassifier(x_train_std, y_train, x_test_std, g = 0.01, c = 25)
    # neural net
    # y_predict = neurNet_classifier(x_train_std, y_train, x_test_std)
    # Adaboost classifier
    # y_predict = adaBoostClassifier(x_train_std, y_train, x_test_std)

    # =======train based on three different subjects using SVM======
    predictions[0] = y_predict
    for i in range(3):
        X = subjects_X[i]
        y = subjects_y[i]
        # standarlization
        x_std = standarlization(X, test_all)
        x_train_std = x_std[0]
        x_test_std = x_std[1]
        y_predict = svmClassifier(x_train_std, y, x_test_std, g = 0.001, c = 5)
        predictions[i + 1] = y_predict

    # ======= do voting from four predictions============
    votes = np.array([0, 0, 0])
    prediction_final = []
    for y_num in range(len(predictions[0])):
        for meth_num in range(4):
            y = int(predictions[meth_num][y_num]) # y is the class label, starts from 1
            print("y{}".format(y))
            votes[y - 1] += 1
        prediction_final.append(np.argmax(votes) + 1)

    result_to_csv(prediction_final, copa + 'sample.csv', is_testing, test_rows)

