from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(0)

# def remove_nan_mean(dataframe, f):
#     x = dataframe[f].mean()
#     # print(x)
#     dataframe[f].fillna(x, inplace=True)
#
#
# def remove_nan_mode(dataframe, f):
#     x = dataframe[f].mode()[0]
#     dataframe[f].fillna(x, inplace=True)
#
#
# def remove_nan_median(dataframe, f):
#     x = dataframe[f].median()
#     dataframe[f].fillna(x, inplace=True)
#
#
# def remove_all_nulls(dataframe):
#     for f in dataframe.columns:
#         if dataframe[f].isnull().sum() > 0:
#             remove_nan_mean(dataframe, f)


def make_boolean(dataframe, f):
    dataframe[f].replace({'Yes': 1, 'No': 0}, inplace=True)


def make_all_boolean(dataframe, list_of_features):
    for lf in list_of_features:
        make_boolean(dataframe, lf)


def one_hot_encoder(dataframe, f):
    one_hot_list = pd.get_dummies(dataframe[[f]])
    res = pd.concat([dataframe, one_hot_list], axis=1)
    res = res.drop([f], axis=1)
    return res


def one_hot_encoder_all(dataframe, list_of_features):
    for lf in list_of_features:
        dataframe = one_hot_encoder(dataframe, lf)
    return dataframe


def min_max_scaling(dataframe):
    min_max_scaler = preprocessing.MinMaxScaler()
    dataframe[dataframe.columns] = min_max_scaler.fit_transform(dataframe)
    return dataframe


def get_labels(dataframe, f):
    res = dataframe[f]
    return res


def get_features(dataframe, f):
    cols = list(dataframe.columns.values)
    cols.pop(cols.index(f))
    res = dataframe[cols]
    return res


def imputation(dataframe, f, typ):
    impute = SimpleImputer(missing_values=np.nan, strategy=typ)
    dataframe[f] = impute.fit_transform(dataframe[f].to_numpy().reshape(-1, 1))
    return dataframe


def impute_all(dataframe):
    for f in dataframe.columns:
        if dataframe[f].isnull().sum() > 0:
            if is_string_dtype(dataframe[f]):
                dataframe = imputation(dataframe, f, 'most_frequent')
            elif is_numeric_dtype(dataframe[f]):
                dataframe = imputation(dataframe, f, 'mean')
    return dataframe


def dataset1_process():

    df = pd.read_csv('datasets/Telco-Customer-Churn.csv', na_values=' ')
    df.drop(['customerID'], axis=1, inplace=True)
    df = impute_all(df)

    transform_bool = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    make_all_boolean(df, transform_bool)

    one_hot_encoding = ['gender', 'InternetService', 'Contract', 'PaymentMethod',
                        'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

    df = one_hot_encoder_all(df, one_hot_encoding)

    drop_extra_col = ['MultipleLines_No phone service', 'OnlineSecurity_No internet service',
                      'OnlineBackup_No internet service', 'DeviceProtection_No internet service',
                      'TechSupport_No internet service', 'StreamingTV_No internet service',
                      'StreamingMovies_No internet service']
    for drop_col in drop_extra_col:
        df.drop([drop_col], axis=1, inplace=True)

    df = min_max_scaling(df)

    label_name = 'Churn'
    initial_feature_names = list(df.columns.values)
    initial_feature_names.remove(label_name)

    initial_features = get_features(df, label_name)
    labels = get_labels(df, label_name)

    training_data, testing_data = train_test_split(df, test_size=0.2, random_state=25)

    training_feature = get_features(training_data, label_name)
    training_label = get_labels(training_data, label_name)

    testing_feature = get_features(testing_data, label_name)
    testing_label = get_labels(testing_data, label_name)

    return training_feature, training_label, testing_feature, testing_label


def dataset2_process():
    train_df = pd.read_csv('datasets/adult/adult.data', header=None, sep=', ', na_values='?', engine='python')
    test_df = pd.read_csv('datasets/adult/adult.test', header=None, sep=', ', na_values='?', engine='python',
                          skiprows=1)
    features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                'greater_than_50']
    train_df.columns = features
    test_df.columns = features
    test_df['greater_than_50'].replace({'<=50K.': '<=50K', '>50K.': '>50K'}, inplace=True)

    train_sample = train_df.shape[0]
    df2 = train_df.merge(test_df, how='outer')

    df2 = impute_all(df2)

    df2['greater_than_50'].replace({'<=50K': 0, '>50K': 1}, inplace=True)
    df2['sex'].replace({'Male': 0, 'Female': 1}, inplace=True)

    one_hot_encoding = ['workclass', 'education', 'marital-status', 'occupation',
                        'relationship', 'race', 'native-country']

    df2 = one_hot_encoder_all(df2, one_hot_encoding)

    df2 = min_max_scaling(df2)

    train_df = df2.iloc[:train_sample, :]
    test_df = df2.iloc[train_sample:, :]

    label_name = 'greater_than_50'

    initial_features = get_features(df2, label_name)
    labels = get_labels(df2, label_name)

    training_feature = get_features(train_df, label_name)
    training_label = get_labels(train_df, label_name)

    testing_feature = get_features(test_df, label_name)
    testing_label = get_labels(test_df, label_name)

    return training_feature, training_label, testing_feature, testing_label


def dataset3_process():
    df3 = pd.read_csv('datasets/creditcard.csv')
    df_pos, df_neg = df3[df3['Class'] == 1], df3[df3['Class'] == 0]
    df_neg = df_neg.sample(n=2000, random_state=0)
    df3 = df_pos.merge(df_neg, how='outer')
    # print(df3.shape)
    df3 = min_max_scaling(df3)

    label_name = 'Class'
    initial_feature_names = list(df3.columns.values)
    initial_feature_names.remove(label_name)

    initial_features = get_features(df3, label_name)
    labels = get_labels(df3, label_name)

    training_data, testing_data = train_test_split(df3, test_size=0.2, random_state=25)

    training_feature = get_features(training_data, label_name)
    training_label = get_labels(training_data, label_name)

    testing_feature = get_features(testing_data, label_name)
    testing_label = get_labels(testing_data, label_name)

    return training_feature, training_label, testing_feature, testing_label


tf1, tl1, tf2, tl2 = dataset1_process()
tf3, tl3, tf4, tl4 = dataset2_process()
tf5, tl5, tf6, tl6 = dataset3_process()


class LR:

    def __init__(self, x, y):
        self.X_arr = np.matrix(x)
        self.y_arr = np.matrix(y)
        self.y_arr[self.y_arr < 0.1] = -1.0
        self.y_arr = self.y_arr.reshape(self.X_arr.shape[0], -1)
        self.w_arr = np.matrix(np.zeros(self.X_arr.shape[1]))
        # self.X_test = np.matrix(p)
        # self.y_test = np.matrix(q)
        # self.y_test[self.y_test < 0.1] = -1.0
        # self.y_test = self.y_test.reshape(self.X_test.shape[0], -1)

    # @staticmethod
    # def set_training_mat(x, y, p, q):
    #     X_arr = np.matrix(x)
    #     y_arr = np.matrix(y)
    #     y_arr[y_arr < 0.1] = -1.0
    #     y_arr = y_arr.reshape(X_arr.shape[0], -1)
    #     # y_arr = y_arr.reshape(-1,1)
    #     w_arr = np.matrix(np.zeros(X_arr.shape[1]))
    #
    #     X_test = np.matrix(p)
    #     y_test = np.matrix(q)
    #     y_test[y_test < 0.1] = -1.0
    #     y_test = y_test.reshape(X_test.shape[0], -1)
    #     return X_arr, y_arr, w_arr, X_test, y_test

    @staticmethod
    def logistic_func(w, x):
        z = np.dot(x, w.T)
        # return 1.0*(np.exp(z) - np.exp(-z)) /(np.exp(z) + np.exp(-z))
        return np.tanh(z)

    @staticmethod
    def loss_func(y, hx):
        loss = np.mean(np.square(y - hx))
        return loss

    @staticmethod
    def grad_desc_update(w, x, y):
        hx = LR.logistic_func(w, x)
        loss = y - hx
        dz = 1 - np.square(hx)
        # print(dz.shape)
        temp2 = np.multiply(loss, dz)
        temp3 = np.dot(temp2.T, x)
        return temp3

    @staticmethod
    def grad_desc(x, y, w, alpha=.1, error=0.5, max_itr=1000):
        hx = LR.logistic_func(w, x)
        loss = LR.loss_func(y, hx)
        loss_list = [loss]

        itr = 1

        while loss > error:

            w = w + (alpha * LR.grad_desc_update(w, x, y)) / x.shape[0]
            hx = LR.logistic_func(w, x)
            loss = LR.loss_func(y, hx)
            loss_list.append(loss)
            itr += 1
            if itr >= max_itr:
                break

        # plt.plot(range(len(loss_list)), loss_list)
        # plt.show()
        return w, itr

    def prediction(self, x):
        pred = self.logistic_func(self.w_arr, x)
        pred_list = [1.0 if i > 0.0 else -1.0 for i in pred]
        return np.array(pred_list).reshape(len(pred_list), -1)

    def perform_lr(self):
        # x_trn, y_trn, w_mat, x_tst, y_tst = LR.set_training_mat(tf1, tl1, tf2, tl2)
        # LR.grad_desc(x_trn, y_trn, w_mat, alpha=0.01, max_itr=10000)
        # y_predicted = LR.prediction(w_mat, x_trn)
        # print("Correctly predicted labels:", np.sum(y_trn == y_predicted))
        # print(accuracy_score(y_trn, y_predicted))
        self.w_arr, itr = LR.grad_desc(self.X_arr, self.y_arr, self.w_arr, alpha=0.1, error=0.5, max_itr=1000)
        # y_predicted = self.prediction(self.X_arr)
        # print(accuracy_score(self.y_arr, y_predicted))
        # print("Correctly predicted labels:", np.sum(self.y_arr == y_predicted))


# lr_instance = LR()
# x_trn, y_trn, w_mat, x_tst, y_tst = LR.set_training_mat(tf1, tl1, tf2, tl2)
# LR.grad_desc(x_trn, y_trn, w_mat, alpha=0.01, max_itr=10000)

lr1 = LR(tf1, tl1)
lr2 = LR(tf2, tl2)
lr3 = LR(tf3, tl3)
lr4 = LR(tf4, tl4)
lr5 = LR(tf5, tl5)
lr6 = LR(tf6, tl6)

# lr1.perform_lr()
# s = lr1.prediction(tf1)
# print(accuracy_score(lr2.y_arr, s))

# lr2.perform_lr()
# s = lr2.prediction(tf2)
# print(accuracy_score(lr2.y_arr, s))


# lr3.perform_lr()
# s = lr3.prediction(tf4)
# print(accuracy_score(lr4.y_arr, s))


# lr5.perform_lr()
# s = lr5.prediction(tf5)
# print(accuracy_score(lr5.y_arr, s))


# s = lr5.prediction(tf6)
# print(accuracy_score(lr5.y_arr, s))


class AdaBoost:

    def __init__(self, ex_feature, ex_label, lr_weak, k):
        self.lr_weak = lr_weak
        self.k = k
        self.ex_feature = ex_feature
        self.ex_label = ex_label
        self.total_samples = ex_feature.shape[0]
        self.total_features = ex_feature.shape[1]
        self.weights = np.zeros((self.total_samples, 1))
        self.weights = (1.0 / self.total_samples) + self.weights
        self.data_feature = pd.DataFrame()
        self.data_label = pd.DataFrame()
        self.LR_models = [LR(ex_feature, ex_label)] * k
        self.hp_weights = np.array(np.zeros(k))

    def resampling(self):

        indices = np.random.choice(self.total_samples, (self.total_samples,), p=self.weights.ravel())
        indices = np.array(indices)
        # indices = pd.DataFrame(indices)
        # indices.to_numpy()
        # indices
        # indices.reshape(-1,1)
        # print(np.unique(indices).shape[0])

        self.data_feature = self.ex_feature.iloc[indices]
        self.data_label = self.ex_label.iloc[indices]

        # print(self.data_feature)
        # print(self.data_label)

    def fit_current_hp(self):

        for idx in range(self.k):
            self.resampling()

            weak_model = LR(self.data_feature, self.data_label)
            weak_model.perform_lr()
            self.LR_models[idx] = weak_model

            # print(weak_model.w_arr)

            y_predicted = weak_model.prediction(self.lr_weak.X_arr)
            print(accuracy_score(self.lr_weak.y_arr, y_predicted))

            error = np.sum(self.weights[y_predicted != self.lr_weak.y_arr])
            # print(error)
            # print(np.sum(y_predicted != self.lr_weak.y_arr), np.sum(y_predicted == self.lr_weak.y_arr))

            if error > 0.5:
                continue
            self.weights[y_predicted == self.lr_weak.y_arr] *= error / (1 - error)

            s = np.sum(self.weights)
            self.weights /= s
            self.hp_weights[idx] = np.log((1-error)/error)

    def predict(self, x):

        total_prediction = np.array(np.zeros(x.shape[0]))
        total_prediction = total_prediction.reshape(x.shape[0], -1)

        for i in range(self.k):

            y_predicted = self.LR_models[i].prediction(x)

            total_prediction += (y_predicted * self.hp_weights[i])

        return np.where(total_prediction > 0.0, 1.0, -1.0)


# ab = AdaBoost(tf1, tl1, lr1,  5)
# ab.fit_current_hp()
# prd = ab.predict(lr2.X_arr)
# print(accuracy_score(lr2.y_arr, prd))

EPS = 1e-7


def matrix(y, y_guessed):

    tn, fp, fn, tp = confusion_matrix(y, y_guessed).ravel()

    accuracy = (tp + tn) / (tn + fp + fn + tp) * 100
    recall = tp / (tp + fn + EPS),
    specificity = tn / (fp + tn + EPS),
    precision = tp / (tp + fp + EPS),
    fdr = fp / (tp + fp + EPS),
    f1_score = tp / (tp + 0.5 * (fp + fn) + EPS)

    print('accuracy ' + str(accuracy))
    print('recall ' + str(recall))
    print('specificity ' + str(specificity))
    print('precision ' + str(precision))
    print('fdr ' + str(fdr))
    print('f1_score ' + str(f1_score))


# matrix(lr1.y_arr, s)
# matrix(lr2.y_arr, s)
# matrix(lr3.y_arr, s)
# matrix(lr4.y_arr, s)
# matrix(lr5.y_arr, s)
# matrix(lr6.y_arr, s)





