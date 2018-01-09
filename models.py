import os

from copy import deepcopy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_fscore_support, \
    precision_recall_curve, average_precision_score

import pandas as pd
import numpy as np

CSV_WALKING_PATH = os.path.split(__file__)[0] + '/Kyoto2007'  # file path with default dataset


def fit_label_encoder(not_float):
    number = LabelEncoder()
    return number.fit_transform(not_float)


def name_correct(func):
    def checking_wrapper(self, name, *args, **kwargs):
        if not isinstance(name, str):
            raise ValueError('Name of model should have type "str"')

        result = func(self, name, *args, **kwargs)
        return result

    return checking_wrapper


class DataProcessor:
    using_features = ['_DestinationPortNumber_', '_SourcePortNumber_', '_Service_', '_Count_', '_SerrorRate_',
                      '_DstHostCount_', '_DstHostSRVCount_', '_DstHostSameSRCPortRate_', '_SameSRVRate_',
                      '_Label_']  # последней всегда должны идти лейблы, если они есть

    def __init__(self, name, attack_ratio=0.05, attack_label=-1, processing_mode=0,
                 data=None, using_features=None, csv_separator='\t',
                 num_classes=2, labeled_data=True):

        """
        :param processing_mode:
                0 - read data from files in given path (non-recursive), type(data) is str
                1 - data is iterable
        """

        self.name = name
        self.attack_ratio = attack_ratio
        self.attack_label = attack_label

        if using_features is not None:
            self.using_features = using_features
        else:
            if not labeled_data:
                self.using_features.pop()

        if data is None:
            self.data = CSV_WALKING_PATH
            self.processing_mode = 0
        else:
            self.processing_mode = processing_mode
            self.data = data
        self.csv_separator = csv_separator
        self.num_classes = num_classes
        self.labeled_data = labeled_data

        self.__check_parameters_correctness()

        if not self.processing_mode:
            self.extract_data_from_path()

        self.fetch_classes()
        self.__make_all_features_numeric()

    def fetch_classes(self):
        if self.num_classes == 2:
            self.data[self.using_features[-1]] = self.data[self.using_features[-1]].apply(
                lambda x: self.attack_label if x * self.attack_label > 0 else x
            )

    def get_labels(self):
        if self.labeled_data:
            return np.array(self.data[self.using_features[-1]])
        else:
            raise AttributeError('Cant give labels from labelless dataset')

    def get_features(self):
        if self.labeled_data:
            features_without_labels = deepcopy(self.using_features)
            features_without_labels.remove(self.using_features[-1])
            return np.array(self.data[features_without_labels])

        return np.array(self.data[self.using_features])

    def __make_all_features_numeric(self):
        for index in self.data.columns.values:
            for item in self.data[index]:
                if item not in [0, '0', '0.0', []]:
                    try:
                        float(item)
                    except (ValueError, TypeError):
                        self.data[index] = fit_label_encoder(self.data[index])
                    break

    def extract_data_from_path(self):
        all_csv_files = []
        df = pd.DataFrame()

        for item in os.walk(self.data):
            for f in item[-1]:
                if f[-1:-5:-1][::-1] == '.csv':
                    all_csv_files.append('/'.join([item[0], f]))

        for csv_file in all_csv_files:
            df = pd.read_csv(csv_file, sep=self.csv_separator)
            try:
                df['_DurationOfConnection_']
            except KeyError:
                pass  # its fine
            else:
                df = df.append(df)

        indices = df[:-1]
        self.data = indices[self.using_features]

    def get_train_test_split(self, test_size=0.33):
        if self.labeled_data:
            return train_test_split(self.get_features(),
                                    self.get_labels(),
                                    test_size=test_size)

        return train_test_split(self.get_features(),
                                test_size=test_size)

    def __check_parameters_type_correctness(self):
        if not isinstance(self.attack_ratio, (int, float)):
            raise ValueError('Parameter attack_ratio should has type int or float')

        try:
            list(self.using_features)
        except TypeError:
            raise ValueError('Parameter attack_probability should be iterable')

        try:
            list(self.data)
        except TypeError:
            raise ValueError('Parameter training_data should be iterable')

        if not isinstance(self.csv_separator, str):
            raise ValueError('Parameter csv_separator should has type str')

    def __check_parameters_value_correctness(self):
        if not 0 < self.attack_ratio < 1:
            raise ValueError('Parameter attack_ratio should be in range (0, 1)')

        if str(self.attack_label) not in ('-1', '1', '-1.0', '1.0'):
            raise ValueError('Parameter attack_label should be in set {-1, 1}')
        self.attack_label = int(self.attack_label)

        if str(self.processing_mode) not in ('0', '1', '0.0', '1.0'):
            raise ValueError('Parameter processing_mode should be in set {0, 1}')
        self.processing_mode = int(self.processing_mode)

    def __check_parameters_consistency(self):
        if self.processing_mode == 0:
            if not isinstance(self.data, str):
                raise ValueError('Parameter training_data should have type str when processing_mode == 0')
        elif self.processing_mode == 1:
            if isinstance(self.data, str):
                raise ValueError('Parameter training_data should not have type str when processing_mode == 1')

    def __check_parameters_correctness(self):
        self.__check_parameters_type_correctness()
        self.__check_parameters_value_correctness()
        self.__check_parameters_consistency()


class Model:
    train_features = None
    test_features = None
    train_labels = None
    test_labels = None

    def __init__(self, name, ml_algorithm, features, labels, parent, test_size=0.33, attack_label=-1):

        self.name = name
        self.ml_algorithm = ml_algorithm()
        self.attack_label = attack_label

        self.__check_ml_algorithm_correctness(ml_algorithm)

        self.train_features, self.test_features, \
        self.train_labels, self.test_labels = train_test_split(features,
                                                               labels,
                                                               test_size=test_size)
        self.parent = parent

        self.fit()

    def __check_ml_algorithm_correctness(self, ml_algorithm):
        try:
            ml_algorithm.fit()
        except AttributeError:
            raise TypeError('Model "%s" has no fit method.' % self.name)
        except TypeError:
            pass

        try:
            ml_algorithm.predict()
        except AttributeError:
            raise TypeError('Model "%s" has no predict method.' % self.name)
        except TypeError:
            pass

        try:
            ml_algorithm.predict_proba()
        except AttributeError:
            print('Warning:\nModel "%s" has no predict_proba method.' % self.name)
        except TypeError:
            pass

    def predict_proba(self, p_value=0.5, data=None):
        print('Model named "%s" predicting proba' % self.name)
        if data is None:
            try:
                predicted = self.ml_algorithm.predict_proba(self.test_features)
                return np.apply_along_axis(
                    lambda x: 1 if x[0] > p_value else 0, 1, predicted  # this mb a bit wrong
                ), self.test_labels
            except AttributeError:
                raise TypeError('Model "%s" has no predict_proba method' % self.name)
        else:
            return self.ml_algorithm.predict_proba(data), None

    def fit(self):
        print('Fitting model named "%s"' % self.name)
        self.ml_algorithm.fit(self.train_features, self.train_labels)

    def predict(self, data=None):
        print('Model named "%s" predicting' % self.name)
        if data is None:
            data = self.test_features
            return self.ml_algorithm.predict(data), self.test_labels
        else:
            if isinstance(data, str):
                features = self.parent.data[data].get_features()
                labels = self.parent.data[data].get_labels()
                return self.ml_algorithm.predict(features), labels
            return self.ml_algorithm.predict(data), None


class ModelsManager:
    models = {}
    data = {}
    metrics = {}

    def __init__(self):
        self.data['kyoto2007'] = DataProcessor('kyoto2007')
        self.__load_gradient_boosting()
        self.__load_logistic_regression()

        self.__add_default_metrics()

    def __add_default_metrics(self):
        default_metric_names = [
            'average_precision_score',
            'accuracy_score',
            'roc_curve',
            'auc',
            'precision_recall_fscore_support',
            'precision_recall_curve'
        ]
        default_metrics = [
            average_precision_score,
            accuracy_score,
            roc_curve,
            auc,
            precision_recall_fscore_support,
            precision_recall_curve
        ]
        for name, metric in zip(default_metric_names, default_metrics):
            self.add_metric(name, metric)

    @name_correct
    def predict(self, model_name, data_name):
        try:
            return self.models[model_name].predict(self.data[data_name].get_features())
        except KeyError:
            if model_name in self.models.keys():
                self.__data_not_found(data_name)
            else:
                self.__model_not_found(model_name)

    @name_correct
    def predict_proba(self, model_name, data_name):
        try:
            return self.models[model_name].predict_proba(self.data[data_name].get_features())
        except KeyError:
            if model_name in self.models.keys():
                self.__data_not_found(data_name)
            else:
                self.__model_not_found(model_name)

    def __data_not_found(self, data_name):
        print('Dataset named "%s" doesnt exist' % data_name)
        print('Available data sets:')
        for data in self.data:
            print(data)

    def __metric_not_found(self, metric_name):
        print('Metric named "%s" doesnt exist' % metric_name)
        print('Available metrics:')
        for metric in self.metrics:
            print(metric)

    def __model_not_found(self, model_name):
        print('Model named "%s" doesnt exist' % model_name)
        print('Available models:')
        for model in self.models:
            print(model)

    @name_correct
    def get_metric_result(self, metric_name, model_name, data_name=None, *args, **kwargs):

        if model_name not in self.models.keys():
            self.__model_not_found(model_name)
            return
        else:
            # пока считаем, что все метрики требуют такие обязательные параметры
            if data_name is not None:
                _, test_features, _, known = self.data[data_name].get_train_test_split()
                predicted, _ = self.models[model_name].predict(data=test_features)

                if self.data[data_name].attack_label != self.models[model_name].attack_label:
                    if self.data[data_name].num_classes == 2:
                        if self.models[model_name].attack_label == -1:
                            predicted *= -1
                        else:
                            known *= -1
                    else:
                        print('Model and data have different attack labels. Deal with it yourself.')
                else:
                    if self.models[model_name].attack_label == -1:
                        predicted *= -1
                        known *= -1

            else:
                predicted, known = self.models[model_name].predict()

        if metric_name in self.metrics.keys():
            return self.metrics[metric_name](known, predicted, *args, **kwargs)
        self.__metric_not_found(metric_name)

    @name_correct
    def add_metric(self, name, metric, replace=False):

        if name in self.metrics.keys():
            if replace:
                print('Metric named "%s" already exist, proceed with replacing' % name)
                self.metrics[name] = metric
            else:
                print('Metric named "%s" already exist, proceed without replacing' % name)
        else:
            print('Add metric named "%s"' % name)
            self.metrics[name] = metric

    @name_correct
    def add_model(self, model_name, ml_algorithm, data_name, replace=False):

        try:
            features = self.data[data_name].get_features()
            labels = self.data[data_name].get_labels()
        except KeyError:
            print('Model named "%s" cant be added since provided data_name "%s" doesnt exist'
                  % (model_name, data_name))
            return

        if model_name in self.models.keys():
            if replace:
                print('Model named "%s" already exist, proceed with replacing' % model_name)
                self.models[model_name] = Model(model_name, ml_algorithm, features,
                                                labels, self,
                                                attack_label=self.data[data_name].attack_label)
            else:
                print('Model named "%s" already exist, proceed without replacing' % model_name)
        else:
            print('Add model named "%s"' % model_name)
            self.models[model_name] = Model(model_name, ml_algorithm, features, labels, self)

    @name_correct
    def add_data(self, name, replace=False, **kwargs):

        def really_add_data(data_dict):
            dp = DataProcessor(name, **kwargs)
            if dp is not None:
                data_dict[name] = dp
            else:
                print('Data named "%s" cant be added due following error:' % name)
                print('ERROR: some of features are not in data or df is empty')

        if name in self.data.keys():
            if replace:
                print('Data named "%s" already exist, proceed with replacing' % name)
                really_add_data(self.data)
            else:
                print('Data named "%s" already exist, proceed without replacing' % name)
        else:
            print('Add data named "%s" ' % name)
            really_add_data(self.data)

    def __load_gradient_boosting(self):
        self.add_model('gb_95-5_notime', GradientBoostingClassifier, 'kyoto2007')

    def __load_logistic_regression(self):
        self.add_model('lr_95-5_notime', LogisticRegression, 'kyoto2007')


if __name__ == '__main__':
    mm = ModelsManager()
    print(mm.get_metric_result('average_precision_score', 'gb_95-5_notime'))
