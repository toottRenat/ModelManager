"""
ТИПА ДЕЛАЕМ АПИ
"""

import os
import pandas as pd
import numpy as np

from copy import deepcopy

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_fscore_support,\
    precision_recall_curve, average_precision_score


# тут лежат разные параметры, которые влияют на работу системы

CSV_WALKING_PATH = 'Kyoto2007'  # путь с файлами для обучение моделей


def fit_label_encoder(not_float):
    number = LabelEncoder()
    return number.fit_transform(not_float)


def check_name_correctness(func):

    def checking_wrapper(self, name, *args, **kwargs):
        if type(name) is not str:
            raise ValueError('Name of model should have type "str"')

        result = func(self, name, *args, **kwargs)
        return result

    return checking_wrapper


"""
training_mode:
0 - данные для обучения читаются из файлов в заданной папке (нерекурсивно), type(training_data) is str
1 - training_data is iterable
2 - 
3 - 
"""


class DataProcessor:
    # todo мб сюда нужно будет добавить методы для скейла, разной обработки и т.д.
    using_features = ['_DestinationPortNumber_', '_SourcePortNumber_',
                      '_Service_', '_Count_', '_SerrorRate_', '_DstHostCount_',
                      '_DstHostSRVCount_', '_DstHostSameSRCPortRate_', '_SameSRVRate_'
                      '_Label_']  # последней всегда должны идти лейблы (или можно сделать их отдельную передачу)

    def __init__(self, name, attack_ratio=0.05, attack_probability=0.9,
                 attack_label=-1, training_mode=0, using_features=None, training_data=None,
                 csv_separator='\t'):

        self.name = name
        self.attack_ratio = attack_ratio
        self.attack_probability = attack_probability
        self.attack_label = attack_label

        if using_features is not None:
            self.using_features = using_features

        if training_data is None:
            self.training_data = CSV_WALKING_PATH
            self.training_mode = 0
        else:
            self.training_mode = training_mode
            self.training_data = training_data
        self.csv_separator = csv_separator

        self.__check_parameters_correctness()

        if not self.training_mode:
            self.training_data = self.get_data_from_path()

        if self.training_data is None:
            pass  # todo handle somehow errors

    def get_labels(self):
        return self.training_data[self.using_features[-1]]

    def get_features(self):
        features_without_labels = deepcopy(self.using_features)
        features_without_labels.pop(self.using_features[-1])
        return self.training_data[features_without_labels]

    def get_data_from_path(self):
        all_csv_files = []
        df = pd.DataFrame()

        for item in os.walk(self.training_data):
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

        try:
            indices = list(df.columns.values)[:-1]
        except KeyError:  # some of features is not in data or df is empty
            return None
        else:
            return indices[self.using_features]

    def __check_parameters_type_correctness(self):
        if type(self.attack_ratio) not in (int, float):
            raise ValueError('Parameter attack_ratio should has type int or float')

        if type(self.attack_probability) not in (int, float):
            raise ValueError('Parameter attack_probability should has type int or float')

        try:
            list(self.using_features)
        except TypeError:
            raise ValueError('Parameter attack_probability should be iterable')

        try:
            list(self.training_data)
        except TypeError:
            raise ValueError('Parameter training_data should be iterable')

        if type(self.csv_separator) is not str:
            raise ValueError('Parameter csv_separator should has type str')

    def __check_parameters_value_correctness(self):
        if not 0 < self.attack_ratio < 1:
            raise ValueError('Parameter attack_ratio should be in range (0, 1)')

        if not 0 < self.attack_probability < 1:
            raise ValueError('Parameter attack_probability should be in range (0, 1)')

        if str(self.attack_label) not in ('-1', '1', '-1.0', '1.0'):
            raise ValueError('Parameter attack_label should be in set {-1, 1}')
        self.attack_label = int(self.attack_label)

        if str(self.training_mode) not in ('0', '1', '0.0', '1.0'):
            raise ValueError('Parameter training_mode should be in set {0, 1}')
        self.training_mode = int(self.training_mode)

    def __check_parameters_consistency(self):
        if self.training_mode == 0:
            if type(self.training_data) is not str:
                raise ValueError('Parameter training_data should have type str when training_mode == 0')
        elif self.training_mode == 1:
            if type(self.training_data) is str:
                raise ValueError('Parameter training_data should not have type str when training_mode == 1')

    def __check_parameters_correctness(self):
        self.__check_parameters_type_correctness()
        self.__check_parameters_value_correctness()
        self.__check_parameters_consistency()


class Model:
    # todo возможно этого класс лишний

    def __init__(self, name, model, features, labels):

        self.name = name
        self.model = model

        self.__check_model_correctness(model)
        self.fit(features, labels)

    def __check_model_correctness(self, model):
        # todo возможно нужны еще какие-либо методы (например, predict_proba)
        try:
            model.fit()
        except AttributeError:
            raise TypeError('Model "%s" has no fit method.' % self.name)
        except TypeError:
            pass

        try:
            model.predict()
        except AttributeError:
            raise TypeError('Model "%s" has no predict method.' % self.name)
        except TypeError:
            pass

    def fit(self, features, labels):
        self.model.fit(features, labels)

    def predict(self, data):
        return self.model.predict(data)


class ModelsManager:
    # todo возможно стоит из него сделать дексриптор или подрубить @property
    # todo возможно нужно также сделать словарь из метрик качества, только они будут функциями
    models = {}
    data = {}

    def __init__(self):
        self.load_gradient_boosting()
        self.load_logistic_regression()

    @check_name_correctness
    def add_model(self, name, model, data_name, replace=False):

        try:
            features = self.data[data_name].get_features()
            labels = self.data[data_name].get_labels()
        except KeyError:
            print('Model named "%s" cant be added since provided data_name "%s" doesnt exist'
                  % (name, data_name))
            return

        if name in self.models.keys():
            if replace:
                print('Model named "%s" already exist, proceed with replacing' % name)
                self.models[name] = Model(name, model, features, labels)
            else:
                print('Model named "%s" already exist, proceed without replacing' % name)
        else:
            print('Add model named "%s"' % name)
            self.models[name] = Model(name, model, features, labels)

    @check_name_correctness
    def add_data(self, name, replace=False, **kwargs):

        def really_add_data(data_dict):
            dp = DataProcessor(name, *kwargs)  # mb nada **
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

    @check_name_correctness
    def get_model_average_precision_score(self, name):
        pass

    @check_name_correctness
    def get_model_precision_recall_fscore_support(self, name):
        pass

    @check_name_correctness
    def get_model_cross_validation(self, name):
        pass

    @check_name_correctness
    def get_model_roc_curve(self, name):
        pass

    @check_name_correctness
    def get_model_auc(self, name):
        pass

    def load_gradient_boosting(self):
        self.add_model('gb', GradientBoostingClassifier)

    def load_logistic_regression(self):
        self.add_model('lr', LogisticRegression)


mm = ModelsManager()
