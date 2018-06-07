import numpy as np
import pandas as pd
import glob
import os
import sys
sys.path.append('../../oboe/automl')
import preprocessing
from sklearn.preprocessing import LabelEncoder
import configparser
from sklearn.preprocessing import Imputer
config = configparser.ConfigParser()
dirname = 'uci_datasets'

if not os.path.exists(dirname):
    os.makedirs(dirname)

def if_exist_string(vec):    
    for entry in vec:
        if type(entry) == str:
            return True
    return False

def LabelEncoder_nonnumeric(vec):
    vec_encoded = []
    unique_values = np.unique(np.array([x for x in vec if x != "NA"]))
    for entry in vec:
        if type(entry) != float:
            vec_encoded.append(np.where(entry == unique_values)[0][0])
        else:
            vec_encoded.append(np.nan)
    vec_encoded = np.array(vec_encoded)
    return vec_encoded

for file in glob.glob('*.data'):
    dataset_name = os.path.splitext(file)[0]
    try:
        with open('{}.data'.format(dataset_name)) as f:
            lines = f.readlines()
        data = []
        for l in lines:    
            l = l.replace(',NA', ',"NA"')
            data.append(list(eval(l)))      

            df = pd.read_csv('{}.data'.format(dataset_name), index_col=0, header=None)
            val = df.values
            le = LabelEncoder()
            categorical = []
            for i in range(val.shape[0]):
                if not 1020304050 in data[i]:
                    for j,v in enumerate(data[i]):
                        if j < len(data[i]) - 1 and j > 0:
                            categorical.append(type(v) == str)
                    break

        #imputer
        for j in range(len(categorical)):
            if categorical[j]:
                if if_exist_string(val[:, j]):
                    val[:, j] = LabelEncoder_nonnumeric(val[:, j])
                else:
                    imp_categorical = Imputer(missing_values='NaN', strategy='most_frequent', axis=0, copy=False)
                    val[:, j] = imp_categorical.fit_transform(val[:, j].reshape(-1, 1)).T
            else:
                imp_numerical = Imputer(missing_values='NaN', strategy='mean', axis=0, copy=False)
                val[:, j] = imp_numerical.fit_transform(val[:, j].reshape(-1, 1)).T

        #labelencoder
        for j in range(val.shape[1]):
            if j == val.shape[1]-1 or categorical[j]:
                val[:, j] = le.fit_transform(val[:, j])

        #one-hot-encoding and standardization
        data_numeric = val[:, :-1]
        data_labels = val[:, -1]
        data_numeric, categorical = preprocessing.pre_process(raw_data=data_numeric, categorical=categorical, impute=False, standardize=True, one_hot_encode=True)
        val = np.append(data_numeric, np.array(data_labels, ndmin=2).T, axis=1)
        pd.DataFrame(val, index=None, columns=None).to_csv('{}/{}.csv'.format(dirname, dataset_name), index=None, header=None)
        print("dataset {} finished \n \n".format(dataset_name))
    except:
        print("!!! error encountered on {} \n \n".format(dataset_name))

