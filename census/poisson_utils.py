import os
import sys
sys.path.insert(1, '../')
import numpy as np
import folktables
import pdb
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import pandas as pd

from scipy.stats import norm
from scipy.optimize import brentq
from scipy.sparse import csc_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from tqdm import tqdm

def acs_filter(df, outcome_name, reg_feat_name):
    df = df[np.bitwise_not(np.isnan(df[outcome_name]))]
    df = df[np.bitwise_not(np.isnan(df[reg_feat_name]))]
    return df

def transform_features(features, ft, enc=None):
    c_features = features.T[ft == "c"].T.astype(str)
    if enc is None:
        enc = OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse=False)
        enc.fit(c_features)
    c_features = enc.transform(c_features)
    features = csc_matrix(np.concatenate([features.T[ft == "q"].T.astype(float), c_features], axis=1))
    return features, enc

def get_data(year,feature_names,outcome_name,regression_feature_name,acs_filter=None,randperm=True):
    # Predict income and regress to time to work
    data_source = folktables.ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA"], download=True)
    if acs_filter is not None:
        acs_data = acs_filter(acs_data, outcome_name, regression_feature_name)
    df_features = acs_data[feature_names]
    df_outcome = acs_data[outcome_name]
    if randperm:
        shuffler = np.random.permutation(df_outcome.shape[0])
        df_features, df_outcome = df_features.iloc[shuffler], df_outcome.iloc[shuffler]
    return df_features, df_outcome

def get_tree(year,feature_names,ft,outcome_name,reg_feat_name,enc=None,transform=False, acs_filter=None):
    try:
        tree = xgb.Booster()
        tree.load_model(f"./.cache/poisson-model{year}.json")
    except:
        features, outcome = get_data(year, feature_names, outcome_name, reg_feat_name, acs_filter=acs_filter)
        if transform:
            print("Transforming features and training tree.")
            tree = train_eval_regressor(transform_features(features, ft, enc)[0], outcome, transform=transform)
        else:
            print("Training tree without transformation.")
            tree = train_eval_regressor(features, outcome)
        os.makedirs("./.cache/", exist_ok=True)
        tree.save_model(f"./.cache/poisson-model{year}.json")
    return tree

def train_eval_regressor(features, outcome, transform=False):
    X_train, X_test, y_train, y_test = train_test_split(features, outcome, test_size=0.1)
    if transform:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
    else:
        dtrain = xgb.DMatrix(X_train.to_numpy(), label=y_train.to_numpy())
        dtest = xgb.DMatrix(X_test.to_numpy(), label=y_test.to_numpy())
    param = {'max_depth': 7, 'eta': 0.1, 'objective': 'count:poisson', 'eval_metric': ['error', 'mae']}
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 500
    tree = xgb.train(param, dtrain, num_round, evallist)
    return tree