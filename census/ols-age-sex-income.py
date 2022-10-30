import os
import numpy as np
import folktables
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

def travel_time_filter(data):
    """
    Filters for the employment prediction task
    """
    df = data
    df = df[df['AGEP'] > 16]
    df = df[df['PWGTP'] >= 1]
    df = df[df['ESR'] == 1]
    return df

def combined_filter(data):
    return travel_time_filter(folktables.adult_filter(data))

def get_data(year,features,outcome, randperm=True):
    # Predict income and regress to time to work
    data_source = folktables.ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA"], download=True)
    income_features = acs_data[features].fillna(-1)
    income = acs_data[outcome].fillna(-1)
    employed = np.isin(acs_data['COW'], np.array([1,2,3,4,5,6,7]))
    if randperm:
        shuffler = np.random.permutation(income.shape[0])
        income_features, income, employed = income_features.iloc[shuffler], income.iloc[shuffler], employed[shuffler]
    return income_features, income, employed

def train_eval_regressor(features, outcome, add_bias=True):
    X_train, X_test, y_train, y_test = train_test_split(features, outcome, test_size=0.1) 
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {'max_depth': 10, 'eta': 0.2, 'objective': 'reg:pseudohubererror', 'eval_metric': ['rmse','mae']}
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 1000
    tree = xgb.train(param, dtrain, num_round, evallist)
    return tree

def ols(features, outcome):
    ols_coeffs = np.linalg.pinv(features).dot(outcome)
    return ols_coeffs

def plot_data(age,income,sex):
    plt.figure()
    ageranges = np.digitize(age, bins=[0,20,30,40,50]) 
    sex = np.array(['female' if s==2 else 'male' for s in sex])
    sns.boxplot(x=ageranges, y=income, hue=sex, showfliers=False)
    plt.gca().set_xticklabels(['0-20','20-30','30-40','40-50','50+'])
    plt.ylabel('income ($)')
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig("./plots/raw_data.pdf")

def get_tree(year=2017):
    try:
        income_tree = xgb.Booster()
        income_tree.load_model(f"./.cache/model{year}.json")
    except:
        income_features_2017, income_2017, employed_2017 = get_data(year=year, features=['AGEP','SCHL','MAR','RELP','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','SEX','RAC1P'], outcome='PINCP')
        age_2017 = income_features_2017['AGEP'].to_numpy()[employed_2017]
        income_2017 = income_2017.to_numpy()[employed_2017]
        sex_2017 = income_features_2017['SEX'].to_numpy()[employed_2017]
        income_features_2017 = income_features_2017.to_numpy()[employed_2017,:]
        income_tree = train_eval_regressor(income_features_2017, income_2017)
        os.makedirs("./.cache/", exist_ok=True)
        income_tree.save_model(f"./.cache/model{year}.json")
    return income_tree

def trial(ols_features_2018, income_2018, predicted_income_2018, ols_coeff_true, N, n, delta, delta1_opt):
    X_labeled, X_unlabeled, Y_labeled, Y_unlabeled, Yhat_labeled, Yhat_unlabeled = train_test_split(ols_features_2018, income_2018, predicted_income_2018, train_size=n)
    X = np.concatenate([X_labeled, X_unlabeled],axis=0)

    naive_estimate = ols(X, np.concatenate([Y_labeled, Yhat_unlabeled], axis=0))

    myopic_estimate = ols(X_labeled, Y_labeled)

    myopic_sigmahat = np.std(np.linalg.pinv(X)[:,:n]*Y_labeled[None,:], axis=1)

    myopic_fluctuations = myopic_sigmahat * ( norm.ppf(1-delta1_opt/2) * np.sqrt(n*(N-n)/N) + norm.ppf(1-(delta-delta1_opt)/2) * np.sqrt( ((N-n)**3) / (N*n) ))

    rectifier = ((N-n)/n)*(np.linalg.pinv(X)[:,:n].dot(Yhat_labeled-Y_labeled))

    modelassisted_estimate = naive_estimate - rectifier 

    sigmahat = np.std(np.linalg.pinv(X)[:,:n]*(Yhat_labeled-Y_labeled)[None,:], axis=1) 
    
    fluctuations = sigmahat * ( norm.ppf(1-delta1_opt/2) * np.sqrt(n*(N-n)/N) + norm.ppf(1-(delta-delta1_opt)/2) * np.sqrt( ((N-n)**3) / (N*n) ))

    naive_error = np.abs(naive_estimate - ols_coeff_true)
    myopic_error = np.abs(myopic_estimate - ols_coeff_true)
    modelassisted_error = np.abs(modelassisted_estimate - ols_coeff_true)

    myopic_width = myopic_fluctuations
    modelassisted_width = fluctuations

    myopic_covered = myopic_error <= myopic_width 
    modelassisted_covered = modelassisted_error <= modelassisted_width

    return naive_error, myopic_error, modelassisted_error, myopic_width, modelassisted_width, myopic_covered, modelassisted_covered

if __name__ == "__main__":
    os.makedirs('./plots', exist_ok=True)
    # Train tree on 2017 data
    np.random.seed(0) # Fix seed for tree
    income_tree = get_tree()
    np.random.seed(0) # Fix seed for evaluation

    # Evaluate tree and plot data in 2018
    income_features_2018, income_2018, employed_2018 = get_data(year=2018, features=['AGEP','SCHL','MAR','RELP','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','SEX','RAC1P'], outcome='PINCP')
    age_2018 = income_features_2018['AGEP'].to_numpy()[employed_2018]
    income_2018 = income_2018.to_numpy()[employed_2018]
    sex_2018 = income_features_2018['SEX'].to_numpy()[employed_2018]
    income_features_2018 = income_features_2018.to_numpy()[employed_2018,:]
    predicted_income_2018 = income_tree.predict(xgb.DMatrix(income_features_2018)) 
    plot_data(age_2018, income_2018, sex_2018)

    # Collect OLS features and do MAI
    ols_features_2018 = np.stack([age_2018, sex_2018], axis=1)
    ols_coeff_true = ols(ols_features_2018, income_2018)
    N = ols_features_2018.shape[0]
    n = 500 
    num_trials = 100
    delta = 0.05
    delta1_opt = brentq(lambda x: (norm.ppf(1-x/2)/norm.ppf(1-(delta-x)/2)) - (N-n)/n, 0, delta)

    for i in range(num_trials):
        naive_error, myopic_error, modelassisted_error, myopic_width, modelassisted_width, myopic_covered, modelassisted_covered = trial(ols_features_2018, income_2018, predicted_income_2018, ols_coeff_true, N, n, delta, delta1_opt)
