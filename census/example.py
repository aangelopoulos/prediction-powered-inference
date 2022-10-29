import os
import numpy as np
import folktables
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

if __name__ == "__main__":
    os.makedirs('./plots', exist_ok=True)
    # Train tree on 2017 data
    income_features_2017, income_2017, employed_2017 = get_data(year=2017, features=['AGEP','SCHL','MAR','RELP','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','SEX','RAC1P'], outcome='PINCP')
    age_2017 = income_features_2017['AGEP'].to_numpy()[employed_2017]
    income_2017 = income_2017.to_numpy()[employed_2017]
    sex_2017 = income_features_2017['SEX'].to_numpy()[employed_2017]
    income_features_2017 = income_features_2017.to_numpy()[employed_2017,:]
    income_tree = train_eval_regressor(income_features_2017, income_2017)

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
    n = 1000

    X_labeled, X_unlabeled, Y_labeled, Y_unlabeled, Yhat_labeled, Yhat_unlabeled = train_test_split(ols_features_2018, income_2018, predicted_income_2018, train_size=n)
    naive_estimate = ols(np.concatenate([X_labeled, X_unlabeled],axis=0), np.concatenate([Y_labeled, Yhat_unlabeled], axis=0))

    print(f"True OLS coefficients: {ols_coeff_true}, Predicted OLS coefficients: {naive_estimate}")
