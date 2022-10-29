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

def get_data(year,features,outcome):
    # Predict income and regress to time to work
    data_source = folktables.ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA"], download=True)
    income_features = acs_data[features].fillna(-1)
    income = acs_data[outcome].fillna(-1)
    employed = np.isin(acs_data['COW'], np.array([1,2,3,4,5,6,7]))
    return income_features, income, employed

def train_eval_regressor(features, outcome, add_bias=True):
    X_train, X_test, y_train, y_test = train_test_split(features, outcome, test_size=1/3) 
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {'max_depth': 6, 'eta': 0.3, 'objective': 'reg:squarederror', 'eval_metric': ['rmse','mae']}
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 100
    tree = xgb.train(param, dtrain, num_round, evallist)
    return tree, X_test, y_test

def ols(features, outcome):
    ols_model = LinearRegression().fit(features,outcome)
    ols_coeffs = ols_model.coef_
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
    os.makedirs('./plots')
    income_features_2017, income_2017, employed_2017 = get_data(year=2017, features=['AGEP','SCHL','MAR','RELP','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','SEX','RAC1P'], outcome='PINCP')
    age_2017 = income_features_2017['AGEP'].to_numpy()[employed_2017]
    income_2017 = income_2017.to_numpy()[employed_2017]
    sex_2017 = income_features_2017['SEX'].to_numpy()[employed_2017]
    plot_data(age_2017, income_2017, sex_2017)
    ols_features = np.stack([age_2017, sex_2017], axis=1)
    ols_coeff_true = ols(ols_features, income_2017)

    #income_tree, X_labeled, y_labeled = train_eval_regressor(income_features_2017, income_2017)
    print(ols_coeff_true)
