import os
import numpy as np
import torch
import folktables
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy.stats import norm
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from tqdm import tqdm

def filter(df):
    df.fillna(-1)
    df = df[(df['PINCP'] >= 0 ) & (df['PRIVCOV'] >= 0)]
    df.loc[:,'PRIVCOV'] = 1-(df.loc[:,'PRIVCOV']-1)
    return df

def get_data(year,features,outcome,filter=None,randperm=True):
    # Predict income and regress to time to work
    data_source = folktables.ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA"], download=True)
    acs_data = filter(acs_data)
    df_features = acs_data[features]
    df_outcome = acs_data[outcome]
    if randperm:
        shuffler = np.random.permutation(df_outcome.shape[0])
        df_features, df_outcome = df_features.iloc[shuffler], df_outcome.iloc[shuffler]
    return df_features, df_outcome

def train_eval_regressor(features, outcome, add_bias=True):
    X_train, X_test, y_train, y_test = train_test_split(features, outcome, test_size=0.1)
    dtrain = xgb.DMatrix(X_train.to_numpy(), label=y_train.to_numpy())
    dtest = xgb.DMatrix(X_test.to_numpy(), label=y_test.to_numpy())
    param = {'max_depth': 7, 'eta': 0.1, 'objective': 'reg:pseudohubererror', 'eval_metric': ['rmse','mae']}
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 1000
    tree = xgb.train(param, dtrain, num_round, evallist)
    return tree

def plot_data(pincp, privcov):
    plt.figure(figsize=(7.5,2.5))
    sns.set_theme(style="white", palette="pastel")
    bins = [20000,40000,60000,80000,100000]
    incomeranges = np.digitize(pincp, bins=bins)
    avgs = [privcov[incomeranges == i].mean() for i in range(len(bins)+1)]
    plt.bar(range(len(bins)+1),avgs)
    plt.gca().set_xticklabels(['', '<20K', '20K-40K','40K-60K','60K-80K','80K-100K', '>100K'])
    plt.ylabel('frac w/private insurance')
    plt.gca().set_ylim([0,1])
    plt.xlabel('household income ($)')
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig("./logistic-plots/raw_data.pdf")

def get_tree(year=2017):
    try:
        tree = xgb.Booster()
        tree.load_model(f"./.cache/logistic-model{year}.json")
    except:
        features_2017, privcov_2017 = get_data(year=2017, features=['AGEP','SCHL','MAR','RELP','DIS','PINCP','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','SEX','RAC1P'], outcome='PRIVCOV', filter=filter)
        tree = train_eval_regressor(features_2017, privcov_2017)
        os.makedirs("./.cache/", exist_ok=True)
        tree.save_model(f"./.cache/logistic-model{year}.json")
    return tree

def trial(X, privcov_2018, predicted_privcov_2018, coeff_true, N, n, delta):
    X_labeled, X_unlabeled, Y_labeled, Y_unlabeled, Yhat_labeled, Yhat_unlabeled = train_test_split(X, privcov_2018, predicted_privcov_2018, train_size=n)

    X = np.concatenate([X_labeled, X_unlabeled], axis=0)

    d = X.shape[1]

    imputed_estimate = logistic(X,X.T@np.concatenate([Y_labeled, Yhat_unlabeled], axis=0))

    classical_estimate = logistic(X, N/n * X_labeled.T@Y_labeled)

    classical_sigmahat = np.std(X_labeled.T*Y_labeled[None,:], axis=1)

    classical_fluctuations = classical_sigmahat * norm.ppf(1-delta/2/d) * np.sqrt(N*(N-n)/n)

    classical_grid = np.linspace(N/n * X_labeled.T@Y_labeled-classical_fluctuations,N/n * X_labeled.T@Y_labeled+classical_fluctuations, 2)

    predictionpowered_XTy = X.T@np.concatenate([Yhat_labeled, Yhat_unlabeled], axis=0)

    predictionpowered_estimate = logistic(X, predictionpowered_XTy)

    predictionpowered_sigmahat = np.std(X_labeled.T*(Y_labeled[None,:]-Yhat_labeled[None,:]), axis=1)

    fluctuations = predictionpowered_sigmahat * norm.ppf(1-delta/2/d) * np.sqrt(N*(N-n)/n)

    predictionpowered_grid = np.linspace(predictionpowered_XTy-fluctuations, predictionpowered_XTy+fluctuations, 2)

    # Interval construction
    classical_outputs = [logistic(X, xty)[0] for xty in classical_grid]
    classical_interval = [min(classical_outputs), max(classical_outputs)]

    predictionpowered_outputs = [logistic(X, xty)[0] for xty in predictionpowered_grid]
    predictionpowered_interval = [min(predictionpowered_outputs), max(predictionpowered_outputs)]

    imputed_error = imputed_estimate[0] - coeff_true[0]
    classical_error = classical_estimate[0] - coeff_true[0]

    classical_width = classical_interval[1]-classical_interval[0]
    predictionpowered_width = predictionpowered_interval[1] - predictionpowered_interval[0]

    classical_covered = (coeff_true[0] >= classical_interval[0]) & (coeff_true[0] <= classical_interval[1])
    predictionpowered_covered = (coeff_true[0] >= predictionpowered_interval[0]) & (coeff_true[0] <= predictionpowered_interval[1])

    return imputed_error, classical_error, classical_width, predictionpowered_width, classical_covered, predictionpowered_covered, classical_sigmahat[0], predictionpowered_sigmahat[0]

def make_histograms(df):
    # Width figure
    plt.figure(figsize=(5.5, 2.5))
    my_palette = sns.color_palette(["#71D26F","#BFB9B9"], 2)
    sns.set_theme(style="white", palette=my_palette)
    kde = sns.kdeplot(df[df["estimator"] != "imputed"], x="width", hue="estimator", fill=True, clip=(0,None), hue_order=["prediction-powered","classical"])
    plt.ylabel("")
    plt.xlabel("width")
    plt.gca().set_yticks([])
    plt.gca().set_yticklabels([])
    kde.legend_.set_title(None)
    sns.despine(top=True,right=True,left=True)
    plt.gca().legend(loc="best", labels=["classical", "prediction-powered"])
    plt.tight_layout()
    plt.savefig('./logistic-plots/width.pdf', bbox_inches="tight")

    cvg_classical = (df[df["estimator"]=="classical"]["covered"]).mean()
    cvg_predictionpowered = (df[df["estimator"]=="prediction-powered"]["covered"]).mean()

    print(f"Classical coverage {cvg_classical}, prediction-powered coverage {cvg_predictionpowered}")

if __name__ == "__main__":
    os.makedirs('./logistic-plots', exist_ok=True)
    try:
        df = pd.read_pickle('./.cache/logistic-results.pkl')
    except:
        # Train tree on 2017 data
        np.random.seed(0) # Fix seed for tree
        tree = get_tree()
        np.random.seed(0) # Fix seed for evaluation

        # Evaluate tree and plot data in 2018
        features_2018, privcov_2018 = get_data(year=2018, features=['AGEP','SCHL','MAR','RELP','DIS','PINCP','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','SEX','RAC1P'], outcome='PRIVCOV', filter=filter)
        predicted_privcov_2018 = tree.predict(xgb.DMatrix(features_2018.to_numpy()))
        pincp_2018 = features_2018['PINCP']
        plot_data(pincp_2018, privcov_2018)

        # Collect logistic features and do MAI
        X = np.stack([pincp_2018, np.ones_like(pincp_2018)], axis=1)
        col_norms = np.linalg.norm(X,axis=0)
        X = X / col_norms[None,:]
        XTy = X.T@privcov_2018
        coeff_true = logistic(X, XTy)

        print(f"True logistic regression coefficients: {coeff_true/col_norms}")
        N = features_2018.shape[0]
        n = 10000
        num_trials = 100
        delta = 0.05

        # Store results
        columns = ["error","width","covered", r'$\sigma$', "estimator"]
        df = pd.DataFrame(np.zeros((num_trials*5,len(columns))), columns=columns)

        for i in tqdm(range(num_trials)):
            imputed_error, classical_error, classical_width, predictionpowered_width, classical_covered, predictionpowered_covered, classical_sigma, predictionpowered_sigma = trial(X, privcov_2018.to_numpy(), predicted_privcov_2018, coeff_true, N, n, delta)
            print(imputed_error, classical_error, classical_width, predictionpowered_width, classical_covered, predictionpowered_covered, classical_sigma, predictionpowered_sigma)
            df.loc[i] = imputed_error/col_norms[0], -1, 0, 0, "imputed"
            df.loc[i+num_trials] = imputed_error/col_norms[0], -1, 0, 0, "imputed"
            df.loc[i+2*num_trials] = classical_error/col_norms[0], classical_width/col_norms[0], int(classical_covered), classical_sigma/col_norms[0], "classical"
            df.loc[i+3*num_trials] = classical_error/col_norms[0], classical_width/col_norms[0], int(classical_covered), classical_sigma/col_norms[0], "classical"
            df.loc[i+4*num_trials] = -1, predictionpowered_width/col_norms[0], int(predictionpowered_covered), predictionpowered_sigma/col_norms[0], "prediction-powered"
        df.to_pickle('./.cache/logistic-results.pkl')
    make_histograms(df)
