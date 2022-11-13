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

def logistic(X, XTy, lr=1e-11):
    betahat = 0.5*np.array([0,0.5])
    for epoch in range(1000):
        p = 1./(1. + np.exp(-X@betahat))
        grad = (X.T@p - XTy)
        betahat = betahat - (lr*1/((epoch+1)**2))*grad # With quadratic learning rate decay
        print(betahat)
    return betahat

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
    avgs = [privcov[incomeranges == i].mean() for i in range(len(bins)+2)]
    plt.bar(range(len(bins)+2),avgs)
    #plt.gca().set_xticklabels(['<20K','20K-40K','40K-60K','60K-80K','80K-100K', '>100K'])
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

def trial(features_2018, privcov_2018, predicted_privcov_2018, coeff_true, N, n, delta):
    features_2018 = np.concatenate([features_2018, np.ones((features_2018.shape[0],1))], axis=1)

    X_labeled, X_unlabeled, Y_labeled, Y_unlabeled, Yhat_labeled, Yhat_unlabeled = train_test_split(features_2018, privcov_2018, predicted_privcov_2018, train_size=n)

    X = np.concatenate([X_labeled, X_unlabeled],axis=0)

    imputed_estimate = logistic(X,X.T@np.concatenate([Y_labeled, Yhat_unlabeled], axis=0))

    classical_estimate = logistic(X_labeled, X_labeled.T@Y_labeled)

    classical_sigmahat = np.std(X_labeled.T*Y_labeled[None,:], axis=1)

    classical_fluctuations = classical_sigmahat * norm.ppf(1-delta/2) * np.sqrt(N*(N-n)/n)

    rectifier = np.abs(X_labeled.T@(Y_labeled-Yhat_labeled))
    pdb.set_trace()

    modelassisted_estimate = logistic(X, X.T@np.concatenate([Yhat_labeled, Yhat_unlabeled], axis=0))

    modelassisted_sigmahat = np.std(X_labeled.T*(Y_labeled[None,:]-Yhat_labeled[None,:]), axis=1)

    fluctuations = modelassisted_sigmahat * norm.ppf(1-delta/2) * np.sqrt(N*(N-n)/n)

    imputed_error = imputed_estimate - coeff_true
    classical_error = classical_estimate - coeff_true
    modelassisted_error = modelassisted_estimate - coeff_true

    classical_width = classical_fluctuations
    modelassisted_width = rectifier + fluctuations

    classical_covered = np.abs(classical_error) <= classical_width
    modelassisted_covered = np.abs(modelassisted_error) <= modelassisted_width

    return imputed_error, classical_error, modelassisted_error, classical_width, modelassisted_width, classical_covered, modelassisted_covered, classical_sigmahat, modelassisted_sigmahat

def make_histograms(df):
    # Error figure
    df["error"] = df["error"].abs()
    fig, axs = plt.subplots(ncols=2, figsize=(7.5, 2.5))
    sns.set_theme(style="white", palette="pastel")
    kde0 = sns.kdeplot(df[df["coefficient"]=="age"][df["estimator"] != "imputed"], ax=axs[0], x="error", hue="estimator", fill=True, clip=(0,None))
    axs[0].axvline(x=df[df["coefficient"]=="age"][df["estimator"] == "imputed"]["error"].mean(), label="imputed", color="#7EAC95")
    axs[0].set_ylabel("")
    axs[0].set_xlabel("error (age coefficient, $/yr of age)")
    axs[0].set_yticklabels([])
    axs[0].set_yticks([])
    kde0.get_legend().remove()
    sns.despine(ax=axs[0],top=True,right=True,left=True)

    sns.kdeplot(df[df["coefficient"]=="sex"][df["estimator"] != "imputed"], ax=axs[1], x="error", hue="estimator", fill=True, clip=(0,None))
    l = axs[1].axvline(x=df[df["coefficient"]=="sex"][df["estimator"] == "imputed"]["error"].mean(), label="imputed", color="#7EAC95")
    axs[1].set_ylabel("")
    axs[1].set_xlabel("error (sex coefficient, $)")
    axs[1].set_yticklabels([])
    axs[1].set_yticks([])
    axs[1].legend(["model-assisted", "classical", "imputed"])
    sns.despine(ax=axs[1],top=True,right=True,left=True)
    fig.suptitle("") # This is here for spacing
    plt.tight_layout()
    plt.savefig('./logistic-plots/err.pdf')

    # Width figure
    fig, axs = plt.subplots(ncols=2, figsize=(7.5, 2.5))
    sns.set_theme(style="white", palette="pastel")
    kde0 = sns.kdeplot(df[df["coefficient"]=="age"][df["estimator"] != "imputed"], ax=axs[0], x="width", hue="estimator", fill=True, clip=(0,None))
    axs[0].set_ylabel("")
    axs[0].set_xlabel("width (age coefficient, $/yr of age)")
    axs[0].set_yticks([])
    axs[0].set_yticklabels([])
    kde0.get_legend().remove()
    sns.despine(ax=axs[0],top=True,right=True,left=True)

    sns.kdeplot(df[df["coefficient"]=="sex"][df["estimator"] != "imputed"], ax=axs[1], x="width", hue="estimator", fill=True, clip=(0,None))
    axs[1].set_ylabel("")
    axs[1].set_xlabel("width (sex coefficient, $)")
    axs[1].set_yticks([])
    axs[1].set_yticklabels([])
    sns.despine(ax=axs[1],top=True,right=True,left=True)
    fig.suptitle("") # This is here for spacing
    plt.tight_layout()
    axs[1].legend(["model-assisted", "classical"], bbox_to_anchor = (1.,1.2) )
    plt.savefig('./logistic-plots/width.pdf')

    # Standard deviation figure
    fig, axs = plt.subplots(ncols=2, figsize=(7.5, 2.5))
    sns.set_theme(style="white", palette="pastel")
    kde0 = sns.kdeplot(df[df["coefficient"]=="age"][df["estimator"] != "imputed"], ax=axs[0], x=r'$\sigma$', hue="estimator", fill=True, clip=(0,None))
    axs[0].set_ylabel("")
    axs[0].set_xlabel("std of summands (age)")
    axs[0].set_yticks([])
    axs[0].set_yticklabels([])
    kde0.get_legend().remove()
    sns.despine(ax=axs[0],top=True,right=True,left=True)

    sns.kdeplot(df[df["coefficient"]=="sex"][df["estimator"] != "imputed"], ax=axs[1], x=r'$\sigma$', hue="estimator", fill=True, clip=(0,None))
    axs[1].set_ylabel("")
    axs[1].set_xlabel("std of summands (sex)")
    axs[1].set_yticks([])
    axs[1].set_yticklabels([])
    sns.despine(ax=axs[1],top=True,right=True,left=True)
    fig.suptitle("") # This is here for spacing
    plt.tight_layout()
    axs[1].legend(["model-assisted", "classical"], bbox_to_anchor = (1.,1.2) )
    plt.savefig('./logistic-plots/stds.pdf')

    cvg_classical_age = (df[(df["estimator"]=="classical") & (df["coefficient"]=="age")]["covered"]).mean()
    cvg_classical_sex = (df[(df["estimator"]=="classical") & (df["coefficient"]=="sex")]["covered"]).mean()
    cvg_modelassisted_age = (df[(df["estimator"]=="model assisted") & (df["coefficient"]=="age")]["covered"]).mean()
    cvg_modelassisted_sex = (df[(df["estimator"]=="model assisted") & (df["coefficient"]=="sex")]["covered"]).mean()

    print(f"Myopic coverage ({cvg_classical_age},{cvg_classical_sex}), model-assisted ({cvg_modelassisted_age},{cvg_modelassisted_sex})")

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
        pincp_2018 = pincp_2018
        X = np.stack([pincp_2018, np.ones_like(pincp_2018)], axis=1)
        XTy = X.T@privcov_2018
        coeff_true = logistic(X, XTy)
        print(f"True logistic regression coefficients: {coeff_true}")

        N = features_2018.shape[0]
        n = 500
        num_trials = 100
        delta = 0.05

        # Store results
        columns = ["error","width","covered", r'$\sigma$', "estimator"]
        df = pd.DataFrame(np.zeros((num_trials*3,len(columns))), columns=columns)

        for i in tqdm(range(num_trials)):
            imputed_error, classical_error, modelassisted_error, classical_width, modelassisted_width, classical_covered, modelassisted_covered, classical_sigma, modelassisted_sigma = trial(pincp_2018[:,None], privcov_2018, predicted_privcov_2018, coeff_true, N, n, delta)
            df.loc[i] = imputed_error[0], -1, 0, 0, "imputed"
            df.loc[i+num_trials] = imputed_error[1], -1, 0, 0, "imputed"
            df.loc[i+2*num_trials] = classical_error[0], classical_width[0], int(classical_covered[0]), classical_sigma[0], "classical"
            df.loc[i+3*num_trials] = classical_error[1], classical_width[1], int(classical_covered[1]), classical_sigma[1], "classical"
            df.loc[i+4*num_trials] = modelassisted_error[0], modelassisted_width[0], int(modelassisted_covered[0]), modelassisted_sigma[0], "model assisted"
            df.loc[i+5*num_trials] = modelassisted_error[1], modelassisted_width[1], int(modelassisted_covered[1]), modelassisted_sigma[1], "model assisted"
        df.to_pickle('./.cache/logistic-results.pkl')
    make_histograms(df)
