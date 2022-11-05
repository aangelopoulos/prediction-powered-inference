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
from tqdm import tqdm

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
    plt.figure(figsize=(7.5,2.5))
    sns.set_theme(style="white", palette="pastel")
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

def trial(ols_features_2018, income_2018, predicted_income_2018, ols_coeff_true, N, n, delta):
    X_labeled, X_unlabeled, Y_labeled, Y_unlabeled, Yhat_labeled, Yhat_unlabeled = train_test_split(ols_features_2018, income_2018, predicted_income_2018, train_size=n)
    X = np.concatenate([X_labeled, X_unlabeled],axis=0)

    imputed_estimate = ols(X, np.concatenate([Y_labeled, Yhat_unlabeled], axis=0))

    classical_estimate = ols(X_labeled, Y_labeled)

    classical_sigmahat = np.std(np.linalg.pinv(X)[:,:n]*Y_labeled[None,:], axis=1)

    classical_fluctuations = classical_sigmahat * norm.ppf(1-delta/2) * np.sqrt(N*(N-n)/n)

    rectifier = (N/n)*(np.linalg.pinv(X)[:,:n].dot(Y_labeled-Yhat_labeled))

    modelassisted_estimate = ols(X, np.concatenate([Yhat_labeled, Yhat_unlabeled], axis=0)) + rectifier 

    modelassisted_sigmahat = np.std(np.linalg.pinv(X)[:,:n]*(Y_labeled-Yhat_labeled)[None,:], axis=1) 
    
    fluctuations = modelassisted_sigmahat * norm.ppf(1-delta/2) * np.sqrt(N*(N-n)/n)

    imputed_error = imputed_estimate - ols_coeff_true
    classical_error = classical_estimate - ols_coeff_true
    modelassisted_error = modelassisted_estimate - ols_coeff_true

    classical_width = classical_fluctuations
    modelassisted_width = fluctuations

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
    plt.savefig('./plots/err.pdf')

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
    axs[1].legend(["model-assisted", "classical"], bbox_to_anchor = (1.1,1.25) )
    plt.savefig('./plots/width.pdf')

    # Standard deviation figure
    fig, axs = plt.subplots(ncols=2, figsize=(7.5, 2.5))
    sns.set_theme(style="white", palette="pastel")
    kde0 = sns.kdeplot(df[df["coefficient"]=="age"][df["estimator"] != "imputed"], ax=axs[0], x=r'$\sigma$', hue="estimator", fill=True, clip=(0,None))
    axs[0].set_ylabel("")
    axs[0].set_xlabel("std of OLS summands (age)")
    axs[0].set_yticks([])
    axs[0].set_yticklabels([])
    kde0.get_legend().remove()
    sns.despine(ax=axs[0],top=True,right=True,left=True)

    sns.kdeplot(df[df["coefficient"]=="sex"][df["estimator"] != "imputed"], ax=axs[1], x=r'$\sigma$', hue="estimator", fill=True, clip=(0,None))
    axs[1].set_ylabel("")
    axs[1].set_xlabel("std of OLS summands (sex)")
    axs[1].set_yticks([])
    axs[1].set_yticklabels([])
    sns.despine(ax=axs[1],top=True,right=True,left=True)
    fig.suptitle("") # This is here for spacing
    plt.tight_layout()
    axs[1].legend(["model-assisted", "classical"], bbox_to_anchor = (1.1,1.25) )
    plt.savefig('./plots/stds.pdf')

    cvg_classical_age = (df[(df["estimator"]=="classical") & (df["coefficient"]=="age")]["covered"]).mean()    
    cvg_classical_sex = (df[(df["estimator"]=="classical") & (df["coefficient"]=="sex")]["covered"]).mean()    
    cvg_modelassisted_age = (df[(df["estimator"]=="model assisted") & (df["coefficient"]=="age")]["covered"]).mean()    
    cvg_modelassisted_sex = (df[(df["estimator"]=="model assisted") & (df["coefficient"]=="sex")]["covered"]).mean()    

    print(f"Myopic coverage ({cvg_classical_age},{cvg_classical_sex}), model-assisted ({cvg_modelassisted_age},{cvg_modelassisted_sex})")

if __name__ == "__main__":
    os.makedirs('./plots', exist_ok=True)
    try:
        df = pd.read_pickle('./.cache/results.pkl')
    except:
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
        print(f"True OLS coefficients: {ols_coeff_true}")
        N = ols_features_2018.shape[0]
        n = 500 
        num_trials = 100
        delta = 0.05

        # Store results
        columns = ["error","width","covered", r'$\sigma$', "estimator","coefficient"]
        df = pd.DataFrame(np.zeros((num_trials*3*2,len(columns))), columns=columns)

        for i in tqdm(range(num_trials)):
            imputed_error, classical_error, modelassisted_error, classical_width, modelassisted_width, classical_covered, modelassisted_covered, classical_sigma, modelassisted_sigma = trial(ols_features_2018, income_2018, predicted_income_2018, ols_coeff_true, N, n, delta)
            df.loc[i] = imputed_error[0], -1, 0, 0, "imputed", "age"
            df.loc[i+num_trials] = imputed_error[1], -1, 0, 0, "imputed", "sex"
            df.loc[i+2*num_trials] = classical_error[0], classical_width[0], int(classical_covered[0]), classical_sigma[0], "classical", "age"
            df.loc[i+3*num_trials] = classical_error[1], classical_width[1], int(classical_covered[1]), classical_sigma[1], "classical", "sex"
            df.loc[i+4*num_trials] = modelassisted_error[0], modelassisted_width[0], int(modelassisted_covered[0]), modelassisted_sigma[0], "model assisted", "age"
            df.loc[i+5*num_trials] = modelassisted_error[1], modelassisted_width[1], int(modelassisted_covered[1]), modelassisted_sigma[1], "model assisted", "sex"
        df.to_pickle('./.cache/results.pkl')
    make_histograms(df)
