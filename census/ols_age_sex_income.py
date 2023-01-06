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

from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from tqdm import tqdm

from concentration import ols, standard_ols_interval, pp_ols_interval

import pdb

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
    X_train, X_test, y_train, y_test = train_test_split(features, outcome, test_size=0.2)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    num_round = 1000

    lossfns = ['reg:squarederror', 'reg:pseudohubererror']

    # TODO: Fix this
    space={
           'max_depth': hp.choice('max_depth', np.arange(3, 30+1, dtype=int)),
           'eta': hp.uniform('eta', 0, 1),
           'objective': hp.choice('objective', lossfns)
          }

    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    def objective(params):
        tree = xgb.train(params, dtrain, num_round, evallist)
        print(params)

        pred = tree.predict(dtest)
        std = np.std(y_test - pred)
        return {'loss': std, 'status': STATUS_OK }

    trials = Trials()

    #best_hyperparams = fmin(fn = objective,
    #                        space = space,
    #                        algo = tpe.suggest,
    #                        max_evals = 100,
    #                        trials = trials)

    #print(best_hyperparams)
    #best_hyperparams['objective'] = lossfns[best_hyperparams['objective']]
    #tree = xgb.train(best_hyperparams, dtrain, num_round, evallist)
    tree = xgb.train({'eta': 0.8, 'max_depth': 30, 'objective': 'reg:pseudohubererror'}, dtrain, 10000, evallist)
    return tree


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
    plt.savefig("./ols-plots/raw_data.pdf")

def get_tree(year=2017):
    try:
        income_tree = xgb.Booster()
        income_tree.load_model(f"./.cache/ols-model{year}.json")
    except:
        income_features_2017, income_2017, employed_2017 = get_data(year=year, features=['AGEP','SCHL','MAR','RELP','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','SEX','RAC1P'], outcome='PINCP')
        age_2017 = income_features_2017['AGEP'].to_numpy()[employed_2017]
        income_2017 = income_2017.to_numpy()[employed_2017]
        sex_2017 = income_features_2017['SEX'].to_numpy()[employed_2017]
        income_features_2017 = income_features_2017.to_numpy()[employed_2017,:]
        income_tree = train_eval_regressor(income_features_2017, income_2017)
        os.makedirs("./.cache/", exist_ok=True)
        income_tree.save_model(f"./.cache/ols-model{year}.json")
    return income_tree

def trial(ols_features_2018, income_2018, predicted_income_2018, n, alpha):
    X_labeled, X_unlabeled, Y_labeled, Y_unlabeled, Yhat_labeled, Yhat_unlabeled = train_test_split(ols_features_2018, income_2018, predicted_income_2018, train_size=n)
    X = np.concatenate([X_labeled, X_unlabeled],axis=0)

    Yhat = np.concatenate([Yhat_labeled, Yhat_unlabeled], axis=0)

    naive_interval = standard_ols_interval(X, Yhat, alpha, sandwich=False)

    classical_interval = standard_ols_interval(X_labeled, Y_labeled, alpha, sandwich=False)

    pp_interval = pp_ols_interval(X_labeled, X_unlabeled, Y_labeled, Yhat_labeled, Yhat_unlabeled, alpha, sandwich=False)

    return naive_interval, classical_interval, pp_interval

def make_plots(df, true):
    # Line plots
    ns = np.sort(np.unique(df["n"]))
    plot_data = df[["coefficient", "estimator","width", "n"]].groupby(["coefficient", "estimator","n"], group_keys=False).mean()["width"].reset_index()

    fig, axs = plt.subplots(ncols=2, figsize=(7.5, 2.5))
    my_palette = sns.color_palette(["#71D26F","#BFB9B9"], 2)
    sns.set_theme(style="white", palette=my_palette)
    lplt = sns.lineplot(data=plot_data[(plot_data["coefficient"] == "age") & (plot_data["estimator"] != "naive")], x="n", y="width", hue="estimator", ax=axs[0], hue_order=["prediction-powered", "classical"])
    axs[0].set_ylabel("mean CI width ($/yr of age)")
    sns.despine(ax=axs[0],top=True,right=True)
    lplt.get_legend().remove()
    lplt = sns.lineplot(data=plot_data[(plot_data["coefficient"] == "sex") & (plot_data["estimator"] != "naive")], x="n", y="width", hue="estimator", ax=axs[1], hue_order=["prediction-powered", "classical"])
    axs[1].set_ylabel("mean CI width ($)")
    sns.despine(ax=axs[1],top=True,right=True)
    lplt.get_legend().set_title(None)
    plt.tight_layout()
    plt.savefig('./ols-plots/width-lineplot.pdf')

    make_histograms(df[df["n"] == ns.min()])

    make_intervals(df[df["n"] == ns.min()], true)

def make_intervals(df, true):
    ci_naive = df[(df["coefficient"] == "age") & (df["estimator"] == "naive")]
    ci_naive = [ci_naive["lb"].mean(), ci_naive["ub"].mean()]
    ci_classical = df[(df["coefficient"] == "age") & (df["estimator"] == "classical")]
    ci_classical = [ci_classical["lb"].mean(), ci_classical["ub"].mean()]
    ci = df[(df["coefficient"] == "age") & (df["estimator"] == "prediction-powered")]
    ci = [ci["lb"].mean(), ci["ub"].mean()]
    my_palette = sns.color_palette(["#71D26F","#BFB9B9","#D0A869"], 3)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7.5,2))
    axs[0].plot([ci[0], ci[1]],[0.8,0.8], linewidth=3, color="#71D26F", label='prediction-powered')
    axs[0].plot([ci_classical[0], ci_classical[1]],[0.5, 0.5], linewidth=3, color="#BFB9B9", label='classical')
    axs[0].plot([ci_naive[0], ci_naive[1]],[0.3, 0.3], linewidth=3, color="#FFCD82", label='naive')
    axs[0].vlines(true[0], ymin=0.0, ymax=1, linestyle="dotted", linewidth=3, label="ground truth", color="#F7AE7C")
    axs[0].set_xlabel("age coefficient")
    axs[0].set_yticks([])
    axs[0].set_yticklabels([])
    axs[0].xaxis.set_tick_params()
    axs[0].set_ylim([0,1])
    axs[0].set_xlim([None, None])
    sns.despine(ax=axs[0],top=True,right=True,left=True)

    # Sex coeff
    ci_naive = df[(df["coefficient"] == "sex") & (df["estimator"] == "naive")]
    ci_naive = [ci_naive["lb"].mean(), ci_naive["ub"].mean()]
    ci_classical = df[(df["coefficient"] == "sex") & (df["estimator"] == "classical")]
    ci_classical = [ci_classical["lb"].mean(), ci_classical["ub"].mean()]
    ci = df[(df["coefficient"] == "sex") & (df["estimator"] == "prediction-powered")]
    ci = [ci["lb"].mean(), ci["ub"].mean()]
    axs[1].plot([ci[0], ci[1]],[0.8,0.8], linewidth=3, color="#71D26F", label='prediction-powered')
    axs[1].plot([ci_classical[0], ci_classical[1]],[0.5, 0.5], linewidth=3, color="#BFB9B9", label='classical')
    axs[1].plot([ci_naive[0], ci_naive[1]],[0.3, 0.3], linewidth=3, color="#FFCD82", label='naive')
    axs[1].vlines(true[0], ymin=0.0, ymax=1, linestyle="dotted", linewidth=3, label="ground truth", color="#F7AE7C")
    axs[1].legend()
    axs[1].set_xlabel("sex coefficient")
    axs[1].set_yticks([])
    axs[1].set_yticklabels([])
    axs[1].xaxis.set_tick_params()
    axs[1].set_ylim([0,1])
    axs[1].set_xlim([None, None])
    sns.despine(ax=axs[1],top=True,right=True,left=True)

    plt.savefig('./ols-plots/intervals.pdf', bbox_inches='tight')
    plt.show()


def make_histograms(df):
    my_palette = sns.color_palette(["#71D26F","#BFB9B9"], 2)

    # Width figure
    fig, axs = plt.subplots(ncols=2, figsize=(7.5, 2.5))
    sns.set_theme(style="white", palette=my_palette)
    kde0 = sns.kdeplot(df[(df["coefficient"]=="age") & (df["estimator"] != "naive")], ax=axs[0], x="width", hue="estimator", hue_order=["prediction-powered", "classical"], fill=True, clip=(0,None), cut=0)
    axs[0].set_ylabel("")
    axs[0].set_xlabel("width (age coefficient, $/yr of age)")
    axs[0].set_yticks([])
    axs[0].set_yticklabels([])
    kde0.get_legend().remove()
    sns.despine(ax=axs[0],top=True,right=True,left=True)

    sns.kdeplot(df[(df["coefficient"]=="age") & (df["estimator"] != "naive")], ax=axs[1], x="width", hue="estimator", hue_order=["prediction-powered", "classical"], fill=True, clip=(0,None), cut=0)
    axs[1].set_ylabel("")
    axs[1].set_xlabel("width (sex coefficient, $)")
    axs[1].set_yticks([])
    axs[1].set_yticklabels([])
    sns.despine(ax=axs[1],top=True,right=True,left=True)
    fig.suptitle("") # This is here for spacing
    plt.tight_layout()
    axs[1].legend(["classical", "prediction-powered"], bbox_to_anchor = (1.,1.2) )
    plt.savefig('./ols-plots/width.pdf')

    cvg_classical_age = (df[(df["estimator"]=="classical") & (df["coefficient"]=="age")]["covered"]).astype(int).mean()
    cvg_classical_sex = (df[(df["estimator"]=="classical") & (df["coefficient"]=="sex")]["covered"]).astype(int).mean()
    cvg_predictionpowered_age = (df[(df["estimator"]=="prediction-powered") & (df["coefficient"]=="age")]["covered"]).astype(int).mean()
    cvg_predictionpowered_sex = (df[(df["estimator"]=="prediction-powered") & (df["coefficient"]=="sex")]["covered"]).astype(int).mean()

    print(f"Classical coverage ({cvg_classical_age},{cvg_classical_sex}), prediction-powered ({cvg_predictionpowered_age},{cvg_predictionpowered_sex})")

if __name__ == "__main__":
    os.makedirs('./ols-plots', exist_ok=True)
    income_features_2018, income_2018, employed_2018 = get_data(year=2018, features=['AGEP','SCHL','MAR','RELP','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','SEX','RAC1P'], outcome='PINCP')
    age_2018 = income_features_2018['AGEP'].to_numpy()[employed_2018]
    income_2018 = income_2018.to_numpy()[employed_2018]
    sex_2018 = income_features_2018['SEX'].to_numpy()[employed_2018]
    income_features_2018 = income_features_2018.to_numpy()[employed_2018,:]
    plot_data(age_2018, income_2018, sex_2018)

    # OLS solution
    ols_features_2018 = np.stack([age_2018, sex_2018], axis=1)
    true = ols(ols_features_2018, income_2018)

    try:
        df = pd.read_pickle('./.cache/ols-results.pkl')
    except:
        # Train tree on 2017 data
        np.random.seed(0) # Fix seed for tree
        income_tree = get_tree()
        np.random.seed(0) # Fix seed for evaluation

        # Evaluate Tree
        predicted_income_2018 = income_tree.predict(xgb.DMatrix(income_features_2018))

        # Collect OLS features and do MAI
        print(f"True OLS coefficients: {true}")
        N = ols_features_2018.shape[0]
        num_n = 50
        ns = np.linspace(2000, 5000, num_n).astype(int)
        num_trials = 100
        alpha = 0.05

        # Store results
        columns = ["lb","ub","covered","estimator","coefficient","n"]

        results = []
        for i in tqdm(range(num_trials)):
            for j in range(ns.shape[0]):
                n = ns[j]
                ii, ci, ppi = trial(ols_features_2018, income_2018, predicted_income_2018, n, alpha)
                temp_df = pd.DataFrame(np.zeros((6,len(columns))), columns=columns)
                temp_df.loc[0] = ii[0][0], ii[1][0], (ii[0][0] <= true[0]) & (true[0] <= ii[1][0]), "naive", "age", n
                temp_df.loc[1] = ci[0][0], ci[1][0], (ci[0][0] <= true[0]) & (true[0] <= ci[1][0]), "classical", "age", n
                temp_df.loc[2] = ppi[0][0], ppi[1][0], (ppi[0][0] <= true[0]) & (true[0] <= ppi[1][0]), "prediction-powered", "age", n
                temp_df.loc[3] = ii[0][1], ii[1][1], (ii[0][1] <= true[1]) & (true[1] <= ii[1][1]), "naive", "sex", n
                temp_df.loc[4] = ci[0][1], ci[1][1], (ci[0][1] <= true[1]) & (true[1] <= ci[1][1]), "classical", "sex", n
                temp_df.loc[5] = ppi[0][1], ppi[1][1], (ppi[0][1] <= true[1]) & (true[1] <= ppi[1][1]), "prediction-powered", "sex", n
                results += [temp_df]
        df = pd.concat(results)
        df["width"] = df["ub"] - df["lb"]
        df.to_pickle('./.cache/ols-results.pkl')
    make_plots(df, true)
