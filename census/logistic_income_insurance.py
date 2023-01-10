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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from tqdm import tqdm

from concentration import logistic, standard_logistic_interval, pp_logistic_interval

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
    sns.set_theme(style="white", palette="pastel", font_scale=1.15)
    bins = [20000,40000,60000,80000,100000]
    incomeranges = np.digitize(pincp, bins=bins)
    avgs = [privcov[incomeranges == i].mean() for i in range(len(bins)+1)]
    plt.bar(range(len(bins)+1),avgs)
    plt.gca().set_xticklabels(['', '<20K', '20K-40K','40K-60K','60K-80K','80K-100K', '>100K'])
    plt.ylabel('frac w/private insurance')
    plt.gca().set_ylim([0,1])
    plt.xlabel('household income ($)')
    sns.despine(top=True, right=True)
    plt.subplots_adjust(top=0.9, bottom=0.28)
    plt.savefig("./logistic-plots/raw_data.pdf")

def trial(X, Y, Yhat, true, n, alpha):
    X_labeled, X_unlabeled, Y_labeled, Y_unlabeled, Yhat_labeled, Yhat_unlabeled = train_test_split(X, Y, Yhat, train_size=n)

    naive_interval = standard_logistic_interval(X, Yhat, alpha)

    classical_interval = standard_logistic_interval(X_labeled, Y_labeled, alpha)

    pp_interval = pp_logistic_interval(X_labeled, X_unlabeled, Y_labeled, Yhat_labeled, Yhat_unlabeled, alpha)

    return naive_interval, classical_interval, pp_interval

def make_plots(df, true):
    # Line plots
    ns = np.sort(np.unique(df["n"]))

    my_palette = sns.color_palette(["#71D26F","#BFB9B9","#D0A869"], 3)
    sns.set_theme(style="white", palette=my_palette, font_scale=1.2)
    fig, axs = plt.subplots(ncols=3, figsize=(11, 2.5))

    make_histograms(df[df["n"] == ns.min()], axs[0])

    make_lineplots(df, axs[1])

    make_intervals(df[df["n"] == ns.min()], true, axs[2])

    #plt.subplots_adjust(right=0.8, left=0.1, top=0.8, bottom=0.2)
    plt.tight_layout()

    plt.savefig('./logistic-plots/results.pdf')

def make_lineplots(df, ax):
    plot_df = df[["estimator","width", "n"]].groupby(["estimator","n"], group_keys=False).mean()["width"].reset_index()
    lplt = sns.lineplot(data=plot_df[plot_df["estimator"] != "naive"], x="n", y="width", hue="estimator", ax=ax, hue_order=["prediction-powered", "classical"])
    ax.set_ylabel("mean width")
    ax.set_xlabel("n")
    ax.xaxis.set_tick_params()
    ax.yaxis.set_tick_params()
    ax.locator_params(tight=True, nbins=4)
    lplt.get_legend().remove()
    sns.despine(ax=ax,top=True,right=True)

def make_intervals(df, true, ax):
    ci_naive = df[df["estimator"] == "naive"]
    ci_naive = [ci_naive["lb"].mean(), ci_naive["ub"].mean()]
    ci_classical = df[df["estimator"] == "classical"]
    ci_classical = [ci_classical["lb"].mean(), ci_classical["ub"].mean()]
    ci = df[df["estimator"] == "prediction-powered"]
    ci = [ci["lb"].mean(), ci["ub"].mean()]

    ax.plot([ci[0], ci[1]],[0.4,0.4], linewidth=20, color="#DAF3DA", path_effects=[pe.Stroke(linewidth=22, offset=(-1,0), foreground="#71D26F"), pe.Stroke(linewidth=22, offset=(1,0), foreground="#71D26F"), pe.Normal()], label='prediction-powered', solid_capstyle='butt')
    ax.plot([ci_classical[0], ci_classical[1]],[0.25, 0.25], linewidth=20, color="#EEEDED", path_effects=[pe.Stroke(linewidth=22, offset=(-1,0), foreground="#BFB9B9"), pe.Stroke(linewidth=22, offset=(1,0), foreground="#BFB9B9"), pe.Normal()], label='no ML', solid_capstyle='butt')
    ax.plot([ci_naive[0], ci_naive[1]],[0.1, 0.1], linewidth=20, color="#FFEACC", path_effects=[pe.Stroke(linewidth=22, offset=(-1,0), foreground="#FFCD82"), pe.Stroke(linewidth=22, offset=(1,0), foreground="#FFCD82"), pe.Normal()], label='naive ML', solid_capstyle='butt')
    ax.vlines(true[0], ymin=0.0, ymax=1, linestyle="dotted", linewidth=3, label="ground truth", color="#F7AE7C")
    ax.set_xlabel("coefficient")
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.xaxis.set_tick_params()
    ax.set_ylim([0,0.5])
    ax.set_xlim([None, None])
    ax.legend(bbox_to_anchor = (1.1,1), borderpad=1, labelspacing = 1)
    sns.despine(ax=ax,top=True,right=True,left=True)

def make_histograms(df, ax):
    # Width figure
    kde0 = sns.kdeplot(df[df["estimator"] != "naive"], ax=ax, x="width", hue="estimator", hue_order=["prediction-powered", "classical"], fill=True, clip=(0,None))
    ax.set_ylabel("")
    ax.set_xlabel("width")
    ax.set_yticks([])
    ax.set_yticklabels([])
    kde0.get_legend().remove()
    sns.despine(ax=ax,top=True,right=True,left=True)

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


if __name__ == "__main__":
    os.makedirs('./logistic-plots', exist_ok=True)
    normalize = True
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
    true = logistic(X, privcov_2018)

    print(f"True logistic regression coefficients: {true}")

    try:
        df = pd.read_pickle('./.cache/logistic-results.pkl')
    except:
        N = features_2018.shape[0]
        num_n = 10
        ns = np.linspace(1000, 10000, num_n).astype(int)
        num_trials = 10
        alpha = 0.05

        # Store results
        columns = ["lb","ub","covered","estimator","n"]

        results = []
        for j in range(ns.shape[0]):
            for i in range(num_trials):
                n = ns[j]
                ii, ci, ppi = trial(X, privcov_2018.to_numpy(), predicted_privcov_2018, true, n, alpha)
                temp_df = pd.DataFrame(np.zeros((3,len(columns))), columns=columns)
                temp_df.loc[0] = ii[0][0], ii[1][0], (ii[0][0] <= true[0]) & (true[0] <= ii[1][0]), "naive", n
                temp_df.loc[1] = ci[0][0], ci[1][0], (ci[0][0] <= true[0]) & (true[0] <= ci[1][0]), "classical", n
                temp_df.loc[2] = ppi[0][0], ppi[1][0], (ppi[0][0] <= true[0]) & (true[0] <= ppi[1][0]), "prediction-powered", n
                results += [temp_df]
        df = pd.concat(results)
        df["width"] = df["ub"] - df["lb"]
        df.to_pickle('./.cache/logistic-results.pkl')

    make_plots(df, true)
