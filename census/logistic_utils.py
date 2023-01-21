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
from ols_utils import transform_features

from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from tqdm import tqdm

def acs_filter(df, outcome_name, reg_feat_name):
    df = df[np.bitwise_not(np.isnan(df[outcome_name]))]
    df = df[np.bitwise_not(np.isnan(df[reg_feat_name]))]
    df[outcome_name] = 1 - (df[outcome_name] - 1)
    assert (np.unique(df[outcome_name]) == np.array([0,1])).all()
    return df

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
        tree.load_model(f"./.cache/logistic-model{year}.json")
    except:
        features, outcome = get_data(year, feature_names, outcome_name, reg_feat_name, acs_filter=acs_filter)
        if transform:
            print("Transforming features and training tree.")
            tree = train_eval_regressor(transform_features(features, ft, enc)[0], outcome, transform=transform)
        else:
            print("Training tree without transformation.")
            tree = train_eval_regressor(features, outcome)
        os.makedirs("./.cache/", exist_ok=True)
        tree.save_model(f"./.cache/logistic-model{year}.json")
    return tree

def train_eval_regressor(features, outcome, transform=False):
    X_train, X_test, y_train, y_test = train_test_split(features, outcome, test_size=0.1)
    if transform:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
    else:
        dtrain = xgb.DMatrix(X_train.to_numpy(), label=y_train.to_numpy())
        dtest = xgb.DMatrix(X_test.to_numpy(), label=y_test.to_numpy())
    param = {'max_depth': 7, 'eta': 0.1, 'objective': 'binary:logistic', 'eval_metric': ['error', 'mae']}
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 500
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
    os.makedirs("./logistic-plots/", exist_ok=True)
    plt.savefig("./logistic-plots/raw_data.pdf")

def make_plots(df, true):
    # Line plots
    ns = np.sort(np.unique(df["n"]))

    my_palette = sns.color_palette(["#71D26F","#BFB9B9","#D0A869"], 3)
    sns.set_theme(style="white", palette=my_palette, font_scale=1.2)
    fig, axs = plt.subplots(ncols=3, figsize=(11, 2.5))

    make_histograms(df[df["n"] == ns.min()], axs[0])

    make_lineplots(df, axs[1])

    make_intervals(df[df["n"] == ns.min()], true, axs[2])

    plt.tight_layout()

    os.makedirs("./logistic-plots/", exist_ok=True)
    plt.savefig('./logistic-plots/results.pdf')

def make_lineplots(df, ax):
    plot_df = df[["estimator","width", "n"]].groupby(["estimator","n"], group_keys=False).mean()["width"].reset_index()
    lplt = sns.lineplot(data=plot_df[plot_df["estimator"] != "imputed"], x="n", y="width", hue="estimator", ax=ax, hue_order=["prediction-powered", "classical"])
    ax.set_ylabel("mean width")
    ax.set_xlabel("n")
    ax.xaxis.set_tick_params()
    ax.yaxis.set_tick_params()
    ax.locator_params(tight=True, nbins=4)
    lplt.get_legend().remove()
    sns.despine(ax=ax,top=True,right=True)

def make_intervals(df, true, ax):
    ci_imputed = df[df["estimator"] == "imputed"]
    epsilon = 1e-7 # Add a bit to the imputed estimate, because otherwise it is not visible on the plot.
    ci_imputed = [ci_imputed["lb"].mean() - epsilon,  ci_imputed["ub"].mean() + epsilon]
    ci_classical = df[df["estimator"] == "classical"]
    ci_classical = [ci_classical["lb"].mean(), ci_classical["ub"].mean()]
    ci = df[df["estimator"] == "prediction-powered"]
    ci = [ci["lb"].mean(), ci["ub"].mean()]

    ax.plot([ci[0], ci[1]],[0.4,0.4], linewidth=20, color="#DAF3DA", path_effects=[pe.Stroke(linewidth=22, offset=(-1,0), foreground="#71D26F"), pe.Stroke(linewidth=22, offset=(1,0), foreground="#71D26F"), pe.Normal()], label='prediction-powered', solid_capstyle='butt')
    ax.plot([ci_classical[0], ci_classical[1]],[0.25, 0.25], linewidth=20, color="#EEEDED", path_effects=[pe.Stroke(linewidth=22, offset=(-1,0), foreground="#BFB9B9"), pe.Stroke(linewidth=22, offset=(1,0), foreground="#BFB9B9"), pe.Normal()], label='classical', solid_capstyle='butt')
    ax.plot([ci_imputed[0], ci_imputed[1]],[0.1, 0.1], linewidth=20, color="#FFEACC", path_effects=[pe.Stroke(linewidth=22, offset=(-1,0), foreground="#FFCD82"), pe.Stroke(linewidth=22, offset=(1,0), foreground="#FFCD82"), pe.Normal()], label='imputed', solid_capstyle='butt')
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
    kde0 = sns.kdeplot(df[df["estimator"] != "imputed"], ax=ax, x="width", hue="estimator", hue_order=["prediction-powered", "classical"], fill=True, clip=(0,None))
    ax.set_ylabel("")
    ax.set_xlabel("width")
    ax.set_yticks([])
    ax.set_yticklabels([])
    kde0.get_legend().remove()
    sns.despine(ax=ax,top=True,right=True,left=True)
