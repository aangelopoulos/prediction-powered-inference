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

import scipy
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
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

def transform_features(features, ft, enc=None):
    c_features = features.T[ft == "c"].T.astype(str)
    if enc is None:
        enc = OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse=False)
        enc.fit(c_features)
    c_features = enc.transform(c_features)
    features = scipy.sparse.csc_matrix(np.concatenate([features.T[ft == "q"].T.astype(float), c_features], axis=1))
    return features, enc

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
        tree = xgb.train(params, dtrain, num_round, evallist, enable_categorical=True)
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
    tree = xgb.train({'eta': 0.3, 'max_depth': 7, 'objective': 'reg:pseudohubererror'}, dtrain, 10000, evallist)
    return tree


def plot_data(age,income,sex):
    plt.figure(figsize=(7.5,2.5))
    sns.set_theme(style="white", palette="pastel", font_scale=1.5)
    ageranges = np.digitize(age, bins=[0,20,30,40,50])
    sex = np.array(['female' if s==2 else 'male' for s in sex])
    sns.boxplot(x=ageranges, y=income, hue=sex, showfliers=False)
    plt.gca().set_xticklabels(['0-20','20-30','30-40','40-50','50+'])
    plt.ylabel('income ($)')
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig("./ols-plots/raw_data.pdf")

def get_tree(year, features, ft, enc=None):
    try:
        income_tree = xgb.Booster()
        income_tree.load_model(f"./.cache/ols-model{year}.json")
    except:
        income_features, income, employed = get_data(year=year, features=features, outcome='PINCP')
        income_tree = train_eval_regressor(transform_features(income_features, ft, enc)[0], income.to_numpy())
        os.makedirs("./.cache/", exist_ok=True)
        income_tree.save_model(f"./.cache/ols-model{year}.json")
    return income_tree

def trial(ols_features_2019, income_2019, predicted_income_2019, n, alpha, sandwich=True):
    X_labeled, X_unlabeled, Y_labeled, Y_unlabeled, Yhat_labeled, Yhat_unlabeled = train_test_split(ols_features_2019, income_2019, predicted_income_2019, train_size=n)
    X = np.concatenate([X_labeled, X_unlabeled],axis=0)

    Yhat = np.concatenate([Yhat_labeled, Yhat_unlabeled], axis=0)

    naive_interval = standard_ols_interval(X, Yhat, alpha, sandwich=sandwich)

    classical_interval = standard_ols_interval(X_labeled, Y_labeled, alpha, sandwich=sandwich)

    pp_interval = pp_ols_interval(X_labeled, X_unlabeled, Y_labeled, Yhat_labeled, Yhat_unlabeled, alpha, sandwich=sandwich)

    return naive_interval, classical_interval, pp_interval

def make_plots(df, true):
    # Line plots
    ns = np.sort(np.unique(df["n"]))

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 5))
    my_palette = sns.color_palette(["#71D26F","#BFB9B9","#D0A869"], 3)
    sns.set_theme(style="white", palette=my_palette, font_scale=1)

    make_intervals(df[df["n"] == ns.min()], true, axs[0,:])

    make_histograms(df[df["n"] == ns.min()], axs[1,:])

    make_lineplots(df, axs[2,:])

    plt.tight_layout()
    plt.savefig('./ols-plots/results.pdf')

def make_lineplots(df, axs):
    plot_df = df[["coefficient", "estimator","width", "n"]].groupby(["coefficient", "estimator","n"], group_keys=False).mean()["width"].reset_index()
    lplt = sns.lineplot(data=plot_df[(plot_df["coefficient"] == "age") & (plot_df["estimator"] != "naive")], x="n", y="width", hue="estimator", ax=axs[0], hue_order=["prediction-powered", "classical"])
    axs[0].set_ylabel("mean width ($/yr)")
    axs[0].set_xlabel("n")
    axs[0].xaxis.set_tick_params()
    axs[0].yaxis.set_tick_params()
    sns.despine(ax=axs[0],top=True,right=True)
    lplt.get_legend().remove()
    lplt = sns.lineplot(data=plot_df[(plot_df["coefficient"] == "sex") & (plot_df["estimator"] != "naive")], x="n", y="width", hue="estimator", ax=axs[1], hue_order=["prediction-powered", "classical"])
    axs[1].set_ylabel("mean width ($)")
    axs[1].set_xlabel("n")
    axs[1].xaxis.set_tick_params()
    axs[1].yaxis.set_tick_params()
    sns.despine(ax=axs[1],top=True,right=True)
    lplt.get_legend().remove()

def make_intervals(df, true, axs):
    ci_naive = df[(df["coefficient"] == "age") & (df["estimator"] == "naive")]
    ci_naive = [ci_naive["lb"].mean(), ci_naive["ub"].mean()]
    ci_classical = df[(df["coefficient"] == "age") & (df["estimator"] == "classical")]
    ci_classical = [ci_classical["lb"].mean(), ci_classical["ub"].mean()]
    ci = df[(df["coefficient"] == "age") & (df["estimator"] == "prediction-powered")]
    ci = [ci["lb"].mean(), ci["ub"].mean()]

    axs[0].plot([ci[0], ci[1]],[0.8,0.8], linewidth=10, color="#DAF3DA", path_effects=[pe.Stroke(linewidth=11, offset=(-0.5,0), foreground="#71D26F"), pe.Stroke(linewidth=11, offset=(0.5,0), foreground="#71D26F"), pe.Normal()], label='prediction-powered', solid_capstyle="butt")
    axs[0].plot([ci_classical[0], ci_classical[1]],[0.5, 0.5], linewidth=10, color="#EEEDED", path_effects=[pe.Stroke(linewidth=11, offset=(-0.5,0), foreground="#BFB9B9"), pe.Stroke(linewidth=11, offset=(0.5,0), foreground="#BFB9B9"), pe.Normal()],  label='classical', solid_capstyle="butt")
    axs[0].plot([ci_naive[0], ci_naive[1]],[0.2, 0.2], linewidth=10, color="#FFEACC", path_effects=[pe.Stroke(linewidth=11, offset=(-0.5,0), foreground="#FFCD82"), pe.Stroke(linewidth=11, offset=(0.5,0), foreground="#FFCD82"), pe.Normal()],  label='imputed', solid_capstyle="butt")
    axs[0].vlines(true[0], ymin=0.0, ymax=1, linestyle="dotted", linewidth=3, label="ground truth", color="#F7AE7C")
    axs[0].set_xlabel("age coeff")
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
    axs[1].plot([ci[0], ci[1]],[0.8,0.8], linewidth=10, color="#DAF3DA", path_effects=[pe.Stroke(linewidth=11, offset=(-0.5,0), foreground="#71D26F"), pe.Stroke(linewidth=11, offset=(0.5,0), foreground="#71D26F"), pe.Normal()], label='prediction-powered', solid_capstyle="butt")
    axs[1].plot([ci_classical[0], ci_classical[1]],[0.5, 0.5], linewidth=10, color="#EEEDED", path_effects=[pe.Stroke(linewidth=11, offset=(-0.5,0), foreground="#BFB9B9"), pe.Stroke(linewidth=11, offset=(0.5,0), foreground="#BFB9B9"), pe.Normal()],  label='classical', solid_capstyle="butt")
    axs[1].plot([ci_naive[0], ci_naive[1]],[0.2, 0.2], linewidth=10, color="#FFEACC", path_effects=[pe.Stroke(linewidth=11, offset=(-0.5,0), foreground="#FFCD82"), pe.Stroke(linewidth=11, offset=(0.5,0), foreground="#FFCD82"), pe.Normal()],  label='imputed', solid_capstyle="butt")
    axs[1].vlines(true[1], ymin=0.0, ymax=1, linestyle="dotted", linewidth=3, label="ground truth", color="#F7AE7C")
    axs[1].set_xlabel("sex coeff")
    axs[1].set_yticks([])
    axs[1].set_yticklabels([])
    axs[1].legend(bbox_to_anchor = (1,1), borderpad=1, fontsize=15)
    axs[1].xaxis.set_tick_params()
    axs[1].locator_params(axis='x', nbins=4)
    axs[1].set_ylim([0,1])
    axs[1].set_xlim([None, None])
    sns.despine(ax=axs[1],top=True,right=True,left=True)

def make_histograms(df, axs):
    # Width figure
    kde0 = sns.kdeplot(df[(df["coefficient"]=="age") & (df["estimator"] != "naive")], ax=axs[0], x="width", hue="estimator", hue_order=["prediction-powered", "classical"], fill=True, clip=(0,None), cut=0)
    axs[0].set_ylabel("")
    axs[0].set_xlabel("width (age coeff, $/yr)")
    axs[0].set_yticks([])
    axs[0].set_yticklabels([])
    kde0.get_legend().remove()
    sns.despine(ax=axs[0],top=True,right=True,left=True)

    kde1 = sns.kdeplot(df[(df["coefficient"]=="age") & (df["estimator"] != "naive")], ax=axs[1], x="width", hue="estimator", hue_order=["prediction-powered", "classical"], fill=True, clip=(0,None), cut=0)
    axs[1].set_ylabel("")
    axs[1].set_xlabel("width (sex coeff, $)")
    axs[1].set_yticks([])
    axs[1].set_yticklabels([])
    kde1.get_legend().remove()
    sns.despine(ax=axs[1],top=True,right=True,left=True)
    #plt.tight_layout()

    cvg_classical_age = (df[(df["estimator"]=="classical") & (df["coefficient"]=="age")]["covered"]).astype(int).mean()
    cvg_classical_sex = (df[(df["estimator"]=="classical") & (df["coefficient"]=="sex")]["covered"]).astype(int).mean()
    cvg_predictionpowered_age = (df[(df["estimator"]=="prediction-powered") & (df["coefficient"]=="age")]["covered"]).astype(int).mean()
    cvg_predictionpowered_sex = (df[(df["estimator"]=="prediction-powered") & (df["coefficient"]=="sex")]["covered"]).astype(int).mean()

    print(f"Classical coverage ({cvg_classical_age},{cvg_classical_sex}), prediction-powered ({cvg_predictionpowered_age},{cvg_predictionpowered_sex})")

if __name__ == "__main__":
    os.makedirs('./ols-plots', exist_ok=True)
    features = ['AGEP','SCHL','MAR','DIS','ESP','CIT','MIG','MIL','ANC1P','NATIVITY','DEAR','DEYE','DREM','SEX','RAC1P', 'SOCP', 'COW']
    ft = np.array(["q", "q", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c"])
    income_features_2019, income_2019, employed_2019 = get_data(year=2019, features=features, outcome='PINCP')
    age_2019 = income_features_2019['AGEP'].to_numpy()
    income_2019 = income_2019.to_numpy()
    sex_2019 = income_features_2019['SEX'].to_numpy()
    income_features_2019, enc = transform_features(income_features_2019, ft)
    plot_data(age_2019[employed_2019], income_2019[employed_2019], sex_2019[employed_2019])

    # OLS solution
    ols_features_2019 = np.stack([age_2019, sex_2019], axis=1)
    true = ols(ols_features_2019, income_2019)

    try:
        df = pd.read_pickle('./.cache/ols-results.pkl')
    except:
        # Train tree on 2018 data
        np.random.seed(0) # Fix seed for tree
        income_tree = get_tree(2018, features, ft, enc=enc)
        np.random.seed(0) # Fix seed for evaluation

        # Evaluate Tree
        predicted_income_2019 = income_tree.predict(xgb.DMatrix(income_features_2019))

        # Collect OLS features and do MAI
        print(f"True OLS coefficients: {true}")
        N = ols_features_2019.shape[0]
        num_n = 10
        ns = np.linspace(100, 2000, num_n).astype(int)
        num_trials = 50
        alpha = 0.05

        # Store results
        columns = ["lb","ub","covered","estimator","coefficient","n"]

        results = []
        for i in tqdm(range(num_trials)):
            for j in range(ns.shape[0]):
                n = ns[j]
                ii, ci, ppi = trial(ols_features_2019, income_2019, predicted_income_2019, n, alpha)
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
