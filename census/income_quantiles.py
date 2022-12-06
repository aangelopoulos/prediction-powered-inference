import os, sys
sys.path.insert(1, '../')
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
from ols_age_sex_income import get_data, get_tree
from concentration import binomial_iid, wsr_iid

def plot_data(income):
    plt.figure(figsize=(7.5,2.5))
    sns.set_theme(style="white", palette="pastel")
    sns.kdeplot(x=income)
    plt.xlabel('income ($)')
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig("./income-quantile-plots/raw_data.pdf")

def trial(income_2018, predicted_income_2018, median_true, N, n, delta):
    Y_labeled, Y_unlabeled, Yhat_labeled, Yhat_unlabeled = train_test_split(income_2018, predicted_income_2018, train_size=n)

    theta_grid = np.linspace(0,100000,500)

    classical_Cthetas = [binomial_iid(n, delta, (Y_labeled <= theta).mean()) for theta in theta_grid]
    classical_indexes = np.array([ (classical_Cthetas[j][0] <= 0.5) & (classical_Cthetas[j][1] >= 0.5) for j in range(theta_grid.shape[0])])
    if classical_indexes.sum() <= 1:
        temp = np.stack(classical_Cthetas,axis=0)
        idx = np.argmin(np.abs(temp[:,1] - 0.5))
        classical_indexes[idx] = True
        classical_indexes[idx-1] = True
    classical_interval = theta_grid[np.where(classical_indexes)[0]]
    classical_interval = np.array([classical_interval.min(), classical_interval.max()])

    Cthetas = [wsr_iid( ((Y_labeled <= theta).astype(float) - (Yhat_labeled <= theta).astype(float) + 1)/2, delta, np.linspace(0.1,0.9,100), step=n)*2-1 for theta in theta_grid]
    d_median = [0.5 - (predicted_income_2018 <= theta).mean() for theta in theta_grid]
    indexes = np.array([ (Cthetas[j][0] <= d_median[j]) & (Cthetas[j][1] >= d_median[j]) for j in range(theta_grid.shape[0])])
    if indexes.sum() <= 1:
        temp = np.stack(Cthetas,axis=0)
        idx = np.argmin(np.abs(temp[:,1] - d_median))
        indexes[idx] = True
        indexes[idx-1] = True
    predictionpowered_interval = theta_grid[np.where(indexes)[0]] # Check if 0.5 - (1/N)\sum_{i=1}^N 1(\tilde{f}_i <= \theta) is inside range
    predictionpowered_interval = np.array([predictionpowered_interval.min(), predictionpowered_interval.max()])

    classical_width = classical_interval.max() - classical_interval.min()
    predictionpowered_width = predictionpowered_interval.max() - predictionpowered_interval.min()

    classical_covered = (classical_interval[0] <= median_true) & (classical_interval[1] >= median_true)
    predictionpowered_covered = (predictionpowered_interval[0] <= median_true) &  (predictionpowered_interval[1] >= median_true)

    return classical_width, predictionpowered_width, classical_covered, predictionpowered_covered

def make_kde(df):
    my_palette = sns.color_palette(["#71D26F","#BFB9B9"], 2)
    # Width figure
    plt.figure()
    sns.set_theme(style="white", palette=my_palette)
    kde = sns.kdeplot(df, x="width", hue="estimator", hue_order=["prediction-powered", "classical"], fill=True, clip=(0,None))
    plt.gca().set_ylabel("")
    plt.gca().set_xlabel("width ($)")
    plt.gca().set_yticks([])
    plt.gca().set_yticklabels([])
    sns.despine(ax=plt.gca(),top=True,right=True,left=True)
    kde.get_legend().remove()
    plt.gca().legend(["classical", "prediction-powered"], bbox_to_anchor = (1.,1.) )
    plt.savefig('./income-quantile-plots/width.pdf')

if __name__ == "__main__":
    os.makedirs('./income-quantile-plots', exist_ok=True)
    try:
        df = pd.read_pickle('./.cache/income-quantile-results.pkl')
    except:
        # Train tree on 2017 data
        np.random.seed(0) # Fix seed for tree
        income_tree = get_tree()
        np.random.seed(0) # Fix seed for evaluation

        # Evaluate tree and plot data in 2018
        income_features_2018, income_2018, _ = get_data(year=2018, features=['AGEP','SCHL','MAR','RELP','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','SEX','RAC1P'], outcome='PINCP')
        income_2018 = income_2018.to_numpy()
        income_features_2018 = income_features_2018.to_numpy()
        predicted_income_2018 = income_tree.predict(xgb.DMatrix(income_features_2018))
        plot_data(income_2018)

        # Compute the true quantiles
        median_true = np.median(income_2018)
        print(f"True median income: {median_true}")
        N = predicted_income_2018.shape[0]
        n = 100
        num_trials = 100
        delta = 0.05

        # Store results
        columns = ["width","covered","estimator"]
        df = pd.DataFrame(np.zeros((num_trials*2,len(columns))), columns=columns)

        for i in tqdm(range(num_trials)):
            classical_width, predictionpowered_width, classical_covered, predictionpowered_covered  = trial(income_2018, predicted_income_2018, median_true, N, n, delta)
            df.loc[i] = classical_width, int(classical_covered), "classical"
            df.loc[i+num_trials] = predictionpowered_width, int(predictionpowered_covered), "prediction-powered"
        df.to_pickle('./.cache/income-quantile-results.pkl')
    pdb.set_trace()
    make_kde(df)
