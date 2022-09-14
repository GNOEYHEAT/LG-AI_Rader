import numpy as np
import pandas as pd
import os
import argparse
from tqdm import trange

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import Lars, LassoLars, OrthogonalMatchingPursuit
from sklearn.linear_model import (
    BayesianRidge,
    ARDRegression,
    PassiveAggressiveRegressor,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import RANSACRegressor, TheilSenRegressor, HuberRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    BaggingRegressor,
)
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from model import STACKING
from utils import set_seeds

import warnings
warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
upper_dir = os.path.join(current_dir, os.pardir)

seed = 1011
set_seeds(seed)

lr = LinearRegression(n_jobs=-1)
ridge = Ridge(random_state=seed)
lasso = Lasso(random_state=seed)
en = ElasticNet(random_state=seed)
lar = Lars(random_state=seed)
llar = LassoLars(random_state=seed)
omp = OrthogonalMatchingPursuit()
br = MultiOutputRegressor(BayesianRidge())
ard = MultiOutputRegressor(ARDRegression())
par = MultiOutputRegressor(PassiveAggressiveRegressor(random_state=seed))
ransac = RANSACRegressor(random_state=seed)
tr = MultiOutputRegressor(TheilSenRegressor(n_jobs=-1, random_state=seed))
huber = MultiOutputRegressor(HuberRegressor())
kr = KernelRidge()
svm = MultiOutputRegressor(SVR())
knn = KNeighborsRegressor(n_jobs=-1)
dt = DecisionTreeRegressor(random_state=seed)
et = ExtraTreeRegressor(random_state=seed)
bagging = BaggingRegressor(n_jobs=-1, random_state=seed)
ets = ExtraTreesRegressor(n_jobs=-1, random_state=seed)
rf = RandomForestRegressor(n_jobs=-1, random_state=seed)
ada = MultiOutputRegressor(AdaBoostRegressor(random_state=seed))
gbr = MultiOutputRegressor(GradientBoostingRegressor(random_state=seed))
hgbr = MultiOutputRegressor(HistGradientBoostingRegressor(random_state=seed))
xgboost = XGBRegressor(tree_method="gpu_hist", gpu_id=0, n_jobs=-1, random_state=seed)
lightgbm = MultiOutputRegressor(LGBMRegressor(n_jobs=-1, random_state=seed))
catboost = MultiOutputRegressor(
    CatBoostRegressor(task_type="GPU", devices="0", verbose=False, random_state=seed)
)
mlp = MLPRegressor(random_state=seed)

def parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaler", default="standard", type=str)
    parser.add_argument('--n_cv', type=int, default=10)
    parser.add_argument("--mode", type=str, default="train", help="train or experiment")

    return parser.parse_args()

def make_datasets(datasets, scaler, mode, seed):
    
    if mode == "test":
        
        X_train, X_test, y_train, y_test = train_test_split(
        datasets[0], datasets[1], random_state=seed
    )

        X_train = pd.get_dummies(X_train, drop_first=True)
        X_test = pd.get_dummies(X_test, drop_first=True)

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    else:
        
        X_train = datasets[0]
        y_train = datasets[1]
        X_test = datasets[2]

        X_train = pd.get_dummies(X_train, drop_first=True)
        X_test = pd.get_dummies(X_test, drop_first=True)

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train

def main(args):

    n_cv = args.n_cv
    mode = args.mode
    scaler = args.scaler

    if scaler == "standard":
        scaler = StandardScaler()
    elif scaler == "minmax":
        scaler = MinMaxScaler()

    # base_ml = [bagging, ets, rf, xgboost, catboost, lightgbm]
    total_datasets = [noise_datasets, clean_datasets]

    for order, datasets in enumerate(total_datasets):

        if mode == "experiment":

            print("Experiment Start..")
            X_train, X_test, y_train, y_test = make_datasets(datasets, scaler, "test", seed)
            experiment_stacking = STACKING(X_train, X_test, y_train, y_test, base_ml, len(base_ml), seed, "test")
            
            experiment_stacking.run_level0()
            _ = experiment_stacking.run_level1()
            best_ml = experiment_stacking.best_n_ml[: experiment_stacking.final_select_n].tolist()
            print(experiment_stacking.scores)
            # For the noise dataset, it was best to use catboost,
            # For the clean dataset, it was best to use catboost, lightgbm, and ets.

        elif mode == "train":
            
            best_ml = [catboost, ]
            level0_save_path = os.path.join(noise_model_path, "level0")
            level1_save_path = os.path.join(noise_model_path, "level1")
        
            if order == 1:
                best_ml = [catboost, lightgbm, ets,]
                level0_save_path = os.path.join(clean_model_path, "level0")
                level1_save_path = os.path.join(clean_model_path, "level1")

            print("Inference Start..")
            X_train, X_test, y_train = make_datasets(datasets, scaler, "real", seed)
            inference_stacking = STACKING(X_train, X_test, y_train, "_", best_ml, len(best_ml), seed, "real")

            inference_stacking.run_level0(save_path = level0_save_path)
            prediction = inference_stacking.run_level1(save = True, save_path = level1_save_path)
            result = prediction.round(3)
            print(result)

        else:
            print("You must enter mode as inference or experiment.")

if __name__ == "__main__":
    
    args = parser()

    noise_X_train_df = pd.read_pickle(os.path.join(upper_dir, "refine/noise/raw", "X_train_df.pkl"))
    noise_X_test_df = pd.read_pickle(os.path.join(upper_dir, "refine/noise/raw", "X_test_df.pkl"))
    noise_y_train_df = pd.read_pickle(os.path.join(upper_dir, "refine/noise/raw", "y_train_df.pkl"))
    noise_datasets = [noise_X_train_df, noise_y_train_df, noise_X_test_df,]

    clean_X_train_df = pd.read_pickle(os.path.join(upper_dir, "refine/clean/raw", "X_train_df.pkl"))
    clean_X_test_df = pd.read_pickle(os.path.join(upper_dir, "refine/clean/raw", "X_test_df.pkl"))
    clean_y_train_df = pd.read_pickle(os.path.join(upper_dir, "refine/clean/raw", "y_train_df.pkl"))
    clean_datasets = [clean_X_train_df, clean_y_train_df, clean_X_test_df, ]

    noise_model_path = os.path.join(upper_dir, "model", "noise")
    clean_model_path = os.path.join(upper_dir, "model", "clean")

    os.makedirs(os.path.join(upper_dir, "model"), exist_ok=True) 
    os.makedirs(noise_model_path, exist_ok=True) 
    os.makedirs(clean_model_path, exist_ok=True)

    os.makedirs(os.path.join(noise_model_path, "level0"), exist_ok=True)
    os.makedirs(os.path.join(noise_model_path, "level1"), exist_ok=True)
    os.makedirs(os.path.join(clean_model_path, "level0"), exist_ok=True)
    os.makedirs(os.path.join(clean_model_path, "level1"), exist_ok=True)

    main(args)
