import numpy as np
import pandas as pd
import os
from scipy.fft import fft

from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from lightgbm import LGBMRegressor

from utils import set_seeds, outier_imputation, split_dataset
from feature import SmoothingTransform, ByLowPassTransform, FourierTransform

current_dir = os.path.dirname(os.path.abspath(__file__))
upper_dir = os.path.join(current_dir, os.pardir)

seed = 1011
set_seeds(seed)

DROP_COLUMNS = ["X_02", "X_04", "X_23", "X_47", "X_48"]

sd_scaler = StandardScaler()

feature_regressor = MultiOutputRegressor(
    LGBMRegressor(
        class_weight="balanced",
        drop_rate=0.9,
        min_data_in_leaf=100,
        max_bin=255,
        n_estimators=500,
        min_sum_hessian_in_leaf=1,
        importance_type="gain",
        learning_rate=0.1,
        bagging_fraction=0.85,
        colsample_bytree=1.0,
        feature_fraction=0.1,
        lambda_l1=5.0,
        lambda_l2=3.0,
        max_depth=9,
        min_child_samples=55,
        min_child_weight=5.0,
        min_split_gain=0.1,
        num_leaves=45,
        subsample=0.75,
        n_jobs=-1,
        random_state=seed,
    )
)

def save_pickle(base_path, X_train, X_test, y_train):

    save_df_list = [X_train, X_test, y_train]
    save_df_name = ["X_train_df.pkl", "X_test_df.pkl", "y_train_df.pkl"]

    for file, name in zip(save_df_list, save_df_name):
        file.to_pickle(os.path.join(base_path, name))

def save_npy(base_path, X_train, X_test, y_train):
    
    save_npy_list = [X_train, X_test, y_train]
    save_npy_name = ["X_train_df.npy", "X_test_df.npy", "y_train_df.npy"]

    for npy, name in zip(save_npy_list, save_npy_name):
        np.save(os.path.join(base_path, name), npy)

def scaling_datasets(datasets, scaler):
        
    X_train = datasets[0]
    X_test = datasets[1]
    y_train = datasets[2]

    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train.values

def main():

    X_df, X_train_df, y_train_df, X_test_df = split_dataset(train, test)

    clean_X_df = outier_imputation(X_df, 30)
    
    X_train_df = clean_X_df.iloc[: len(X_train_df)].reset_index(drop=True)
    X_test_df = clean_X_df.iloc[len(X_train_df) :].reset_index(drop=True)
    y_train_df = y_train_df.copy()

    smoothing = SmoothingTransform(
    X_train=X_train_df, y_train=y_train_df, X_test=X_test_df, search_max_value=5000
    )

    lowpass = ByLowPassTransform(
    X_train=X_train_df, y_train=y_train_df, X_test=X_test_df, max_N=10
    )

    fourier = FourierTransform(
    X_train=X_train_df, y_train=y_train_df, X_test=X_test_df, scaler=sd_scaler, fft=fft
    )

    smoothing.serach_best_param_by_feature_importance(feature_regressor)
    smoothing.serach_best_param_by_corr()
    lowpass.serach_best_param()

    # Exponential Smoothing & Moving Average DataFrame by feature importance
    ft_smoothing_X_train = smoothing.data_transform("train", "ft")
    ft_smoothing_X_test = smoothing.data_transform("test", "ft")

    # Exponential Smoothing & Moving Average DataFrame by correlation
    corr_smoothing_X_train = smoothing.data_transform("train", "corr")
    corr_smoothing_X_test = smoothing.data_transform("test", "corr")

    # lowpass smoothing by correlation
    lowpass_X_train = lowpass.data_transform("train")
    lowpass_X_test = lowpass.data_transform("test")

    # raw scale -> standard scale -> fourier transform 
    scale_fft_X_train = fourier.data_transform("train", "scale")
    scale_fft_X_test = fourier.data_transform("test", "scale")
    # raw scale -> fourier transform 
    raw_fft_X_train = fourier.data_transform("train", "raw")
    raw_fft_X_test = fourier.data_transform("test", "raw")

    # Make noise dataframe (all feature + Exponential Smoothing & Moving Average DataFrame by feature importance)
    noise_X_train_df = pd.concat([X_train_df, ft_smoothing_X_train, lowpass_X_train, scale_fft_X_train, raw_fft_X_train,], axis=1)
    noise_X_test_df = pd.concat([X_test_df, ft_smoothing_X_test, lowpass_X_test, scale_fft_X_test, raw_fft_X_test,], axis=1)
    print(noise_X_train_df.shape, noise_X_test_df.shape)
    save_pickle(os.path.join(upper_dir, "refine/noise", "raw"), noise_X_train_df, noise_X_test_df, y_train_df)

    scale_noise_X_train_df, scale_noise_X_test_df, scale_y_train_df = scaling_datasets([noise_X_train_df, noise_X_test_df, y_train_df], sd_scaler)
    save_npy(os.path.join(upper_dir, "refine/noise", "scale"), scale_noise_X_train_df, scale_noise_X_test_df, scale_y_train_df)
    
    # Make clean dataframe (all feature + Exponential Smoothing & Moving Average DataFrame by correlation)
    clean_X_train_df = pd.concat([X_train_df, corr_smoothing_X_train, lowpass_X_train, scale_fft_X_train, raw_fft_X_train,], axis=1)
    clean_X_test_df = pd.concat([X_test_df, corr_smoothing_X_test, lowpass_X_test, scale_fft_X_test, raw_fft_X_test,], axis=1)
    print(clean_X_train_df.shape, clean_X_test_df.shape)
    save_pickle(os.path.join(upper_dir, "refine/clean", "raw"), clean_X_train_df, clean_X_test_df, y_train_df)

    scale_clean_X_train_df, scale_clean_X_test_df, scale_y_clean_df = scaling_datasets([clean_X_train_df, clean_X_test_df, y_train_df], sd_scaler)
    save_npy(os.path.join(upper_dir, "refine/clean", "scale"), scale_clean_X_train_df, scale_clean_X_test_df, scale_y_clean_df)
    
if __name__ == "__main__":

    train = pd.read_csv(os.path.join(upper_dir, "open", "train.csv")).drop(
    columns=DROP_COLUMNS
    )

    test = pd.read_csv(os.path.join(upper_dir, "open", "test.csv")).drop(
        columns=DROP_COLUMNS
    )

    os.makedirs(os.path.join(upper_dir, "refine"), exist_ok=True) 
    os.makedirs(os.path.join(upper_dir, "refine", "noise"), exist_ok=True) 
    os.makedirs(os.path.join(upper_dir, "refine", "noise", "raw"), exist_ok=True) 
    os.makedirs(os.path.join(upper_dir, "refine", "noise", "scale"), exist_ok=True) 
    os.makedirs(os.path.join(upper_dir, "refine", "clean"), exist_ok=True) 
    os.makedirs(os.path.join(upper_dir, "refine", "clean", "raw"), exist_ok=True) 
    os.makedirs(os.path.join(upper_dir, "refine", "clean", "scale"), exist_ok=True) 

    main()
