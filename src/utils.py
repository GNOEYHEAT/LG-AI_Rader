import os
import random
import numpy as np
import pandas as pd
from tqdm import trange

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings("ignore")

seed = 1011

def set_seeds(seed=seed):
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def split_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame):
    
    # Drop column name
    drop_columns_list = ["ID"]
    # Features & labels containing strings
    feature_str = "X"
    label_str = "Y"

    # drop column
    train_df = train_df.drop(drop_columns_list, axis=1)
    test_df = test_df.drop(drop_columns_list, axis=1)

    # split dataframe based on column category
    X_train_df = train_df.filter(regex="X")
    y_train_df = train_df.filter(regex="Y")
    X_test_df = test_df.filter(regex="X")

    # feature dataframes concatnate
    X_df = pd.concat([X_train_df, X_test_df], axis=0).reset_index(drop=True)

    return X_df, X_train_df, y_train_df, X_test_df

def outier_imputation(inp_df, point_threshold):
    print("Outier Imputation START..")
    df = inp_df.copy()
    column_list = df.columns.tolist()

    for index in trange(len(column_list)):
        col_name = column_list[index]

        # scaling Data
        scaler = StandardScaler()
        np_scaled = scaler.fit_transform(df[col_name].values.reshape(-1, 1))
        data = pd.DataFrame(np_scaled)

        # train isolation forest
        outliers_fraction = float(0.1)
        model = IsolationForest(contamination=outliers_fraction, random_state=seed)
        model.fit(data)

        # anomaly define
        score_arr = abs(model.score_samples(data[0].values.reshape(-1, 1)))
        bool_arr = score_arr == score_arr.max()
        data["anomaly"] = bool_arr

        if data["anomaly"].sum() > point_threshold:

            pass

        else:
            test = df[[col_name]]
            test["anomaly"] = bool_arr
            anomaly_index_format = test[test["anomaly"] == 1].index.tolist()

            before_time_stamp = 5
            inputation_value = []
            for ind in anomaly_index_format:
                value = test.loc[ind - before_time_stamp : ind][col_name].mean()
                inputation_value.append(value)
            test.loc[test["anomaly"] == 1, [col_name]] = inputation_value
            df[col_name] = test[col_name]

    return df