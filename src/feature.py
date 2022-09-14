import pandas as pd
import numpy as np
from tqdm import trange
from scipy import signal

class SmoothingTransform:
    
    def __init__(self, X_train, y_train, X_test, search_max_value):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test

        self.search_max_value = search_max_value

        self.best_mean_com = None
        self.best_std_com = None
        self.best_mean_window = None
        self.best_std_window = None

        self.best_com_ft = None
        self.best_window_ft = None

    def serach_best_param_by_corr(self):

        search_list = np.linspace(0, self.search_max_value, 100)

        ewm_mean_score_list = []
        ewm_std_score_list = []
        rolling_mean_score_list = []
        rolling_std_score_list = []

        # EWM
        print("Search Best EWM Parameter by correlation...")
        for idx in trange(len(search_list)):

            COM = search_list[idx]

            mean_ewm_X_train_df = self.X_train.ewm(com=COM, adjust=True).mean()

            std_ewm_X_train_df = self.X_train.ewm(com=COM, adjust=True).std()
            std_ewm_X_train_df.loc[0, :] = std_ewm_X_train_df.loc[1, :]

            mean_df = pd.concat([mean_ewm_X_train_df, self.y_train], axis=1)
            std_df = pd.concat([std_ewm_X_train_df, self.y_train], axis=1)

            mean_score = mean_df.corr().filter(regex="Y").abs().iloc[:47].sum().sum()
            std_score = std_df.corr().filter(regex="Y").abs().iloc[:47].sum().sum()

            ewm_mean_score_list.append(mean_score)
            ewm_std_score_list.append(std_score)

        BEST_COM_MEAN = int(search_list[np.array(ewm_mean_score_list).argmax()])
        BEST_COM_STD = int(search_list[np.array(ewm_std_score_list).argmax()])
        self.best_mean_com = BEST_COM_MEAN
        self.best_std_com = BEST_COM_STD

        # Rolling
        print("Search Best Rolling Parameter by correlation...")

        search_list = np.linspace(1, self.search_max_value, 100)

        for idx in trange(len(search_list)):

            window = int(search_list[idx])
            mean_rolling_X_train_df = self.X_train.rolling(window=window).mean()
            mean_rolling_X_train_df.loc[: window - 2, :] = mean_rolling_X_train_df.loc[
                window - 1, :
            ].values

            std_rolling_X_train_df = self.X_train.rolling(window=window).std()
            std_rolling_X_train_df.loc[: window - 2, :] = std_rolling_X_train_df.loc[
                window - 1, :
            ].values

            mean_df = pd.concat([mean_rolling_X_train_df, self.y_train], axis=1)
            std_df = pd.concat([std_rolling_X_train_df, self.y_train], axis=1)

            mean_score = mean_df.corr().filter(regex="Y").abs().iloc[:47].sum().sum()
            std_score = std_df.corr().filter(regex="Y").abs().iloc[:47].sum().sum()

            rolling_mean_score_list.append(mean_score)
            rolling_std_score_list.append(std_score)

        BEST_ROLLING_MEAN = int(search_list[np.array(rolling_mean_score_list).argmax()])
        BEST_ROLLING_STD = int(search_list[np.array(rolling_std_score_list).argmax()])

        self.rolling_std_score_list = rolling_std_score_list

        self.best_mean_window = BEST_ROLLING_MEAN
        self.best_std_window = BEST_ROLLING_STD

    def serach_best_param_by_feature_importance(self, model):

        y_col = self.y_train.columns.tolist()
        n_split = 100

        com_list = np.linspace(0.1, 20, n_split)
        window_list = np.linspace(1, 100, n_split)

        score_list = []

        print("Search Best EWM Parameter by feature importance...")
        for idx in trange(len(com_list)):

            total = 25
            mean = 0
            len_y_col = len(y_col)

            COM = com_list[idx]

            ewm_X_train_df = self.X_train.ewm(com=COM, adjust=True).mean()
            ewm_X_train_df.columns = [
                "ewm_mean_%s" % (col) for col in ewm_X_train_df.columns.tolist()
            ]

            temp_X_train_df = pd.concat([self.X_train, ewm_X_train_df], axis=1)
            model.fit(temp_X_train_df, self.y_train)

            for n in range(len_y_col):

                feature_imp = pd.DataFrame(
                    sorted(
                        zip(
                            model.estimators_[n].feature_importances_,
                            temp_X_train_df.columns,
                        )
                    ),
                    columns=["Value", "Feature"],
                )

                temp_feature_imp = feature_imp.sort_values(
                    by="Value", ascending=False
                ).iloc[:total]
                cnt = temp_feature_imp[
                    temp_feature_imp["Feature"].str.contains("ewm_mean")
                ].shape[0]

                mean += cnt / total
            print("COM = %s" % COM, "SCORE = %s" % (mean / len_y_col))
            score_list.append(mean / len_y_col)

        best_index = np.array(score_list).argmax()
        BEST_COM_FT = int(com_list[best_index])
        self.best_com_ft = BEST_COM_FT

        score_list = []

        print("Search Best Rolling Parameter by feature importance...")
        for idx in trange(len(window_list)):

            total = 25
            mean = 0
            len_y_col = len(y_col)

            window = int(window_list[idx])

            mean_rolling_X_train_df = self.X_train.rolling(window=window).mean()
            mean_rolling_X_train_df.loc[: window - 2, :] = mean_rolling_X_train_df.loc[
                window - 1, :
            ].values
            mean_rolling_X_train_df.columns = [
                "rolling_mean_%s" % (col) for col in self.X_train.columns.tolist()
            ]

            temp_X_train_df = pd.concat([self.X_train, mean_rolling_X_train_df], axis=1)
            model.fit(temp_X_train_df, self.y_train)

            for n in range(len_y_col):

                feature_imp = pd.DataFrame(
                    sorted(
                        zip(
                            model.estimators_[n].feature_importances_,
                            temp_X_train_df.columns,
                        )
                    ),
                    columns=["Value", "Feature"],
                )
                temp_feature_imp = feature_imp.sort_values(
                    by="Value", ascending=False
                ).iloc[:total]
                cnt = temp_feature_imp[
                    temp_feature_imp["Feature"].str.contains("rolling_mean")
                ].shape[0]

                mean += cnt / total
            print("window = %s" % window, "SCORE = %s" % (mean / len_y_col))
            score_list.append(mean / len_y_col)

        best_index = np.array(score_list).argmax()
        BEST_WINDOW_FT = int(window_list[best_index])
        self.best_window_ft = BEST_WINDOW_FT

    def data_transform(self, data_type="train", feature_type="corr"):

        X = self.X_train

        if data_type == "test":
            X = self.X_test

        if feature_type == "corr":
            mean_ewm_X_df = X.ewm(com=self.best_mean_com, adjust=True).mean()
            mean_ewm_X_df.columns = [
                "ewm_mean_%s_corr" % (col) for col in mean_ewm_X_df.columns.tolist()
            ]

            std_ewm_X_df = X.ewm(com=self.best_std_com, adjust=True).std()
            std_ewm_X_df.loc[0, :] = std_ewm_X_df.loc[1, :]
            std_ewm_X_df.columns = [
                "ewm_std_%s_corr" % (col) for col in std_ewm_X_df.columns.tolist()
            ]

            mean_rolling_X_df = X.rolling(window=self.best_mean_window).mean()
            mean_rolling_X_df.loc[
                : self.best_mean_window - 2, :
            ] = mean_rolling_X_df.loc[self.best_mean_window - 1, :].values
            mean_rolling_X_df.columns = [
                "rolling_mean_%s_corr" % (col)
                for col in mean_rolling_X_df.columns.tolist()
            ]

            std_rolling_X_df = X.rolling(window=self.best_std_window).std()
            std_rolling_X_df.loc[: self.best_std_window - 2, :] = std_rolling_X_df.loc[
                self.best_std_window - 1, :
            ].values
            std_rolling_X_df.columns = [
                "rolling_std_%s_corr" % (col)
                for col in std_rolling_X_df.columns.tolist()
            ]

            all_X_df = pd.concat(
                [mean_ewm_X_df, std_ewm_X_df, mean_rolling_X_df, std_rolling_X_df],
                axis=1,
            )

        else:
            mean_ewm_X_df = X.ewm(com=self.best_com_ft, adjust=True).mean()
            mean_ewm_X_df.columns = [
                "ewm_mean_%s_ft" % (col) for col in mean_ewm_X_df.columns.tolist()
            ]

            std_ewm_X_df = X.ewm(com=self.best_com_ft, adjust=True).std()
            std_ewm_X_df.loc[0, :] = std_ewm_X_df.loc[1, :]
            std_ewm_X_df.columns = [
                "ewm_std_%s_ft" % (col) for col in std_ewm_X_df.columns.tolist()
            ]

            mean_rolling_X_df = X.rolling(window=self.best_window_ft).mean()
            mean_rolling_X_df.loc[: self.best_window_ft - 2, :] = mean_rolling_X_df.loc[
                self.best_window_ft - 1, :
            ].values
            mean_rolling_X_df.columns = [
                "rolling_mean_%s_ft" % (col)
                for col in mean_rolling_X_df.columns.tolist()
            ]

            std_rolling_X_df = X.rolling(window=self.best_window_ft).std()
            std_rolling_X_df.loc[: self.best_window_ft - 2, :] = std_rolling_X_df.loc[
                self.best_window_ft - 1, :
            ].values
            std_rolling_X_df.columns = [
                "rolling_std_%s_ft" % (col) for col in std_rolling_X_df.columns.tolist()
            ]

            all_X_df = pd.concat(
                [mean_ewm_X_df, std_ewm_X_df, mean_rolling_X_df, std_rolling_X_df],
                axis=1,
            )

        return all_X_df

class DifferencingTransform:
    
    def __init__(self, X_train, y_train, X_test, trial):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.trial = trial
        self.best_cnt = None

    def serach_best_param(self):

        X_train = self.X_train
        cnt = 1
        score_list = []

        for _ in trange(self.trial):
            X_train = X_train.diff()

            temp_X_train = X_train.copy()
            temp_X_train.iloc[:cnt, :] = temp_X_train.iloc[cnt, :].values
            XY = pd.concat([temp_X_train, self.y_train], axis=1)
            score = XY.corr().filter(regex="Y").abs().iloc[:47].sum().sum()
            score_list.append(score)

            cnt += 1

        self.best_cnt = np.array(score_list).argmax() + 1

    def data_transform(self, _type="train"):

        X = self.X_train

        if _type == "test":
            X = self.X_test

        for _ in range(self.best_cnt):
            X = X.diff()

        X.iloc[: self.best_cnt, :] = X.iloc[self.best_cnt, :].values

        return X

class ByLowPassTransform:
    
    def __init__(self, X_train, y_train, X_test, max_N):

        self.X_train = X_train
        self.feature_list = X_train.columns.tolist()
        self.y_train = y_train
        self.X_test = X_test
        self.max_N = max_N
        self.best_param = None
        self.score = None
        self.param = None

    def serach_best_param(self):

        N_list = np.linspace(1, self.max_N, self.max_N)
        lowpass_list = np.linspace(0.1, 0.99, 100)
        score_list = []
        param_list = []
        
        print("Search Best Low Pass Filter Parameter by feature importance...")

        for idx in trange(len(lowpass_list)):
            lowpass = lowpass_list[idx]

            for N in N_list:
                # print(N, lowpass)
                beta0, beta1 = signal.butter(N, lowpass, btype="lowpass")
                temp_X = pd.DataFrame()

                for feature in self.feature_list:
                    feature_values = self.X_train[feature].values

                    trans_feature_values = signal.filtfilt(beta0, beta1, feature_values)
                    temp_X[feature] = trans_feature_values

                XY = pd.concat([temp_X, self.y_train], axis=1)
                score = XY.corr().filter(regex="Y").abs().iloc[:47].sum().sum()
                # print(score)
                score_list.append(score)
                param_list.append((N, lowpass))

        BEST_PARAM = param_list[np.array(score_list).argmax()]
        self.best_param = BEST_PARAM
        self.score = np.array(score_list)
        self.param = param_list

    def data_transform(self, _type="train"):

        X = self.X_train

        if _type == "test":
            X = self.X_test

        n, lowpass = self.best_param
        beta0, beta1 = signal.butter(n, lowpass, btype="lowpass")

        temp_X = pd.DataFrame()

        for feature in self.feature_list:
            feature_values = X[feature].values
            trans_feature_values = signal.filtfilt(beta0, beta1, feature_values)
            temp_X[feature] = trans_feature_values

        temp_X.columns = ["lowpass_%s" % (col) for col in temp_X.columns.tolist()]

        return temp_X

class FourierTransform:
    
    def __init__(self, X_train, y_train, X_test, scaler, fft):
        
        self.colum_names = X_train.columns.tolist()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        
        self.scaler = scaler
        self.scaled_X_train = self.scaler.fit_transform(X_train)
        self.scaled_X_test = self.scaler.transform(X_test)
        
        self.fft = fft
    
    def data_transform(self, data_type="train", _type="scale"):
        
        X = self.X_train
        scale_X = self.scaled_X_train
        
        if data_type == "test":
            X = self.X_test
            scale_X = self.scaled_X_test
        
        
        fft_X = self.fft(X).astype("float32")
        scale_fft_X = self.fft(scale_X).astype("float32")
        
        pd_fft_X = pd.DataFrame(fft_X, columns = self.colum_names)
        pd_fft_X.columns = ["raw_fft_%s" % (col) for col in pd_fft_X.columns.tolist()]
        
        pd_scale_fft_X = pd.DataFrame(scale_fft_X, columns = self.colum_names)
        pd_scale_fft_X.columns = ["scale_fft_%s" % (col) for col in pd_scale_fft_X.columns.tolist()]
        
        if _type == "scale":
            return pd_scale_fft_X
        
        else:
            return pd_fft_X