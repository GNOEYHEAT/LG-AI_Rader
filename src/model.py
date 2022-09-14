import numpy as np
import pandas as pd
import time
import pickle
import os

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class STACKING:
    
    def __init__(
        self, X_train, X_test, y_train, y_test, model_list, best_n, seed, _type
    ):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_list = model_list
        self.best_n = best_n
        self.seed = seed
        self.n_folds = 10
        self._type = _type

        self.meta_ml_X_train = None
        self.meta_ml_X_test = None
        self.score = None
        self.final_select_n = None

    def get_stacking_ml_datasets(
                                self,
                                model,
                                X_train_n,
                                y_train_n,
                                X_test_n,
                                n_folds,
                                fitting=True,
                                save=False,
                                save_path="",
                                model_index=""):
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        
        train_fold_pred = np.zeros((X_train_n.shape[0], y_train_n.shape[1]))
        test_pred = np.zeros((X_test_n.shape[0], y_train_n.shape[1], n_folds))
        
        for folder_counter, (train_index, valid_index) in enumerate(kf.split(X_train_n, y_train_n)):
        
            X_tr = X_train_n[train_index]
            y_tr = y_train_n[train_index]
            X_te = X_train_n[valid_index]
            
            if fitting:
                model.fit(X_tr, y_tr)

            if save:
                model_save_path = os.path.join(save_path, "%s_%s.pkl"%(model_index, folder_counter))

                with open(model_save_path, "wb") as f:
                    pickle.dump(model, f)

            train_fold_pred[valid_index] = model.predict(X_te)
            test_pred[:, :, folder_counter] = model.predict(X_test_n)
            
        test_pred_mean = np.mean(test_pred, axis=2)
        
        return train_fold_pred, test_pred_mean

    def lg_nrmse(self, gt, preds):

        all_nrmse = []

        for idx in range(gt.shape[1]):
            rmse = mean_squared_error(gt[:, idx], preds[:, idx], squared=False)
            nrmse = rmse / np.mean(np.abs(gt[:, idx]))
            all_nrmse.append(nrmse)

        score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:14])

        return score

    def run_level0(self, save_path=""):

        meta_ml_X_train = []
        meta_ml_X_test = []

        print("Start Level0 Modeling Start..")

        for index, estimator in enumerate(self.model_list):

            print("Number %s :" % (index), estimator)
            start = time.time()

            if self._type == "real":
              temp_X_train, temp_X_test = self.get_stacking_ml_datasets(
                  estimator, self.X_train, self.y_train.values, self.X_test, self.n_folds, True, True, save_path, index
              )
              meta_ml_X_train.append(temp_X_train)
              meta_ml_X_test.append(temp_X_test)
              end = time.time()
              print(f"{end - start:.2f} sec")

            else:
              temp_X_train, temp_X_test = self.get_stacking_ml_datasets(
                  estimator, self.X_train, self.y_train.values, self.X_test, self.n_folds
              )
              meta_ml_X_train.append(temp_X_train)
              meta_ml_X_test.append(temp_X_test)
              end = time.time()
              print(f"{end - start:.2f} sec")

        if self._type == "real":

            self.meta_ml_X_train = np.array(meta_ml_X_train)
            self.meta_ml_X_test = np.array(meta_ml_X_test)

            return "Level0 Compeleted-!"

        self.scores = {}

        for idx, estimator in enumerate(self.model_list):
            self.scores[idx] = self.lg_nrmse(
                self.y_test.values, estimator.predict(self.X_test)
            )

        score_index = np.array(list(self.scores.values()), dtype=np.float64).argsort()
        score_sort_model_list = np.array(self.model_list)[score_index]
        best_n_ml = score_sort_model_list[: self.best_n]

        self.score_index = score_index
        self.score_sort_model_list = score_sort_model_list
        self.best_n_ml = best_n_ml

        self.meta_ml_X_train = np.array(meta_ml_X_train)[score_index[: self.best_n]]
        self.meta_ml_X_test = np.array(meta_ml_X_test)[score_index[: self.best_n]]

    def run_level1(self, save=False, save_path=""):

        if self._type == "real":

            meta_clf = LinearRegression()

            temp_X_train = self.meta_ml_X_train.mean(axis=0)
            temp_X_test = self.meta_ml_X_test.mean(axis=0)

            meta_clf.fit(temp_X_train, self.y_train)

            if save:
              model_save_path = os.path.join(save_path, "meta_clf.pkl")
              with open(model_save_path, "wb") as f:
                      pickle.dump(meta_clf, f)

            prediction = meta_clf.predict(temp_X_test)

            return prediction

        else:
            selected_n = self.meta_ml_X_train.shape[0]
            lg_nrmse_list = []

            for n in range(1, selected_n + 1):

                print("If you mix %s of the best models" % (n))
                meta_clf = LinearRegression()

                temp_X_train = self.meta_ml_X_train[:n, :, :].mean(axis=0)
                temp_X_test = self.meta_ml_X_test[:n, :, :].mean(axis=0)

                meta_clf.fit(temp_X_train, self.y_train)
                prediction = meta_clf.predict(temp_X_test)
                lg_nrmse_all = self.lg_nrmse(self.y_test.values, prediction)
                lg_nrmse_list.append(lg_nrmse_all)
                print("lg_nrmse_all: ", lg_nrmse_all)

            final_select_n = np.array(lg_nrmse_list).argmin() + 1
            print("The best number is %s" % (final_select_n))
            self.final_select_n = final_select_n

            temp_X_train = self.meta_ml_X_train[:final_select_n, :, :].mean(axis=0)
            temp_X_test = self.meta_ml_X_test[:final_select_n, :, :].mean(axis=0)

            meta_clf.fit(temp_X_train, self.y_train)
            prediction = meta_clf.predict(temp_X_test)

            return prediction