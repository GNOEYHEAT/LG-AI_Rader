import numpy as np
import pandas as pd
import os
import pickle
from tqdm import trange

import warnings
warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
upper_dir = os.path.join(current_dir, os.pardir)

def ensemble_clean_noise(clean_path, noise_path, submission_path):
    
    clean = pd.read_csv(clean_path)
    noise = pd.read_csv(noise_path)
    submission = pd.read_csv(submission_path)

    ensemble_result = np.mean(
        np.array([clean.iloc[:, 1:].values, noise.iloc[:, 1:].values]), axis=0
    ).round(3)

    for idx, col in enumerate(submission.columns):
        if col == "ID":
            continue
        submission[col] = ensemble_result[:, idx - 1]

    return submission

def inference_by_trained_model(
    X_test_path, level0_save_path, level1_save_path, submission_path
):
    X_test = np.load(X_test_path)
    level0_model_list = os.listdir(level0_save_path)
    level0_model_cnt = len(set([model.split("_")[0] for model in level0_model_list]))
    level0_model_cv_cnt = len(
        set([model.replace(".pkl", "").split("_")[-1] for model in level0_model_list])
    )

    y_pred = []

    for model_index in trange(level0_model_cnt):

        model_path_index = os.path.join(level0_save_path, "%s") % (model_index)
        load_name_path = model_path_index + "_%s.pkl"

        for folder_counter in range(level0_model_cv_cnt):

            with open(load_name_path % (folder_counter), "rb") as f:
                model = pickle.load(f)

            y_pred.append(model.predict(X_test))

    meta_ml_X_test = np.stack(y_pred).mean(axis=0)

    with open(os.path.join(level1_save_path, "meta_clf.pkl"), "rb") as f:
        meta_clf = pickle.load(f)

    prediction = meta_clf.predict(meta_ml_X_test).round(3)
    submission = pd.read_csv(submission_path)

    for idx, col in enumerate(submission.columns):
        if col == "ID":
            continue
        submission[col] = prediction[:, idx - 1]

    return submission

def main():

    refine_path = os.path.join(upper_dir, "refine")
    model_path = os.path.join(upper_dir, "model")
    output_path = os.path.join(upper_dir, "output")
    submission_path = os.path.join(upper_dir, "open", "sample_submission.csv")

    order_list = ["clean", "noise"]

    for order in order_list:
        X_test_path = os.path.join(refine_path, order, "scale", "X_test_df.npy")
        level0_save_path = os.path.join(model_path, order, "level0")
        level1_save_path = os.path.join(model_path, order, "level1")
        submission = inference_by_trained_model(X_test_path, level0_save_path, level1_save_path, submission_path)
        print(submission.head())
        submission.to_csv(os.path.join(upper_dir, "output/%s/submission.csv"%(order)), index=False)

    clean_path = os.path.join(output_path, "clean/submission.csv")
    noise_path = os.path.join(output_path, "noise/submission.csv")
    final_submission = ensemble_clean_noise(clean_path, noise_path, submission_path)
    final_submission.to_csv(os.path.join(output_path, "final/submission.csv"), index=False)

if __name__ == "__main__":

    output_path = os.path.join(upper_dir, "output")
    os.makedirs(output_path, exist_ok=True) 
    os.makedirs(os.path.join(output_path, "clean"), exist_ok=True) 
    os.makedirs(os.path.join(output_path, "noise"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "final"), exist_ok=True)

    main()