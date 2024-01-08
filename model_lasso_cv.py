import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
import numpy as np


DATASET_FILE = "combined.csv"
CUTOFF_DATE = pd.to_datetime("2020-03-01")
ALPHAS = [0.01, 0.1, 0.15, 0.5, 0.75, 1.0, 2, 5, 10.0]


if __name__ == "__main__":
    df = pd.read_csv("combined.csv")
    df["TIME"] = pd.to_datetime(df["TIME"])

    # limit samples till pandemic end
    df = df[df["TIME"] < CUTOFF_DATE]

    # drop time and station id
    training_df = df.drop(["TIME"], axis=1)

    # input and output
    x = training_df.drop(["USAGE"], axis=1)
    y = training_df["USAGE"]

    # normalize x values
    x = (x - x.min()) / (x.max() - x.min())

    tscv = TimeSeriesSplit(n_splits=5)
    lasso_cv = LassoCV(cv=tscv, alphas=ALPHAS)
    lasso_cv.fit(x, y)

    best_alpha = lasso_cv.alpha_
    print("BEST ALPHA:", best_alpha)
    # print("MSE PATH:", lasso_cv.mse_path_)

    # Get MSE scores for each alpha
    mse_scores = lasso_cv.mse_path_
    mean_mse_scores = np.mean(mse_scores, axis=1)
    std_mse_scores = np.std(mse_scores, axis=1)

    res = []
    # Display the MSE scores for each alpha
    for i, alpha in enumerate(lasso_cv.alphas_):
        res.append({"Alpha": alpha, "Mean MSE": mean_mse_scores[i], "STD MSE": std_mse_scores[i]})

    
    print(pd.DataFrame(res))

    # Plot mean and standard deviation of MSE scores
    plt.figure(figsize=(8, 6))
    plt.errorbar(lasso_cv.alphas_, mean_mse_scores, yerr=std_mse_scores, fmt='-b', ecolor='r', elinewidth=2, label="MSE (STD)")
    plt.xlabel('Alpha')
    plt.ylabel('MSE (Mean)')
    plt.title('MSE Mean and STD')
    plt.show()
