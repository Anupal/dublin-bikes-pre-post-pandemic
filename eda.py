import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import sys

DATASET_FILE = "combined.csv"


if __name__ == "__main__":
    # check args
    print("CMD:", sys.argv)
    if len(sys.argv) < 2 or sys.argv[1] not in ("in", "out"):
        print("Please provide in/out argument!")
        sys.exit(1)
    output_feature = "USAGE_OUT" if sys.argv[1] == "out" else "USAGE_IN"

    df = pd.read_csv(DATASET_FILE)

    # drop time and station id
    df = df.drop(["TIME", "AVAILABLE BIKE STANDS"], axis=1)

    # # normalize all values
    df = (df - df.min()) / (df.max() - df.min())

    # input and output
    x = df.drop(["USAGE_OUT", "USAGE_IN"], axis=1)
    y = df[output_feature]

    print("Input Params")
    print(x)

    print("\nOutput Params")
    print(y)

    print("\n--- PEARSON CORRELATION ---\n")
    print(df[list(x.columns) + [output_feature]].corr()[output_feature])

    # print("\n--- LINEAR REGRESSION ---\n")
    # model = LinearRegression().fit(x, y)

    # print("Intercept", model.intercept_)

    # coefficients = model.coef_
    # res = []
    # # Display the coefficients for each input feature
    # for i, column in enumerate(x.columns):
    #     res.append({
    #         "Parameter": column,
    #         "Coefficient": coefficients[i],
    #         "Weight":  coefficients[i] * (x[column].max() - x[column].min() ) / x[column].std()
    #     })

    # print(pd.DataFrame(res))

    # print("\n--- RANDOM FOREST ---\n")
    # model = RandomForestRegressor().fit(x, y)

    # feature_importances = model.feature_importances_
    # res = []
    # # Display the feature_importances for each input feature
    # for i, column in enumerate(x.columns):
    #     res.append({
    #         "Parameter": column,
    #         "Importance": feature_importances[i],
    #     })

    # print(pd.DataFrame(res))