import pandas as pd
import numpy as np
from itertools import product
from xgboost import XGBRegressor
from data_preprocessing import process_data
import json
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


'''Load config'''
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


'''Load and Split data'''
data = process_data(config).toPandas()

features = data.drop(columns=["Weekly_Sales"], inplace=False)
target = data[["Weekly_Sales"]]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)
y_train = y_train.squeeze()
y_test = y_test.squeeze()


'''Training'''
n_estimators = [500, 750, 1000, 1500, 2000]
learning_rate = [0.01, 0.03, 0.05, 0.07, 0.09]
max_depth = [6, 8, 10, 12]
reg_alpha = [0, 1, 2]
reg_lambda = [0, 1, 2, 3, 4]
objectives = ["reg:squarederror", "reg:pseudohubererror", "reg:squaredlogerror"]

param_combinations = list(
    product(
        n_estimators,
        learning_rate,
        max_depth,
        reg_alpha,
        reg_lambda,
        objectives
    )
)

total_combinations = (
        len(n_estimators)
        * len(learning_rate)
        * len(max_depth)
        * len(reg_alpha)
        * len(reg_lambda)
        * len(objectives)
    )

print(f"Total combinations: {total_combinations}")

best_rmse = float("inf")
best_params = None
best_model = None

results = []

for i, params in enumerate(param_combinations, start=1):
    n_est, lr, depth, alpha, lamb, objective = params

    model = XGBRegressor(
        n_estimators=n_est,
        learning_rate=lr,
        max_depth=depth,
        reg_alpha=alpha,
        reg_lambda=lamb,
        objective=objective,
        random_state=42,
        tree_method="hist",
        enable_categorical=True,
        n_jobs=-1
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    results.append({
        "n_estimators": n_est,
        "learning_rate": lr,
        "max_depth": depth,
        "reg_alpha": alpha,
        "reg_lambda": lamb,
        "objective": objective,
        "tree_method": "hist",
        "enable_categorical": True,
        "RMSE": rmse
    })

    if rmse < best_rmse:

        best_rmse = rmse
        best_params = {
            "n_estimators": n_est,
            "learning_rate": lr,
            "max_depth": depth,
            "reg_alpha": alpha,
            "reg_lambda": lamb,
            "objective": objective,
            "tree_method": "hist",
            "enable_categorical": True,
        }

        best_model = model

    if i % 100 == 0 :
        results_df = pd.DataFrame(results)
        os.makedirs("models", exist_ok=True)
        results_df.to_csv(config["model"]["exhaustive_result_path"], index=False)
        print( 
                f"{i}/{total_combinations} completed | "
                f"Current RMSE: {rmse:,.2f} | "
                f"Best RMSE: {best_rmse:,.2f}"
            )
   
model_path = config["model"]["model_path"]
best_model.save_model(model_path)

params_path = config["model"]["params_path"]
with open(params_path, "w") as f:
    json.dump(best_params, f, indent=4)

print("Best model and parameters saved.")