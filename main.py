import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

df = pd.read_csv(os.path.join("datasets","dataset_cell_growth_4factor.csv"))

features = df.columns[1:]
targets = df.columns[0]
x = df.loc[:, features].to_numpy()
y = df.loc[:, targets].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#============================================
#   Linear regression
#============================================
model_linear = make_pipeline(StandardScaler(), LinearRegression())
linear_cv_results = cross_validate(model_linear, x_train, y_train, cv=5,
                                   scoring=('neg_root_mean_squared_error', "r2"),
                                   return_train_score=True)

linear_train_rmse = np.mean(-linear_cv_results["train_neg_root_mean_squared_error"] )
linear_train_r2 = np.mean(linear_cv_results["train_r2"])
linear_valid_rmse = np.mean(-linear_cv_results["test_neg_root_mean_squared_error"])
linear_valid_r2 = np.mean(linear_cv_results["test_r2"])

#============================================
#   Polynomial regression
#============================================
model_polynomial = make_pipeline(StandardScaler(),
                                 PolynomialFeatures(include_bias=False), LinearRegression())
param_grid_pol = dict(polynomialfeatures__degree = [i for i in range(1,6)])
model_polynomial_gcsv = GridSearchCV(model_polynomial, param_grid=param_grid_pol, cv=5,
                                    scoring={"rmse":"neg_root_mean_squared_error", "r2":"r2"},
                                    refit = "rmse", return_train_score=True)
model_polynomial_gcsv.fit(x_train, y_train)

df_polynomial = pd.DataFrame(model_polynomial_gcsv.cv_results_)
columns_view = ["rank_test_rmse","param_polynomialfeatures__degree",
    "mean_test_rmse", "std_test_rmse","mean_train_rmse", "std_train_rmse",
    "mean_test_r2", "mean_train_r2"]

# Examine polynomial model weights
best_polynomial = model_polynomial_gcsv.best_estimator_
coefs = best_polynomial["linearregression"].coef_
intercept = best_polynomial["linearregression"].intercept_
feature_names = [name.split(" [")[0].replace(" ", "_") for name in features]
terms = best_polynomial["polynomialfeatures"].get_feature_names_out(feature_names)


#============================================
#   k-nearest neighbors
#============================================
model_knn = make_pipeline(StandardScaler(), KNeighborsRegressor())
param_grid_knn =dict (kneighborsregressor__n_neighbors= [i for i in range (1,31)],
                                        kneighborsregressor__weights = ["uniform", "distance"])
model_knn_gcsv = GridSearchCV(model_knn, param_grid = param_grid_knn, cv=5,
                              scoring= {"rmse":"neg_root_mean_squared_error", "r2":"r2"},
                              refit = "rmse", return_train_score=True)
model_knn_gcsv.fit(x_train, y_train)

df_knn = pd.DataFrame(model_knn_gcsv.cv_results_)
df_knn = df_knn.sort_values(by="rank_test_rmse", ascending=True)

columns_view_knn = [
    "rank_test_rmse", "param_kneighborsregressor__n_neighbors",
    "param_kneighborsregressor__weights","mean_test_rmse", "std_test_rmse",
    "mean_train_rmse", "std_train_rmse","mean_test_r2", "mean_train_r2"]


# =====================================
# Final test evaluation
# =====================================
y_test_pred = model_knn_gcsv.predict(x_test)
test_rmse = root_mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)


# =====================================
# PRINT RESULTS
# =====================================
os.makedirs("tables", exist_ok=True)

# Linear regression
print("\nLinear regression: 5-fold cross-validation")
print(f"  Train      | RMSE | {linear_train_rmse:.3f} d^-1")
print(f"  Validation | RMSE | {linear_valid_rmse:.3f} d^-1")
print(f"  Train      | R²   | {linear_train_r2:.3f}")
print(f"  Validation | R²   | {linear_valid_r2:.3f}")

# Polynomial regression
print("\nPolynomial regression: 5-fold cross-validation")
#print(df_polynomial.loc[:, columns_view].to_string(index=False))
print(" - Best degree:", model_polynomial_gcsv.best_params_["polynomialfeatures__degree"])
print(f" - Best mean validation RMSE: " f"{-model_polynomial_gcsv.best_score_:.3f} d^-1")
print(f" - Mean validation R²: " f"{df_polynomial.loc[model_polynomial_gcsv.best_index_, 'mean_test_r2']:.3f}")

    # Output a summary table
table = df_polynomial.loc[:, columns_view].to_string(index=False, float_format="{:.3f}".format)
with open(os.path.join("tables", "polynomial_cv_results.txt"), "w") as file:
    file.write(table)

    # Get polynomial model weights
print("\nBest polynomial model (standardized inputs):")
print(f"{targets} =")
print(f"{intercept:+8.3f}")

for coef, term in zip(coefs, terms):
    print(f"{coef:+8.3f} * {term.replace(' ', ' * ')}")

# k-nearest neighbors
print("\nkNN regression: 5-fold cross-validation")
print(" - Best number of neighbors:", model_knn_gcsv.best_params_["kneighborsregressor__n_neighbors"])
print(" - Best weights:", model_knn_gcsv.best_params_["kneighborsregressor__weights"])
print(f" - Best mean validation RMSE: {-model_knn_gcsv.best_score_:.3f} d^-1")
print(f" - Mean validation R²: " f"{df_knn.loc[model_knn_gcsv.best_index_, 'mean_test_r2']:.3f}")

    # Output a summary table
table = df_knn.loc[:, columns_view_knn].to_string(index=False, float_format="{:.3f}".format)
with open(os.path.join("tables", "knn_cv_results.txt"), "w") as file:
    file.write(table)

# Final test evaluation
print("\nFinal kNN model: performance of Test set")
print(f"  Test | RMSE | {test_rmse:.3f} d^-1")
print(f"  Test | R²   | {test_r2:.3f}")


# ==========================
#   Plot Validation Curves
# ==========================

os.makedirs("figures", exist_ok=True)

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
LINESTYLES = ["-", ":", "--", "-."]
FIG_WIDTH = 14.39 / 2.54
FIG_HEIGHT = 9.2 / 2.54
fontsize_xy_label = 14
fontsize_ticks = 12
fontsize_legend = 12

# Polynomial Regression
df_sub = df_polynomial.sort_values(by="param_polynomialfeatures__degree", ascending=True)
degrees = df_sub.loc[:, "param_polynomialfeatures__degree"].to_numpy(dtype=int)
rmse_train = -df_sub.loc[:, "mean_train_rmse"].to_numpy()
rmse_valid = -df_sub.loc[:, "mean_test_rmse"].to_numpy()

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(FIG_WIDTH, FIG_HEIGHT),dpi=200, layout="constrained")

ax.plot(degrees, rmse_train,label="Train (Polynomial)", color=COLORS[0], linestyle="-")
ax.plot(degrees, rmse_valid,label="Valid (Polynomial)", color=COLORS[0], linestyle="--")
ax.set_xlabel("Polynomial degree", fontsize=fontsize_xy_label)
ax.set_ylabel("Growth Rate RMSE [d⁻¹]", fontsize=fontsize_xy_label)
ax.set_xticks(degrees)
ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
ax.legend(ncols=1, fontsize=fontsize_legend)
ax.set_ylim(bottom=0)
ax.set_title("Polynomial regression: validation curve", fontsize=14)

fig.savefig(os.path.join("figures", "validation_curve_polynomial.png"))
plt.close(fig)

# k-Nearest Neighbors
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(FIG_WIDTH, FIG_HEIGHT),dpi=200, layout="constrained")

for i, weight in enumerate(param_grid_knn["kneighborsregressor__weights"]):
    mask_weight = (df_knn.loc[:, "param_kneighborsregressor__weights"] == weight)
    df_sub = df_knn.loc[mask_weight, :]
    df_sub = df_sub.sort_values(by="param_kneighborsregressor__n_neighbors", ascending=True)
    n_neighbors = df_sub.loc[:, "param_kneighborsregressor__n_neighbors"].to_numpy(dtype=int)
    rmse_train = -df_sub.loc[:, "mean_train_rmse"].to_numpy()
    rmse_valid = -df_sub.loc[:, "mean_test_rmse"].to_numpy()
    ax.plot(n_neighbors, rmse_train,label=f"Train (kNN: {weight})", color=COLORS[i], linestyle="-")
    ax.plot(n_neighbors, rmse_valid,label=f"Valid (kNN: {weight})", color=COLORS[i], linestyle="--" )

ax.set_xlabel("k", fontsize=fontsize_xy_label)
ax.set_ylabel("Growth Rate RMSE [d⁻¹]", fontsize=fontsize_xy_label)
ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
ax.legend(ncols=1, fontsize=fontsize_legend)
ax.set_ylim(bottom=-0.005)
ax.set_title("kNN regression: validation curves", fontsize=14)

fig.savefig(os.path.join("figures", "validation_curve_knn.png"))
plt.close(fig)
