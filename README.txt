### Cell growth prediction using Machine Learning
This Python project predicts the specific growth rate of cells from temperature, pH, glucose concentration, and lactate concentration data. It compares linear regression, polynomial regression, and k-nearest neighbors regression using a simulated dataset. Cross-validation guides model selection, while a separate test set evaluates the selected model. The program exports tables and figures to support the comparison.

# Highlights
•	Build regression pipelines for cell growth prediction.
•	Scale culture variables within cross-validation pipelines.
•	Optimize polynomial degree and kNN hyperparameters.
•	Evaluate predictions using RMSE and R².
•	Export validation curves and model comparison tables.
# Environment
•	Python: 3.14.7
•	Matplotlib: 3.11.0
•	NumPy: 2.5.2
•	pandas: 3.0.5
•	scikit-learn: 1.9.0

## Code design
# Data loading and variable selection
The workflow starts when main.py reads datasets/dataset_cell_growth_4factor.csv into a pandas DataFrame. The dataset contains 318 samples. The first column contains the target, specific growth rate in d⁻¹. The remaining columns contain the four inputs: temperature, pH, glucose concentration, and lactate concentration.
The inputs and target are selected separately and converted to NumPy arrays. Separating the target from the inputs prevents the measured growth rate from being used as an input to its own prediction.

# Training and test split
The data are divided into a training set containing 80% of the observations and a test set containing 20%. Setting random_state = 0 makes the split reproducible.
The training set is used to fit and compare models and select hyperparameters. The test set is reserved for the final evaluation of the selected kNN model.

# Scaling and model pipelines
Each model is placed in a pipeline with StandardScaler. Scaling centers each input around its training mean and divides it by its training standard deviation.

# Setting and optimizing the models
Placing the scaler inside each pipeline ensures that it is fitted only on the training portion of each cross-validation fold. The validation observations are transformed using those training statistics. The polynomial pipeline also includes PolynomialFeatures, which creates powers and interaction terms before linear regression is fitted.
-	Linear regression: The program evaluates linear regression as a baseline using five-fold cross-validation on the training set. In each fold, the model is fitted on four parts and validated on the remaining part. Training and validation RMSE and R² are averaged across the five folds.
-	Polynomial regression: GridSearchCV evaluates polynomial degrees from 1 to 5 using five-fold cross-validation. It records training and validation RMSE and R² for each degree, then selects the degree with the lowest mean validation RMSE.
-	kNN regression: GridSearchCV evaluates neighbor counts from 1 to 30 with both uniform and distance weighting, giving 60 configurations. Each configuration is evaluated using five-fold cross-validation. The program records training and validation RMSE and R², then selects the configuration with the lowest mean validation RMSE.
For all models, lower RMSE indicates smaller prediction errors, while higher R² indicates better performance relative to a mean-based prediction. The program reverses RMSE signs to positive values when reporting errors in the terminal and figures.

# Final evaluation and code output
With refit="rmse", each search automatically refits its best pipeline on the full training set. The script then uses the selected kNN pipeline to predict the reserved test observations. Test RMSE and R² provide the final performance assessment. The script evaluates kNN at this stage; the justification for choosing this model is presented in the analysis below.
The program prints the main results and saves the search results in the tables directory. It also saves two PNG validation curves in the figures directory for:
-	Polynomial regression: [Polynomial degree vs  RMSE]
-	kNN regression: [ number_of_neighbors(k) vs RMSE] for both uniform and distance.

## Analysis