#!/usr/bin/env python
# coding: utf-8

# # Prediction of VaR and Dynamic Portfolio Risk Hedging
# 
# 

# Mouhamed DIALLO
# 

# # Presentation

# This project aims to enhance portfolio risk management by predicting Value at Risk (VaR) and implementing dynamic hedging strategies using machine learning. Traditional VaR models lack adaptability in volatile markets, so this project applies advanced methods like Bayesian Neural Networks and Q-Learning to capture complex patterns and improve responsiveness. By leveraging a diverse financial dataset, the project seeks to provide accurate, real-time risk predictions and adaptable hedging solutions for financial professionals managing multi-asset portfolios.

# # Step 1: Data Exploration, Preparation and VaR Prediction with ML

# We begin by importing the dataset:

# In[ ]:


import pandas as pd
from google.colab import drive
data = pd.read_csv("Stock Market Dataset.csv", delimiter=',')
display(data)


# In[ ]:


print(data.info())


# In[ ]:


print(data.describe())


# In[ ]:


data.shape


# ## DATA CLEANING

# Then, we prepare and clean the dataset for analysis. First, we ensure that price columns (like Bitcoin_Price, Ethereum_Price) are correctly formatted as floating-point numbers by removing commas and converting any object-type data. Missing values are then assessed, and columns with substantial gaps (e.g., Platinum_Price and Platinum_Vol.) are removed. The data is further cleaned by filling remaining missing values with the previous day’s values (backfill method) and checked for data quality by identifying any negative values, which would be unusual for prices, and any duplicate rows:

# In[ ]:


# Convert price columns to float if necessary
price_columns = [
    'Bitcoin_Price', 'Ethereum_Price',
    'S&P_500_Price', 'Nasdaq_100_Price', 'Berkshire_Price',
    'Gold_Price', 'Platinum_Price'
]

for col in price_columns:
    if data[col].dtype == 'object':  # Check if the column is of type object (string)
        # Remove commas to clean the values
        data[col] = data[col].str.replace(',', '', regex=False)
        # Convert the column to numeric, coercing invalid values to NaN, then to float
        data[col] = pd.to_numeric(data[col], errors='coerce').astype(float)


# In[ ]:


print(data.info())


# In[ ]:


data.shape


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Convert the date column to datetime, if applicable
# Replace 'date_column' with the name of your date column
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # Optional
data['day_of_year'] = data['Date'].dt.dayofyear  # Create a numeric column for the day of the year
data['year'] = data['Date'].dt.year  # Create a numeric column for the year

# Select only numerical columns
numerical_data = data.select_dtypes(include=['int64', 'float64'])

# Compute the correlation matrix
correlation_matrix = numerical_data.corr()

# Option 1: Heatmap without annotations
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, cmap='coolwarm', square=True, cbar=True)
plt.title("Correlation Matrix without Annotations")
plt.show()

# Option 2: Heatmap with filtered significant correlations
threshold = 0.6  # Adjust the threshold as needed
filtered_corr_matrix = correlation_matrix[(correlation_matrix >= threshold) | (correlation_matrix <= -threshold)]
plt.figure(figsize=(15, 12))
sns.heatmap(filtered_corr_matrix, annot=True, cmap='coolwarm', square=True, cbar=True)
plt.title(f"Correlation Matrix (Correlations > |{threshold}|)")
plt.show()


# In[ ]:


print("Missing Values : \n",data.isnull().sum())


# In[ ]:


# Drop columns from the dataset
data.drop(['Platinum_Price', 'Platinum_Vol.'], axis=1, inplace=True)

print(data.isnull().sum())


# In[ ]:


# Completeness: Checking for missing values
missing_data = data.isnull().sum()
print("Completeness (Missing Values):\n", missing_data[missing_data > 0])


# In[ ]:


import numpy as np
# Measures for Data Quality
# Accuracy: Checking for unusual or erroneous values (e.g., negative prices)
accuracy_issues = data[(data.select_dtypes(include=[np.number]) < 0).any(axis=1)]
print("Accuracy Issues (Negative Values):\n", accuracy_issues)


# In[ ]:


# Consistency: Checking for duplicates and inconsistent data
duplicates = data.duplicated().sum()
print(f"Number of Duplicate Rows: {duplicates}")


# In[ ]:


#Allows replacing the current value with the previous day's value
data.fillna(method='bfill', inplace=True)
print(data.isnull().sum())


# Finally, the code calculates daily logarithmic returns for the S&P 500 indices to analyze their performance and volatility over time:

# In[ ]:


#Calculation of Logarithmic Returns for S&P 500

data['S&P_500_Log_Returns'] = np.log(data['S&P_500_Price'] / data['S&P_500_Price'].shift(1))


# In[ ]:


data.head()


# ## Traditional VaR calculation for the entire period

# Currently, VaR is calculated on the complete set of returns without applying a moving window, so VaR remains constant for each row.
# Using a sliding window (optional): If you want VaR to vary day by day, you can use a sliding window (e.g., 252 days) to recalculate VaR for each day.

# Historical Method:

# In[ ]:


# Initialize a list to store VaR values with the first 29 elements as 0
VaR_actual = [0] * 29

# Calculate the 5th percentile VaR over a rolling 30-day window
for i in range(len(data['S&P_500_Log_Returns']) - 29):
    x = data['S&P_500_Log_Returns'][i : i + 30].quantile(q=0.05, interpolation='nearest')
    VaR_actual.append(x)


# Monte Carlo Method:

# In[ ]:


# Monte Carlo simulation parameters
num_simulations = 10000
VaR_monte_carlo = [0] * 29
confidence_level = 0.05

# Calculate Monte Carlo VaR over a rolling 30-day window
for i in range(len(data['S&P_500_Log_Returns']) - 29):
    window_data = data['S&P_500_Log_Returns'][i : i + 30]
    mean_return = np.mean(window_data)
    std_dev = np.std(window_data)

    # Simulate returns
    simulated_returns = np.random.normal(mean_return, std_dev, num_simulations)

    # Calculate VaR from simulations at the specified confidence level
    var_value = -np.quantile(simulated_returns, 1 - confidence_level)
    VaR_monte_carlo.append(var_value)


# In[ ]:


len(VaR_actual), len(data) , len(VaR_monte_carlo)


# In[ ]:


# Join the observed VaR vector to the log return dataframe
pd.options.mode.chained_assignment = None  # default='warn'
data['VaR_S&P_500'] = VaR_actual
data['VaR_S&P_500_montecarlo'] = VaR_monte_carlo


# In[ ]:





# In[ ]:


# Plot log returns and VaR

plt.figure(figsize=(10, 9))
plt.plot(data.index, data['S&P_500_Log_Returns'], label='S&P500 Return', linewidth=1.5, linestyle='-')
#plt.plot(data.index, data['VaR_actual_new'], label="Actual VaR", linewidth=1.5, linestyle='-')
plt.plot(data.index, data['VaR_S&P_500'], label="Historical VaR", linewidth=1.5, linestyle='-', color='orange')
plt.plot(data.index, data['VaR_S&P_500_montecarlo'], label="Monte Carlo VaR", linewidth=1.5, linestyle='-', color='red')
plt.grid(True)
plt.ylabel('Log Return', color='black', size=15)
plt.xlabel('Time', color='black', size=15)
plt.legend(loc='upper left', fontsize=12)
plt.show()


# In[ ]:


# verification of values
print(data[['Date', 'S&P_500_Log_Returns',  'VaR_S&P_500', 'VaR_S&P_500_montecarlo']].head(100))


# In[ ]:


# Remove rows where VaR = 0 (missing)
df = data[(data['VaR_S&P_500'] != 0)]


# In[ ]:


data.head(5)
data.shape


# ## Prédiction de la VaR avec machine learning

# In[ ]:


# Prepare data for VaR prediction using machine learning
# Feature columns include prices and volumes for various stocks, and the target column is historical VaR
features = data[[
    'Apple_Price', 'Apple_Vol.',
    'Tesla_Price', 'Tesla_Vol.',
    'Microsoft_Price', 'Microsoft_Vol.',
    'Google_Price', 'Google_Vol.',
    'Amazon_Price', 'Amazon_Vol.'
]]

# Target column for prediction (adjust as needed)
target = data['VaR_S&P_500']


# In[ ]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Initialize the linear regression model
linear_model = LinearRegression()

# Perform 5-fold cross-validation to compute R2 scores on the training set
scores = cross_val_score(linear_model, X_train, y_train, scoring='r2', cv=5)

# Train the model on the training data
linear_model.fit(X_train, y_train)

# Predict target values for the test data
y_pred = linear_model.predict(X_test)

# Print the model's performance metrics
print("Linear Regression")
print("Cross-validated R2 score:", scores.mean())  # Average R2 from cross-validation
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))  # Root Mean Squared Error
print("Test MAE:", mean_absolute_error(y_test, y_pred))  # Mean Absolute Error
print("Test R2:", r2_score(y_test, y_pred))  # R2 score on test data


# In[ ]:


# Importing GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Initialize the Gradient Boosting Regressor with specified parameters
gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)

# Perform cross-validation to calculate the R2 score on the training set
scores = cross_val_score(gb_model, X_train, y_train, scoring='r2', cv=5)

# Train the Gradient Boosting model on the training data
gb_model.fit(X_train, y_train)

# Predict target values for the test data
y_pred = gb_model.predict(X_test)

# Print model performance metrics
print("Gradient Boosting Regressor")
print("Cross-validated R2 score:", scores.mean())  # Average R2 score from cross-validation
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))  # Root Mean Squared Error on test data
print("Test MAE:", mean_absolute_error(y_test, y_pred))  # Mean Absolute Error on test data
print("Test R2:", r2_score(y_test, y_pred))  # R2 score on test data


# In[ ]:


from sklearn.tree import DecisionTreeRegressor

# Initialize the Decision Tree Regressor with specified parameters
dt_model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=42)

# Perform 5-fold cross-validation to compute R2 scores on the training set
scores = cross_val_score(dt_model, X_train, y_train, scoring='r2', cv=5)

# Train the model on the training data
dt_model.fit(X_train, y_train)

# Predict target values for the test data
y_pred = dt_model.predict(X_test)

# Print the model's performance metrics
print("Decision Tree Regressor")
print("Cross-validated R2 score:", scores.mean())  # Average R2 from cross-validation
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))  # Root Mean Squared Error
print("Test MAE:", mean_absolute_error(y_test, y_pred))  # Mean Absolute Error
print("Test R2:", r2_score(y_test, y_pred))  # R2 score on test data


# In[ ]:


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Model parameters
n_estimators = 1000
early_stopping_rounds = 10  # Stop after 10 rounds with no improvement
best_score = float('inf')   # Initial high score for comparison
no_improvement_count = 0    # Counter for no improvement rounds

# Initialize the XGBoost Regressor
xgb_model = XGBRegressor(
    objective='reg:squarederror',
    max_depth=3,
    learning_rate=0.1,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42
)

# Track validation scores
validation_scores = []

# Training with manual early stopping
for i in range(1, n_estimators + 1):
    # Set the current number of estimators
    xgb_model.set_params(n_estimators=i)
    xgb_model.fit(X_train, y_train)

    # Validate the model on the test set
    y_val_pred = xgb_model.predict(X_test)
    val_score = mean_squared_error(y_test, y_val_pred)

    # Record validation score
    validation_scores.append(val_score)

    # Check for improvement
    if val_score < best_score:
        best_score = val_score
        no_improvement_count = 0  # Reset counter
    else:
        no_improvement_count += 1

    # Stop if no improvement for the specified number of rounds
    if no_improvement_count >= early_stopping_rounds:
        print(f"Early stopping at iteration {i}")
        break

# Final evaluation after early stopping
y_pred = xgb_model.predict(X_test)
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Test R2:", r2_score(y_test, y_pred))


# In[ ]:


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Initialize and train the Ridge regression model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_ridge = ridge_model.predict(X_test)

# Compute evaluation metrics
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

# Print metrics
print("Test RMSE:", rmse_ridge)  # Root Mean Squared Error
print("Test MAE:", mae_ridge)    # Mean Absolute Error
print("Test R2:", r2_ridge)      # R2 Score


# In[ ]:


from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def svm_prediction(X_train, X_test, y_train, y_test):
    # Initialize the SVR model with an RBF kernel
    model = SVR(kernel='rbf')  # RBF kernel is used, but 'linear' or others can also be selected

    # Perform cross-validation to calculate the mean R² score
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    # Train the model on the training dataset
    model.fit(X_train, y_train)

    # Predict the target variable for the test dataset
    y_pred = model.predict(X_test)

    # Compute and print evaluation metrics
    print("Support Vector Regression")
    print("Cross-validated R2 score:", scores.mean())  # Mean R2 score from cross-validation
    print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))  # Root Mean Squared Error
    print("Test MAE:", mean_absolute_error(y_test, y_pred))  # Mean Absolute Error
    print("Test R2:", r2_score(y_test, y_pred))  # R2 Score on test data

# Call the function with the training and test datasets
svm_prediction(X_train, X_test, y_train, y_test)


# In[ ]:


# Initialize the models
models = {
    "Linear Regression": LinearRegression(),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
    "Decision Tree": DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=42),
    "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, learning_rate=0.1, reg_alpha=0.1, reg_lambda=0.1, random_state=42),
    "Ridge": Ridge(alpha=1.0),
}

# Create a list to store results
results = []

# Loop through each model to compute metrics
for model_name, model in models.items():
    # Cross-validated R2
    scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=5)
    cv_r2 = scores.mean()

    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Add results to the list
    results.append({
        "Model": model_name,
        "Cross-validated R2": cv_r2,
        "Test RMSE": rmse,
        "Test MAE": mae,
        "Test R2": r2
    })

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)

# Display results
print(results_df)

# Visualization: Comparing the models
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# RMSE Chart
axes[0].bar(results_df["Model"], results_df["Test RMSE"])
axes[0].set_title("Model Comparison - RMSE")
axes[0].set_ylabel("RMSE")

# MAE Chart
axes[1].bar(results_df["Model"], results_df["Test MAE"])
axes[1].set_title("Model Comparison - MAE")
axes[1].set_ylabel("MAE")

# R² Chart
axes[2].bar(results_df["Model"], results_df["Test R2"])
axes[2].set_title("Model Comparison - R²")
axes[2].set_ylabel("R²")

plt.tight_layout()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

# Initialization of models
models = {
    "Linear Regression": LinearRegression(),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
    "Decision Tree": DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=42),
    "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, learning_rate=0.1, reg_alpha=0.1, reg_lambda=0.1, random_state=42),
    "Ridge": Ridge(alpha=1.0),
}

# Training and storing predictions for each model
predictions = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Predict test data
    predictions[model_name] = y_pred  # Store predictions

# Create subplots for visualizing each model's predictions against actual values
fig, axes = plt.subplots(len(models), 1, figsize=(12, 3 * len(models)), sharex=True)

for i, (model_name, y_pred) in enumerate(predictions.items()):
    # Plot actual values
    axes[i].plot(y_test.reset_index(drop=True), label="Actual Values", color='black', linewidth=1.5)
    # Plot predicted values for the current model
    axes[i].plot(y_pred, label=f"Prediction - {model_name}", linestyle='--', color='blue')
    axes[i].set_title(f"Comparison of Actual Values and Predictions - {model_name}")
    axes[i].set_ylabel("Value")
    axes[i].legend()

# Configure x-axis label for the last subplot and adjust layout
axes[-1].set_xlabel("Time Index")
plt.tight_layout()
plt.show()


# In[ ]:





# # Stage 2 : Advanced Techniques for Accurate and Reliable VaR Predictions

# In[ ]:


# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer

# External libraries
from xgboost import XGBRegressor
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import chi2

# Optimization library
get_ipython().system('pip install optuna')
import optuna


# ##Grid Search Optimization and Evaluation of Regression Models

# ### Hyperparameter optimization for the Gradient Boosting model

# In[ ]:


# Initialize the Gradient Boosting model
gb_model = GradientBoostingRegressor(random_state=42)

# Define the hyperparameter grid for Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=gb_model,
    param_grid=param_grid,
    scoring='r2',
    cv=3,            # Reduce the number of folds in cross-validation
    n_jobs=-1,
    verbose=1
)
# Execute the search on the training data
grid_search.fit(X_train, y_train)

# Best hyperparameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Evaluate the optimized model on the test set
best_gb_model = grid_search.best_estimator_
y_test_pred = best_gb_model.predict(X_test)

# Compute performance metrics
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

# Results
optimization_results = {
    "Best Parameters": best_params,
    "Best Cross-validated R2": best_score,
    "Test R2": test_r2,
    "Test RMSE": test_rmse,
    "Test MAE": test_mae
}

optimization_results


# ###Optimization and Evaluation of a Decision Tree Regression Model

# In[ ]:


# Hyperparameter grid for Decision Tree
dt_param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}

# Configure Grid Search for Decision Tree
dt_model = DecisionTreeRegressor(random_state=42)
dt_grid_search = GridSearchCV(
    estimator=dt_model,
    param_grid=dt_param_grid,
    scoring='r2',
    cv=3,
    n_jobs=-1,
    verbose=1
)

# Execute Grid Search for Decision Tree
print("Hyperparameter optimization for Decision Tree...")
dt_grid_search.fit(X_train, y_train)

# Best hyperparameters and score
dt_best_params = dt_grid_search.best_params_
dt_best_score = dt_grid_search.best_score_

# Evaluate Decision Tree on the test set
dt_best_model = dt_grid_search.best_estimator_
dt_y_test_pred = dt_best_model.predict(X_test)
dt_test_r2 = r2_score(y_test, dt_y_test_pred)
dt_test_rmse = np.sqrt(mean_squared_error(y_test, dt_y_test_pred))
dt_test_mae = mean_absolute_error(y_test, dt_y_test_pred)

# Results for Decision Tree
dt_results = {
    "Best Parameters": dt_best_params,
    "Best Cross-validated R2": dt_best_score,
    "Test R2": dt_test_r2,
    "Test RMSE": dt_test_rmse,
    "Test MAE": dt_test_mae
}
dt_results


# ###Optimization and Evaluation of an XGBoost Regression Model

# In[ ]:


# Hyperparameter grid for XGBoost
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Configure Grid Search for XGBoost
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=xgb_param_grid,
    scoring='r2',
    cv=3,
    n_jobs=-1,
    verbose=1
)

# Execute Grid Search for XGBoost
print("Hyperparameter optimization for XGBoost...")
xgb_grid_search.fit(X_train, y_train)

# Best hyperparameters and score
xgb_best_params = xgb_grid_search.best_params_
xgb_best_score = xgb_grid_search.best_score_

# Evaluate XGBoost on the test set
xgb_best_model = xgb_grid_search.best_estimator_
xgb_y_test_pred = xgb_best_model.predict(X_test)
xgb_test_r2 = r2_score(y_test, xgb_y_test_pred)
xgb_test_rmse = np.sqrt(mean_squared_error(y_test, xgb_y_test_pred))
xgb_test_mae = mean_absolute_error(y_test, xgb_y_test_pred)

# Results for XGBoost
xgb_results = {
    "Best Parameters": xgb_best_params,
    "Best Cross-validated R2": xgb_best_score,
    "Test R2": xgb_test_r2,
    "Test RMSE": xgb_test_rmse,
    "Test MAE": xgb_test_mae
}
xgb_results


# 
# ###Comparison of Predictions from Optimized Regression Models

# In[ ]:


# Best hyperparameters found
best_gb_model = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.6, random_state=42
)

best_dt_model = DecisionTreeRegressor(
    max_depth=7, min_samples_leaf=5, min_samples_split=2, random_state=42
)

best_xgb_model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=50, max_depth=5, learning_rate=0.2, subsample=1.0,
    colsample_bytree=0.8, random_state=42
)

# Train models on the training set
best_gb_model.fit(X_train, y_train)
best_dt_model.fit(X_train, y_train)
best_xgb_model.fit(X_train, y_train)

# Make predictions for each model
y_test_pred_gb = best_gb_model.predict(X_test)  # Gradient Boosting
y_test_pred_dt = best_dt_model.predict(X_test)  # Decision Tree
y_test_pred_xgb = best_xgb_model.predict(X_test)  # XGBoost

# Create a dictionary with predictions
optimized_predictions = {
    "Gradient Boosting": y_test_pred_gb,
    "Decision Tree": y_test_pred_dt,
    "XGBoost": y_test_pred_xgb
}

# Visualize predictions for each model
fig, axes = plt.subplots(len(optimized_predictions), 1, figsize=(12, 4 * len(optimized_predictions)), sharex=True)

for i, (model_name, y_pred) in enumerate(optimized_predictions.items()):
    # Plot actual values
    axes[i].plot(y_test.reset_index(drop=True), label="Actual Values", color='black', linewidth=1.5)
    # Plot predicted values
    axes[i].plot(y_pred, label=f"Predictions - {model_name}", linestyle='--', color='blue', linewidth=1.5)
    axes[i].set_title(f"Comparison of Actual and Predicted Values - {model_name}", fontsize=14)
    axes[i].set_ylabel("VaR_S&P_500", fontsize=12)
    axes[i].legend(fontsize=10)
    axes[i].grid(True)

# Configure the X-axis for the last subplot
axes[-1].set_xlabel("Index", fontsize=12)
plt.tight_layout()
plt.show()


# ##Optuna: Bayesian optimization mentioned.
# 
# Enables a more efficient exploration of the hyperparameter space compared to an exhaustive grid search.

# ###Hyperparameter Optimization for XGBoost using Optuna

# In[ ]:


# Objective function for Optuna
def objective_xgb(trial):
    # Define the hyperparameter search space
    param_grid = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0)
    }

    # Initialize the model with the proposed hyperparameters
    model = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        **param_grid
    )

    # Perform cross-validation on the training data
    scores = cross_val_score(model, X_train, y_train, scoring=make_scorer(r2_score), cv=5, n_jobs=-1)

    # Return the mean score (maximizing R²)
    return np.mean(scores)

# Create the Optuna study to optimize hyperparameters
study = optuna.create_study(direction='maximize', study_name="XGB Optimization")
study.optimize(objective_xgb, n_trials=50, n_jobs=-1)

# Optimization results
best_params_optuna = study.best_params
best_score_optuna = study.best_value

# Display the best hyperparameters and score
best_params_optuna, best_score_optuna


# ###Hyperparameter Optimization for Gradient Boosting using Optuna

# In[ ]:


# Objective function for Optuna - Gradient Boosting
def objective_gb(trial):
    # Define the hyperparameter search space
    param_grid = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0)
    }

    # Initialize the model with the proposed hyperparameters
    model = GradientBoostingRegressor(random_state=42, **param_grid)

    # Perform cross-validation on the training data
    scores = cross_val_score(model, X_train, y_train, scoring=make_scorer(r2_score), cv=5, n_jobs=-1)

    # Return the mean score (maximizing R²)
    return np.mean(scores)



# Create the Optuna studies
study_gb = optuna.create_study(direction='maximize', study_name="Gradient Boosting Optimization")

# Optimize the hyperparameters
study_gb.optimize(objective_gb, n_trials=50, n_jobs=-1)

# Optimization results
best_params_gb = study_gb.best_params
best_score_gb = study_gb.best_value

# Display the results
best_params_gb, best_score_gb


# ###Hyperparameter Optimization for Decision Tree using Optuna

# In[ ]:


# Objective function for Optuna - Decision Tree
def objective_dt(trial):
    # Define the hyperparameter search space
    param_grid = {
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20)
    }

    # Initialize the model with the proposed hyperparameters
    model = DecisionTreeRegressor(random_state=42, **param_grid)

    # Perform cross-validation on the training data
    scores = cross_val_score(model, X_train, y_train, scoring=make_scorer(r2_score), cv=5, n_jobs=-1)

    # Return the mean score (maximizing R²)
    return np.mean(scores)

study_dt = optuna.create_study(direction='maximize', study_name="Decision Tree Optimization")
study_dt.optimize(objective_dt, n_trials=50, n_jobs=-1)

best_params_dt = study_dt.best_params
best_score_dt = study_dt.best_value

best_params_dt, best_score_dt


# ### Recover and display the best parameters and score after optimization :

# In[ ]:


# Study for Gradient Boosting
study_gb = optuna.create_study(direction='maximize', study_name="Gradient Boosting Optimization")
study_gb.optimize(objective_gb, n_trials=20, n_jobs=-1)  # Reduce to 20 trials for faster execution

# Study for Decision Tree
study_dt = optuna.create_study(direction='maximize', study_name="Decision Tree Optimization")
study_dt.optimize(objective_dt, n_trials=20, n_jobs=-1)  # Reduce to 20 trials for faster execution

# Optimization results
best_params_gb = study_gb.best_params
best_score_gb = study_gb.best_value

best_params_dt = study_dt.best_params
best_score_dt = study_dt.best_value

# Display the results
best_params_gb, best_score_gb, best_params_dt, best_score_dt


# In[ ]:


# Optimal hyperparameters found by Optuna
best_params_gb = study_gb.best_params
best_params_dt = study_dt.best_params
best_params_xgb = study.best_params

# Add additional model-specific parameters if necessary (e.g., random_state)
best_params_gb['random_state'] = 42
best_params_dt['random_state'] = 42
best_params_xgb['random_state'] = 42


# In[ ]:


# Initialize models with the optimized hyperparameters
model_gb = GradientBoostingRegressor(**best_params_gb)
model_dt = DecisionTreeRegressor(**best_params_dt)
model_xgb = XGBRegressor(**best_params_xgb)

# Train the models on the training data
model_gb.fit(X_train, y_train)
model_dt.fit(X_train, y_train)
model_xgb.fit(X_train, y_train)


# In[ ]:


# reminder of predictions on the test data
y_pred_gb = model_gb.predict(X_test)
y_pred_dt = model_dt.predict(X_test)
y_pred_xgb = model_xgb.predict(X_test)


# In[ ]:


# Function to display performance and plots
def plot_results(y_test, y_pred, model_name):
    plt.figure(figsize=(15, 5))

    # Plot 1: Comparison of Predictions vs Actual Values
    plt.subplot(1, 3, 1)
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predictions")
    plt.title(f"{model_name}: Predictions vs Actual")

    # Plot 2: Residual Distribution
    plt.subplot(1, 3, 2)
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True, bins=30)
    plt.title(f"{model_name}: Residual Distribution")
    plt.xlabel("Error (Residuals)")

    # Plot 3: Density of Predictions
    plt.subplot(1, 3, 3)
    sns.kdeplot(y_test, label="Actual Values", color="blue", fill=True)
    sns.kdeplot(y_pred, label="Predictions", color="orange", fill=True)
    plt.title(f"{model_name}: Density")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Display the results for each model
plot_results(y_test, y_pred_gb, "Gradient Boosting")
plot_results(y_test, y_pred_dt, "Decision Tree")
plot_results(y_test, y_pred_xgb, "XGBoost")


# In[ ]:


# Calculate metrics
metrics = {
    "Model": ["Gradient Boosting", "Decision Tree", "XGBoost"],
    "R²": [
        r2_score(y_test, y_pred_gb),
        r2_score(y_test, y_pred_dt),
        r2_score(y_test, y_pred_xgb),
    ],
    "RMSE": [
        mean_squared_error(y_test, y_pred_gb, squared=False),
        mean_squared_error(y_test, y_pred_dt, squared=False),
        mean_squared_error(y_test, y_pred_xgb, squared=False),
    ],
}

# Create a DataFrame
results_df = pd.DataFrame(metrics)

# Display the table
print(results_df)


# ## kupiec & Christoffersen test

# The Kupiec test checks if the proportion of exceedances matches the expected probability, ensuring the model estimates overall risk correctly. The Christoffersen test assesses if exceedances are independent over time. Together, they evaluate the accuracy and reliability of VaR predictions.

# In[ ]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42)


# In[ ]:


print(type(X_test))


# In[ ]:


def kupiec_test(data, alpha):
    """
    Performs the Kupiec test (Proportion of Failures Test).

    Parameters:
    - data: DataFrame containing the 'VaR_Breach' column.
    - alpha: VaR confidence level (e.g., 0.05 for a 95% VaR).

    Returns:
    - LR_pof: Test statistic.
    - p_value: P-value of the test.
    """
    n = len(data)
    x = data['VaR_Breach'].sum()
    pi = x / n
    pi0 = alpha
    # Handle cases where pi is 0 or 1
    if pi == 0:
        pi = 1e-10
    if pi == 1:
        pi = 1 - 1e-10
    # Likelihood ratio statistic
    LR_pof = -2 * (
        np.log(((1 - pi0) ** (n - x) * pi0 ** x) /
               ((1 - pi) ** (n - x) * pi ** x))
    )
    p_value = 1 - chi2.cdf(LR_pof, df=1)
    return LR_pof, p_value

def christoffersen_test(data, alpha):
    """
    Performs the Christoffersen test for independence and conditional coverage.

    Parameters:
    - data: DataFrame containing the 'VaR_Breach' column.
    - alpha: VaR confidence level.

    Returns:
    - LR_independence: Independence test statistic.
    - p_value_independence: P-value of the independence test.
    - LR_conditional: Conditional coverage test statistic.
    - p_value_conditional: P-value of the conditional coverage test.
    """
    breaches = data['VaR_Breach'].astype(int).values
    n = len(breaches)

    # Remove the first observation to align lags
    breaches = breaches[1:]
    n = len(breaches)

    # Lagged breaches
    breaches_lagged = data['VaR_Breach'].astype(int).shift(1).values[1:]

    # Count transitions
    n00 = np.sum((breaches_lagged == 0) & (breaches == 0))
    n01 = np.sum((breaches_lagged == 0) & (breaches == 1))
    n10 = np.sum((breaches_lagged == 1) & (breaches == 0))
    n11 = np.sum((breaches_lagged == 1) & (breaches == 1))

    # Total counts
    n0 = n00 + n01
    n1 = n10 + n11

    # Transition probabilities
    pi01 = n01 / n0 if n0 > 0 else 0
    pi11 = n11 / n1 if n1 > 0 else 0
    pi1 = (n01 + n11) / (n0 + n1)

    # Likelihood ratio statistics
    # Independence test
    L0 = ((1 - pi1) ** (n00 + n10)) * (pi1 ** (n01 + n11))
    L1 = ((1 - pi01) ** n00) * (pi01 ** n01) * ((1 - pi11) ** n10) * (pi11 ** n11)
    # Handle cases where probabilities are zero
    if L0 == 0 or L1 == 0:
        LR_independence = 0
    else:
        LR_independence = -2 * np.log(L0 / L1)
    p_value_independence = 1 - chi2.cdf(LR_independence, df=1)

    # Conditional coverage test
    LR_pof, _ = kupiec_test(data, alpha)
    LR_conditional = LR_pof + LR_independence
    p_value_conditional = 1 - chi2.cdf(LR_conditional, df=2)

    return LR_independence, p_value_independence, LR_conditional, p_value_conditional


# In[ ]:


# List of models and their predictions
models = {
    'Gradient Boosting': y_pred_gb,
    'Decision Tree': y_pred_dt,
    'XGBoost': y_pred_xgb
}

# VaR confidence level (e.g., 95%)
alpha = 0.05

# Loop through models to apply the tests
for model_name, y_pred in models.items():
    print(f"\n=== Results for the model {model_name} ===")

    # 1. Create a DataFrame for the test set
    data_test = data.loc[X_test.index].copy()

    # 2. Assign the VaR predictions to the test DataFrame
    data_test['VaR_predicted'] = y_pred

    # 3. Calculate breaches for the predicted VaR
    data_test['VaR_Breach'] = data_test['S&P_500_Log_Returns'] < data_test['VaR_predicted']

    # 4. Drop any missing values (if necessary)
    data_test = data_test.dropna(subset=['VaR_Breach'])

    # Check if there are enough data points for the tests
    if data_test['VaR_Breach'].sum() == 0:
        print("No VaR breaches detected. Tests cannot be performed.")
        continue

    # 5. Perform the Kupiec test on the test set
    LR_pof_ml, p_value_ml = kupiec_test(data_test, alpha)
    print(f"Kupiec test statistic: {LR_pof_ml:.4f}")
    print(f"Kupiec test p-value: {p_value_ml:.4f}")

    # 6. Perform the Christoffersen test on the test set
    LR_ind_ml, p_val_ind_ml, LR_cond_ml, p_val_cond_ml = christoffersen_test(data_test, alpha)

    print(f"Christoffersen independence test statistic: {LR_ind_ml:.4f}")
    print(f"Independence test p-value: {p_val_ind_ml:.4f}")

    print(f"Christoffersen conditional coverage test statistic: {LR_cond_ml:.4f}")
    print(f"Conditional coverage test p-value: {p_val_cond_ml:.4f}")


# === Results for the model Gradient Boosting ===
# 
# Kupiec test statistic: 1.5788
# 
# Kupiec test p-value: 0.2089
# 
# Christoffersen independence test statistic: 0.0101
# 
# Independence test p-value: 0.9201
# 
# Christoffersen conditional coverage test statistic: 1.5889
# 
# Conditional coverage test p-value: 0.4518
# 
# === Results for the model Decision Tree ===
# Kupiec test statistic: 3.1464
# 
# Kupiec test p-value: 0.0761
# 
# Christoffersen independence test statistic: 0.1327
# 
# Independence test p-value: 0.7157
# 
# Christoffersen conditional coverage test statistic: 3.2790
# 
# Conditional coverage test p-value: 0.1941
# 
# === Results for the model XGBoost ===
# Kupiec test statistic: 0.9813
# 
# Kupiec test p-value: 0.3219
# 
# Christoffersen independence test statistic: 0.9945
# 
# Independence test p-value: 0.3186
# 
# Christoffersen conditional coverage test statistic: 1.9759
# 
# Conditional coverage test p-value: 0.3723

# In[ ]:





# 

# In[ ]:





# # Stage 3

# In[ ]:


pip install PyWavelets


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Conv1D, MaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pywt


# ## LSTM

#  Cette fonction prépare les données pour un modèle LSTM :
#  1. Nettoie les données en supprimant les valeurs manquantes.
#  2. Convertit les colonnes de features en float.
#  3. Calcule les rendements logarithmiques pour les colonnes de prix.
#  4. Normalise les features et la cible avec MinMaxScaler.
#  5. Crée des séquences temporelles sur une fenêtre glissante (window_size).
#  6. Renvoie les séquences (X, y) et le scaler pour la cible.
# 

# In[ ]:


def prepare_lstm_data(data, features_columns, target_column, window_size=30):
    # Nettoyage et préparation des données
    data = data.dropna(subset=features_columns + [target_column])

    # Convertir les colonnes en float
    for col in features_columns:
        data[col] = data[col].replace(',', '', regex=True).astype(float)

    # Calcul des rendements logarithmiques
    for col in features_columns:
        if 'Price' in col:
            data[f'{col}_Log_Returns'] = np.log(data[col] / data[col].shift(1))

    # Mise à jour des features
    updated_features_columns = features_columns + [f'{col}_Log_Returns' for col in features_columns if 'Price' in col]
    data = data.dropna()

    # Préparation des séquences
    features = data[updated_features_columns].values
    target = data[target_column].values

    # Normalisation
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()
    features_scaled = scaler_features.fit_transform(features)
    target_scaled = scaler_target.fit_transform(target.reshape(-1, 1))

    # Création des séquences
    X, y = [], []
    for i in range(window_size, len(features_scaled)):
        X.append(features_scaled[i-window_size:i])
        y.append(target_scaled[i])

    return np.array(X), np.array(y), scaler_target


# Crée un modèle LSTM séquentiel avec 3 couches LSTM, des Dropouts pour régularisation,
# et des couches Dense pour la sortie. Compile avec Adam, MSE comme perte, et MAE comme métrique.
# 

# In[ ]:


def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, activation='relu', return_sequences=True),
        Dropout(0.3),
        LSTM(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# In[ ]:




def train_and_evaluate_model(X, y, test_size=0.2, random_state=42):
    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Création du modèle
    model = create_lstm_model(X_train.shape[1:])

    # Callback pour early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    # Entraînement
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    # Prédictions
    y_pred = model.predict(X_test)

    return y_test, y_pred, model, history


# Cette fonction entraîne et évalue un modèle LSTM :
# 1. Divise les données en ensembles d'entraînement et de test.
# 2. Crée un modèle LSTM avec la fonction `create_lstm_model`.
# 3. Configure un callback EarlyStopping pour arrêter l'entraînement
# si la perte de validation ne s'améliore pas pendant 15 epochs.
# 4. Entraîne le modèle sur les données d'entraînement avec une validation croisée.
# 5. Prédit les valeurs sur l'ensemble de test.
# 6. Retourne les vraies valeurs, les prédictions, le modèle entraîné, et l'historique d'entraînement.
# 

# In[ ]:


def evaluate_model(y_test, y_pred, scaler_target):
    # Dénormalisation des prédictions et valeurs réelles
    y_test_original = scaler_target.inverse_transform(y_test)
    y_pred_original = scaler_target.inverse_transform(y_pred)

    # Métriques de performance
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    r2 = r2_score(y_test_original, y_pred_original)

    print(f"RMSE: {rmse}")
    print(f"R² Score: {r2}")

    # Visualisation
    plt.figure(figsize=(12,6))
    plt.plot(y_test_original, label='Valeurs Réelles')
    plt.plot(y_pred_original, label='Prédictions', linestyle='--')
    plt.title('Prédictions LSTM vs Valeurs Réelles')
    plt.legend()
    plt.show()

    return rmse, r2


# Utilisation

# In[ ]:


features_columns = [
    'Apple_Price', 'Apple_Vol.',
    'Tesla_Price', 'Tesla_Vol.',
    'Microsoft_Price', 'Microsoft_Vol.'
]
target_column = 'VaR_S&P_500'


# Préparation des données

# In[ ]:


X, y, scaler_target = prepare_lstm_data(data, features_columns, target_column)


# Entraînement et évaluation

# In[ ]:


y_test, y_pred, model, history = train_and_evaluate_model(X, y)


# Évaluation finale

# In[ ]:


rmse, r2_LSTM = evaluate_model(y_test, y_pred, scaler_target)


# ## Hybrid model CNN-LSTM with wavelet denoising

# Cette fonction applique un débruitage de données avec la méthode wavelet :
# 1. Décompose les données en coefficients wavelet à l'aide de `wavedec`.
# 2. Calcule un seuil adaptatif basé sur l'estimation de bruit des coefficients.
# 3. Applique un seuillage doux aux coefficients de détail pour réduire le bruit.
# 4. Reconstruit les données débruitées avec `waverec`.
# 5. Gère les erreurs potentielles et retourne les données d'origine en cas de problème.
# 

# In[ ]:


def wavelet_denoising(data, wavelet='db4', level=2):
    # Débruitage wavelet avec gestion des erreurs
    try:
        coeffs = pywt.wavedec(data, wavelet, level=level)
        # Seuillage doux des coefficients
        sigma = (1/0.6745) * np.median(np.abs(coeffs[-1]))
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], value=sigma, mode='soft')
        denoised_data = pywt.waverec(coeffs, wavelet)
        return denoised_data[:len(data)]
    except Exception as e:
        print(f"Erreur lors du débruitage wavelet : {e}")
        return data


# 1. Cette fonction crée des séquences glissantes pour des séries temporelles.  
# 2. Divise les features en segments de longueur `window_size`.  
# 3. Associe chaque segment à une valeur cible située juste après la fenêtre.  
# 4. Retourne deux tableaux NumPy : les séquences (`X`) et leurs cibles (`y`).  
# 

# In[ ]:


def create_sequences(features, target, window_size=10):
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i+window_size])
        y.append(target[i+window_size])
    return np.array(X), np.array(y)


# 1. Construit un modèle combinant CNN et LSTM pour analyser des séries temporelles.  
# 2. Ajoute une couche `Conv1D` et `MaxPooling1D` pour extraire et réduire les caractéristiques.  
# 3. Intègre deux couches LSTM pour capturer les dépendances temporelles, avec Dropout pour régularisation.  
# 4. Ajoute des couches Dense pour les relations non linéaires et une couche de sortie pour la prédiction.  
# 5. Compile le modèle avec Adam (learning rate 0.001), MSE comme perte, et MAE comme métrique.  
# 

# In[ ]:


def build_advanced_model(input_shape):
    model = Sequential([
        # Couche CNN pour l'extraction de caractéristiques
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),

        # Couche LSTM pour capturer les dépendances temporelles
        LSTM(100, activation='tanh', return_sequences=True),
        Dropout(0.3),

        LSTM(50, activation='tanh'),
        Dropout(0.2),

        # Couches denses pour l'apprentissage non linéaire
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),

        # Couche de sortie
        Dense(1)
    ])

    # Configuration de l'optimiseur avec un taux d'apprentissage adaptatif
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model


# 1. Entraîne et évalue un modèle avancé sur des séries temporelles.  
# 2. Configure EarlyStopping et ReduceLROnPlateau pour régularisation et ajustement du taux d'apprentissage.  
# 3. Construit et entraîne le modèle avec validation croisée.  
# 4. Prédit les valeurs sur le test, calcule RMSE et R².  
# 5. Visualise les prédictions vs valeurs réelles et affiche les résultats.  
# 6. Retourne RMSE, R² et le modèle entraîné.  
# 

# In[ ]:


def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_name='Advanced Model'):
    # Création des callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001
    )

    # Construction et entraînement du modèle
    model = build_advanced_model(X_train.shape[1:])

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Prédictions
    y_pred = model.predict(X_test).flatten()

    # Calcul des métriques
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Visualisation
    plt.figure(figsize=(12,6))
    plt.plot(y_test, label='Valeurs Réelles', color='blue')
    plt.plot(y_pred, label='Prédictions', color='red', linestyle='--')
    plt.title(f'Prédictions vs Valeurs Réelles ({model_name})')
    plt.xlabel('Échantillons')
    plt.ylabel('Valeur')
    plt.legend()
    plt.show()

    print(f"Résultats pour {model_name}:")
    print(f"RMSE: {rmse}")
    print(f"R² Score: {r2}")

    return rmse, r2, model


# Colonnes de features et target

# In[ ]:



features_columns = [
    'Apple_Price', 'Apple_Vol.',
    'Tesla_Price', 'Tesla_Vol.',
    'Microsoft_Price', 'Microsoft_Vol.'
]
target_column = 'VaR_S&P_500'


# Débruitage de la target

# In[ ]:


data[target_column] = wavelet_denoising(data[target_column].values)


# Normalisation

# In[ ]:


scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()


# Préparation des features et de la target

# In[ ]:


X = scaler_features.fit_transform(data[features_columns])
y = scaler_target.fit_transform(data[target_column].values.reshape(-1, 1)).flatten()


# Création des séquences

# In[ ]:


X_seq, y_seq = create_sequences(X, y, window_size=10)


# Division des données

# In[ ]:



X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)


# Entraînement et évaluation du modèle

# In[ ]:


rmse, r2_CNN_LSTM, model = train_and_evaluate_model(X_train, y_train, X_test, y_test)


# The results show that the denoised GRU and CNN-LSTM models have relatively similar performance but still yield negative 𝑅² scores, indicating that the models fail to effectively capture relationships within the data.

# ## Transformer

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# 1. Implémente un encodage positionnel pour un modèle Transformer.
# 2. Calcule une matrice d'encodage positionnel basée sur des sinus et cosinus pour représenter les positions dans une séquence.
# 3. Stocke la matrice d'encodage (pe) en tant que tenseur non paramétrique avec `register_buffer`.
# 4. Ajoute l'encodage positionnel aux embeddings d'entrée dans la méthode `forward`.
# 

# In[ ]:


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]


# 1. Implémente un Transformer avancé pour séries temporelles.  
# 2. Inclut une couche d'entrée avec GELU et normalisation (`LayerNorm`).  
# 3. Ajoute un encodage positionnel pour les séquences.  
# 4. Utilise un encodeur Transformer configurable avec GELU et Dropout.  
# 5. Génère une prédiction à partir du dernier pas temporel via des couches résiduelles avec Dropout et normalisation.  
# 

# In[ ]:


class AdvancedTransformerModel(nn.Module):
    def __init__(self, input_dim, seq_length,
                 d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1):
        super(AdvancedTransformerModel, self).__init__()

        # Input embedding with more robust normalization
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()  # Using GELU instead of ReLU for potentially better performance
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer encoder layers with modifications
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu'  # Changed to GELU
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output layers with residual connection
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        x = self.input_layer(x)  # Embed input
        x = self.positional_encoding(x)  # Add positional encoding

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Take the last time step
        x = x[:, -1, :]

        # Output layers
        return self.output_layers(x)


# 1. Normalise les cibles pour améliorer la stabilité de l'entraînement.  
# 2. Prépare les datasets et dataloaders pour l'entraînement et la validation.  
# 3. Initialise le modèle Transformer avancé et le transfère sur GPU si disponible.  
# 4. Configure l'optimiseur AdamW avec une décroissance de poids et un programmeur de taux d'apprentissage (cosine annealing).  
# 5. Utilise la perte Huber (SmoothL1) pour une régression robuste.  
# 6. Implémente une boucle d'entraînement avec validation et Early Stopping pour éviter le surapprentissage.  
# 7. Sauvegarde le meilleur modèle selon la perte de validation.  
# 8. Charge le meilleur modèle pour l'évaluation finale.  
# 9. Inverse la normalisation des prédictions pour calculer RMSE et R².  
# 10. Visualise les prédictions finales comparées aux valeurs réelles.  
# 11. Retourne RMSE, R², et le modèle entraîné.  
# 

# In[ ]:


def train_and_evaluate_transformer(X_train, y_train, X_test, y_test, epochs=150):
    # Normalize targets to improve training stability
    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    # Prepare datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_scaled, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test_scaled, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model initialization
    input_dim = X_train.shape[2]
    model = AdvancedTransformerModel(input_dim=input_dim, seq_length=X_train.shape[1])
    model.to(device)

    # Optimizer with lower learning rate and adaptive optimization
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4, amsgrad=True)

    # Learning rate scheduler with cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Huber Loss for more robust regression
    criterion = nn.SmoothL1Loss()

    # Training loop with early stopping
    best_val_loss = float('inf')
    patience = 15
    trigger_times = 0

    # Training and validation
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_true = [], []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch).squeeze()
                val_loss += criterion(predictions, y_batch).item()

                val_preds.extend(predictions.cpu().numpy())
                val_true.extend(y_batch.cpu().numpy())

        val_loss /= len(test_loader)
        scheduler.step()

        # Inverse transform predictions and true values
        val_preds_original = target_scaler.inverse_transform(np.array(val_preds).reshape(-1, 1)).flatten()
        val_true_original = target_scaler.inverse_transform(np.array(val_true).reshape(-1, 1)).flatten()

        # Calculate R²
        r2 = r2_score(val_true_original, val_preds_original)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), 'best_transformer_model.pth')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping!")
                break

    # Load best model
    model.load_state_dict(torch.load('best_transformer_model.pth'))

    # Final evaluation
    model.eval()
    final_preds, final_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch).squeeze()
            final_preds.extend(predictions.cpu().numpy())
            final_true.extend(y_batch.cpu().numpy())

    # Inverse transform final predictions and true values
    final_preds_original = target_scaler.inverse_transform(np.array(final_preds).reshape(-1, 1)).flatten()
    final_true_original = target_scaler.inverse_transform(np.array(final_true).reshape(-1, 1)).flatten()

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(final_true_original, final_preds_original))
    r2 = r2_score(final_true_original, final_preds_original)

    # Visualization
    plt.figure(figsize=(12,6))
    plt.plot(final_true_original, label='Valeurs Réelles', color='blue')
    plt.plot(final_preds_original, label='Prédictions', color='red', linestyle='--')
    plt.title('Prédictions du Modèle Transformer')
    plt.xlabel('Échantillons')
    plt.ylabel('Valeur')
    plt.legend()
    plt.show()

    print(f"RMSE Final: {rmse}")
    print(f"R² Final: {r2}")

    return rmse, r2, model


# 1. Cette fonction prépare des données séquentielles pour des modèles basés sur des séries temporelles.  
# 2. Applique une normalisation Min-Max sur les colonnes de features et la cible.  
# 3. Crée des séquences de longueur spécifiée (`sequence_length`) à partir des features.  
# 4. Associe chaque séquence à la valeur cible correspondante (valeur suivante après la séquence).  
# 5. Convertit les séquences et les cibles en tableaux NumPy.  
# 6. Retourne les séquences, les cibles, et les scalers pour les features et la cible.  
# 

# In[ ]:


def prepare_sequential_data(df, features_columns, target_column, sequence_length=10):

    # Feature Scaling
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # Scale features
    features_scaled = feature_scaler.fit_transform(df[features_columns])

    # Scale target
    target_scaled = target_scaler.fit_transform(df[[target_column]])

    # Prepare sequences
    X, y = [], []
    for i in range(len(df) - sequence_length):
        # Create sequence of features
        X.append(features_scaled[i:i+sequence_length])
        # Target is the next value after the sequence
        y.append(target_scaled[i+sequence_length][0])

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    return X, y, feature_scaler, target_scaler


# 1. Sélection des colonnes de features (prix et volumes pour Apple, Tesla, Microsoft) et de la cible (`VaR_S&P_500`).  
# 2. Préparation des données séquentielles avec normalisation et création de séquences temporelles de longueur 10.  
# 3. Division des données en ensembles d'entraînement (80%) et de test (20%) avec `train_test_split`.  
# 4. Entraînement d'un modèle Transformer avancé avec la fonction `train_and_evaluate_transformer`.  
# 5. Évaluation des performances du modèle avec les métriques RMSE et R², et visualisation des prédictions.  
# 

# In[ ]:


features_columns = ['Apple_Price', 'Apple_Vol.', 'Tesla_Price', 'Tesla_Vol.', 'Microsoft_Price', 'Microsoft_Vol.']
target_column = 'VaR_S&P_500'

# Prepare sequential data
X, y, feature_scaler, target_scaler = prepare_sequential_data(df, features_columns, target_column, sequence_length=10)

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rmse, r2_tr, model = train_and_evaluate_transformer(X_train, y_train, X_test, y_test)


# ## Comparison between the 3 models

# In[ ]:


modeles = ['LSTM', 'CNN-LSTM', 'Transformer']
r2_scores = [r2_LSTM, r2_CNN_LSTM, r2_tr]  # Utilisez vos valeurs réelles de R²

plt.figure(figsize=(10, 6))
bars = plt.bar(modeles, r2_scores)
plt.title('Comparison of R² scores between LSTM, CNN-LSTM and Transformer', fontsize=15)
plt.ylabel('Score R²', fontsize=12)
plt.xlabel('Models', fontsize=12)

# Ajout des valeurs sur chaque barre
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontsize=10)

# Ajout d'une ligne à y=0 pour référence
plt.axhline(y=0, color='r', linestyle='--')

plt.tight_layout()
plt.show()


# In[ ]:




