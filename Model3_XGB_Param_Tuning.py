import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor

# Read the prepared dataset
data = pd.read_csv('Datasets/Cleaned_prepared_data.csv')

# Selecting the features and targets
features = data[['latitude', 'longitude', 'baro_altitude', 'ground_speed', 'track', 'vertical_rate', 'Climbing', 'Descending', 'Cruise']]
targets = data[['latitude_in_10min', 'longitude_in_10min', 'baro_altitude_in_10min']]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.01, random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [280, 300, 320],
    'learning_rate': [0.04, 0.05, 0.06],
    'max_depth': [6, 7, 8],
    'subsample': [0.4, 0.6, 0.8],
    'colsample_bytree': [0.95, 1.0, 1.05]
}

# Creating the Grid Search
model = XGBRegressor(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=10, scoring='neg_mean_absolute_error')

# Performing hyperparameter tuning
grid_search.fit(X_train, y_train)

# return best parameters
print("Best parameters:", grid_search.best_params_)

# Using the model with the best parameters found for further predictions or analysis
best_model = grid_search.best_estimator_
print(best_model)