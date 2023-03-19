from deepNN.deep_nn import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('../lotto_numbers.csv')

# Predict all numbers with Stacking
X = data[['date']]
y = data.drop(columns=['date'])

# Create numerical features from date
X['year'] = X['date'].str[:4].astype(int)
X['month'] = X['date'].str[5:7].astype(int)
X['day'] = X['date'].str[8:10].astype(int)
X = X.drop(columns=['date'])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost model
xgb_model = MultiOutputRegressor(xgb.XGBRegressor(random_state=42))
xgb_model.fit(X_train, y_train)

# KerasRegressor model
keras_model = KerasRegressor(build_fn=create_keras_model2, epochs=100, batch_size=16, verbose=0)
keras_model.fit(X_train, y_train)

# RandomForestRegressor model
rf_model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
rf_model.fit(X_train, y_train)

# Make predictions with each model
xgb_predictions_train = xgb_model.predict(X_train)
keras_predictions_train = keras_model.predict(X_train)
rf_predictions_train = rf_model.predict(X_train)

# Combine predictions into a single array
stacked_predictions_train = np.column_stack((xgb_predictions_train, keras_predictions_train, rf_predictions_train))

# Fit a model on the combined predictions
meta_model = LinearRegression()
meta_model.fit(stacked_predictions_train, y_train)

# Make predictions on test data
xgb_predictions_test = xgb_model.predict(X_test)
keras_predictions_test = keras_model.predict(X_test)
rf_predictions_test = rf_model.predict(X_test)

# Combine predictions into a single array
stacked_predictions_test = np.column_stack((xgb_predictions_test, keras_predictions_test, rf_predictions_test))
final_predictions = meta_model.predict(stacked_predictions_test)

# Evaluate the model
mse = mean_squared_error(y_test, final_predictions, multioutput='raw_values')
print("Stacking Model Mean Squared Error for each number:", mse)

# Print predictions
final_predictions_rounded = np.round(final_predictions).astype(int)
print("Stacking Predicted Numbers:\n", final_predictions_rounded)