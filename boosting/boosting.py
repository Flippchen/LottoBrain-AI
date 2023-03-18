import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
import numpy as np

# Predict super number with XGBoost
# Load data
data = pd.read_csv('../lotto_numbers.csv')

# Split the data into input features (X) and target variable (y)
number_columns = ['number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6']
X = data[number_columns]
y = data['super_number']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model and fit
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions
xgb_predictions = xgb_model.predict(X_test)
xgb_mse = mean_squared_error(y_test, xgb_predictions)
print("XGBoost Mean Squared Error:", xgb_mse)


# Predict all numbers with XGBoost
X = data[['date']]
y = data.drop(columns=['date'])

X['year'] = X['date'].str[:4].astype(int)
X['month'] = X['date'].str[5:7].astype(int)
X['day'] = X['date'].str[8:10].astype(int)
X = X.drop(columns=['date'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = MultiOutputRegressor(xgb.XGBRegressor(random_state=42))
xgb_model.fit(X_train, y_train)

xgb_predictions = xgb_model.predict(X_test)
xgb_mse = mean_squared_error(y_test, xgb_predictions)
print("XGBoost Mean Squared Error:", xgb_mse)

xgb_mse = mean_squared_error(y_test, xgb_predictions, multioutput='raw_values')
print("XGBoost Mean Squared Error for each number:", xgb_mse)

xgb_predictions_rounded = np.round(xgb_predictions).astype(int)
print("XGBoost Predicted Numbers:\n", xgb_predictions_rounded)