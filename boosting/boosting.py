import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
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