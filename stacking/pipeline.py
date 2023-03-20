from deepNN.deep_nn import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class KerasTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        self.model.fit(X, y)
        return self

    def transform(self, X, y=None):
        return self.model.predict(X)


class MultiOutputTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        self.model.fit(X, y)
        return self

    def transform(self, X, y=None):
        return self.model.predict(X)


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

# Create the base models
xgb_base = MultiOutputTransformer(xgb.XGBRegressor(random_state=42))
keras_base = KerasTransformer(KerasRegressor(build_fn=create_keras_model2, epochs=100, batch_size=16, verbose=0))
rf_base = MultiOutputTransformer(RandomForestRegressor(random_state=42))

# Create the base models transformer with the base models
base_models_transformer = ColumnTransformer(
    transformers=[
        ('xgb', xgb_base, list(range(3))),
        ('keras', keras_base, list(range(3))),
        ('rf', rf_base, list(range(3)))
    ],
    remainder='drop'
)
# create the stacking pipeline
stacking_pipeline = Pipeline([
    ('base_models', base_models_transformer),
    ('meta_model', LinearRegression())
])

# Fit the stacking pipeline
stacking_pipeline.fit(X_train, y_train)

# Evaluate the model
final_predictions = stacking_pipeline.predict(X_test)
mse = mean_squared_error(y_test, final_predictions, multioutput='raw_values')
print("Stacking Model Mean Squared Error for each number:", mse)

# Print predictions
final_predictions_rounded = np.round(final_predictions).astype(int)
print("Stacking Predicted Numbers:\n", final_predictions_rounded)
