from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import pandas as pd

# Load data
data = pd.read_csv('../lotto_numbers.csv')
number_columns = ['number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6']

# Split the data into input features (X) and target variable (y)
X = data[number_columns]
y = data['super_number']
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create model
def create_keras_model():
    model = Sequential()
    model.add(Dense(64, input_dim=6, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# Use KerasRegressor to wrap the model and fit it to the training data
keras_model = KerasRegressor(build_fn=create_keras_model, epochs=100, batch_size=16, verbose=0)
keras_model.fit(X_train, y_train)

# Make predictions and calculate the mean squared error
keras_predictions = keras_model.predict(X_test)
keras_mse = mean_squared_error(y_test, keras_predictions)
print("KerasRegressor Mean Squared Error:", keras_mse)
