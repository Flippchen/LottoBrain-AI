import pandas as pd
from sklearn.linear_model import LinearRegression

#Load the data from csv file
data = pd.read_csv('lotto_numbers.csv')

# Split the data into input features (X) and target variable (y)
X = data.drop(['date', 'super_number'], axis=1)
y = data[['number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6', 'super_number']]

# Train a linear regression model on the data
model = LinearRegression()
model.fit(X, y)

# Make a prediction for the next lotto draw
last_draw = X.iloc[-1]
#next_draw = model.predict(last_draw.values.reshape(1, -1))

number_to_predict = [[1, 2, 3, 4, 5, 6]]
next_draw = model.predict(number_to_predict)

# Print the predicted next lotto numbers
print('Predicted next lotto numbers:', [int(num) for num in next_draw[0]])
