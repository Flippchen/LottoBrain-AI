from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_csv('../lotto_numbers.csv')

number_columns = ['number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6']
number_data = data[number_columns]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(number_data)

k = 10  # You can experiment with different values for the number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(scaled_data)


def predict_cluster(kmeans, scaler, numbers):
    scaled_numbers = scaler.transform([numbers])
    cluster = kmeans.predict(scaled_numbers)
    return cluster[0]


new_numbers = [3, 12, 18, 30, 32, 49]  # Replace with your own numbers
predicted_cluster = predict_cluster(kmeans, scaler, new_numbers)
print(f"The predicted cluster for the numbers {new_numbers} is {predicted_cluster}.")


