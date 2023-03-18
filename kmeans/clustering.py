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


new_numbers = [3, 12, 18, 30, 32, 49]
predicted_cluster = predict_cluster(kmeans, scaler, new_numbers)
print(f"The predicted cluster for the numbers {new_numbers} is {predicted_cluster}.")


pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)


def plot_clusters(reduced_data, kmeans):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=kmeans.labels_, palette='viridis', s=50, edgecolor='k', alpha=0.7)
    sns.scatterplot(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1], s=200, color='red', marker='x', linewidth=2, label='Cluster Centers')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('K-means Clustering of Lottery Numbers')
    plt.legend()
    fig = plt.gcf()
    plt.show()
    fig.savefig('images/lotto_clusters.png')


plot_clusters(reduced_data, kmeans)