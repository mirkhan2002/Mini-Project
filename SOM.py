import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from minisom import MiniSom  
df = pd.read_csv(r'C:\Users\khanm\Downloads\Mall_Customers.csv')
df.head()
plt.style.use('fivethirtyeight')
age = df['Age'].tolist()
spending_score = df['Spending Score (1-100)'].tolist()
fig = plt.figure(figsize=(6,6))
plt.scatter(age, spending_score)
plt.suptitle("Scatter Plot of Age and Spending Score")
plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.show()
age = df['Age'].tolist()
annual_income = df['Annual Income (k$)'].tolist()
fig = plt.figure(figsize=(6,6))
plt.scatter(age, annual_income)
plt.suptitle("Scatter Plot of Age and Annual Income")
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
plt.show()
annual_income = df['Annual Income (k$)'].tolist()
spending_score = df['Spending Score (1-100)'].tolist()
fig = plt.figure(figsize=(6,6))
plt.scatter(annual_income, spending_score)
plt.suptitle("Scatter Plot of Annual Income & Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()
df.isnull().sum()
features = df[['Annual Income (k$)', 'Spending Score (1-100)']]
data = features.values
data.shape
som_shape = (1, 5)
som = MiniSom(som_shape[0], som_shape[1], data.shape[1], sigma=0.5, learning_rate=0.5)
max_iter = 1000
q_error = []
t_error = []
for i in range(max_iter):
    rand_i = np.random.randint(len(data))
    som.update(data[rand_i], som.winner(data[rand_i]), i, max_iter)
    q_error.append(som.quantization_error(data))
    t_error.append(som.topographic_error(data))
plt.plot(np.arange(max_iter), q_error, label='quantization error')
plt.plot(np.arange(max_iter), t_error, label='topographic error')
plt.ylabel('Quantization error')
plt.xlabel('Iteration index')
plt.legend()
plt.show()
winner_coordinates = np.array([som.winner(x) for x in data]).T
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
plt.figure(figsize=(10,8))
for c in np.unique(cluster_index):
    plt.scatter(data[cluster_index == c, 0],
                data[cluster_index == c, 1], label='cluster='+str(c), alpha=.7)
for centroid in som.get_weights():
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', 
                s=10, linewidths=20, color='k')
plt.title("Clusters of Customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
num_clusters = som_shape[1]
cluster_labels = cluster_index
df['Cluster'] = cluster_labels
num_rows = int(np.ceil(num_clusters / 2))
num_cols = min(2, num_clusters)
plt.figure(figsize=(10, 6))
for cluster in range(num_clusters):
    plt.subplot(num_rows, num_cols, cluster + 1)
    plt.hist(df['Annual Income (k$)'][df['Cluster'] == cluster], bins=10, alpha=0.7)
    plt.title('Cluster ' + str(cluster))
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
winner_coordinates = np.array([som.winner(x) for x in data]).T
cluster_labels = np.ravel_multi_index(winner_coordinates, som_shape)
df['Cluster'] = cluster_labels
df.to_csv('clustered_customers.csv', index=False)
cluster_counts = df['Cluster'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
plt.bar(cluster_counts.index, cluster_counts.values)
plt.title('Cluster Occurrences')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.xticks(cluster_counts.index)
plt.show()