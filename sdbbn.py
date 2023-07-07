import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import plotly.graph_objects as go
df = pd.read_csv(r"C:\Users\khanm\Mini Project\clustered_customers.csv")
df.head()
fig = px.scatter(df, x='Age', y='Spending Score (1-100)', color='Cluster',
                 title='Scatter Plot of Age and Spending Score')
fig.show()
fig = px.scatter(df, x='Age', y='Annual Income (k$)', color='Cluster',
                 title='Scatter Plot of Age and Annual Income')
fig.show()
fig = px.scatter(df, x='Annual Income (k$)', y='Spending Score (1-100)', color='Cluster',
                 title='Scatter Plot of Annual Income and Spending Score')
fig.show()

features = df[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = MinMaxScaler()
data = scaler.fit_transform(features)
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
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(max_iter), y=q_error, mode='lines', name='Quantization error'))
fig.add_trace(go.Scatter(x=np.arange(max_iter), y=t_error, mode='lines', name='Topographic error'))
fig.update_layout(
    title='Quantization and Topographic Error',
    xaxis_title='Iteration index',
    yaxis_title='Error',
    hovermode='x'
)
fig.show()
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
winner_coordinates = np.array([som.winner(x) for x in data])
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
cluster_labels = cluster_index + 1
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