import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans

df = pd.read_csv(r'C:\Users\khanm\Downloads\Mall_Customers.csv')

X = df[['Age', 'Spending Score (1-100)']].values

model = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=111, algorithm='elkan')
model.fit(X)

labels = model.labels_ + 1
centroids = model.cluster_centers_

df['Cluster'] = labels

fig = go.Figure()

for cluster_num in range(1, 5):
    cluster_data = df[df['Cluster'] == cluster_num]
    fig.add_trace(go.Scatter(
        x=cluster_data['Age'],
        y=cluster_data['Spending Score (1-100)'],
        mode='markers',
        marker=dict(
            size=10,
            color=cluster_num,
            line=dict(width=1, color='black')
        ),
        name=f'Cluster {cluster_num}',
        hovertemplate='Age: %{x}<br>Spending Score: %{y}'
    ))

fig.add_trace(go.Scatter(
    x=centroids[:, 0],
    y=centroids[:, 1],
    mode='markers',
    marker=dict(
        size=12,
        color='red',
        line=dict(width=1, color='black')
    ),
    name='Centroids',
    hovertemplate='Centroid<br>Age: %{x}<br>Spending Score: %{y}'
))

fig.update_layout(
    title='Interactive K-Means Clustering',
    xaxis_title='Age',
    yaxis_title='Spending Score (1-100)',
    showlegend=True,
    hovermode='closest'
)

fig.show()
