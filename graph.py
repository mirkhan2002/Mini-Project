import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
df = pd.read_csv(r'C:\Users\khanm\Downloads\Mall_Customers.csv')
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