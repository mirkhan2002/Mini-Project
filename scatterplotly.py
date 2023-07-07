import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
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
