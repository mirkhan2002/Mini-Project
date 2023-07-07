import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv(r'C:\Users\khanm\Downloads\Mall_Customers.csv')

X_age_spending = df[['Age', 'Spending Score (1-100)']].values

inertia = []
for n in range(1, 11):
    model = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=111, algorithm='lloyd')
    model.fit(X_age_spending)
    inertia.append(model.inertia_)

plt.figure(1, figsize=(15, 6))
plt.plot(np.arange(1, 11), inertia, 'o')
plt.plot(np.arange(1, 11), inertia, '-', alpha=0.5)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()
