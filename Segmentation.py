import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
df = pd.read_csv(r'C:\Users\khanm\Downloads\Mall_Customers.csv')
df.head()
plt.figure(1 , figsize = (15 , 6)) # sets the dimensions of image
n = 0 
X_age_spending = df[['Age' , 'Spending Score (1-100)']].iloc[: , :].values # extracts only age and spending score information from the dataframe
inertia = []
for n in range(1 , 11):
    model_1 = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 , max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan')) # use predefined Kmeans algorithm
    model_1.fit(X_age_spending) # fit the data into the model
    inertia.append(model_1.inertia_)
    plt.figure(1 , figsize = (15 ,6)) # set dimension of image
plt.plot(np.arange(1 , 11) , inertia , 'o') # Mark the points with a solid circle
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5) # connect remaining points with a line
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia') # label the x and y axes
plt.show() # display
