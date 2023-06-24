import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r'C:\Users\khanm\Downloads\Mall_Customers.csv')
df.head()
plt.figure(1 , figsize = (15 , 6)) # sets the dimensions of image
n = 0 
plt.figure(1 , figsize = (15 , 5))
sns.countplot(y = 'Gender' , data = df)
plt.show()