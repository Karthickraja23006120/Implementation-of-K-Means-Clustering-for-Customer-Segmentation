# Ex-7: Implementation-of-K-Means-Clustering-for-Customer-Segmentation
## DATE:
## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import libraries: Use libraries such as pandas, sklearn, matplotlib, and seaborn.
2. Load and preprocess the data: Load your customer data and scale it for better performance of K-Means.
3. Elbow method: Use the elbow method to determine the optimal number of clusters (K).
4. Fit K-Means: Apply the K-Means algorithm with the chosen number of clusters.
5. Visualize the clusters: Plot the customer segments on a 2D plot for easy interpretation.
6. Interpret the results: Analyze the customer segments to understand their characteristics.
7. Predict clusters for new customers: Use the trained K-Means model to predict the cluster for new data.
## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Karthick Raja K
RegisterNumber: 212223240066
*/
```
```Python
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Mall_Customers.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.cluster import KMeans
wcss=[]   #Within-Cluster Sum of Squares.

for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init="k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.xlabel("No.of Clusters")
plt.ylabel("wcss")
plt.title("ELBOW METHOD GRAPH")
plt.show()
km = KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])
y_pred=km.predict(data.iloc[:,3:])
y_pred
data["cluster"]=y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]

plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="green",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="purple",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="blue",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="gold",label="cluster4")
plt.legend()
plt.title("Customer Segments")
```

## Output:
![image](https://github.com/RahulM2005R/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/166299886/6bfe56ab-cb96-410a-9db4-f55e7e43505e)
![image](https://github.com/RahulM2005R/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/166299886/26e99a9b-155e-48f9-a406-202c4f0c1424)
![image](https://github.com/RahulM2005R/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/166299886/04b7dfd0-ff27-41d3-90db-dad7be88b8ad)
![image](https://github.com/RahulM2005R/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/166299886/aa6e54de-57c1-4904-b819-efdde42f450a)
![image](https://github.com/RahulM2005R/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/166299886/6895d451-b3ff-480c-bac8-474cfde2a5a1)
![image](https://github.com/RahulM2005R/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/166299886/a6c8e7be-831b-419c-a85b-adf385d039c8)
![image](https://github.com/RahulM2005R/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/166299886/d8a90b81-5169-46f1-8efe-83d016abd023)




## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
