# -------------------------------------------------------------------------
# AUTHOR: Daeyoung Hwang
# FILENAME: clustering.py
# SPECIFICATION: find the k with the highets silhouette score and find the homogeneity score for
# k-means and agglomerative clustering
# FOR: CS 4210- Assignment #5
# TIME SPENT: 1 hr
# -----------------------------------------------------------*/

# importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None)  # reading the data by using Pandas library

# assign your training data to X_training feature matrix
X_training = np.array(df.values)

silhouette_scores = []
# run kmeans testing different k values from 2 until 20 clusters
for k in range(2, 21):
    # Use:  kmeans = KMeans(n_clusters=k, random_state=0)
    #      kmeans.fit(X_training)
    # --> add your Python code
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_training)

    # for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
    # find which k maximizes the silhouette_coefficient
    # --> add your Python code here
    silhouette_scores.append(silhouette_score(X_training, kmeans.labels_))

# plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
# --> add your Python code here
k_list = [i for i in range(2, 21)]

plt.plot(k_list, silhouette_scores)
plt.show()
best_k = silhouette_scores.index(max(silhouette_scores)) + 2
print('best k:', best_k)


# reading the validation data (clusters) by using Pandas library
# --> add your Python code here

val_df = pd.read_csv('testing_data.csv', header=None)

# assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
# --> add your Python code here
labels = np.array(val_df.values).reshape(1, -1)[0]
# Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
# --> add your Python code here
# rung agglomerative clustering now by using the best value o k calculated before by kmeans
# Do it:
agg = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
agg.fit(X_training)

# Calculate and print the Homogeneity of this agglomerative clustering
print("Agglomerative Clustering Homogeneity Score = " + metrics.homogeneity_score(labels, agg.labels_).__str__())
