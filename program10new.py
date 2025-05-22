 # ----------- Import required libraries ------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# ----------- Load the Wisconsin Breast Cancer Dataset ------------
# Load the dataset
data = pd.read_csv('breast_cancer_data.csv')  # Correct path to uploaded file

# Assume 'target' is the label column
X = data.drop(columns=['target'])  # Features: all except 'target'
y = data['target']  # Target column

# ----------- Preprocess the Data ------------
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------- Apply K-means Clustering ------------
# Apply KMeans clustering with 2 clusters (malignant and benign)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# ----------- Get Clustering Results ------------
y_kmeans = kmeans.predict(X_scaled)

# ----------- Visualize the Clustering Results ------------
# Reduce dimensionality using PCA to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Since cluster_centers_ are still in original feature space, we also need to transform them
cluster_centers_pca = pca.transform(kmeans.cluster_centers_)

# Plotting the clusters in 2D
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', label='Clustered Data',alpha=0.6)
plt.scatter(cluster_centers_pca[:, 0], cluster_centers_pca[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.title("K-means Clustering of Wisconsin Breast Cancer Data")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()

# ----------- Evaluate the Clustering Result ------------
# Because KMeans clusters are arbitrarily labeled, need to align them with true labels
# Find out the correct mapping using confusion matrix
cm = confusion_matrix(y, y_kmeans)
print("Confusion Matrix:\n", cm)

# Choose the best mapping (this assumes two clusters only)
if cm[0, 0] + cm[1, 1] < cm[0, 1] + cm[1, 0]:
    # If wrong mapping, invert labels
    y_kmeans = 1 - y_kmeans

# Now calculate accuracy
print(f"Accuracy of K-means clustering: {accuracy_score(y, y_kmeans):.2f}")
