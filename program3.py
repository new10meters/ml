import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset from the external CSV file
file_path = "iris_data.csv"  # Ensure this file is in the correct location
df = pd.read_csv(file_path)
# Step 2: Standardize the features (excluding the 'Species' column)
features = ['sepal length (cm)', 'sepal width (cm)', 
            'petal length (cm)', 'petal width (cm)']
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

print("\nCovariance Matrix:\n")
cov_matrix = np.cov(df_scaled.T)
print(cov_matrix)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\nEigenvalues:", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)

# Step 3: Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled)

# Step 4: Create a DataFrame with the PCA components and add the species column
df_pca = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
df_pca['Species'] = df['Species']

# Step 5: Visualize the 2D PCA result
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Species',
                data=df_pca, palette='Set1'  , s=100, marker='o')
plt.title('PCA of Iris Dataset (4 features to 2)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Species')
plt.show()
