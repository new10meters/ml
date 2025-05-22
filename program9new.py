import numpy as np  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.naive_bayes import GaussianNB  
from sklearn.metrics import accuracy_score  
from sklearn.decomposition import PCA  
import matplotlib.pyplot as plt  

# ----------- Load Data from CSV File ------------  

data = pd.read_csv('olivetti_faces.csv')  

# Separate features and labels  
X = data.iloc[:, :-1].values  # All columns except the last one as features  
y = data.iloc[:, -1].values    # Last column as labels  

# ----------- Optional: Reduce Dimensions using PCA ------------  
pca = PCA(n_components=50)  # Reduce to 50 components  
X_pca = pca.fit_transform(X)  

# ----------- Split Data into Training and Testing Sets ------------  
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)  

# ----------- Train the Naive Bayes Classifier ------------  
nb_classifier = GaussianNB()  
nb_classifier.fit(X_train, y_train)  

# ----------- Evaluate the Model ------------  
y_pred = nb_classifier.predict(X_test)  
accuracy = accuracy_score(y_test, y_pred)  

print(f"Accuracy of Naive Bayes Classifier: {accuracy:.2f}")  

# ----------- Predict New Sample(s) ------------  
sample_indices = [0, 5, 10]  # Sample indices to test on  
for idx in sample_indices:  
    print(f"\nPredicted Label for Sample {idx} : {nb_classifier.predict([X_test[idx]])[0]}  and its true labels is - {y_test[idx]} ")  
    
    # Since PCA has reduced dimensionality, we can't visualize the PCA images directly  
    # We need the original images, which we might need to keep before PCA  
    original_image = X[idx].reshape(64, 64)  # Reshape the original (non-PCA) image data  
    plt.imshow(original_image, cmap='gray')  # Plot the original image  
    plt.title(f"True Label: {y_test[idx]}, Predicted Label: {nb_classifier.predict([X_test[idx]])[0]}")  
    plt.show() 