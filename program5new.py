import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.model_selection import train_test_split   
from sklearn.neighbors import KNeighborsClassifier   
from sklearn.metrics import accuracy_score  

# Generate random values  
values = np.random.rand(100)  

print("\nValues:")  
print(values)  

# Generate labels based on conditions  
labels = []  
for i in values[:50]:   
    if i <= 0.5:  
        labels.append('Class1')  
    else:  
        labels.append('Class2')  

labels += [None] * 50  # Add None for the unlabeled data points  
print("\nlabels:")  
print(labels)  

# Create DataFrame  
data = {  
    "Point": [f"x{i+1}" for i in range(100)],   
    "Value": values,  
    "Label": labels  
}  
print("\n Data Frame:")  
df = pd.DataFrame(data)  
print(df)  

# Plot histograms for numerical columns  
num_col = df.select_dtypes(include=['int', 'float']).columns  
for col in num_col:  
    df[col].hist(figsize=(12, 8), bins=10, edgecolor='black')  # Change to df[col]  
    plt.title(f"Histogram for {col}", fontsize=16)  # Use f-string for title  
    plt.xlabel(col)  
    plt.ylabel("Frequency")  
    plt.show()  

# Split data into labeled and unlabeled  
labeled_df = df[df["Label"].notna()]  
X_train = labeled_df[["Value"]]   
y_train = labeled_df["Label"]  

unlabeled_df = df[df["Label"].isna()].copy()  # Create a copy to avoid warning  
X_test = unlabeled_df[["Value"]]  

# Generate true labels for testing accuracy  
true_labels = ["Class1" if x <= 0.5 else "Class2"
               for x in values[50:]]  

# Train KNN and store predictions  
k_values = [1, 2, 3, 4, 5, 20, 30]   
results = {}  
accuracies = {}  

for k in k_values:  
    knn = KNeighborsClassifier(n_neighbors=k)  
    knn.fit(X_train, y_train)  
    predictions = knn.predict(X_test)  
    results[k] = predictions  

    # Calculate accuracy and store results  
    accuracy = accuracy_score(true_labels, predictions) * 100   
    accuracies[k] = accuracy  
    print(f"Accuracy for k={k}: {accuracy:.2f}%")  

    # Assign predictions back to the DataFrame for current k  
    unlabeled_df[f"Label_k{k}"] = predictions  

# Drop the 'Label' column and create df1  
df1 = unlabeled_df.drop(columns=['Label'], axis=1)   
print("\n DataFrame with 'Label':")  
print(df1)