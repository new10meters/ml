import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier, export_text  
from sklearn.metrics import accuracy_score  

# ----------- Load Data from CSV File ------------  
# Read the dataset from a CSV file  
# Adjust the file path as necessary  
data = pd.read_csv('breast_cancer_data.csv')  

# Assuming 'target' is the name of the last column  
X = data.drop(columns='target')  
y = data['target']  

# ----------- Split Data into Training and Testing Sets ------------  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# ----------- Train the Decision Tree Classifier ------------  
# Create a DecisionTreeClassifier model  
dt_classifier = DecisionTreeClassifier(random_state=42)  

# Train the model  
dt_classifier.fit(X_train, y_train)  

# ----------- Evaluate the Model ------------  
# Predict the test set results  
y_pred = dt_classifier.predict(X_test)  

# Calculate accuracy  
accuracy = accuracy_score(y_test, y_pred)  
print(f"Accuracy of Decision Tree Classifier: {accuracy:.2f}")  

# ----------- Visualize Decision Tree (Optional) ------------  
# Display decision tree rules  
print("\nDecision Tree Rules:")  
tree_rules = export_text(dt_classifier, feature_names=list(X.columns))  
print(tree_rules)  
 
# ----------- Classify a New Sample ------------  
# Example of a new sample (modify this sample as needed)  
new_sample = np.array([  
    14.6, 21.7, 94.7, 577.6, 0.102, 0.125, 0.078,   
    0.057, 0.159, 0.06, 0.02, 0.039, 0.029, 0.046,  
    0.063, 0.07, 0.113, 0.144, 0.24, 0.104, 0.111,  
    0.051, 0.002, 0.004, 0.002, 0.007, 0.015, 0.014,  
    0.027, 0.023  
]).reshape(1, -1)  

# Convert the new sample to a DataFrame with feature names  
new_sample_df = pd.DataFrame(new_sample, columns=X.columns)  

# Predict the class for the new sample  
prediction = dt_classifier.predict(new_sample_df)  

# Print the prediction  
class_label = 'Malignant' if prediction[0] == 1 else 'Benign'  # Update according to the dataset  
print(f"\nThe predicted class for the new sample is: {class_label}")  