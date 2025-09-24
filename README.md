# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Necessary Libraries and Load Data.

2.Split Dataset into Training and Testing Sets.

3.Train the Model Using Stochastic Gradient Descent (SGD).

4.Make Predictions and Evaluate Accuracy.

5.Generate Confusion Matrix.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Pelleti Sindhu Sri
RegisterNumber: 212224240113
*/


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


iris = load_iris()
X = iris.data
y = iris.target
print("Feature Names:", iris.feature_names)
print("Target Names:", iris.target_names)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, random_state=42)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))


sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d',
            xticklabels=iris.target_names, yticklabels=iris.target_names, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SGD Classifier")
plt.show()


new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  
new_sample_scaled = scaler.transform(new_sample)
prediction = model.predict(new_sample_scaled)
print("Predicted Class for New Sample:", iris.target_names[prediction[0]])

```

## Output:

<img width="961" height="444" alt="Screenshot 2025-09-24 222007" src="https://github.com/user-attachments/assets/b05d9f28-c3a4-445e-bad7-e8ca677d29a7" />

<img width="685" height="588" alt="Screenshot 2025-09-24 222016" src="https://github.com/user-attachments/assets/3a6b65f4-567e-463d-b088-37605f152718" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
