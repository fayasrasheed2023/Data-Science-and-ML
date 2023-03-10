import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv("iris.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
print(X)
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
