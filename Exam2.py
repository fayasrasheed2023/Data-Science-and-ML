import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv("customer_data.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
print(X)
print(y)
data.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)
y_pred = regression.predict(X_test)


""""
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
"""