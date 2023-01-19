import numpy as np 
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


X, y = datasets.fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)

kf = KFold(n_splits=10, random_state=1)
for train_ix, test_ix in kf.split(X):
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]

myLinearModel = LinearRegression().fit(X_train, y_train)
y_hat = myLinearModel.predict(X_test)

def MSE(targets, predicted):
    return np.mean(np.square(targets - predicted))

#print("MSE (Python):", MSE(y_test, y_hat))
#print("MSE (scikit-learn):", mean_squared_error(y_test, y_hat))