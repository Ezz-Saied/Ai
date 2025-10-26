import numpy as np
import pandas as pd

data = pd.read_csv('data.csv', header=None)
data.columns = ['X1', 'X2', 'X3', 'y']

X = data[['X1', 'X2', 'X3']].values
y = data[['y']].values

X_mean, X_std = X.mean(axis=0), X.std(axis=0)
y_mean, y_std = y.mean(), y.std()
X = (X - X_mean) / X_std
y = (y - y_mean) / y_std

X_b = np.c_[np.ones((X.shape[0], 1)), X]

theta = np.random.randn(X_b.shape[1], 1)

learning_rate = 1e-5
n_epochs = 100
m = len(X_b)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        theta -= learning_rate * gradients

print("Learned Parameters (theta):", theta.ravel())

x_new = np.array([[80, 75, 152]])  
x_new_scaled = (x_new - X_mean) / X_std
x_new_b = np.c_[np.ones((1, 1)), x_new_scaled]
y_pred_scaled = x_new_b.dot(theta)
y_pred = (y_pred_scaled * y_std) + y_mean
print("Example Prediction:", y_pred[0][0])

