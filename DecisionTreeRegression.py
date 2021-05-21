import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# DATA
dataset = pd.read_csv("data.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# SCALING
""" from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y.reshape(-1,1)).ravel() """

# FITTING DECISION TREE TO DATASET - " CREATE REGRESSOR "
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0) # use default params for this exercise but specify random_state
regressor.fit(x,y)

# VISUALIZE ---
# you have to smooth
x_opt = np.arange(min(x), max(x), 0.01)
x_opt = x_opt.reshape(len(x_opt), 1)
# then graph
plt.scatter(x, y, color = "red")
plt.plot(x_opt, regressor.predict(x_opt), color = "blue")
plt.show()

# PREDICT SPECIFIC VALUE
prediction = regressor.predict([[6.5]])
print(prediction[0])
