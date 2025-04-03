import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 🔢 Generare date
values = np.round(np.random.uniform(-10, 10, size=100), 3)
data_in = np.array(values)
noise = np.round(np.random.uniform(-1, 1, size=100), 3)

data_out = 4 * data_in ** 2 + 2 * data_in + 7 + noise

sorted_indices = np.argsort(data_in)
data_in_sorted = data_in[sorted_indices]
data_out_sorted = data_out[sorted_indices]

fig, ax = plt.subplots()
fig.canvas.manager.set_window_title("Polynomial regression")
ax.set_title('Data for polynomial regression model')
ax.scatter(data_in_sorted, data_out_sorted, color="green", label='Valori reale')

data_in_train, data_in_test, data_out_train, data_out_test = train_test_split(
    data_in, data_out, test_size=0.3, random_state=42
)

data_in_train = data_in_train.reshape(-1, 1)
data_in_test = data_in_test.reshape(-1, 1)

poly = PolynomialFeatures(degree=2)
data_in_train_poly = poly.fit_transform(data_in_train)
data_in_test_poly = poly.transform(data_in_test)

model = LinearRegression()
model.fit(data_in_train_poly, data_out_train)

data_out_predict = model.predict(data_in_test_poly)
sorted_test_indices = np.argsort(data_in_test.flatten())
data_in_test_sorted = data_in_test[sorted_test_indices]
data_out_predict_sorted = data_out_predict[sorted_test_indices]

ax.plot(data_in_test_sorted, data_out_predict_sorted, color="red", label='Predicții model')

ax.legend()

mse = mean_squared_error(data_out_test, data_out_predict)
print("MSE:", mse)
print("w (coef_):", model.coef_)
print("b (intercept_):", model.intercept_)

plt.show()
