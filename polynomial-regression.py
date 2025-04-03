import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


values = np.round(np.random.uniform(-10, 10, size=100), 3)
data_in = np.array(values)
noise = np.round(np.random.uniform(-1, 1, size=100), 3)

data_out_2 = 4 * data_in**2 + 2 * data_in + 7 + noise
data_out_3 = 0.5 * data_in**3 - 2 * data_in**2 + 3 * data_in + 5 + noise

sorted_indices = np.argsort(data_in)
data_in_sorted = data_in[sorted_indices]
data_out_sorted = data_out_2[sorted_indices]
data_out_3_sorted = data_out_3[sorted_indices]

fig, (ax, ax1) = plt.subplots(2, 1, figsize=(6.5, 10))
fig.canvas.manager.set_window_title("Polynomial regression")

ax.set_title('Regresie polinomială grad 2')
ax.scatter(data_in_sorted, data_out_sorted, color="green", label='Valori reale')

data_in_train, data_in_test, data_out_train, data_out_test = train_test_split(
    data_in, data_out_2, test_size=0.3, random_state=42
)
data_in_train = data_in_train.reshape(-1, 1)
data_in_test = data_in_test.reshape(-1, 1)

poly2 = PolynomialFeatures(degree=2)
data_in_train_poly2 = poly2.fit_transform(data_in_train)
data_in_test_poly2 = poly2.transform(data_in_test)

model2 = LinearRegression()
model2.fit(data_in_train_poly2, data_out_train)
data_out_predict2 = model2.predict(data_in_test_poly2)

sorted_test_indices = np.argsort(data_in_test.flatten())
data_in_test_sorted = data_in_test[sorted_test_indices]
data_out_predict2_sorted = data_out_predict2[sorted_test_indices]

ax.plot(data_in_test_sorted, data_out_predict2_sorted, color="red", label='Predicții model (grad 2)')
ax.legend()

ax1.set_title('Regresie polinomială grad 3')
ax1.scatter(data_in_sorted, data_out_3_sorted, color="green", label='Valori reale')

data_in_train3, data_in_test3, data_out_3_train, data_out_3_test = train_test_split(
    data_in, data_out_3, test_size=0.3, random_state=42
)
data_in_train3 = data_in_train3.reshape(-1, 1)
data_in_test3 = data_in_test3.reshape(-1, 1)

poly3 = PolynomialFeatures(degree=3)
data_in_train_poly3 = poly3.fit_transform(data_in_train3)
data_in_test_poly3 = poly3.transform(data_in_test3)

model3 = LinearRegression()
model3.fit(data_in_train_poly3, data_out_3_train)
data_out_predict3 = model3.predict(data_in_test_poly3)

x_line = np.linspace(-10, 10, 100).reshape(-1, 1)
x_line_poly3 = poly3.transform(x_line)
y_line_3 = model3.predict(x_line_poly3)
ax1.plot(x_line, y_line_3, color="blue", label="Predicții model (grad 3)")
ax1.legend()

mse2 = mean_squared_error(data_out_test, data_out_predict2)
mse3 = mean_squared_error(data_out_3_test, data_out_predict3)

print("------ MODEL GRAD 2 ------")
print("MSE grad 2:", mse2)
print("w2 (coef_):", model2.coef_)
print("b2 (intercept_):", model2.intercept_)

print("\n------ MODEL GRAD 3 ------")
print("MSE grad 3:", mse3)
print("w3 (coef_):", model3.coef_)
print("b3 (intercept_):", model3.intercept_)

plt.tight_layout()
plt.show()
