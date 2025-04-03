import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



rand_vals = np.random.randint(1,10, size = 100)
x = np.array(rand_vals)
noise  = np.round(np.random.uniform(-1,1, size = 100), 2)
y = 3 * x + 7 + noise
sorted_indices = np.argsort(x)
x_sorted = x[sorted_indices]
y_sorted = y[sorted_indices]
fig, ax = plt.subplots()
fig.canvas.manager.set_window_title("Grafic regresie liniară")
line_data = ax.set_title('Data for regression model')
ax.scatter(x_sorted,y_sorted, color = "green", label = 'Valori reale')
   

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)# "The answer to life, the universe and everything is... 42."
x_train = x_train.reshape(-1, 1)  # make 2D for the model bcs you can recive more features in the model
x_test = x_test.reshape(-1, 1)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
line_model = ax.plot(x_test,y_pred, color = "red", label='Predictii model')
ax.legend()
plt.show() 
print("MSE:", mse)