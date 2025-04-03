import matplotlib.pyplot as plt
import numpy as np

rand_vals = np.random.randint(1,10, size = 100)
x = np.array(rand_vals)
noise  = np.round(np.random.uniform(-1,1, size = 100), 2)
y = 3 * x + 7 + noise
sorted_indices = np.argsort(x)
x_sorted = x[sorted_indices]
y_sorted = y[sorted_indices]
fig, ax = plt.subplots()
fig.canvas.manager.set_window_title("Grafic regresie liniară")
ax.set_title('Data for regression model')
ax.plot(x_sorted,y_sorted)
plt.show()    
