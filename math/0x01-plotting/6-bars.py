#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

col = ["Farrah", "Fred", "Felicia"]
row = [("apples", "red"), ("bananas", "yellow"),
       ("oranges", "#ff8000"), ("peaches", "#ffe5b4")]
y = np.zeros(len(col))

plt.title("Number of Fruit per Person")
plt.ylabel("Quantity of Fruit")
plt.ylim(0, 80)
plt.legend()

for i in range(len(row)):
    plt.bar(col, fruit[i], 0.5, bottom=y, color=row[i][1], label=row[i][0])
    y += fruit[i]

plt.show()
