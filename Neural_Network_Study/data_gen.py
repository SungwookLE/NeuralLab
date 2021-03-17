import numpy as np
import matplotlib.pyplot as plt
import csv

# Prepare data_set
data_a = np.random.randn(100, 2)
data_a[(data_a[:, 0] >= -0.1) & (data_a[:, 1] >= -0.1)] = - \
    0.5 * abs(np.random.rand(1, 2))

data_b = np.random.randn(100, 2)
data_b[(data_b[:, 0] <= 0.1) | (data_b[:, 1] <= 0.1)
       ] = 0.5 * abs(np.random.rand(1, 2))

label_a = np.ones((100, 1))
label_b = np.zeros((100, 1))

group_a = np.concatenate((data_a, label_a), axis=1)
group_b = np.concatenate((data_b, label_b), axis=1)

data_set = np.concatenate((group_a, group_b), axis=0)
features = data_set[:, 0:2]
labels = data_set[:, 2]

# SAVE TO CSV
file = open("dataset.csv", mode="w")
writer = csv.writer(file)
for row in data_set:
    writer.writerow(row)

# VISUALIZATION
plt.scatter(group_a[:, 0], group_a[:, 1], color='red',
            facecolors='none', label='group_a')
plt.scatter(group_b[:, 0], group_b[:, 1], color='blue',
            facecolors='none', label='group_b')
#plt.tight_layout()
plt.xlim((-3, 3))
plt.ylim((-3, 3))
plt.legend()
plt.title('Classification Data')
plt.show()