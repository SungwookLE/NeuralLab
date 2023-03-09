import numpy as np
import matplotlib.pyplot as plt
import csv

if __name__ == "__main__":

    # 1. Prepare Classification dataSet: {Size_Features: 2 , Size_OutputLabel: 1}
    data_a = np.random.randn(100, 2)*10
    data_a[(data_a[:, 0] >= 0) & (data_a[:, 1] >= 0)] = - \
        0.5 * abs(np.random.rand(1, 2))*10

    data_b = np.random.randn(100, 2)*10
    data_b[(data_b[:, 0] <= 0) | (data_b[:, 1] <= 0)
        ] = 0.5 * abs(np.random.rand(1, 2))*10

    label_a = np.ones((100, 1))
    label_b = np.zeros((100, 1))

    group_a = np.concatenate((data_a, label_a), axis=1)
    group_b = np.concatenate((data_b, label_b), axis=1)

    data_set = np.concatenate((group_a, group_b), axis=0)

    # 2. Save to csv
    file = open("dataSet.csv", mode="w")
    writer = csv.writer(file)
    for row in data_set:
        writer.writerow(row)

    # 3. VISUALIZATION
    plt.scatter(group_a[:, 0], group_a[:, 1], color='red',
                facecolors='none', label='group_a')
    plt.scatter(group_b[:, 0], group_b[:, 1], color='blue',
                facecolors='none', label='group_b')

    plt.xlim((-30, 30))
    plt.ylim((-30, 30))
    plt.legend()
    plt.title('DataSet')
    plt.savefig("./data/dataShow.png")
    plt.show()