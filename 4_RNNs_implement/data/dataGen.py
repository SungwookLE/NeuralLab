import numpy as np
import matplotlib.pyplot as plt
import csv
import math

if __name__ == "__main__":

    # 1. Prepare TimeSeries dataSet
    time = np.arange(0, 10, step=0.1).reshape(-1,1)
    noise_acc = np.random.randn(time.size,1)*0.1
    noise_pos = np.random.randn(time.size,1)*0.2
    init_vel = 10

    acc = np.sin(2*math.pi*0.2*time)*5 + noise_acc
    vel = np.add.accumulate(0.1*acc) + init_vel
    pos = np.add.accumulate(0.1*vel) + noise_pos
    vel_GT = np.add.accumulate(0.1*np.sin(2*math.pi*0.2*time)*5) + init_vel

    data_set = np.concatenate((time, acc, pos, vel_GT), axis=1)
    print(data_set.shape)

    # 2. Save to csv
    file = open("./data/dataSet.csv", mode="w")
    writer = csv.writer(file)
    writer.writerow(["time", "acc", "pos", "vel(ground truth)"])
    for row in data_set:
        writer.writerow(row)

    # 3. VISUALIZATION
    plt.figure()
    plt.subplot(3,1,1)
    plt.grid()
    plt.title("acc")
    plt.plot(data_set[1:, 0], data_set[1:, 1])
    plt.subplot(3,1,2)
    plt.grid()
    plt.title("pos")
    plt.plot(data_set[1:, 0], data_set[1:, 2])
    plt.subplot(3,1,3)
    plt.grid()
    plt.title("vel(gt)")
    plt.plot(data_set[1:, 0], data_set[1:, 3])
    plt.savefig("./data/dataShow.png")
    plt.show()