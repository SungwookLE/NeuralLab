import numpy as np
import matplotlib.pyplot as plt
import csv


if __name__ =="__main__":

    group_a=[]
    group_b=[]

    # READ FROM CSV
    file = open("./results/resultNeural.csv", mode="r", encoding='utf-8')
    
    reader = csv.reader(file)
    Initial = dict()
    Result =dict()

    # GET RESULTS (LAYER1)
    line = next(reader)
    line = next(reader)
    Result["W_h1[0]"] = float(line[4])
    Result["W_h1[1]"] = float(line[5])
    Result["B_h1"] = float(line[6])

    Result["W_h2[0]"] = float(line[7])
    Result["W_h2[1]"] = float(line[8])
    Result["B_h2"] = float(line[9])

    
    next(reader) # row eliminate
    for row in reader:
        if (float(row[2]) == 1):
            group_a.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])])
        else:
            group_b.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])])
    group_a= np.array(group_a)
    group_b= np.array(group_b)

    x_plot = np.linspace(-30,30,100)
    
    y_plot = -Result["W_h1[0]"] / Result["W_h1[1]"] * x_plot - Result["B_h1"] / Result["W_h1[1]"]
    plt.plot(x_plot, y_plot, 'r-.', linewidth=1.0, label='h1')
    y_plot = -Result["W_h2[0]"] / Result["W_h2[1]"] * x_plot - Result["B_h2"] / Result["W_h2[1]"]
    plt.plot(x_plot, y_plot, 'b-.', linewidth=1.0, label='h2')

    # VISUALIZATION
    plt.scatter(group_a[:, 0], group_a[:, 1], color='red',
                facecolors='none', label='group_a')
    plt.scatter(group_b[:, 0], group_b[:, 1], color='blue',
                facecolors='none', label='group_b')

    count = 0
    for i in range(len(group_a[:,0])):
        if (group_a[i,2] == group_a[i,3]):
            plt.plot(group_a[i,0], group_a[i,1], '.r')
            count+=1

    for i in range(len(group_b[:,0])):
        if (group_b[i,2] == group_b[i,3]):
            plt.plot(group_b[i,0], group_b[i,1], '.b')
            count +=1

    plt.xlim((-30, 30))
    plt.ylim((-30, 30))
    plt.legend()
    plt.text(-2,-2,str(count) + "/" +str(len(group_a[:,0])+len(group_b[:,0])) + ": " +str(count/(len(group_a[:,0])+len(group_b[:,0]))*100)+"%")
    plt.title('Classification Data')
    plt.savefig('results/resultNeuralShow.png')
    plt.show()