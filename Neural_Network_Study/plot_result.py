import numpy as np
import matplotlib.pyplot as plt
import csv
def forward_propagation(X, W, B, opt="sigmoid"):
    
    def activate_sigmoid(x):
        res = 1.0/(1.0+np.exp(-x))
        return res

    def activate_relu(x):
        res = np.zeros((len(x),1))
        for i in range(len(x)):
            if (x[i]>=0):
                res[i] = x[i]
            else:
                res[i] = 0
        return res

    res = X[:,0]*W[0] + X[:,1]*W[1] + B

    if (opt == "relu"):
        ans = activate_relu(res)
    else: #(opt == "sigmoid")
        ans = activate_sigmoid(res)

    return ans

group_a=[]
group_b=[]
# READ FROM CSV
file = open("dataset.csv", mode="r", encoding='utf-8')
reader = csv.reader(file)
for row in reader:
    
    if (float(row[2]) == 1):
        group_a.append([float(row[0]), float(row[1])])
    else:
        group_b.append([float(row[0]), float(row[1])])

group_a= np.array(group_a)
group_b= np.array(group_b)

"""
X2 = -W[0]/W[1]*X1 - B/W[1]
"""
# INSERT RESULTS
Initial={"W[0]":0.28877, "W[1]":-0.202874, "B":-0.127324}
Result={"W[0]":-0.489742, "W[1]":-0.342007, "B":0.402725}
score_a = forward_propagation(group_a, W=[Result["W[0]"], Result["W[1]"]], B=Result["B"], opt="relu")
score_b = forward_propagation(group_b, W=[Result["W[0]"], Result["W[1]"]], B=Result["B"], opt="relu")

# VISUALIZATION
plt.scatter(group_a[:, 0], group_a[:, 1], color='red',
            facecolors='none', label='group_a')
plt.scatter(group_b[:, 0], group_b[:, 1], color='blue',
            facecolors='none', label='group_b')

x_plot = np.linspace(-3,3,100)

y_plot = -Initial["W[0]"] / Initial["W[1]"] * x_plot - Initial["B"] / Initial["W[1]"]
plt.plot(x_plot, y_plot, 'k-.', linewidth=0.5, label='initial')

y_plot = -Result["W[0]"] / Result["W[1]"] * x_plot - Result["B"] / Result["W[1]"]
plt.plot(x_plot, y_plot, 'k-.', linewidth=1.5, label='result')

count = 0
for i in range(len(score_a)):
    if (score_a[i] >= 0.5):
        plt.plot(group_a[i,0], group_a[i,1], '.r')
        count+=1
for i in range(len(score_b)):
    if (score_b[i] < 0.5):
        plt.plot(group_b[i,0], group_b[i,1], '.b')
        count +=1

#plt.tight_layout()
plt.xlim((-3, 3))
plt.ylim((-3, 3))
plt.legend()
plt.text(-2,-2,str(count) + "/" +str(len(group_a[:,0])+len(group_b[:,0])) + ": " +str(count/(len(group_a[:,0])+len(group_b[:,0]))*100)+"%")
plt.title('Classification Data')
plt.show()



