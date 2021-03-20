import numpy as np
import matplotlib.pyplot as plt
import csv

group_a=[]
group_b=[]
# READ FROM CSV
file = open("result.csv", mode="r", encoding='utf-8')
reader = csv.reader(file)
next(reader) # row eliminate
for row in reader:
    
    if (float(row[2]) == 1):
        group_a.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])])
    else:
        group_b.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])])

group_a= np.array(group_a)
group_b= np.array(group_b)

"""
X2 = -W[0]/W[1]*X1 - B/W[1]
"""
x_plot = np.linspace(-30,30,100)

# INSERT RESULTS (LAYER1)
"""
Initial={"W[0]":-0.298642, "W[1]":-0.00951062, "B":0.424167}
Result={"W[0]":-1.95113, "W[1]":-1.75355, "B":0.63973}

y_plot = -Initial["W[0]"] / Initial["W[1]"] * x_plot - Initial["B"] / Initial["W[1]"]
plt.plot(x_plot, y_plot, 'k-.', linewidth=0.5, label='initial')
y_plot = -Result["W[0]"] / Result["W[1]"] * x_plot - Result["B"] / Result["W[1]"]
plt.plot(x_plot, y_plot, 'k-.', linewidth=1.5, label='result')
"""

# INSERT RESULTS (LAYER2)
Result1={"W_h1[0]":0.126182, "W_h1[1]":-5.30623, "B_h1":-3.2251}
Result2={"W_h2[0]":1.71375, "W_h2[1]":0.23397, "B_h2":0.375245}

y_plot = -Result1["W_h1[0]"] / Result1["W_h1[1]"] * x_plot - Result1["B_h1"] / Result1["W_h1[1]"]
plt.plot(x_plot, y_plot, 'r-.', linewidth=1.0, label='h1')
y_plot = -Result2["W_h2[0]"] / Result2["W_h2[1]"] * x_plot - Result2["B_h2"] / Result2["W_h2[1]"]
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

plt.tight_layout()
plt.xlim((-30, 30))
plt.ylim((-30, 30))
plt.legend()
plt.text(-2,-2,str(count) + "/" +str(len(group_a[:,0])+len(group_b[:,0])) + ": " +str(count/(len(group_a[:,0])+len(group_b[:,0]))*100)+"%")
plt.title('Classification Data')
plt.show()