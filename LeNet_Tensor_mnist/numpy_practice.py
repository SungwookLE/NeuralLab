import numpy as np

a = np.array((
    
    
    
    ([1,2,3,4,5,6]),
    ([1,2,3,4,5,6]),
    ([1,2,3,4,5,6])
    

))

a = np.vstack((a,a))

print(a)
print(a.shape)
print()

a=np.pad(a, ((0,0),(2,2),(2,2),(0,0)), 'constant')

print(a)