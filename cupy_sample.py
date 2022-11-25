import numpy as np
import cupy as cp
import time

n = 2000
t1 = time.time()
a = np.random.rand(n,n)
b = np.random.rand(n,n)
np.dot(a,b)
t2 = time.time()
a = cp.random.rand(n,n)
b = cp.random.rand(n,n)
cp.dot(a,b)
t3 = time.time()
print ('np:',t2-t1)
print ('cp:',t3-t2)