import numpy as np
import matplotlib.pyplot as plt

A = []
B = []

corr = np.correlate(A, B, "full")
delay = corr.argmax() - (len(B) - 1)
print(str(delay))
print(corr.max())


plt.subplot(4, 1, 1)
plt.ylabel("A")
plt.plot(A)

plt.subplot(4, 1, 2)
plt.ylabel("B")
plt.plot(B, color="g")

plt.subplot(4, 1, 3)
plt.ylabel("fit")
plt.plot(np.arange(len(A)), A)
plt.plot(np.arange(len(B)) + delay, B)
plt.xlim([0, len(A)])

plt.subplot(4, 1, 4)
plt.ylabel("corr")
plt.plot(np.arange(len(corr)) - len(B) + 1, corr, color="r")
plt.xlim([0, len(A)])

plt.show()
