import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PATH = "C:/Users/Mr-Fish/Desktop/Image_and_ImageData/"
I = mpimg.imread(PATH + "I.png", 0)
E = mpimg.imread(PATH + "E.png", 0)
key1 = mpimg.imread(PATH + "key1.png", 0)
key2 = mpimg.imread(PATH + "key2.png", 0)

[m,n] = I.shape
X = []
Y = []
for i in range(0,m):
    for j in range(0,n):
        X.append([key1[i,j], key2[i,j], I[i,j]])
        Y.append(E[i,j])
X = np.array(X, dtype = np.int64)
Y = np.array(Y, dtype = np.int64)

w = [1,1,1]                                                    # initial weight vector
w = np.array(w, dtype = np.float64)

stop = 0.000000001                                             # stop condition
Eta = 0.0000000001                                             # adjustment parameter

while (1):
    change = Eta * -2 * np.dot(X.T,Y - np.dot(X,w))
    w = w - change
    if (np.dot(change,change) < stop):
        break
        
Eprime = mpimg.imread(PATH + "Eprime.png", 0)
key1 = np.array(key1); key2 = np.array(key2); Eprime = np.array(Eprime);
Image = (1/w[2])*Eprime - (w[0]/w[2])*key1 - (w[1]/w[2])*key2


plt.imshow(Image, cmap='Greys_r')
plt.axis('off')
plt.show()
