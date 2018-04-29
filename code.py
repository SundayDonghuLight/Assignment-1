import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import linear_model

PATH = "C:/Users/Mr-Fish/Desktop/Image_and_ImageData/"
I = mpimg.imread(PATH + "I.png")
E = mpimg.imread(PATH + "E.png")
key1 = mpimg.imread(PATH + "key1.png")
key2 = mpimg.imread(PATH + "key2.png")

[m,n] = I.shape

I_array = I.reshape(1, m*n)
E_array = E.reshape(1, m*n)
key1_array = key1.reshape(1, m*n)
key2_array = key2.reshape(1, m*n)

X = []
Y = []
for i in range(0, m*n):
    tem = [key1_array[0,i], key2_array[0,i], I_array[0,i]]
    X.append(tem)
    Y.append(E_array[0,i])
X = np.array(X)
Y = np.array(Y)

