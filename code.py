import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import linear_model

PATH = "C:/Users/Mr-Fish/Desktop/Image_and_ImageData/"
I = mpimg.imread(PATH + "I.png", 0)
E = mpimg.imread(PATH + "E.png", 0)
key1 = mpimg.imread(PATH + "key1.png", 0)
key2 = mpimg.imread(PATH + "key2.png", 0)

[m,n] = I.shape

I_array = I.reshape(1, m*n)
E_array = E.reshape(1, m*n)
key1_array = key1.reshape(1, m*n)
key2_array = key2.reshape(1, m*n)

X = []
Y = []
for i in range(0, m*n):
    tem = [E_array[0,i], key1_array[0,i], key2_array[0,i]]
    X.append(tem)
    Y.append(I_array[0,i])
X = np.array(X)
Y = np.array(Y)

clf = linear_model.SGDClassifier(alpha=0.00001, max_iter=10000).fit(X,Y)

Eprime = mpimg.imread(PATH + "Eprime.png", 0)
Eprime_array = Eprime.reshape(1, m*n)
P = []
for i in range(0, m*n):
    tem = [Eprime_array[0,i], key1_array[0,i], key2_array[0,i]]
    P.append(tem)
P = np.array(P)

Image = clf.predict(P)
Image = Image.reshape(m,n)

plt.imshow(Image, cmap='Greys_r')
plt.axis('off')
plt.show()
