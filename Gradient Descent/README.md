# Gradient Descent Method

<strong>Loss function "L(w)" : </strong><p>
           <img src="http://chart.googleapis.com/chart?cht=tx&chl=\sum_{1}^{m*n}\left[y_{i}-x_{i1}w_{1}-x_{i2}w_{2}-x_{i3}w_{3}\right]^{2}" style="border:none;">
<p>
即使用Sum of squared errors (SSE) 平方誤差總和 來做為權衡 w 好壞的標準<p>
若[key1 key2 I][w1 w2 w3]t 所估測出來的 y_hat 與E圖片的灰階差越多則 L(w) 越大<p>
　　  <p>
<strong>Learning Algorithm:</strong> Perceptron Learning Algorithm(gradient descent method)<p>
           <img src="http://chart.googleapis.com/chart?cht=tx&chl=w\left(t\right)=w\left(t-1\right)-\eta\frac{\partial+L\left(w\right)}{\partial+w}"style="border:none;">
 　　  <p>
      
 ### 程式說明
<ol>
<li>
載入套件: <ul>
<li>numpy: 能在python中很直覺的實現數學上各種矩陣運算的強大套件</li>
<li>matplotlib.pyplot: 能進行繪圖，在這用來顯示和儲存最後的圖像</li>
<li>matplotlib.image: 用來將圖片讀入並以矩陣形式儲存其灰階階度</li>
</ul></li>
<li>
讀入影像並存入對應的變數中
<pre><code>PATH = "C:/Users/Mr-Fish/Desktop/Image_and_ImageData/"
I = mpimg.imread(PATH + "I.png", 0)
E = mpimg.imread(PATH + "E.png", 0)
key1 = mpimg.imread(PATH + "key1.png", 0)
key2 = mpimg.imread(PATH + "key2.png", 0)
</pre></code></li>
<li>
以一個 m*n by 3 的矩陣 X 儲存[key1 key2 I]的數據，<p>
每一列的元素分別對應到3個圖片相同位置像素的灰度。<p>
向量Y儲存圖片E的數據，並都轉換成np.array方便接下來的數學運算。
<pre><code>[m,n] = I.shape
X = []
Y = []
for i in range(0,m):
    for j in range(0,n):
        X.append([key1[i,j], key2[i,j], I[i,j]])
        Y.append(E[i,j])
X = np.array(X, dtype = np.int64)
Y = np.array(Y, dtype = np.int64)
</pre></code></li>
<li>
隨便設定一個權重向量 w 的初值，Eta是影響Gradient Descent每次下降幅度的參數，stop為停止條件(容許誤差)。
<pre><code>w = [1,1,1]                                                    # initial weight vector
w = np.array(w, dtype = np.float64)
stop = 0.000000001                                             # stop condition
Eta = 0.0000000001                                             # adjustment parameter
</pre></code></li>
<li>
Perceptron Learning Algorithm，在變動向量的範數(Norm)小於停止條件前不斷調整 w 權重向量，<p>
調正方是為減去 L(w) 對 w 的偏微分乘上調整參數 Eta，若切線斜率為正，w 會調小，為負則調大，<p>
故只要Loss function連續且可微，w 就會不斷趨近能使 L(w) 最小的局部最佳解。
<p>其微分形式可參考: <a href="https://ccjou.wordpress.com/2013/05/31/%E7%9F%A9%E9%99%A3%E5%B0%8E%E6%95%B8/">矩陣導數</a></p>
<pre><code>while (1):
    change = Eta * -2 * np.dot(X.T,Y - np.dot(X,w))
    w = w - change
    if (np.dot(change,change) < stop):
        break
</pre></code></li>
<li>
將待解碼(預測)的圖片Eprime讀入並使用前面估測出來的 w 權重代入關係式算出原圖 Image 並顯示出來
<pre><code>Eprime = mpimg.imread(PATH + "Eprime.png", 0)
key1 = np.array(key1); key2 = np.array(key2); Eprime = np.array(Eprime);
Image = (1/w[2])*Eprime - (w[0]/w[2])*key1 - (w[1]/w[2])*key2

plt.imshow(Image, cmap='Greys_r')
plt.axis('off')
plt.show()
</pre></code></li>
</ol>

