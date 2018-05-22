# Gradient Descent Method

<strong>Loss function "L(w)" : </strong><p>
           <img src="http://chart.googleapis.com/chart?cht=tx&chl=\sum_{1}^{m*n}\left[y_{i}-x_{i1}w_{1}-x_{i2}w_{2}-x_{i3}w_{3}\right]^{2}" style="border:none;">
<p>
即使用Sum of squared errors (SSE) 平方誤差總和 來做為權衡 w 好壞的標準<p>
若[key1 key2 I][w1 w2 w3]t 所估測出來的 y_hat 與E圖片的灰階差越多則 L(w) 越大<p>
　　  <p>
<strong>Learning Algorithm:</strong> Perceptron Learning Algorithm(gradient descent method)<p>
           <img src="http://chart.googleapis.com/chart?cht=tx&chl=w\left(t%2B1\right)=w\left(t\right)-\eta\frac{\partial+L\left(w\right)}{\partial+w}"style="border:none;">
 　　  <p>
      
 ### 程式說明
<ol>
<li>
載入套件: <ul>
<li>numpy: 能在python中很直覺的實現數學上各種矩陣運算的強大套件</li>
<li>matplotlib.pyplot: 能進行繪圖，在這用來顯示和儲存最後的圖像</li>
<li>matplotlib.image: 用來將圖片讀入並以矩陣形式儲存其灰階階度</li>
</ul>
<pre><code>import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
</pre></code></li>
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
Perceptron Learning Algorithm，在變動向量自己的內積小於停止條件前不斷調整 w 權重向量，<p>
調正方式為減去 L(w) 對 w 的偏微分乘上調整參數 Eta，若切線斜率為正，w 會調小，為負則調大，<p>
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
 　　  <p>

### 運行結果
　　<strong> w = [w1,w2,w3] = [0.24987121, 0.6598644 , 0.09034653] </strong><p>
    ![Aaron Swartz](https://github.com/SundayDonghuLight/Assignment-1/raw/master/Gradient%20Descent/%E5%9C%96%E7%89%87%E8%A7%A3%E7%A2%BC_GD.png)
　　  <p>
    
### 心得與過程困難
　　要說的話過程中真的可以說是處處碰壁了吧XD，不熟悉的語言，新接觸的知識，和從未使用過的平台環境，帶來了種種的困難，好在github不是太難學，跟著辦完帳號後的新手教學就能學會主要該會的功能了。說個題外話，演算法或以前寫程式時常常需要上網查找或參考，就有留意到很多人都是把程式碼傳到github這個平台上了，所以這次能藉著這個機會強迫自己學會並成為其中的一員對我來說其實也是個不亞於了解機器學習算法的收穫呢。<p>
　　最開始撰寫的時候一來有點不知道從何下手，二來想善用python的優勢，豐富的機器學習套件，結果在丟入參數時value error各種error，在研究了SGDClassifier的說明文件好一番時間後終於把資料餵進去時真的感動到都想大叫了，雖然明明沒做什麼只是用了別人寫好的套件，且辨識度還實在不是太好，但一想到寫出了人生第一個學習程式還是非常的開心，這時還沉浸在這美好的誤會裡沒有醒悟，因為用SGDClassifier還真的要等他學一斷時間呢~<p>
　　關鍵的轉捩點是在我終於把它與過去我所學過的回歸分析連結在一起的瞬間，這個線性關係式怎麼看都像多元線性回歸的模型去掉截距項而已，而且因為已經明確知道是線性方程式了，迴歸係數必為1，誤差(deviation)也不存在。這樣在用最小平方法回歸分析時可經由2階偏導數和二次型(Quadratic form)的正定性質知道Loss function會是一個凹函數，只有一處微分為0的點且為函數值最小的全域最佳解。這樣就可以確定如果圖片沒解析出來絕對不是沒學好，而是程式本身就有錯了，最後終於檢查到了在進行兩個size很大的正整數矩陣相乘時因為配置的記憶體空間不足而造成了算數溢位並影響了結果。<p>
　　對我來說這是一份很有趣的作業，站在完成的立場來看的話確實程式碼都短短的看起來並不是說多難，但從中卻讓我學習到了許多的東西，也更加堅定了想增加自己學習廣度和深度的心情，我蠻喜歡的一個老師就常常說當一個問題解出答案時才是有趣的開始，有時是答案本身就蘊含著出人預料的意義，有時是可以看到不一樣的解法不一樣的思維，每種方法可能都有它所適合的問題，如果只是解出了題目就不再去思考果然是相當可惜的。<p>


