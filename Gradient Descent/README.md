# Gradient Descent Method

<strong>Loss function "L(w)" : </strong><p>
           <img src="http://chart.googleapis.com/chart?cht=tx&chl=\sum_{1}^{m*n}\left[y_{i}-x_{i1}w_{1}-x_{i2}w_{2}-x_{i3}w_{3}\right]^{2}" style="border:none;">
<p>
即使用Sum of squared errors (SSE) 平方誤差總和 來做為權衡 w 好壞的標準<p>
若[key1 key2 I ][w1 w2 w3]t 所估測出來的 y_hat 與E圖片的灰階差越多則 L(w) 越大<p>
　　  <p>
<strong>Learning Algorithm:</strong> Perceptron Learning Algorithm(gradient descent method)<p>
           <img src="http://chart.googleapis.com/chart?cht=tx&chl=w\left(t\right)=w\left(t-1\right)-\eta\frac{\partial+L\left(w\right)}{\partial+w}"style="border:none;">
 
 
 ### 程式說明
