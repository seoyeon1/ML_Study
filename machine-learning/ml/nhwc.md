# NHWC

이전까지는 1차원 data를 가지고 기계학습을 했었다면 이제부터는 **N차원 data**를 가지고 학습을 해보려고 한다.

N차원 데이터를 학습하기에 앞서서 알아둬야할 개념들이 있는데 아래서 설명하겠다.

### 🤔 What is NHWC?

* **N** : a number of sample \#샘플 
* **H** : height \#높이
* **w** : width \#너비
* **C** : a number of channel \#채널 수

### 실



```text
import tensorflow as tf
import numpy as np
import random

tf.random.set_random_seed(0)
np.random.seed(0)
random.seed(0)

from tensorflow.examples.tutorials.mnist import input_data

#784 data -> img

mnist = input_data.read_data_sets('MINST_DATA/', one_hot=True)
nb_classes = 10

X = tf.compat.v1.placeholder(tf.float32, shape = [None, 784])#28 * 28=784
Y= tf.compat.v1.placeholder(tf.float32, shape = [None, nb_classes])#TensorShape([Dimension(None), Dimension(10)])



X_img = tf.reshape(X, shape =[-1, 28, 28, 1])#NHWC
#TensorShape([Dimension(None), Dimension(28), Dimension(28), Dimension(1)])
w1 = tf.compat.v1.get_variable('w1', shape = [3, 3, 1, 32], #HWCN   3x3 filter  weight name, filtershape, filter수:32개, initial value
                               initializer=tf.contrib.layers.xavier_initializer())#hw -> odd. so, 3x3 사용. chanel수 -> 2진수 사용(8,16, 32, 64, 128, ...)
L1 = tf.nn.conv2d(X_img, w1, strides=[1,1], padding='SAME')#h와w가 변경되지 않게 padd을 주려고. stride는 대체적으로 h, w만 정해줌
#L1's shape = [None, 28, 28, 32]
L1 = tf.nn.relu(L1)
```

**NHWC** X **HWCN** =&gt; **NHWC**\(=chanel\)

HWCN 여기의 채널 수\(c\) =  nhwc의 채널수\(c\)



{% hint style="info" %}
**실습 결**

7 layers & epoch : 3

test accuracy before training 0.1036 0 epoch & validation cost 0.09746909 sess save: 0.09746909 1 epoch & validation cost 0.05528703 sess save: 0.05528703 2 epoch & validation cost 0.057354487

test accuracy after training 0.9845
{% endhint %}

