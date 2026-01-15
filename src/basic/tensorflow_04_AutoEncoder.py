# 오토인코더
# 대표적인 비지도 학습을 위한 인공신경망 구조 중 하나.
# 비지도 학습은 어떤 값을 예측하거나 분류하는 것이 목적인 지도학습과는 다르게,
# 데이터의 숨겨진 구조를 발견하는 것이 목표인 학습 방법.
# 입력층과 출력층의 노드 수가 같은 것이 특징.(은닉층은 노드 수는 적음)
# 따라서 오토인코더의 출력은 원본 데이터를 재구축한 결과가 됨.
# 은닉층의 노드 수가 입력층의 노드 수보다 적은 게 키 포인트인데,
# 원본 데이터에서 불필요한 특징들을 제거한 압축된 특징들을 학습한다는 의미임.
# 압축된 특징을 나타내는 은닉층의 출력값을 분류기의 입력으로 사용하면 더 좋은 분류 성능을 나타낼 수 있음. how? 불필요한 특징이 제거됐기 때문.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from TensorFlow.src.basic.tensorflow_03_ANN import hidden1_size

# 1. MNIST 데이터(기초 예제 데이터) 구성
#   train_data = 6만개, test_data = 1만개. numpyArray type으로 내려줌.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#   기본적으로 integer 타입으로 내려주기 때문에 타입을 float으로 변경.
x_train, x_test = x_train.astype("float32"), x_test.astype("float32")
#   flattening: 데이터를 1차원으로 펼쳐주는 작업. 1차원이어야 SoftmaxRegression 함수를 적용할 수 있음.
#   이 예제 데이터의 경우 1개의 데이터가 28 x 28의 2차원 배열로 이뤄져 있음. 따라서 x_train의 경우 60000 x 28 x 28로 이뤄져 있고, x_test는 10000 x 28 x 28.
#   1개의 데이터에 대해서 1차원으로 펼쳐주면 784가 됨. (28 * 28)
#   앞의 -1은 '매직 넘버'로서 맨 앞의 사이즈(x_train의 경우 60000, x_test의 경우 10000)를 자동으로 맞춰 줌. 그러니까 각각 -1이 아니라 60000, 10000으로 넣어줘도 무방함.
#   [60000, 28, 28]의 형태가 [60000, 784] 형태로 변경됨.
x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])
#   데이터의 요소들은 기본적으로 0~255의 값을 갖는데, 이를 0~1값을 갖도록 변경.
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. 학습에 필요한 설정값 정의
learning_rate = 0.02
training_epochs = 50
batch_size = 256
display_step = 1
examples_to_show = 10       # 실행 결과, 위는 원본 데이터, 아래는 재구축 데이터가 담겨 보이게 되는데, 그걸 몇 개로 보여줄지 결정하는 변수
input_size = 784
hidden1_size = 256
hidden2_size = 128
