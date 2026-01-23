# 컨볼루션 신경망(CNN)
#   이미지 분야를 다루는데 최적화된 인공 신경망 구조.
#   이미지는 이미지에 찍힌 대상의 '특징'을 뽑아내는 게 중요함. 그래야 그 대상을 잘 인식할 수 있게 됨.
#   컨볼루션 신경망은 크게 컨볼루션층과 풀링층으로 구성됨. (풀링은 서브샘플링이라고도 함.)
#   컨볼루션(합성곱)은 특징 추출, 풀링은 차원 축소 개념.
#   원본 이미지 -> 컨볼루션 -> 풀링 -> 컨볼루션 -> 풀링 -> ... 후 1차원 벡터 생성 -> ANN input layer에 투입 -> Softmax Regression
#   원본 이미지에서 특징 추출 후 1차원 벡터를 획득, 1차원 벡터를 인공 신경망에 넣어 SoftMaxRegression 수행, 해당 이미지가 어떤 이미지에 부합할 확률이 높은지 확인 가능.
# 컨볼루션(합성곱)
#   원본 행렬 데이터에, 커널(필터)라 불리는 행렬 데이터를 곱한 후 모두 더하여 Scalar를 생성하는 것.
#   컨볼루션을 수행하면, 커널(필터)의 종류에 따라 원본 행렬 데이터의 '특징'이 추출돼 활성화 맵으로 만들어짐.
#   원래 초기에는 사람이 커널을 직접 고안해서 필요할 때마다 커널을 골라서 사용했음.
#   요즘에는 커널 자체를 파라미터로 간주. 필요한 커널을 '학습'을 통해 만들어내는 추세.
#   이렇게 추출된 활성화맵은 원본 이미지에서 드러나지 않았던 특징들을 도드라지게 보여줌.
# 풀링
#   최대값 풀링, 평균값 풀링, 최소값 풀링 세 가지가 있음.
#   이미지의 차원을 축소함. ex) 4x4 행렬을 1x1로 변형.
# CNN의 하이퍼파라미터
#   Wout: 원본 이미지에서 필터를 거쳐 뽑아낼 이미지의 너비
#   Hout: 원본 이미지에서 필터를 거쳐 뽑아낼 이미지의 높이
#   K: 사용할 필터의 수
#   그 결과는 Wout x Hout 크기를 갖는 이미지 K개 생성됨.

import tensorflow as tf

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
#   one-hot encoding 적용
y_train, y_test = tf.one_hot(y_train, depth=10), tf.one_hot(y_test, depth=10)

# 2. 학습 데이터 준비
#   (x, y) 형태의 데이터셋 튜플 생성
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#   한 번의 epoch이 끝날 때마다 60000개 데이터를 섞은 후 batch size로 나눔.
train_data = train_data.repeat().shuffle(60000).batch(50)
train_data_iter = iter(train_data)

# 3. CNN 모델 정의
class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        # 이 모델에서 학습의 대상이 되는 파라미터가 바로 filter.
        # Conv2D: 아래의 경우 5x5 픽셀 크기의 필터가 1픽셀씩 이미지를 필터링하며, 그 결과로 축소된 이미지들을 원본 이미지의 크기와 맞추기 위해 패딩을 넣음.
        #         padding 은 same 대신 valid를 넣을 수 있는데, 이 경우 축소된 이미지에 1픽셀씩만 패딩을 추가하게 됨.
        # MaxPool2D: 이미지를 2x2 크기의 필터로 차원을 축소함. 즉 길이 기준으로는 1/2 크기로, 넓이 기준으로는 1/4 크기로 축소. 가장 큰 값을 기준으로 축소됨.
        self.conv_layer_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), strides=1, padding="same", activation='relu')
        self.pool_layer_1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)

        self.conv_layer_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=1, padding="same", activation='relu')
        self.pool_layer_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)

        self.flatten_layer = tf.keras.layers.Flatten()
        self.fc_layer_1 = tf.keras.layers.Dense(units=1024, activation='relu')

        self.output_layer = tf.keras.layers.Dense(10, activation=None)

    def call(self, x):
        # MNIST 데이터를 3차원 형태로 reshape. MNIST 데이터는 GrayScale 이미지이기 때문에 3번째 차원(컬러 채널)의 값은 1임.
        # 아래 코드를 거치면 28 x 28 크기의 이미지가 60000장 생성됨. -1은 원래 데이터의 값(60,000)을 그대로 가져오라는 뜻이며, 1은 흑백을 뜻함(컬러는 3)
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        # 아래 코드를 거치면 이미지 1장에 대해서 28 x 28 크기의 이미지가 32개 생성됨.
        h_conv1 = self.conv_layer_1(x_image)
        # 아래 코드를 거치면 이미지 1장에 대해서 14 x 14 크기의 이미지가 32개 생성됨.
        h_pool1 = self.pool_layer_1(h_conv1)
        # 아래 코드를 거치면 이미지 1장에 대해서 14 x 14 크기의 이미지가 64개 생성됨.
        h_conv2 = self.conv_layer_2(h_pool1)
        # 아래 코드를 거치면 이미지 1장에 대해서 7 x 7 크기의 이미지가 64개 생성됨.
        h_pool2 = self.pool_layer_2(h_conv2)
        # 아래 코드를 거치면 이미지 한 장에 대한 정보(7 x 7 x 64)가 1차원으로 펼쳐짐. 3136
        h_pool2_flatten = self.flatten_layer(h_pool2)
        # 아래 코드를 거치면 활성화 함수를 통해 특징이 추상화(압축)됨. 3136 -> 1024
        f_fc1 = self.fc_layer_1(h_pool2_flatten)
        # 아래 코드를 거치면 활성화 함수를 통해 특징이 추상화(압축)됨. 1024 -> 10
        logits = self.output_layer(f_fc1)
        # SoftMax 함수를 통해 어떤 숫자인지 확률을 계산한 후 예측값을 뽑아냄.
        y_pred = tf.nn.softmax(logits)
        return y_pred, logits

# 4. cross-entropy 손실 함수 정의
@tf.function
def cross_entropy_loss(logits, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

# 5. 옵티마이저 정의
#   경사하강법을 수행하는 optimizer. 1e-4 = 0.0001
optimizer = tf.optimizers.Adam(learning_rate=1e-4)

# 6. 최적화를 위한 function 정의
def train_step(model, x, y_actual):
    with tf.GradientTape() as tape:
        y_pred, logits = model.call(x)
        loss = cross_entropy_loss(logits, y_actual)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# 7. 모델 정확도 출력 함수 정의
@tf.function
def compute_accuracy(y_pred, y_actual):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_actual, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

# 8. 학습 수행
CNN_model = CNN()
for i in range(10000):
    # 50개씩 MNIST 데이터 load.
    batch_x, batch_y = next(train_data_iter)
    # 100 step마다 training 데이터셋에 대한 정확도 출력
    if i % 100 == 0:
        y_batch_pred, logits = CNN_model.call(batch_x)
        train_accuracy = compute_accuracy(y_batch_pred, batch_y)
        print("반복(Epoch) %d, 정확도(Accuracy): %f" % (i, train_accuracy))
    train_step(CNN_model, batch_x, batch_y)

# 9. 학습 종료 후 학습된 모델의 정확도 출력.
y_test_pred, y_test_logits = CNN_model.call(x_test)
accuracy = compute_accuracy(y_test_pred, y_test)
print("정확도(Accuracy): %f" % accuracy)
