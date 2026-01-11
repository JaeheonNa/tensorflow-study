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

# 2. 데이터셋을 mini-batch 단위로 묶음
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#   repeat(): train_data가 다 떨어지면, 그러니까 epoch 횟수가 데이터 사이즈를 넘어가면, 다시 처음부터 반복해서 데이터를 사용하겠다는 옵션.
#   suffle(): 전체 60000개 데이터를 모두 섞겠다.
train_data = train_data.repeat().shuffle(60000).batch(100)
train_data_iter = iter(train_data)

# 3. tf.keras.Model을 이용해 Softmax Regression 모델 정의(가설 정의)
class SoftmaxRegression(tf.keras.Model):
    def __init__(self):
        super(SoftmaxRegression, self).__init__()
#   Dense: 상위 Layer의 모든 입력이 빠짐 없이 연결되는 Layer. W * x + b 형태의 API. 여기 위에 activation 함수를 입힐 수도, 안 입힐 수도 있음.
#   이 때, W는 입력 데이터 수 x 출력 데이터 수 형태의 행렬임. 이 예제에서는 (784, 10) 형태.
#   units: output dimension. 출력 뉴런 갯수.
#   activation: 활성화 함수 지정 안 함.
#   kernel_initializer: W 값을 0으로 지정.
#   bias_initializer: b 값을 0으로 지정.
        self.softmax_layer = tf.keras.layers.Dense(units=10,
                                                   activation=None,
                                                   kernel_initializer='zeros',
                                                   bias_initializer='zeros')
    def call(self, x):
        logits = self.softmax_layer(x)
        return tf.nn.softmax(logits)

# 4. cross-entropy 손실함수 정의
#   분류 문제에 있어서는 MSE보다 cross-entropy가 더 성능이 좋다고 알려져있음.
@tf.function
def cross_entropy(y_pred, y_actual):
    return tf.reduce_mean(-tf.reduce_sum(y_actual * tf.math.log(y_pred), axis=[1]))
    # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_actual)) 이 함수를 사용하려면, logits에 softmax가 적용이 안 된 logits을 넘겨줘야 함.

# 5. 옵티마이저 정의
optimizer = tf.optimizers.SGD(0.5)

# 6. 최적화 function 정의
#   여기서 model은 tf.keras.Model을 상속받은 모델. trainable_variables라는 속성을 사용할 수 있음.
def train_step(model: tf.keras.Model, x, y_actual):
    with tf.GradientTape() as tape:
        y_pred = model.call(x)
        loss = cross_entropy(y_pred, y_actual)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# 7. 모델 정확도 출력 함수 정의
#   예측값과 실제값을 중 일치하는 것들의 수를 비교해 0~1사이의 정확도를 반환.
#   y_pred는 softmax 함수를 적용했기 때문에 벡터 요소들이 '0~1 사이의 값'을 가짐.
#   y_actual는 one-hot encoding을 적용했기 때문에 '0 또는 1값'을 가짐.
#   argmax는 벡터 요소들 중 가장 큰 수를 가진 index를 반환함.
#   correct_prediction는 True 혹은 False로 이뤄진 배열.
#   correct_prediction_float은 1.0 혹은 0.0으로 이뤄진 배열.
#   accuracy는 correct_prediction_float의 요소를 모두 더한 후, correct_prediction_float의 크기로 나눈 값.
@tf.function
def compute_accuracy(y_pred, y_actual):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_actual, 1))
    correct_prediction_float = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction_float)
    return accuracy

# 8. 학습 수행
SoftmaxRegression_model = SoftmaxRegression()
for i in range(1000):
    batch_xs, batch_ys = next(train_data_iter)
    train_step(SoftmaxRegression_model, batch_xs, batch_ys)

# 9. 모델의 정확도 측정
print("정확도: %f" % compute_accuracy(SoftmaxRegression_model.call(x_test), y_test))
