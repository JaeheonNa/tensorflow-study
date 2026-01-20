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

# 2. 학습을 위한 설정값 정의
#   전체 학습 데이터 6만 개를 256의 크기를 가지는 부분 데이터 약 235개로 나눠서 학습을 수행.
#   235개 부분 데이터들이 각각 30번씩 학습에 투입됨.
#   한 번 학습을 진행할 때 0.002의 학습률을 가짐.
learning_rate = 0.002   # 학습률
num_epochs = 30         # 전체 데이터 세트 기준 학습 횟수
batch_size = 256        # 부분 데이터 세트 크기
display_step = 1        # 손실함수 출력 주기
input_size = 784    # 인공신경망의 입력층 노드 크기
hidden1_size = 256  # 인공신경망의 은닉층 노드 크기
hidden2_size = 256  # 인공신경망의 은닉층 노드 크기
output_size = 10    # 인공신경망의 출력층 노드 크기

# 3. 학습 데이터 준비
#   (x, y) 형태의 데이터셋 튜플 생성
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#   한 번의 epoch이 끝날 때마다 60000개 데이터를 섞은 후 batch size로 나눔.
train_data = train_data.shuffle(60000).batch(batch_size)

# 4. W, b 초기화 함수 정의
#   어떤 방법으로 가중치 W와 편향 b를 초기화할지 정하는 함수.
#   평균 0, 편차 1을 갖는 가우스 분포에서 랜덤으로 뽑아 W, b를 초기화 함.
def random_normal_initializer_with_stddev_1():
    return tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None)

# 5. ANN 모델 정의
class ANN(tf.keras.Model):
    def __init__(self):
        super(ANN, self).__init__()
        self.hidden_layer_1 = tf.keras.layers.Dense(hidden1_size,
                                                    activation='relu',
                                                    kernel_initializer=random_normal_initializer_with_stddev_1(),
                                                    bias_initializer=random_normal_initializer_with_stddev_1())
        self.hidden_layer_2 = tf.keras.layers.Dense(hidden2_size,
                                                    activation='relu',
                                                    kernel_initializer=random_normal_initializer_with_stddev_1(),
                                                    bias_initializer=random_normal_initializer_with_stddev_1())
        self.output_layer = tf.keras.layers.Dense(output_size,
                                                    activation=None,
                                                    kernel_initializer=random_normal_initializer_with_stddev_1(),
                                                    bias_initializer=random_normal_initializer_with_stddev_1())

    def call(self, x):
        H1_output = self.hidden_layer_1(x)
        H2_output = self.hidden_layer_2(H1_output)
        logits = self.output_layer(H2_output)
        return logits

# 6. cross-entropy 손실함수 정의
#   분류 문제에 있어서는 MSE보다 cross-entropy가 더 성능이 좋다고 알려져있음.
@tf.function
def cross_entropy_loss(y_pred, y_actual):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_actual))

# 7. 옵티마이저 정의
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

# 8. 최적화 function 정의
#   여기서 model은 tf.keras.Model을 상속받은 모델. trainable_variables라는 속성을 사용할 수 있음.
@tf.function
def train_step(model, x, y_actual):
    with tf.GradientTape() as tape:
        y_pred = model.call(x)
        loss = cross_entropy_loss(y_pred, y_actual)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# 9. 모델 정확도 출력 함수 정의
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
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

# 10. 학습 수행
ANN_model = ANN()
for epoch in range(num_epochs):
    average_loss = 0.0
    total_batch = int(x_train.shape[0] / batch_size) # 1 epoch에서 수행되는 학습 횟수
    for batch_x, batch_y in train_data:
        train_step(ANN_model, batch_x, batch_y)
        current_loss = cross_entropy_loss(ANN_model.call(batch_x), batch_y)
        average_loss += current_loss / total_batch
    if epoch % display_step == 0:
        print("반복(Epoch): %d, 손실 함수(Loss): %f" % ((epoch + 1), average_loss))

# 11. 모델의 정확도 측정
print("정확도(Accuracy): %f" % compute_accuracy(ANN_model.call(x_test), y_test))