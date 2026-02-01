# 드롭아웃(Drop out)
#   오버피팅(overfitting)을 막기 위한 regularization 기법 중 하나.
#   인공 신경막의 각 레이어에서 사용되는 노드들 중 일부를 랜덤으로 작동하지 않도록 하는 기법.
#   더 적은 노드를 사용하기 때문에 autoEncoder 작용 원리와 같이 데이터의 추상화를 더 강하게 함.
#   드롭할 노드는 학습 때마다(1 train step마다) 랜덤하게 변경됨.
#   단 학습 과정에서는 드롭아웃을 수행하지만, 검증 과정에서는 드롭아웃을 수행하지 않음.

import tensorflow as tf

# 1. CIFAR-10 데이터(예제 데이터) 구성
#   [50000, 32, 32, 3] 형태의 데이터. 즉 32 x 32 사이즈의 컬러 이미지 5만 장.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train.astype("float32"), x_test.astype("float32")
x_train, x_test = x_train / 255.0, x_test / 255.0
# Scalar 형태의 레이블(0~9)을 One-hot encoding 형태로 변환
# squeeze = 불필요한 차원을 줄여주는 api.
#   ex) [[[0, 1]], [[1, 0]]] --> [[0, 1], [1, 0]]
#       (2, 1, 2) 인 shape에서 axis=1을 squeeze하면 (2,2)로 변경되는 식.
# expend_dim = 차원을 늘려주는 api. squeeze와 반대.
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

# 2. 학습 데이터 준비
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train_one_hot))
train_data = train_data.repeat().shuffle(50000).batch(128)
train_data_iter = iter(train_data)

test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test_one_hot))
test_data = test_data.batch(1000)
test_data_iter = iter(test_data)

# 3. CNN 모델 정의
class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        # 이 모델에서 학습의 대상이 되는 파라미터가 바로 filter.
        # Conv2D: 아래의 경우 5x5 픽셀 크기의 필터가 1픽셀씩 이미지를 필터링하며, 그 결과로 축소된 이미지들을 원본 이미지의 크기와 맞추기 위해 패딩을 넣음.
        #         padding 은 same 대신 valid를 넣을 수 있는데, 이 경우 축소된 이미지에 1픽셀씩만 패딩을 추가하게 됨.
        # MaxPool2D: 이미지를 2x2 크기의 필터로 차원을 축소함. 즉 길이 기준으로는 1/2 크기로, 넓이 기준으로는 1/4 크기로 축소. 가장 큰 값을 기준으로 축소됨.
        self.conv_layer_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides=1, padding="same", activation='relu')
        self.pool_layer_1 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2)

        self.conv_layer_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides=1, padding="same", activation='relu')
        self.pool_layer_2 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2)

        self.conv_layer_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", activation='relu')
        self.conv_layer_4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", activation='relu')
        self.conv_layer_5 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", activation='relu')

        self.flatten_layer = tf.keras.layers.Flatten()
        self.fc_layer_1 = tf.keras.layers.Dense(units=384, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)

        self.output_layer = tf.keras.layers.Dense(10, activation=None)

    def call(self, x, is_training):
        h_conv1 = self.conv_layer_1(x)
        h_pool1 = self.pool_layer_1(h_conv1)
        h_conv2 = self.conv_layer_2(h_pool1)
        h_pool2 = self.pool_layer_2(h_conv2)
        h_conv3 = self.conv_layer_3(h_pool2)
        h_conv4 = self.conv_layer_4(h_conv3)
        h_conv5 = self.conv_layer_5(h_conv4)
        h_pool2_flatten = self.flatten_layer(h_conv5)
        f_fc1 = self.fc_layer_1(h_pool2_flatten)
        h_fc1_dropout = self.dropout(f_fc1, training=is_training) # training일 때만 동작 함.
        logits = self.output_layer(h_fc1_dropout)
        y_pred = tf.nn.softmax(logits)
        return y_pred, logits

# 4. cross-entropy 손실 함수 정의
@tf.function
def cross_entropy_loss(logits, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

# 5. 옵티마이저 정의
#   경사하강법을 수행하는 optimizer. 1e-4 = 0.0001
optimizer = tf.optimizers.RMSprop(learning_rate=1e-3)

# 6. 최적화를 위한 function 정의
def train_step(model, x, y_actual, is_training):
    with tf.GradientTape() as tape:
        y_pred, logits = model.call(x, is_training)
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
        y_batch_pred, logits = CNN_model.call(batch_x, False)
        train_accuracy = compute_accuracy(y_batch_pred, batch_y)
        train_loss = cross_entropy_loss(logits, batch_y)
        print("반복(Epoch) %d, 정확도(Accuracy): %f, 손실함수(Loss): %f" % (i, train_accuracy, train_loss))
    train_step(CNN_model, batch_x, batch_y, True)

# 9. 학습 종료 후 학습된 모델의 정확도 출력.
test_accuracy = 0.0
for i in range(10):
    test_batch_x, test_batch_y = next(test_data_iter)
    pred_batch_y, logits = CNN_model.call(test_batch_x, False)
    test_accuracy += compute_accuracy(pred_batch_y, test_batch_y).numpy()
test_accuracy /= 10
print("정확도(Accuracy): %f" % test_accuracy) # 정확도(Accuracy): 0.713500