# CheckPointManager
# 생성한 모델의 parameter들을 disk에 저장할 수 있도록, 쉽게 불러오도록 돕는 API.
# 98 ~ 121  Line 참고.

import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train.astype("float32"), x_test.astype("float32")
x_train, x_test = x_train / 255.0, x_test / 255.0

y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train_one_hot))
train_data = train_data.repeat().shuffle(50000).batch(128)
train_data_iter = iter(train_data)

test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test_one_hot))
test_data = test_data.batch(1000)
test_data_iter = iter(test_data)

class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
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

@tf.function
def cross_entropy_loss(logits, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

optimizer = tf.optimizers.RMSprop(learning_rate=1e-3)

def train_step(model, x, y_actual, is_training):
    with tf.GradientTape() as tape:
        y_pred, logits = model(x, is_training=is_training)
        loss = cross_entropy_loss(logits, y_actual)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

@tf.function
def compute_accuracy(y_pred, y_actual):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_actual, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

CNN_model = CNN()

temp_input = tf.zeros((1, 32, 32, 3))
_ = CNN_model(temp_input, is_training=False)

SAVE_DIR = "./model"
ckpt = tf.train.Checkpoint(step=tf.Variable(0), model=CNN_model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, directory=SAVE_DIR, max_to_keep=5)
latest_ckpt = tf.train.latest_checkpoint(SAVE_DIR)

if latest_ckpt is not None:
    ckpt.restore(latest_ckpt).expect_partial()
    print(f"복구 성공! 현재 Step: {ckpt.step.numpy()}")
    y_batch_pred, logits = CNN_model(x_test, is_training=False)
    train_accuracy = compute_accuracy(y_batch_pred, y_test_one_hot)
    print("테스트 데이터 정확도(Accuracy): %f" % (train_accuracy))
    exit()

while int(ckpt.step) < (10000 + 1):
    batch_x, batch_y = next(train_data_iter)
    if ckpt.step % 100 == 0:
        ckpt_manager.save(checkpoint_number=ckpt.step)
        y_batch_pred, logits = CNN_model(batch_x, is_training=False)
        train_accuracy = compute_accuracy(y_batch_pred, batch_y)
        train_loss = cross_entropy_loss(logits, batch_y)
        print("반복(Epoch) %d, 정확도(Accuracy): %f, 손실함수(Loss): %f" % (ckpt.step, train_accuracy, train_loss))
    train_step(CNN_model, batch_x, batch_y, True)
    ckpt.step.assign_add(1)

test_accuracy = 0.0
for i in range(10):
    test_batch_x, test_batch_y = next(test_data_iter)
    pred_batch_y, logits = CNN_model(test_batch_x, is_training=False)
    test_accuracy += compute_accuracy(pred_batch_y, test_batch_y).numpy()
test_accuracy /= 10
print("정확도(Accuracy): %f" % test_accuracy) # 정확도(Accuracy): 0.721900