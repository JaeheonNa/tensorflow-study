import tensorflow as tf

# 0. 선형회귀 모델(Wx + b)의 W값과 b값을 정의함.
#   이 경우 random.normal distribution 선택. (random.normal distribution: 정규분포. 가우스 분포, 가우시안 분포라고도 함. )
#   Normal Distribution(정규 분포)에서 랜덤한 값을 뽑아내서 초기값을 할당하는 로직임.
#   1 ~ 9 분위가 존재할 때, 정규분포이므로, 5분위 값이 초기값으로 선택될 확률이 가장 높은 것임.
#   shape에는 파라미터의 shape을 지정함. 이 경후 선형회귀모델이므로 x값을 투입했을 때, y값이 나오는 형태이므로 차원을 1로 지정.
W = tf.Variable(tf.random.normal(shape=[1]))
b = tf.Variable(tf.random.normal(shape=[1]))

# 1. 가설 정의 (모델 정의)
@tf.function
def linear_model(x):
    return W*x + b

# 2. 손실 함수 정의
#   여기에서는 MSE로 정의함.
@tf.function
def mse_loss(y_pred, y_actual):
    # Mean Squared Error
    # 손실 제곱들의 평균을 반환함.
    return tf.reduce_mean(tf.square(y_pred - y_actual))

# 3. 최적화
#   SGD(Stochastic Gradient Descent): 최적화 알고리즘을 갖고 있는 클래스 중 하나.
#   Mini-Batch Gradient Descent를 수행해주는 가장 기본적인 옵티마이저.
#   전체 데이터 세트 대신 무작위로 선택된 하나의 데이터 샘플 또는 작은 배치(Mini-Batch)를 사용해 매개변수를 업데이트하는 방식.
#   Adam, RMSprop 옵티마이저의 경우 부분 최적화 문제를 어느 정도 해결할 수 있는, 조금 더 발전된 옵티마이저임.
optimizer = tf.optimizers.SGD(learning_rate=0.01)

#   최적화 function 정의
#   train_step: 한 번의 gradient descent를 수행해주는 함수.
@tf.function
def train_step(x, y_actual):
    # with문이 끝나면 tape 안에는 입력부터 loss까지 가는 '연산 그래프(Graph)'가 저장(녹화)됨.
    with tf.GradientTape() as tape:
        y_pred = linear_model(x)
        loss = mse_loss(y_pred, y_actual)
    gradients = tape.gradient(loss, [W, b])      # 녹화한 내용을 바탕으로, loss(그래프 형태)를 W와 b에 대해 미분. W의 경사와 b의 경사를 구함.
    optimizer.apply_gradients(zip(gradients, [W, b]))    # 실제 여기서 W, b값의 갱신이 일어남.

# 4. 학습
x_train = [1, 2, 3, 4]
y_train = [2, 4, 6, 8]

# 1000번 학습을 수행함.
for _ in range(1000):
    train_step(x_train, y_train)

# 5. 테스트
x_test = [3.5, 5, 5.5, 6]
#   예상값: [7, 10, 11, 12]
#   @tf.function이 적용된 함수는 tensor타입을 리턴함으로,
#   numpy()를 적용해야 로그에 찍힘.
print(linear_model(x_test).numpy())