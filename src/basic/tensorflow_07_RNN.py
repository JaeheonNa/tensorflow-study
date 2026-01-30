# 순환 신경망(RNN)
#   시계열 데이터(현재의 데이터가 앞, 뒤 데이터와 연관 관계를 가지는 데이터)를 다루기에 최적화된 인공 신경망.
#   자연어 처리(NLP: Natural Language Processing) 문제에 주로 사용되는 인공신경망 구조.
#   그 외에도 주가, 파형으로 표현되는 음성 데이터 등이 있음.
#   ANN에서 은닉층에서 자기 자신으로 돌아가는 구간이 추가된 구조.
#   현재(t)의 결과가 다음(t+1)에 영향을 미치고, t+1의 결과가 그 다음(t+2)에 영향을 미치는 과정이 끊임없이 반복되는 구조.
#   이렇게 되면 이전 상태에 대한 정보를 일종의 메모리 형태로 저장할 수 있음.
# 경사도 사라짐 문제
#   오류 역전파 문제는 활성화 함수 중 relu를 사용함으로써 어느정도 해결이 가능했지만,
#   RNN의 경우는 조금 다른 측면에서 경사도 사라짐 문제가 있음.
#   t에서의 학습결과가 t+n번째의 학습들에 의해 점점 희석되는 것.
#   즉 장기 기억력을 가지지 못하다는 것.
#   이를 해결하기 위해 장/단기 기억 네트워크(LSTM: Long-Short Term Memory Network)가 제안됨.
# LSTM
#   은닉층의 각각의 노드를 인풋게이트, 포겟게이트, 아웃풋게이트로 구성된 메모리 블럭이라는 구조로 대체함.
#   포겟게이트를 통해 t-1의 학습결과를 받고, 인풋게이트를 통해 t의 학습결과를 받을 수 있을 때,
#   포겟게이트는 열고 인풋게이트는 닫을 경우, 과거의 데이터로 인한 학습결과가 현재의 데이터에 의해 희석되는 현상을 완화할 수 있음.
#   게이트는 완전이 닫힐 수도(0), 완전히 열릴 수도(1), 적절하게 열릴 수도(0~1) 있음. 최적화 대상임.
#   단, 아웃풋게이트까지 고려하면 연산량이 많아 무겁다는 단점이 존재함. 이를 해결하기 위해 GRU(Gate Recurrent Unit)이 제안 됨.
# GRU
#   아웃풋게이트를 제거한 경량화 버전의 LSTM.
# 임베딩(Embedding)
#   자연어 처리 문제를 다룰 때 널리 사용되는 기법.
#   One-hot Encoding은 자연어를 처리기 부적합 함.
#   가령 10,000개의 단어에 대해서 One-hot Encoding을 하게 되면, 하나의 단어를 표현하는 데이터가 10,000 x 1이 되며, 해당 벡터의 값은 9,999개가 0이고 1이 한 개로 이뤄질 것임.
#   이를 Sparse(희박)하다라고 표현하며, 낭비되는 표현력이 많음.
#   임베딩은 Sparse한 데이터를 Dense하게 변환하는 기법.
#   원본 데이터(One-hot Encoding)에 Dense한 임베딩 행렬을 곱하는 방식임.
#   가령 10,000 x 1 형태의 원본데이터에 10,000 x 250 형태의 임베딩 행렬을 곱하면, 250 x 1형태의 Dense한 데이터를 뽑아낼 수 있음.
#   좋은 임베딩 행렬을 곱해서 나온 임베딩 벡터는 그 형태 자체도 의미를 가짐. (비슷한 단어에 대한 임베딩 벡터는 서로 비슷하게 생김)
# Char-RNN
#   Language Modeling이란, 어떤 단어가 왔을 때, 그 다음에 어떤 단어가 올지를 맞추는 작업인데,
#   Char-RNN의 경우는 그 단위가 '단어'가 아니라 '문자'가 됨.
#   알파벳 자모 26개에 대한 One-hot Encoding 데이터에 대해 임베딩을 수행한 후 RNN 모델에 투입,
#   뽑아낸 데이터에 대해서 argmax를 통해 '그 다음에 올 확률이 가장 높은 문자'의 One-hot Encoding 데이터에 대해 다시 임베딩을 수행한 후 RNN 모델에 투입,, 반복하는 방식.


# google에서 만든 라이브러리로, tensorflow 2.0과 자주 묶여서 사용됨.
from absl import app
import tensorflow as tf

import numpy as np
import os
import time

# input data와 input data를 한 글자씩 뒤로 민 target data를 생성하는 utility 함수 정의.
def split_input_target(chunk):
    # ex) "Hello -> input_text = "Hell", target_text = "ello"
    input_text = chunk[:-1] # 원본 input text
    target_text = chunk[1:] # 하나씩 뒤로 민 text
    return input_text, target_text

# 학습에 사용될 설정값 정의
data_dir = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
batch_size = 64
seq_length = 100 # RNN에게 한 번에 보여줄 문장의 길이. 너무 길면 학습이 어렵고, 너무 짧으면 문맥 파악을 못 함. 여기서는 100글자를 보고 101번째를 맞추게 하겠다는 뜻.
embedding_dim = 256
hidden_size = 1024
num_epochs = 10

# 학습에 사용할 txt load
text = open(data_dir, 'rb').read().decode('utf-8')
# character 들을 뽑아서 집합으로 생성.
voca = sorted(set(text))
voca_size = len(voca)
char2idx = { c : i for i, c in enumerate(voca) }
idx2char = np.array(voca)

# 텍스트의 char를 Integer로 치환 후 배치 단위로 데이터 세트를 생성.
text_as_int = np.array([char2idx[c] for c in text])
# 숫자 리스트를 '스트림(Stream)' 형태로 변경
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
# 스트림에서 흘러나오는 글자들을 101개씩(100+1) 묶음.
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
# 101개짜리 묶음(청크)마다 앞서 정의한 함수(split_input_target)를 적용
dataset_source = sequences.map(split_input_target)

class RNN(tf.keras.Model):
    def __init__(self, batch_size):
        super(RNN, self).__init__()
        # 입력값을 의미를 갖는 벡터 형태로 변경.
        self.embedding_layer = tf.keras.layers.Embedding(voca_size, embedding_dim)
        # RNN 수행
        self.hidden_layer_1 = tf.keras.layers.LSTM(hidden_size, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform')
        # 추상화
        self.output_layer = tf.keras.layers.Dense(voca_size)

    def call(self, x):
        embedded_input = self.embedding_layer(x)
        features = self.hidden_layer_1(embedded_input)
        logits = self.output_layer(features)
        return logits

def sparse_cross_entropy_loss(labels, logits):
    #이 때 labels는 one-hot encoding이 아님. Integer encoding이며, sparse_categorical_crossentropy가 one-hot encoding을 해줌.
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))

optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(model, inputs, targets):
    with tf.GradientTape() as tape:
        logits = model(inputs)
        loss = sparse_cross_entropy_loss(targets, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def generate_text(model, start_string):
    # 생성할 character의 수.
    num_sampling = 4000

    # start_string을 integer 형태로 변환.
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # 생성된 string을 저장할 배열 초기화
    text_generated = []

    # temperature가 높으면 더욱 다양한 텍스트를, 낮으면 더욱 정확한 텍스트를 생성함.
    temperature = 1.0

    model.hidden_layer_1.reset_states()
    for i in range(num_sampling):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    file = open("sampling_rnn_result.txt", "w")
    file.write(start_string + "".join(text_generated))
    file.close()
    return (start_string + "".join(text_generated))

def main(_):
    RNN_model = RNN(batch_size=batch_size)

    # 하나의 데이터를 뽑아서 데이터의 문제 여부를 확인. sanity check.
    for input_example_batch, target_example_batch in dataset_source.batch(batch_size).take(1):
        example_batch_predictions = RNN_model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    # 모델 정보 출력
    RNN_model.summary()

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5")

    for epoch in range(num_epochs):
        start = time.time()
        # 10000분의 1의 확률로 하나씩 뽑아 줄세움(섞는 과정). 이후 batch_size 크기로 묶음.
        epoch_dataset = dataset_source.shuffle(10000).batch(batch_size, drop_remainder=True)
        hidden = RNN_model.hidden_layer_1.reset_states()
        for (batch_n, (input, target)) in enumerate(epoch_dataset):
            loss = train_step(RNN_model, input, target)
            if batch_n % 100 == 0:
                template = "Epoch {}, Batch {}, Loss: {:.4f}"
                print(template.format(epoch + 1, batch_n, loss))
        if(epoch+1) % 5 == 0:
            RNN_model.save_weights(checkpoint_prefix.format(epoch=epoch))

        print("Epoch {}, Loss: {:.4f}".format(epoch + 1, loss))
        print("Time taken for 1 epoch {} sec\n".format(time.time() - start))

    last_checkpoint_path = checkpoint_prefix.format(epoch=epoch)
    RNN_model.save_weights(last_checkpoint_path)
    print("트레이닝이 끝났습니다.")

    # 1. 샘플링용 모델 생성 (배치 사이즈 1)
    sampling_RNN_model = RNN(batch_size=1)

    # 2. [핵심] build() 대신 가짜 데이터를 한 번 흘려보내서 '진짜' 빌드를 합니다.
    # 이렇게 하면 내부 레이어들이 가중치를 받을 준비를 완벽히 마칩니다.
    dummy_input = tf.ones((1, 1), dtype=tf.int64)
    sampling_RNN_model(dummy_input)

    # 3. 이제 가중치를 로드합니다. (경로 재확인)
    last_checkpoint_path = checkpoint_prefix.format(epoch=num_epochs - 1)
    sampling_RNN_model.load_weights(last_checkpoint_path)

    # 4. 제대로 로드되었는지 파라미터 수를 확인합니다. (0이 아니어야 함)
    sampling_RNN_model.summary()

    print("샘플링을 시작합니다.")
    # 시작 문구도 공백보다는 셰익스피어 스타일로 주면 좋습니다.
    print(generate_text(sampling_RNN_model, 'ROMEO: '))

if __name__ == '__main__':
    app.run(main)