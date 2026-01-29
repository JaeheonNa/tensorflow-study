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


# python2와  호환성을 맞추기 위한 모듈...?
from __future__ import absolute_import, division, print_function, unicode_literals

# google에서 만든 라이브러리로, tensorflow 2.0과 자주 묶여서 사용됨.
from absl import app
import tensorflow as tf

import numpy as np
import os
import time

# input data와 input data를 한 글자씩 뒤로 민 target data를 생성하는 utility 함수 정의.
def split_input_target(chunk):
    input_text = chunk[:-1] # 원본 input text
    target_text = chunk[1:] # 하나씩 뒤로 민 text
    return input_text, target_text

# 학습에 사용될 설정값 정의
data_dir = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
batch_size = 64
seq_length = 100 # RNN은 시계열 데이터를 다루기 때문에 기존에 사용하던 데이터와 비교해 '시간'이라는 차원이 하나 더 추가됨. 몇 개의 글자를 하나의 시계열로 볼 것인가?
embedding_dim = 256
hidden_size = 1024
num_epochs = 10

# 학습에 사용할 txt load
text = open(data_dir, 'rb').read().decode('utf-8')
voca = sorted(set(text)) # character 들을 뽑아서 집합으로 생성.
voca_size = len(voca)
char2idx = { c : i for i, c in enumerate(voca) }
idx2char = np.array(voca)