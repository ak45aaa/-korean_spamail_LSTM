import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# 샘플 데이터 (0: 일반 메시지, 1: 스팸 메시지)
messages = pd.read_csv('test_csv.csv', index_col=False)
messages.columns =['Text', 'Label']
labels = messages['Label']

# 토크나이저 생성 및 텍스트 인덱싱
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(messages['Text'])  # 'Text' 컬럼만 사용하여 토크나이저 학습
sequences = tokenizer.texts_to_sequences(messages['Text'])  # 'Text' 컬럼만 사용하여 시퀀스 생성token

# 시퀀스 데이터 패딩 (길이를 맞춰주기 위함)
max_sequence_length = max(len(seq) for seq in sequences)
sequences_padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_length)

# 레이블을 넘파이 배열로 변환
labels_np = np.array(labels)

# LSTM 모델 구성
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=256, input_length=max_sequence_length),
    tf.keras.layers.LSTM(256),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습
model.fit(sequences_padded, labels_np, epochs=10, batch_size=2)

# 새로운 메시지 입력 및 분류
new_message = ["앙"]
new_sequence = tokenizer.texts_to_sequences(new_message)
new_sequence_padded = tf.keras.preprocessing.sequence.pad_sequences(new_sequence, maxlen=max_sequence_length)
prediction = model.predict(new_sequence_padded)

for i in prediction:
    if i[0] >= 0.5:
        print("스팸 메시지입니다.")
    else:
        print("일반 메시지입니다.")