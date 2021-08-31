# RNN Example Description

# 1. 데이터 수집

[Text Data](https://www.notion.so/3f6dfb8a5ce74b3ba40418b6c3b03e98)

# 2. 데이터 이해

- 11개의 샘플로 이루어져 있으며, X의 입력값과 y의 출력값을 가진다.

# 3. 데이터 전처리

> **Tokenizing**

```python
t = Tokenizer()
t.fit_on_texts([text])
vocab_size = len(t.word_index) + 1
# 케라스 토크나이저의 정수 인코딩은 인덱스가 1부터 시작하지만,
# 케라스 원-핫 인코딩에서 배열의 인덱스가 0부터 시작하기 때문에
# 배열의 크기를 실제 단어 집합의 크기보다 +1로 생성해야하므로 미리 +1 선언
print('단어 집합의 크기 : %d' % vocab_size)
print(t.word_index)
'''
{'말이': 1, '경마장에': 2, '있는': 3, '뛰고': 4, '있다': 5, '그의': 6, '법이다': 7, '가는': 8, '고와야': 9, '오는': 10, '곱다': 11}
'''
```

- 텍스트를 토큰화 하고, 각 단어들에 인덱스를 부여한다.

> **Integer Encoding**

```python
sequences = list()
for line in text.split('\n'): # Wn을 기준으로 문장 토큰화
    encoded = t.texts_to_sequences([line])[0]
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)

print('학습에 사용할 샘플의 개수: %d' % len(sequences))
print(sequences)
'''
[[2, 3], [2, 3, 1], [2, 3, 1, 4], [2, 3, 1, 4, 5], [6, 1], [6, 1, 7], [8, 1], [8, 1, 9], [8, 1, 9, 10], [8, 1, 9, 10, 1], [8, 1, 9, 10, 1, 11]]
'''
```

- 텍스트를 토큰화 하고, 빈도수에 따라 인덱스를 부여해줬었다. 이를 가지고 문장안의 단어 배치를 부여된 인덱스의 배열로 구성을 한다.

> **Padding**

```python
max_len=max(len(l) for l in sequences) # 모든 샘플에서 길이가 가장 긴 샘플의 길이 출력
print('샘플의 최대 길이 : {}'.format(max_len))
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
print(sequences)
'''
[[ 0  0  0  0  2  3]
 [ 0  0  0  2  3  1]
 [ 0  0  2  3  1  4]
 [ 0  2  3  1  4  5]
 [ 0  0  0  0  6  1]
 [ 0  0  0  6  1  7]
 [ 0  0  0  0  8  1]
 [ 0  0  0  8  1  9]
 [ 0  0  8  1  9 10]
 [ 0  8  1  9 10  1]
 [ 8  1  9 10  1 11]]
'''
```

- 전체적으로 길이가 다른 배열들의 길이를 맞춰준다.

> **Separate Last Word**

```python
sequences = np.array(sequences)
X = sequences[:,:-1]
y = sequences[:,-1]
print(X)
'''
[[ 0  0  0  0  2]
 [ 0  0  0  2  3]
 [ 0  0  2  3  1]
 [ 0  2  3  1  4]
 [ 0  0  0  0  6]
 [ 0  0  0  6  1]
 [ 0  0  0  0  8]
 [ 0  0  0  8  1]
 [ 0  0  8  1  9]
 [ 0  8  1  9 10]
 [ 8  1  9 10  1]]
'''
print(y)
'''
[ 3  1  4  5  1  7  1  9 10  1 11]
'''
```

- 각 배열의 마지막 요소를 분리한다. 그리고 각 배열에 따른 다음 단어 레이블(label)로 때어낸 마지막 요소를 사용한다.

> **One-hot encoding**

```python
y = to_categorical(y, num_classes=vocab_size)
print(y)
'''
[[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
'''
```

# 4. 인공 신경망 디자인

```python
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_len-1)) # 레이블을 분리하였으므로 이제 X의 길이는 5
model.add(SimpleRNN(32))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

- 자연어 처리에서의 RNN Model은 기본적으로 입력층, 임베딩층(투사층), 은닉층, 출력층의 구성을 가진다.
- 임베딩층(투사층)에서는 lookup table이라고 부르는 초기 생성시에는 각 단어에 따른 랜덤한 행렬들이 존재한다.

  → 입력 데이터 차원 \* 입베딩층 차원의 곱으로, 해당 단어의 고유 임베딩 벡터를 생성해준다.

  → 자연어 처리에서는 해당 층이 있어, 유사 단어들을 판단할 수 있게된다.

- 입력으로 사용되는 X의 차원은 5, 임베딩층의 차원은 10, 그리고 은닉 상태의 크기는 32가 된다.
- 각 은닉상태의 활성화 함수로는 softmax 함수를 사용하였다.
- 출력층으로는 단어의 개수만큼 출력을 뽑아내어, 각 단어들이 등장할 확률로 나타나게 했고, 오차를 구하기 위한 함수로는 categorical_crossentropy, 가중치와 편향을 맞추어 가기 위한 함수는 adam을 사용했다.

# 5. 훈련

```python
model.fit(X, y, epochs=200, verbose=2)
```

# 6. 테스팅 및 활용

```python
def sentence_generation(model, t, current_word, n): # 모델, 토크나이저, 현재 단어, 반복할 횟수
    init_word = current_word # 처음 들어온 단어도 마지막에 같이 출력하기위해 저장
    sentence = ''
    for _ in range(n): # n번 반복
        encoded = t.texts_to_sequences([current_word])[0] # 현재 단어에 대한 정수 인코딩
        encoded = pad_sequences([encoded], maxlen=5, padding='pre') # 데이터에 대한 패딩
        result = model.predict_classes(encoded, verbose=0)
    # 입력한 X(현재 단어)에 대해서 Y를 예측하고 Y(예측한 단어)를 result에 저장.
        for word, index in t.word_index.items():
            if index == result: # 만약 예측한 단어와 인덱스와 동일한 단어가 있다면
                break # 해당 단어가 예측 단어이므로 break
        current_word = current_word + ' '  + word # 현재 단어 + ' ' + 예측 단어를 현재 단어로 변경
        sentence = sentence + ' ' + word # 예측 단어를 문장에 저장
    # for문이므로 이 행동을 다시 반복
    sentence = init_word + sentence
    return sentence
```
