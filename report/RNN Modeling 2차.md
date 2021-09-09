# RNN Modeling 2차

# 1. 1차 RNN Modeling의 문제점

- 자연어 처리 학습에 적합한 RNN Process를 선택했다.
- 활성화 함수, 손실 함수, 옵티마이저의 선택이 잘 못 됐다.
- **하지만 그 전에 근본적인 문제는 각 패턴에 따른 전력들의 다음 시간대 전력을 예측하는데에 고려하는 케이스가 너무 많다는 것 이다.**

# 2. 문제에 대한 해결책 제시

> **각 패턴에 따른 전력들의 다음 시간대 전력을 예측하는데에 고려하는 케이스가 너무 많다는 문제점**

- 여기에 정규화 등등을 이용할 수도 있겠지만, 해당 문제점에 클러스터링 결과물을 이용해보려고 한다.
- 우리는 앞서 kmeans-euclidean-cosine 연구를 통해 클러스터링 결과물을 DB에 저장을 해놨었다. 이를 이용하여 전처리를 한다.

# 3. 데이터 구조 설계

[Household Training Data Structure (cluster label)](https://www.notion.so/db71ba78f5694e779a2e9b242209ea46)

> **Any Household Pattern**

```python
[179., 172., 178., 177., 184., 191., 238., 294., 381., 242., 156., 151., 206., 135., 146., 136., 124., 136., 127., 134., 142., 131., 183., 126.]
```

> **Clustering Pattern**

```python
[150, 144, 153, 141, 147, 137, 137, 267, 438, 412, 296, 290, 252, 315, 236, 283, 386, 481, 432, 426, 406, 382, 286, 178]
```

- 위와 같이 하나의 가구의 어떠한 날짜에 대한 패턴이 존재할 때, 해당 패턴은 클러스터링 결과로 아래와 같은 클러스터링 패턴을 가지고 있었다고 가정한다.

> **Training Data Pattern**

```python
[ 179., 150 ]
[ 179., 144 ]
[ 179., 172., 153 ]
[ 179., 172., 178., 141 ]
# ...
```

- 그러면 위와 같이 **[ 하나의 가구에 주어진 시간에 따른 패턴 + 다음 시간대의 클러스터링 결과 라벨 ]** 의 구성으로 훈련 데이터를 구성한다.

# 4. 데이터 전처리

- 데이터 전처리는 위와 같은 배열을 가지고 다음 순서대로 진행한다. ( 구성만 다를 뿐, 전처리 과정은 이전 RNN Modeling과 똑같기 때문에 상세 내용은 생략한다. )

1. **[ 계절 + 요일 + 온도 + 습도 + 새로 구성한 훈련데이터 패턴 ]** 의 훈련 벡터 구성
   - 해당 과정에서 계절의 정수인코딩, 음수 온도 처리와 같은 과정들이 들어간다.
2. **벡터로 구성된 행렬의 차원을 일정하게 하기 위한 zero-padding 처리**

   ```
   [[  0.   0.   0. ...  31.  73. 365.]
    [  0.   0.   0. ...  73. 365. 340.]
    [  0.   0.   0. ... 365. 340. 353.]
    ...
    [  0.   0.   1. ... 325. 164. 167.]
    [  0.   1.   2. ... 164. 167. 145.]
    [  1.   2.   6. ... 167. 145. 141.]]
   ```

3. 훈련 벡터 안에서 입력 벡터와, 출력 벡터 (real result) 구분

   ```python
   train_X = training_samples[:,:-1]
   train_y = training_samples[:,-1]
   ```

4. categorical_crossentropy 처리를 위한 출력 벡터 one-hot encoding

   ```python
   [[0., 0., 0., ..., 0., 0., 0.],
    [0., 0., 0., ..., 0., 0., 0.],
    [0., 0., 0., ..., 0., 0., 0.],
    ...,
    [0., 0., 0., ..., 0., 0., 0.],
    [0., 0., 0., ..., 0., 0., 0.],
    [0., 0., 0., ..., 0., 0., 0.]]
   ```

# 5. 인공신경망 모델링

- 후에 각 날짜별 패턴 학습 모델과 결과물을 비교하기 위해 똑같이 설계한다.
- 입력층 → 임베딩층 → 은닉층 → 출력층
- activation function - softmax function
- loss function - categorical_crossentropy
- optimizer - adam
- 출력층의 뉴런 개수는 해당 모델의 은닉층에서 뽑아내는 출력의 개수와 이전 모델의 은닉층 출력 개수가 다르므로, 다르게 설정한다. (에러난다)

```python
cluster_model = Sequential(name="rnn-model-1-101-1602")
cluster_model.add(Embedding(len(cluster_one_hot_y), 10, input_length=28))
cluster_model.add(LSTM(128))
cluster_model.add(Dense(2964, activation='softmax'))
cluster_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

cluster_model.summary()
```

# 6. RNN Model Second Test

![Untitled](RNN%20Modeling%202%E1%84%8E%E1%85%A1%20729fd0ac0de347a0910ec214902cf39b/Untitled.png)

non-cluster RNN Model

![Untitled](RNN%20Modeling%202%E1%84%8E%E1%85%A1%20729fd0ac0de347a0910ec214902cf39b/Untitled%201.png)

cluster RNN Model

- 좌측은 첫번째 모델링했던 개별 패턴을 학습시킨 것이고, 우측은 이번에 새로 모델링한 클러스터링 라벨을 출력값으로 해서 학습시킨 결과이다.
- 2개는 다른 가구의 결과물이어서 결과물 자체는 다르지만 패턴이 비슷한가를 놓고 봤을 때는 우측이 어느정도 따라갔다고 육안상으로 확인할 수 있었다.

# 7. non-cluster-rnn-model vs cluster-rnn-model
