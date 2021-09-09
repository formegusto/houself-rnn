# non-cluster-rnn-model vs cluster-rnn-model

- 해당 테스트는 같은 가구에서의 두 모델의 정확도 차이를 비교하기 위해 진행했다.
- 초기 훈련데이터 설정은 똑같지만 후에 각 모델에 맞는 전처리 과정에서 달라진다.
  - 3가지 이상의 전력값을 가진 패턴이 아니라면 제외시킨다. **(동일)**
  - 클러스터 패턴을 출력값으로 사용하는 cluster-rnn-model은 클러스터링 과정에서 제거 했었던 아웃라이어 패턴들을 제외시키고 훈련데이터를 구성한다.
- rnn-model의 설정은 완전히 똑같이 구성한다. 다만 **출력층의 뉴런수만 다르다.**

  ```python
  # non-cluster-rnn-model
  non_cluster_model = Sequential(name="rnn-model-1-101-1602")
  non_cluster_model.add(Embedding(value_size, 10, input_length=28))
  non_cluster_model.add(LSTM(128))
  non_cluster_model.add(Dense(value_size, activation='softmax'))
  non_cluster_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  # cluster-rnn-model
  cluster_model = Sequential(name="rnn-model-1-101-1602")
  cluster_model.add(Embedding(len(cluster_one_hot_y), 10, input_length=28))
  cluster_model.add(LSTM(128))
  cluster_model.add(Dense(2964, activation='softmax'))
  cluster_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  ```

- 테스트용 데이터셋은 같은 데이터셋을 사용하도록 한다.

# 1. Compare Visualization Plot

![Untitled](non-cluster-rnn-model%20vs%20cluster-rnn-model%2020ff9dc00e8949e7b91c0e17698da352/Untitled.png)

non-cluster-rnn-model

![Untitled](non-cluster-rnn-model%20vs%20cluster-rnn-model%2020ff9dc00e8949e7b91c0e17698da352/Untitled%201.png)

cluster-rnn-model

- 두 개의 모델의 결과물을 비교해봤을 때, 육안상으로는 결과가 반반해보였다. 첫 번째 모델이 좀 더 따라간 케이스도 있었고, 두 번째 모델이 패턴을 더 따라간 케이스도 있었다.

# 2. mean-distance, mean-similarity

- 육안상으로는 평가가 힘들어서, 이들을 평가해줄 수 있는 어떠한 지표가 필요했는데, 여기서 euclidean-distance와 cosine similarity를 이용해서 두 모델의 실제 데이터와 예측 데이터의 거리와 방향의 유사도의 평균을 비교하는 표를 구성해봤다.

![Untitled](non-cluster-rnn-model%20vs%20cluster-rnn-model%2020ff9dc00e8949e7b91c0e17698da352/Untitled%202.png)

- 해당 표의 power-info는 예측을 위해 해당 날짜에 실제 값이 어느정도 주어졌을 때의 정도를 나타낸다.

  1 → 1시간, 3 → 3시간 ...

- 그렇게 큰 차이는 보여주지 않았지만 클러스터링 라벨을 훈련 데이터에 썼던 쪽의 모델이 평균거리와, 평균방향성이 더 우월함을 보여줬다.
- 클러스터링의 결과를 훈련에 쓰인 쪽이, 개별 패턴 훈련보다 살짝 좋은 성능을 보여줬지만, 현재 rnn-model의 문제점과 함께 해당 방식이 후에 문제를 보일 수도 있다는 점을 아래에 정리한다.
  1. 현재 모델들은 자연어 처리에서의 rnn-modeling 방식을 사용했기 때문에, 숫자 시퀀스 데이터 예측에는 맞지 않는 모델일수 있다. → 학습 후에 재 설계
  2. 클러스터링을 사용한 rnn-model은 클러스터링 결과에 의존도가 높기 때문에, 후에 숫자 시퀀스 데이터 예측에 맞는 설계를 하면 위의 결과는 뒤집혀질수도 있다.
