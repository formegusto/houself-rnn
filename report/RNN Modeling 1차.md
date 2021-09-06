# RNN Modeling

# 1. 데이터베이스 구조

> **General Household Data Structure**

![https://user-images.githubusercontent.com/52296323/128653444-e9eee492-59c5-49f3-a4a3-dc2ecafd4e16.png](https://user-images.githubusercontent.com/52296323/128653444-e9eee492-59c5-49f3-a4a3-dc2ecafd4e16.png)

![https://user-images.githubusercontent.com/52296323/128653450-1d75c2ec-e99e-46e5-90ce-730d7c597019.png](https://user-images.githubusercontent.com/52296323/128653450-1d75c2ec-e99e-46e5-90ce-730d7c597019.png)

- 가구별로 unique_id, uid, timeslot array 의 형태로 데이터베이스에 저장했다.

> **Data Structure After Merging Timeslot**

![https://user-images.githubusercontent.com/52296323/128653455-aa679110-aba8-4c3a-a4dc-382f382569b5.png](https://user-images.githubusercontent.com/52296323/128653455-aa679110-aba8-4c3a-a4dc-382f382569b5.png)

- 가구 하나를 클러스터링을 위해서 다음과 같이 96개의 타임슬롯을 4개씩 합쳐서 1시간 단위의 24개의 타임슬롯 배열로 가공했었다.

> **Data Structure After KMeans-Clustering**

![](https://user-images.githubusercontent.com/52296323/132186264-b6df0035-910d-49ee-9b24-07dab0d79ee1.png)

- 클러스터링 결과는 하나의 가구의 날짜별로 label로 붙여서 배열로 저장해놨었다.

# 2. Power Pattern Process Description

## Remind. 텍스트 생성기 인공신경망 프로세스

- 자연어 처리(텍스트 생성기)를 위한 학습 과정은 다음과 같았다.

  1. 데이터 구조 분석
  2. 데이터 전처리
     1. 토큰화
     2. 정수 인코딩
     3. 패딩
     4. 원-핫 벡터 변환
     5. 문장(input x) 에 대한 다음 단어(output y)를 분리
  3. 인공신경망 모델링

     > **일반적으로 자연어 처리를 위한 RNN Model은 다음과 같은 형태를 나타냈다.**

     > **입력층 - 임베딩층 - 은닉층 - 출력층**

  4. 테스트

> **자연어 처리는 숫자 처리와**는 다르게 **기계가 잘 이해**할 수 있도록 **토큰화, 정수인코딩, 원-핫 벡터와 같은 전처리**의 과정과 유사도를 판단하기 위해서 **임베딩층의 과정**을 거쳤다.

> **전력 패턴의 예측은 자연어 처리보다는 인공신경망 모델링이 더 수월할 것이라고 생각한다. 우선은 자연어 처리의 전처리 과정에서 쓰였던, 토큰화, 정수 인코딩, 원-핫 벡터 변환의 과정은 거치지 않아도 된다.**

> **하지만 단어를 벡터화 시켜서 서로 다른 단어의 유사도를 판단하는데 쓰이는 임베딩층의 출력인 임베딩 벡터는 쓰일 수도 있다고 생각을 한다. 전력은 엑셀 데이터 상의 전력 단위는 0.001이기 때문이다.**

# 3. 수요 예측**을 위한 RNN Modeling**

## 1. 2021. 08. 31. 1차 모델링 계획

> **클러스터링 라벨 결과를 포함하지 않은 인공신경망**

1. **훈련 데이터 구조**

   [Household Training Data Structure (non-cluster label)](https://www.notion.so/4ecd730e99404ba5a90213cd4745c692)

   - RNN의 의미를 알기 전 까지는 클러스터의 라벨을 학습의 출력결과로 사용하면 된다고 생각을 했었다. 하지만 전**력패턴 이라는 것은 이미 RNN의 관점에서 라벨을 가지고 있었다. ( 다음 시퀀스의 전력 값 )**
   - 계절, 요일, 날씨, 평균온도, 평균습도와 실질적인 시퀀스 데이터 X를 통해서 다음 y를 예측하는 RNN 훈련을 진행할 것 이다.
   - 계절, 요일, 날씨, 평균온도, 평균습도는 입력으로 사용될 시퀀스 데이터가 무조건 가지고 있어야 하는 필수 데이터이다.

   > 결과적으로, input으로 사용이 될, sequence list는 계절 번호, 날짜 번호, 날씨 번호, 평균온도, 평균습도 그리고 1시간 단위의 24개의 전력 패턴으로 구성 시킬 것이다.

   ```python
   [season_no, day_no, weather_no, avg_ta, avg_rhm, ...pattern_datas]
   ```

2. **훈련데이터 전처리 프로세싱**

   > **merging**

   ![](https://user-images.githubusercontent.com/52296323/132186298-6fbd8a54-b236-4215-a230-6a56d471d1b7.png)

   - 이전에 kmeans 에서 했던 것 처럼 1시간 단위의 24개 timeslot으로 병합한다.
   - 이 과정에서 변화가 별로 없는 패턴, 1자로 이어지는 아웃라이어들은 제거한다.

   > **To Integer Encoding From Weather Data**

   ```python
   # season_encoding_func
   def get_season_no(month):
       if month in [3,4,5]:
           return 1 # 봄
       elif month in [6,7,8]:
           return 2 # 여름
       elif month in [9,10,11]:
           return 3 # 가을
       elif month in [12,1,2]:
           return 4 # 겨울

   # day_encoding_logic
   wt_datas['day_no'] = [weather.weekday() + 1 for weather in wt_datas['date']]

   # weather_encoding_dict
   {
    '눈': 5,
    '박무': 4,
    '비': 3,
    '소나기': 7,
    '안개': 10,
    '안개비': 11,
    '연무': 2,
    '진눈깨비': 12,
    '채운': 8,
    '특이사항 없음': 1,
    '햇무리': 6,
    '황사': 9
   }
   ```

   - 계절에 대한 정수인코딩 처리는 시즌별 인덱스 부여
   - 날짜에 대한 정수인코딩 처리는 요일 부여

     → 후에 zero-padding 처리를 위해, 0은 속하지 않도록 구성

   - 날씨는 자연어 계열에 속하기 때문에, 일반적으로 자연어를 정수인코딩 할 때 사용하는 방법인, 빈도수 체크 후 오름차순 정렬 후 인덱스 부여

   ![](https://user-images.githubusercontent.com/52296323/132186334-ab5fa54e-d01b-438e-8f5c-8794b22ed59f.png)

   - 추가적으로 겨울이나 봄의 경우 영하 온도의 데이터를 가지고 있는데, 이는 훈련과정에서 에러를 일으키므로 전체적으로 제일 낮은 영하온도 만큼 올려준다.

   ```python
   # 영하 온도 전처리
   min_ta = min(wt_datas['avg_ta'])
   wt_datas['avg_ta'] += (min_ta * -1 + 1)
   min(wt_datas['avg_ta'])
   ```

   > **Set Padding Datas**

   ```
   Samples Before Padding Process
   [list([1, 2, 2, '20.4', '72.8', 0.341])
    list([1, 2, 2, '20.4', '72.8', 0.341, 0.337])
    list([1, 2, 2, '20.4', '72.8', 0.341, 0.337, 0.324]) ...
    list([1, 2, 6, '15.5', '52.8', 0.045, 0.044, 0.059, 0.058, 0.033, 0.054, 0.059000000000000004, 0.049, 0.04, 0.057999999999999996, 0.056, 0.03, 0.057999999999999996, 0.06, 0.038, 0.048, 0.062, 0.051000000000000004, 0.038, 0.06099999999999999, 0.059000000000000004, 0.03])
    list([1, 2, 6, '15.5', '52.8', 0.045, 0.044, 0.059, 0.058, 0.033, 0.054, 0.059000000000000004, 0.049, 0.04, 0.057999999999999996, 0.056, 0.03, 0.057999999999999996, 0.06, 0.038, 0.048, 0.062, 0.051000000000000004, 0.038, 0.06099999999999999, 0.059000000000000004, 0.03, 0.055999999999999994])
    list([1, 2, 6, '15.5', '52.8', 0.045, 0.044, 0.059, 0.058, 0.033, 0.054, 0.059000000000000004, 0.049, 0.04, 0.057999999999999996, 0.056, 0.03, 0.057999999999999996, 0.06, 0.038, 0.048, 0.062, 0.051000000000000004, 0.038, 0.06099999999999999, 0.059000000000000004, 0.03, 0.055999999999999994, 0.057999999999999996])]

   Tranining Sample Size : 8760
   Tranining Sample MAX_LEN : 29

   Final Samples
   [['0' '0' '0' ... '20.4' '72.8' '0.341']
    ['0' '0' '0' ... '72.8' '0.341' '0.337']
    ['0' '0' '0' ... '0.341' '0.337' '0.324']
    ...
    ['0' '0' '1' ... '0.06099999999999999' '0.059000000000000004' '0.03']
    ['0' '1' '2' ... '0.059000000000000004' '0.03' '0.055999999999999994']
    ['1' '2' '6' ... '0.03' '0.055999999999999994' '0.057999999999999996']]
   ```

   - 시간 단위로 예측할 것이기 때문에, 시간별로 리스트를 나누었었다.
   - 각기 크기가 다르기 때문에 시간을 같게 만들어주기 위하여 zero-padding 처리를 해준다.

   > **Set Training Datas**

   ```
   Input Data For Training
   [['0' '0' '0' ... '2' '20.4' '72.8']
    ['0' '0' '0' ... '20.4' '72.8' '0.341']
    ['0' '0' '0' ... '72.8' '0.341' '0.337']
    ...
    ['0' '0' '1' ... '0.038' '0.06099999999999999' '0.059000000000000004']
    ['0' '1' '2' ... '0.06099999999999999' '0.059000000000000004' '0.03']
    ['1' '2' '6' ... '0.059000000000000004' '0.03' '0.055999999999999994']]

   Output Data For Training
   ['0.341' '0.337' '0.324' ... '0.03' '0.055999999999999994'
    '0.057999999999999996']
   ```

   - 시간(t-1)에 대한 값들까지를 입력 X로 사용하고 다음 시간에 대한 출력값(Y, 실제값)을 분리한다.

   > 추가 본 one-hot vector

   - 자연어 처리에서만 필요할지 알았지만, 각 패턴의 결과인 어떠한 시간대의 y는 여러가지의 형태를 띈다. 이러한 선택지들을 머신러닝, 딥러닝에서는 클래스라고 부른다. 각각의 결과에 고유한 라벨링을 위해 one-hot vector로 변환한다.

   ```python
   one_hot_y = to_categorical(train_y,num_classes=value_size)
   one_hot_y
   ```

3. **훈련데이터 / 테스트데이터**

   - 후에 모델의 성능을 점검해보기 위한 테스트데이터를 계절별 10% 씩 때어내는 작업을 진행했다.

   ```python
   def get_season_no(month):
       if month in [3,4,5]:
           return 1 # 봄
       elif month in [6,7,8]:
           return 2 # 여름
       elif month in [9,10,11]:
           return 3 # 가을
       elif month in [12,1,2]:
           return 4 # 겨울

   test_merge_datas = pd.DataFrame();

   for i in range(1,2):
       filter_list = list(filter(lambda date: get_season_no(date.month) == i, merge_datas.columns))
       test_list_idx = list()
       while True:
           filter_data = filter_list[ran.randrange(0,len(filter_list))]
           if filter_data not in test_list_idx:
               test_list_idx.append(filter_data)

           if len(test_list_idx) >= (len(filter_list) * 15 / 100):
               break;
       test_merge_datas = pd.concat([test_merge_datas, merge_datas[test_list_idx]], axis=1)
       merge_datas.drop(test_list_idx, axis=1, inplace=True)

   test_merge_datas
   ```

4. **RNN Model**

   ```python
   model = Sequential()
   model.add(Embedding(value_size, 10, input_length=28))
   model.add(LSTM(128))
   model.add(Dense(value_size, activation='softmax'))
   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   ```

   - 임베딩층을 제거하면 훈련하는데에 소요시간이 오래걸려서 임베딩층을 넣어서 차원을 축소시켜 학습을 빠르게 할 수 있도록 했다.
   - 활성화 함수로는 소프트맥스 함수를 사용했고, 손실함수로는 여러가지 케이스 중에서 선택하는 categorical_crossentropy 함수를 사용하도록 구성했다. 그리고 가중치와 편향의 변화는 adam 함수가 담당한다.

# 4. RNN Model First Test

![](https://user-images.githubusercontent.com/52296323/132186386-b1a314bb-4a42-4d0c-8952-40d4c71579d8.png)

- 가중치 업데이트 횟수는 400회를 주었다. 그리고 정확도는 98%를 기록하면서 마무리 했다.

```python
for i in range(0, round(len(test_X) / 24)):
    start_idx = 24 * i
    end_idx = (24 * (i + 1))

    real_pattern = test_X[end_idx-1][-23:]
    real_pattern = np.append(real_pattern, [test_y[end_idx-1]])

    test_real.append(real_pattern)
#     print("real_pattern:",real_pattern)

    predict_pattern = test_X[start_idx+power_info][-power_info:].tolist()

    for p in range(start_idx + power_info, end_idx):
        result = model_2.predict_classes([test_X[p].tolist()])
        predict_pattern.append(y[result[0]])
    test_predict.append(predict_pattern)
```

![](https://user-images.githubusercontent.com/52296323/132186420-64335bc1-05ac-4400-8138-cdb41e68bca5.png)

![](https://user-images.githubusercontent.com/52296323/132186427-57eea32c-4a1f-4d88-89fc-a0eba6b7adab.png)

![](https://user-images.githubusercontent.com/52296323/132186434-5bfdf1b6-211a-4d85-86de-5eecaa75a27b.png)

![](https://user-images.githubusercontent.com/52296323/132186438-b63b030c-009b-46f3-9c17-44309555c117.png)

- 그리고 왼쪽부터 차례대로 날짜, 날씨정보는 고정시킨채 1시간 정도의 실제 측정 값을 주었을 때, 5시간, 10시간, 15시간과 같이 예측에 실제 전력 측정 값이 얼마나 주어지느냐에 따른 결과를 지켜봤다. 파란색은 실제값, 빨간색은 예측값을 뜻한다.

  → 보시다시피 1개만 주어진 데이터에서 겹치는 부분이 많이 없었기에 시도를 해봤다.

- 어느정도의 실제 측정 값이 주어지면, 월등하게는 아니지만 예측은 어느정도되는 것 같다.
- 하지만 완벽한 예측은 당연히 힘들었는데 아래와 같은 이유들을 생각해봤다.

  - 다양한 예측값들이 존재한다.
  - 훈련에 사용된 데이터의 갯수가 부족한 것일 수도 있다.
  - 자연어 처리를 기반으로 모델을 구성했다. ( 원-핫 인코딩, 임베딩층 등,, )

  → 숫자 시퀀스 데이터에 대한 RNN Model 구성, 활성화 함수, 옵티마이저, 손실 함수 등 더 정확히 알고 사용하는 것이 중요할 것 같다.
