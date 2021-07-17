# 프로그래머스 이미지 분석
###### <strong> 미술작품 이미지 분류(category 7 classes)

* 언어 : Python
* 프로그램 : Colab
* 소스코드 : efficientnet.ipynb
* 최종모델 : 2021-05-25_15-46-35-03-0.36.h5
* 사용한 모듈 : tensorflow, keras, matplotlib, datetime, efficientnet
* 주요 함수 :

```

 os.makedirs(path, exist_ok) : 폴더생성  datetime.now().strftime('%Y-%m-%d_%H-%M-%S') : 현재날짜시간출력
 ImageDataGenerator(rotation_range, width_shift_range, height_shift_range, horizontal_flip) : Data Augumentation를 위한 데이터 증강
 flow_from_directory(path, target_size, class_mode, seed, batch_size, subset) : 디렉토리 내 데이터 Load 후 ImageDataGenerator 적용
 EfficientNetB0(include_top, weights, input_shape) : efficientnet 모델 생성  Flatten() : 1차원 벡터화  Dense(units, activation, name) : FC 레이어 추가
 Dropout() : 드랍아웃  Model(Input, Output).compile(loss, optimizer, metrics) : 신경망 모델 인스턴스 생성
 EarlyStopping(monitor, mode, patience, verbose) : 모델조기종료  ModelCheckPoint(path, monitor, save_best_only, mode, verbose) : 베스트 모델 저장
 model.fit_generator(generator, steps_per_epoch, epochs, validation_data, validation_steps, callback) : 학습
 
 ```
 

----------------------------------------

### 데이터

* 데이터 정보

<img src = "https://user-images.githubusercontent.com/72690336/119535194-b7ce7c80-bdc2-11eb-8187-d270bb76082d.png" width="50%" height="50%">

-1698장(train 1361장, validation 337장으로 분리)
-label : 0~6 ['dog', 'elephant', 'giraffw', 'guitar', 'horse', 'house', 'person']

* 오늘의 모델 구조

<img src = "https://user-images.githubusercontent.com/72690336/119537719-4e9c3880-bdc5-11eb-9682-dc309266afe5.png" width="50%" height="50%">

  * Baseline으로 MNesNet 구조를 따른다.
  * 다른 구조에 비해 적은 수의 파라미터
  * 현재 이미지 분류 알고리즘 중 가장 좋은 성능을 보이는 모델

----------------------------------------

### CNN - EfficientNet

* 모델 성능
  * train : loss: 0.6839 - accuracy: 0.7820
  * validation : val_loss: 0.3589 - val_accuracy: 0.9139

-epochs의 진행에 따라 loss값을 기준으로 5회이상 향상되지 않으면 모델 조기종료 -> 8회차 종료 -> 3회차 best 모델 선정(2021-05-25_15-46-35-03-0.36.h5)
-validation의 loss와 accuracy가 성능이 더 높다. 과적합 문제는 없는 것으로 보임

* 학습 모델 그래프

<img src = "https://user-images.githubusercontent.com/72690336/119537252-d0d82d00-bdc4-11eb-982e-ddf20e5615e0.png" width="40%" height="40%">  <img src = "https://user-images.githubusercontent.com/72690336/119537335-e4839380-bdc4-11eb-96ae-f80461112d0b.png" width="40%" height="40%">

-epochs가 3을 넘어가는 순간부터 validation 그래프 역행 -> 계속 학습 진행 시 overfitting 발생 -> 조기종료


---------------------------------------------


### 개선할 점

* train 셋의 성능이 너무 낮은 단계에서 EarlyStopping 작용 -> ImageGenerator 옵션 일부 제거(일반화 축소)
* 흑백으로 변환했을 때 가시적으로 더 구분이 잘 되었다. 실제 분석에 적용 후 모델 비교
* 이미지 샤프닝 처리를 통해 그림의 특징을 더 강조한 후 분석 실시 -> 모델 향상에 도움이 되는지 비교
* 다른 알고리즘 사용 후 모델비교
* cross-validation을 통해 성능을 높일 수 있을까?
