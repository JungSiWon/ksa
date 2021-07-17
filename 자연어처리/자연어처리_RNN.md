# 자연어처리
###### <strong> RNN을 이용한 자연어처리

* 언어 : Python
* 프로그램 : Colab
* 소스코드 : 데이터전처리.ipynb, RNN_SMS.ipynb, data_loader.py
* 데이터 : sms.tsv
* 사용한 모듈 : torch, torchvision, torchtext, matplotlib, pandas
* 주요 클래스 :

```

 DataLoader(object)  RNN(nn.module)  ComputeAccr(dloader,imodel)
 
 ```
 

----------------------------------------

### 데이터

* 데이터 정보

<img src = "https://user-images.githubusercontent.com/72690336/126024309-bc7fd091-198a-467e-97fc-c28ba7d51227.png" width="50%" height="50%">

-tsv 파일(tab으로 구분된 text파일)
-5574 row
-label : 'ham', 'spam'
-sms : 가사 최대 256글자로 설정후 slice

* 오늘의 모델 구조

<img src = "https://user-images.githubusercontent.com/72690336/126024418-60e37a32-95cf-4994-b289-c4ab0ce2a11d.png" width="70%" height="70%">

  * Embedding layer à LSTM à Linear(fc) layer à Softmax
  * 앞 뒤 데이터와 관련성 있는 데이터에 적합한 RNN구조 사용
  * Embedding layer
 
 <img src = "https://user-images.githubusercontent.com/72690336/126024963-3209836a-81f4-4fbb-95c3-b86ad4c56d47.png" width="40%" height="40%"> <img src = "https://user-images.githubusercontent.com/72690336/126024990-6adc25b5-dc02-4b9f-b141-838ed2fa40b7.png" width="40%" height="40%">



----------------------------------------

### 주요 클래스 및 모듈

* data_loader.py
  * 오픈소스
  * 데이터 로드 후 train, valid 셋 분리
  * 학습시킬 때 batch_size만큼 끊어서 로드 
  * 조건
    * 두 개의 필드로 구성된 데이터
    * TAB으로 구분되어 있는 데이터

* RNN
  * Embedding layer
  * LSTM 4 층
  * activation function : Log Softmax
    * LogSoftmax + NLLLOSS instead of Softmax + CrossEntropy
 
* ComputeAccr
  * y와 pred 비교를 통해 accuracy 산정
 
 
### 분석 절차
 1. 데이터 전처리
   * 클래스 파악
   * 중복 제거
   * sms 최대 256 글자 slice
   * shuffle
   * train, test 분리
 
 2. 데이터 로드
   * 하이퍼파라미터 세팅
   * DataLoader를 사용하여 train 데이터 train, valid로 나눠서 load
   * DataLoader를 사용하여 test 데이터 load
 
 3. 데이터 분석
   * 모델 선언
   * loss, optimizer 정의 - nn.NLLLoss, Adam
   * 학습
   * 테스트 
 
 
### 분석결과
 * epoch 10회차 학습 모델 성능
   * Loss: 0.0296
   * Accuracy: 97.31


---------------------------------------------


### 개선할 점

* cuda를 사용하여 data_loader 사용 시 모듈 수정이 필요함
* Bidirectional을 사용한 모델 성능과 비교
* 과적합은 없는지 검증
