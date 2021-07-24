# WGAN
###### <strong> WGAN의 개념과 코딩

* 언어 : Python
* 프로그램 : Colab
* 소스코드 : DCGAN.ipynb
* 사용한 모듈 : matplotlib, torch, torchvison, tqdm
* 주요 함수 :

```
 gen_block, dis_block, class Generator, class Critic, get_gen_loss, get_crit_loss, show_tensor_images
 get_gradient, gradient_penalty
 ```
 

----------------------------------------

## 데이터

* CelebA
  * 얼굴 이미지
  * 64x64, 얼굴로 이미지 crop 후 사용

----------------------------------------

## WGAN
 
### WGAN의 기본원리
 
 * 정보이론의 개념
   * m : 정보를 포함하는 message
   * p(m) : m의 확률
   * I(m) : m에 대한 self-information
 
<p align="center"><img src = "https://user-images.githubusercontent.com/72690336/123518921-2b75e900-d6e3-11eb-8e1e-ad50bdc45dfe.png" width="40%" height="40%">

 * Entropy의 개념
   * 메시지 m의 집합 M에 대한 정보량의 평균값 -> <p align="center"><img src = "https://user-images.githubusercontent.com/72690336/123519077-19487a80-d6e4-11eb-89ec-b5fc389b8c04.png" width="40%" height="40%">
 
 * Cross Entropy
   * 두 확률 분포 p와 q에 대해서 q를 이용해서 p를 설명할 때 필요한 정보량
     * p=q -> 최소의 정보량
     * p!= -> 최대의 정보량
 
<p align="center"><img src = "https://user-images.githubusercontent.com/72690336/123519166-9c69d080-d6e4-11eb-940e-fb800041d8d5.png">
 
### WGAN의 개념
 
 * Wasserstein distance
   * Cross entropy를 이용한 distribution의 거리 측정의 문제점
     * gradient가 0에 가까워질수록 gradient 소실 문제 발생
   * 해결책
     * 두 분포의 차이를 두 분포 사이의 거리로 정의
     * 하나의 분포를 다른 분포로 옮기는 작업을 earth move로 간주 -> Earth Mover's Distance로 정의
   * Discriminator -> Critic
     * discriminator return 0(Fake) or 1(real)
     * critic returns the quality of the result in (0,1)
 
 * WassersteinGAN의 Loss 함수
   * Real data의 분포와 fake data의 분포의 차이  
 ![image](https://user-images.githubusercontent.com/72690336/123519740-1a7ba680-d6e8-11eb-8eb3-2a58fcbc7878.png)
   * Gradient exploding
 
 <p align="center"><img src = "https://user-images.githubusercontent.com/72690336/123519804-66c6e680-d6e8-11eb-9fbf-f6d4b01da3e2.png" width="60%" height="60%">
  
 * Gradient penalty
   * Lipschitz 연속 함수
     * 두 점 사이의 거리가 일정한 비율(K) 이상으로 증가하지 않는 함수
   * 1-Lipschitz 연속 함수(1-L 연속)
     * K = 1
     * Gradient의 크기의 절대값이 1보다 작거나 같은 함수
     * Gradient 소실을 피하기 위한 조건으로 사용 -> critic의 학습 능력 제한(weight clipping)
   * critic의 gradient에 대한 regularization을 이용하여 1-L 연속을 유지
     * G(z)와 x를 보간해서 x' 생성 -> G(z)의 품질을 높여서 critic의 학습 속도 조절
 
### WGAN의 구조
 * Generator
   * gen block
     * parameter : input_channels, ouput_channels, kernel, final_layer
     * 구성요소 : transposed convolution + batch_norm + ReLU
     * final_layer : transposed convolution + tanh
     * H_out = (H_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + outpadding + 1
   * gen block을 사용하여 초기 1x1 size 벡터를 이미지의 size만큼 맞춰줌
  
 * Critic
   * crit block
     * parameter : input_channels, output_channels, kernel, stride, final_layer
     * 구성요소 : convolution + batch_norm + LeakyReLU(0.2)
     * final_layer : convolution
     * H_out = └(Hin + 2* padding - dilation * (kernel_size-1)-1)/stride +1┘
     * 마지막 channel의 수 = 1

---------------------------------------------


### 코드 설명

* generator block
   ```python
      def gen_block(self, in_channel, out_channel, kernel_size=4, padding=1, stride=2, dilation=1, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=2, dilation=1, padding=1),
                nn.Tanh(),
            )
     ```
  * input 차원과 output 차원을 입력받음
  * ConvTransposed layer와 batch norm, ReLU 함수로 구성
  * kernel_size=4, padding=1, stride=2, dilation=1
  * 한 층을 거칠때 마다 2배씩 차원이 커짐(첫번째 층에서 padding=0 사용시 4배 증가)
     
* Genrator
     ```python
        class Generator(nn.Module):
          def __init__(self, z_dim=10, im_chan=3, hidden_dim=32):
              super(Generator, self).__init__()
              self.z_dim = z_dim
              # Build the neural network
              self.gen = nn.Sequential(
                  self.gen_block(z_dim, hidden_dim*8, padding=0),
                  self.gen_block(hidden_dim*8, hidden_dim*4),
                  self.gen_block(hidden_dim*4, hidden_dim*2),
                  self.gen_block(hidden_dim*2, hidden_dim),
                  self.gen_block(hidden_dim, im_chan, final_layer=True),
              )
     ```
  * 5개의 generator block
  * 마지막 블록 거친 후 size = 64x64
     
* crit block
   ```python
      def crit_block(self, in_, out, kernel_size=4, stride=2, padding=1, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(in_, out, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_, out, kernel_size=kernel_size, stride=stride, padding=padding)
            )
     ```
  * input 차원과 output 차원을 입력받음
  * Convolution Layer와 BatchNorm, LeakyReLU 함수로 구성
  * kernel_size=4, stride=2, padding=1, 한 층을 거칠때 마다 차원 반으로 감소
     
* Critic
  ```python
     class Critic(nn.Module):
       def __init__(self, im_chan=3, hidden_dim=16):
           super(Critic, self).__init__()
           self.crit = nn.Sequential(
               self.crit_block(im_chan, hidden_dim),
               self.crit_block(hidden_dim, hidden_dim*2),
               self.crit_block(hidden_dim*2, hidden_dim*4),
               self.crit_block(hidden_dim*4, hidden_dim*8),
               self.crit_block(hidden_dim*8, 1, final_layer=True),
           )
     ```
  * 3개의 discriminator block으로 구성
  
* Loss
    ```python
       def get_gen_loss(crit_fake_pred):
         gen_loss = -1 * torch.mean(crit_fake_pred)
         return gen_loss

       def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
         crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp
         return crit_loss
    ```
  
* Gradient
    ```python
       def get_gradient(crit, real, fake, epsilon):
         mixed_images = real * epsilon + fake * (1 - epsilon)
         mixed_scores = crit(mixed_images)

         gradient = torch.autograd.grad(
             inputs = mixed_images,
             outputs = mixed_scores,

             grad_outputs = torch.ones_like(mixed_scores),
             create_graph = True,
             retain_graph = True,
         ) [0]
         return gradient
  
       def gradient_penalty(gradient):
       gradient = gradient.view(len(gradient), -1)
       gradient_norm = gradient.norm(2, dim=1)

       penalty = torch.mean((gradient_norm -1)**2)
       return penalty
    ```
     
* 이후 초기화, 데이터로딩, Optimizer 생성, 모델 Training 순으로 진행
      
      
### 샘플 이미지
* 원본 이미지(real)
      
<p align="center"><img src = "https://user-images.githubusercontent.com/72690336/123521371-2d46a900-d6f1-11eb-95b2-6cb5a42df109.png" width="30%" height="30%">

* 생성이미지(fake)

<p align="center"><img src = "https://user-images.githubusercontent.com/72690336/123521553-5451aa80-d6f2-11eb-8da5-edbdd09af2c7.png" width="30%" height="30%">
 
 -학습이 제대로 이루어지지 않음.
 
* loss 그래프
 
 <p align="center"><img src = "https://user-images.githubusercontent.com/72690336/123521613-bd392280-d6f2-11eb-9f3c-fa06cf694d3b.png" width="30%" height="30%">

 
* 모델 성능(step 1900)
  * Generator loss: 120.19047988891602, Critic loss: -4.923906676292419
    * Loss 값이 -가 나오기도 함
    * 아직 학습이 제대로 이루어지지 않음. 더 많은 step 학습 후 결과 비교 필요
    * 원본이미지가 너무 어둡고 흐릿함 -> 더 높은 size와 이미지처리후 결과 더 좋아질 것으로 예상
