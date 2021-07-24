# pix2pix
###### <strong> pix2pix의 개념

* 언어 : Python
* 프로그램 : Colab
* 소스코드 : CycleGAN.ipynb
* 사용한 모듈 : torch, tqdm, torchvision, matplotlib
* 주요 함수 및 클래스 :

```
 클래스 : ContractingBlock, ResidualBlock, ExpandingBlock, Generator, Discriminator, ImageDataset
 함수 : get_disc_loss, get_gen_adversarial_loss, get_identity_loss, get_gen_loss, weights_init, show_gensor_images, train
 ```
 

----------------------------------------

## 데이터

* horse2zebra
  * 말과 얼룩말 이미지
  * 286 size, 256 size로 crop, RandomHorizontalFlip 이미지를 랜덤으로 수평으로 뒤집음

----------------------------------------

## CycleGAN
  
### CycleGAN의 기본원리
 
 * pix2pix의 개념
   * 한 영상으로부터 새로운 스타일의 영상을 생성하는 기법 ex) 이미지 복원, 흑백to컬러화, 항공사진to지도 등에 사용

 * pix2pix의 배경
   * Conditional GAN을 발전시킴
   * Conditional GAN의 Generator
 
<p align="center"><img src = "https://user-images.githubusercontent.com/72690336/126859105-de619f7c-eb05-43cd-97fd-a9496a8cb812.png" width="60%" height="60%">
 
 *
   * pix2pix의 generator와 discriminator
 
<p align="center"><img src = "https://user-images.githubusercontent.com/72690336/126859146-7b84c1b0-c1d4-4bc5-82b0-ec53462efe6d.png" width="50%" height="50%"><img src = "https://user-images.githubusercontent.com/72690336/126859156-0a48675c-25e6-4314-b936-b713237e8050.png" width="50%" height="50%">

### pix2pix의 구조
* 전체 구조
  * G는 스케치 (x)에서 컬러 영상 (G(x))를 생성
  * D는 합성된 컬러 영상 G(x), 또는 실제 컬러영상 (y)와 x를 비교해서 fack/real 판별

### pix2pix의 구성요소
 
#### Generator
* U-net구조를 이용한 generator
  * 각 encoder block을 거칠수록 image size가 반으로 감소(보통 256 size의 이미지를 8개의 encoder block을 사용해 1차원으로 축소)
  * 각 encoder block는 convolution - batch norm - Leaky ReLU로 구성
  * 각 decoder block을 거칠수록 image size가 2배로 증가(1차원 크기의 벡터를 8개의 decoder block으로 원래 이미지 size인 256d으로 복구)
  * 각 decoder block는 transposed convolution - batch norm - ReLU로 구성
 
 <p align="center"><img src = "https://user-images.githubusercontent.com/72690336/126859470-4154bd5a-10f9-4c4b-8cca-11174f0856cc.png" width="70%" height="70%">
  
* Encoder-decoder 구조
  * deconv-net이라고도 불림
  * Convolution VS Transposed convolution
  
  <p align="center"><img src = "https://user-images.githubusercontent.com/72690336/126859561-d63db898-d785-40ec-a30f-1da76433c28d.png" width="60%" height="60%">
   
* Skip connection
  * Forward pass: encoder의 정보를 decode에 전달
  * Backward pass: encoder의 gradient flow를 개선

#### Discriminator
* PatchGAN 구조를 사용
  * 전통적 GAN엥서는 discriminator가 전체 영상에 대해서 Real/Fake를 판정
  * PatchGAN에서는 영상을 patch로 분할하여 각 영역의 Real/Fake를 판정
 
   
#### loss 함수 
* 전통적인 CGAN loss(adversarial loss, GAN loss) + pixex distance loss
  * pix2pix의 loss 함수
   
<p align="center"><img src = "https://user-images.githubusercontent.com/72690336/126860037-df7c3d3a-e4e6-40ab-b928-f6a85a8a5cce.png" width="40%" height="40%">
 
 * Adversarial loss(L<sub>cGAN</sub>(G,D))
 
<p align="center"><img src = "https://user-images.githubusercontent.com/72690336/126860166-af6e40c4-e368-4c0f-ab17-b058c8a3efe3.png" width="50%" height="50%">

 * Pixel distance loss(L<sub>L1</sub>(G,D))
   * 생성된 영상 (G(x,z))와 groundtruth 영상 (y)와의 픽셀간의 차이
 
<p align="center"><img src = "https://user-images.githubusercontent.com/72690336/126860277-b2168785-62a8-4644-8425-a7ff1dcc8600.png" width="40%" height="40%">
 
### pix2pix의 한계와 극복
* Paired image엥서만 적용 가능 -> unpaired image-to-image translation

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
