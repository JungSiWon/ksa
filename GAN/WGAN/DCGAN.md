# DCGAN
###### <strong> DCGAN의 개념과 코딩

* 언어 : Python
* 프로그램 : Colab
* 소스코드 : DCGAN.ipynb
* 사용한 모듈 : matplotlib, torch, torchvison, tqdm
* 주요 함수 :

```
 gen_block, dis_block, class Generator, class Discriminator, get_gen_loss, get_disc_loss, show_tensor_images 
 ```
 

----------------------------------------

## 데이터

* MNIST

----------------------------------------

## DCGAN
 
### Vanilla GAN의 한계
 
 * 생성 이미지가 만족스럽지 못함 -> DCGAN사용
 * 불안정한 학습 -> WGAN Gradient Penalty 사용
 * 통제불가능한 무작위카테고리 이미지 생성 -> CGAM 사용

  
### DCGAN의 기본원리
 
 * 합성곱(Convolution) 연산
   * 주어진 함수에 g(kernel, filter)를 곱해서 더하는 연산
   * 영상에서는 입력 영상에 작은 크기의 kernel 영상을 pixel-by-pixel로 곱해서 더한 영상을 생성하는 과정을 의미함
 
<p align="center"><img src = "https://user-images.githubusercontent.com/72690336/122898701-0e10ea00-d386-11eb-82fd-a99f2d2c29f9.png" width="40%" height="40%">

 * Convolution의 주요 속성
   * size(kernel의 크기) : 3x3, 5x5, 7x7, etc
   * stride(kernel의 적용단위) : 1, 2, 3, etc
   * padding(입력영상의 주변) : None, 0, 1, etc
 
 * Transposed Convolution
   * Convolution filter의 transposed matrix를 곱함
   * 영상의 크기가 커지는 연산
 
### DCGAN(Deep Convolution 의 개념
 * DCGAN = GAN + Convolution Layer
 
### DCGAN의 구조
 * Generator
 
 <p align="center"><img src = "https://user-images.githubusercontent.com/72690336/123206451-b0e87600-d4f6-11eb-95d0-9f2d6cd6afc8.png" width="60%" height="60%">

 * gen block
   * parameter : input_channels, ouput_channels, kernel, final_layer
   * 구성요소 : transposed convolution + batch_norm + ReLU
   * final_layer : transposed convolution + tanh
   * H_out = (H_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + outpadding + 1
  
 * Discriminator
   * disc block
     * parameter : input_channels, output_channels, kernel, stride, final_layer
     * 구성요소 : convolution + batch_norm + LeakyReLU(0.2)
     * final_layer : convolution
   * 3개의 disc block
  
### DCGAN Loss 함수
  * Loss 함수

 <p align="center">        min max V(D,G) = E<sub>x ~ pdata(x)</sub>[logD(x)] + E<sub>z ~ pz(z)</sub>[log(1-D(G(z)))]

  * Generator loss
    * log(1-D(G(z)))
  * Discriminator loss
    * logD(x)
### GAN Training
  * 훈련 목표
    * 훈련 데이터 x와 일치하는 G(z)를 생성하는 것 P<sub>data</sub> = P<sub>z</sub>, 잘 생성되면 D()는 x와 G(z)를 구분하지 못함 = 값 : 1/2
    * GAN은 min max V(G,D)를 추구하기 때문에 수렴이 어려울 수 있지만, 최적의 경우에 수렴되는 값이 있음을 증명
    
  * 훈련의 어려움
    * D와 G를 동시에 훈련해야함 : D는 max, G는 min을 추구
    * 훈련 초기 생성되는 G(z)는 품질이 안좋은 -> 매번 0에 가까우면 log(1-D(G(z))이 0에 가까운 값을 가짐 -> gradient descent를 적용하기 힘듦
    * 위와 같은 문제을 방지하기 위해 초기에는 log(1-D(G(z))를 사용하지 말고 D(G(z))를 최대화 시키는 방향으로 훈련(후자가 값이 더 크다)

---------------------------------------------


### 코드 설명

* generator block
   ```python
      def gen_block(self, in_channel, out_channel, kernel_size=3, stride=2, final_layer=False):
         if not final_layer:
             return nn.Sequential(
                 nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride),
                 nn.BatchNorm2d(out_channel),
                 nn.ReLU(inplace=True),
             )
         else:
             return nn.Sequential(
                 nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride),
                 nn.Tanh(),
             )
     ```
  * input 차원과 output 차원을 입력받음
  * ConvTransposed layer와 batch norm, ReLU 함수로 구성
  * kernel_size=3, stride=2
  * 한 층을 거칠때 마다 2배씩 차원이 커짐
     
* Genrator
     ```python
        class Generator(nn.Module):
           def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
               super(Generator, self).__init__()
               self.z_dim = z_dim
               # Build the neural network
               self.gen = nn.Sequential(
                   self.gen_block(z_dim, hidden_dim*4),
                   self.gen_block(hidden_dim*4, hidden_dim*2, kernel_size=4, stride=1),
                   self.gen_block(hidden_dim*2, hidden_dim),
                   self.gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
               )
     ```
  * 4개의 generator block
     
* discriminator block
   ```python
      def disc_block(self, in_, out, kernel_size=4, stride=2, final_layer=False):
         if not final_layer:
             return nn.Sequential(
                 nn.Conv2d(in_, out, kernel_size, stride),
                 nn.BatchNorm2d(out),
                 nn.LeakyReLU(0.2, inplace=True)
             )
         else:
             return nn.Sequential(
                 nn.Conv2d(in_, out, kernel_size, stride)
             )
     ```
  * input 차원과 output 차원을 입력받음
  * Convolution Layer와 BatchNorm, LeakyReLU 함수로 구성
  * 한 층을 거칠때 마다 차원 반으로 감소
     
* Discriminator
  ```python
     class Discriminator(nn.Module):
       def __init__(self, im_chan=1, hidden_dim=16):
           super(Discriminator, self).__init__()
           self.disc = nn.Sequential(
               self.disc_block(im_chan, hidden_dim),
               self.disc_block(hidden_dim, hidden_dim*2),
               self.disc_block(hidden_dim*2, 1, final_layer=True),
           )
     ```
  * 3개의 discriminator block으로 구성
  
* Generator Loss
    ```python
       def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
         fake_noise = get_noise(num_images, z_dim, device=device) # z
         fake = gen(fake_noise) # G(z)
         disc_fake_pred = disc(fake) # D(G(z))
         # compare fake_pred & ones
         gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
         return gen_loss
    ```
* Discriminator Loss
    ```python
       def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, divice):
         fake_noise = get_noise(num_images, z_dim, device=device) # z
         fake = gen(fake_noise) # G(z)
         disc_fake_pred = disc(fake.detach()) # D(G(z))
         # compare fake_pred & zeros
         disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred)) # log(1-D(G(z)))
         disc_real_pred = disc(real)
         # compare real_pred & ones
         disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred)) # log(D(x))
         disc_loss = (disc_fake_loss + disc_real_loss)/2

         return disc_loss
    ```
     
* 이후 초기화, 데이터로딩, Optimizer 생성, 모델 Training 순으로 진행
      
      
### 샘플 이미지
* 원본 이미지(real)
      
<p align="center"><img src = "https://user-images.githubusercontent.com/72690336/122808816-4eca1e00-d308-11eb-8eb3-ab3a3777fcd6.png" width="30%" height="30%">

* 생성이미지(fake)

<p align="center"><img src = "https://user-images.githubusercontent.com/72690336/123210147-8f8a8880-d4fc-11eb-8a34-6af16819b8d7.png" width="30%" height="30%">
 
 -지난번의 Vanilla GEN에 비해 훨씬 더 숫자다운 모습을 보인다.
 
* 모델 성능(step 11500)
  * Generator loss: 0.7071045048236847, discriminator loss: 0.6988933897018442
    * Vanilla GEN보다 훨씬 더 낮은 Generator loss 값을 보인다. -> 훨씬 더 정교한 이미지 생성
