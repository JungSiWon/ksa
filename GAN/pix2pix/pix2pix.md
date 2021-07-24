# pix2pix
###### <strong> pix2pix의 개념

* 언어 : Python
* 프로그램 : Colab
* 소스코드 : U-NET.ipynb
* 사용한 모듈 : torch, tqdm, torchvision, matplotlib
* 주요 함수 및 클래스 :

```
 클래스 : ContractingBlock, ExpandingBlock, FeatureMapBlock, UNet, Generator, Discriminator, ImageDataset
 함수 : crop, show_gensor_images, train
 ```
 

----------------------------------------

## 데이터

* 세포 이미지
  * 세포 촬영 이미지 volumn, label
  * 512 size, 373 size로 crop

----------------------------------------

## UNet
  
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

### UNet train 결과
* 원본 이미지(real)

<p align="center"><img src = "https://user-images.githubusercontent.com/72690336/126861061-0ba6e007-0b36-4a89-9d0d-ca010c59e7bc.png" width="40%" height="40%">

* 생성이미지(fake)

<p align="center"><img src = "https://user-images.githubusercontent.com/72690336/126861078-f49c1115-8b3d-490b-8980-247c53618590.png" width="40%" height="40%">

 
* 모델 성능(step 1300, epoch 162)
  * U-Net loss : 0.0617
    * 매우 작은 loss값에 육안으로 식별 불가할 정도의 비슷한 이미지 생성
    * Genrator와 Discriminator가 따로 없다.
