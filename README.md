# 알라미 앱 리뷰 데이터의 문장 요약 및 감성분석(긍정/부정) 키워드 분석

## 프로젝트 소개
* 알라미 앱 리뷰 데이터(리뷰내용 + 별점)을 활용해 다음 3가지 분석을 진행
1. 리뷰 속에 담긴 사람의 긍정 / 부정 감성을 파악하여 분류할 수 있는 감성 분류 예측 모델을 만듦
2. 만든 모델을 활용해 긍정 / 부정 키워드를 출력해, 유저가 느낀 알라미 앱의 장,단점을 파악함
3. 100문장 이상의 리뷰 내용을 문장 요약하여 유저의 리뷰 내용을 한눈에 파악함


* 멤버: 본인(1명)
* 역할: 문장요약 모델 및 감성분류 예측 모델 개발
* 개발 기간: 23.04.07-10
<br>

**개발 방향 및 목표는 아래와 같다.**
1. 긍정/부정 분류 예측 모델 개발
2. 긍정/부정 키워드 출력
3. Transformer 를 통한 100문장 이상의 리뷰 요약
4. 추후, 운영자를 위한 대시보드 개발
<br>

## 작업 순서
* 데이터 불러오기
* 한국어 텍스트 데이터 전처리
  * 형태소 분석
  * 불용어 제거
* 감성 분류
  * Train/Test 나누기
  * 모델 학습
  * 긍정/부정 키워드 분석
* 결론
 
 <br>

## 데이터 구조

* [데이터 출처](https://play.google.com/store/apps/details?id=droom.sleepIfUCan&hl=ko&gl=US) : 구글플레이 앱

* 실제 사용 라벨은 11개 (plastic 제외) 제외 이유는 학습 시, **'plastic 라벨의 이미지 파일이 존재하지 않는다'** 오류 발생   
1. **모델 학습 시간이 오래 소요되는 점**
2. **오류 해결에 어려움** 으로 해당 파일은 삭제하여 진행하는 것으로 결정함

|userName|content|score|thumbsUpCount|reviewCreatedVersion|at|replyContent|
|------|---|---|---|---|---|---|
|txt|txt|float|float|float|date|txt|


<br>

## 기술 스택
#### Environment
<img src="https://img.shields.io/badge/Google Colab-F9AB00?style=for-the-badge&logo=Google Colab&logoColor=white"/> <img src="https://img.shields.io/badge/windows-0078D6?style=for-the-badge&logo=windows&logoColor=white"/>


#### Development
<img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white"> <img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> 
 
<br>
  
## Prerequisite

import shutil   
import tensorflow as tf   
import tensorflow_datasets as td   
from tensorflow.keras.preprocessing.image import ImageDataGenerator   
import numpy as np   
import matplotlib.pylab as plt   
from keras.engine import input_layer   
from tensorflow.keras.utils import plot_model   
import matplotlib.pyplot as plt   




---
<br>

## 목차
1. [구현 기능](#구현-기능)
2. [사용법](#사용법)
3. [결과](#결과)
4. [배운점 & 아쉬운 점](#배운점-&-아쉬운-점)
5. [이후의 계획](#이후의-계획)
<br>

## 구현 기능
1. 딥러닝 기반의 쓰레기 이미지 분류 모델 구축
<br>

## 사용법

**구글 드라이브 마운트**
아래 코드 실행 시, 구글 드라이브에 마운트해 드라이브 파일을 업로드 혹은 다운로드받을 수 있음
```
from google.colab import drive
drive.mount("/content/drive") # 
```
<br>

**이미지 zip 파일 압축 해제**   
압축 파일의 위치(구글 드라이브 - 내 드라이브 위치)   
데이터셋 파일을 구글 드라이브에 저장하지 않음, 15,000장의 파일을 구글 드라이브를 통해 읽어오면 저장하는 속도가 매우 느림   
따라서, 구글 드라이브의 압축 파일 한 개를 읽어오는 시간은 오래 걸리지 않고 압축을 풀면서 생성되는 15,000장의 파일을 **코랩 환경에 저장하면 파일을 읽는 시간이 훨씬 단축됨**
``` 
drive_path = '/content/drive/MyDrive/Colab Notebooks/project/'
source_filename = drive_path + 'dataset/archive (5).zip'
```
<br>

**저장할 경로**
```
extract_folder = '/content/drive/MyDrive/Colab Notebooks/project/dataset/'
```
<br>

**압축 해제**
```
import shutil
shutil.unpack_archive(source_filename, extract_folder)
```
<br>

**Functional API 모델 생성**
```
from keras.engine import input_layer
# Build Model - functional_api

input_layer = tf.keras.Input(shape=(224,224, 3), name = 'InputLayer') # 입력 레이어

x1 = tf.keras.layers.Conv2D(32,(3,3), padding='same',activation='relu')(input_layer) # relu 활성화 함수 
x2 = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(x1) # maxpooling2D : input 차원을 줄임
x3 = tf.keras.layers.Conv2D(64,(3,3), padding='same', activation='relu')(x2)
x4 = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(x3)
x5 = tf.keras.layers.Conv2D(32,(3,3), padding='same', activation='relu')(x4)
x6 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x5)
x7 = tf.keras.layers.Flatten(name = 'Flatten')(x6)
x8 = tf.keras.layers.Dense(64,activation='relu', name = 'Dense1')(x7)
x9 = tf.keras.layers.Dropout(0.4, name = 'Dropout')(x8)
x10 = tf.keras.layers.Dense(num_classes,activation='softmax', name = 'OutputLayer')(x9)


# Create Model
fun_model = tf.keras.Model(inputs=input_layer, outputs=x10, name='FunctionalModel')
```
<br>


## 결과
1) 모델 정확도 67.9%

![image](https://user-images.githubusercontent.com/122415320/235335209-b12f9abe-8fc1-45cb-8ba2-e818aefc01c5.png)

<br>


2) Train/Test 모델 손실 및 정확도 그래프

* 과대적합이나 과소적합이 거의 발생하지 않고 학습이 잘 진행된 것을 확인할 수 있음
![그래프](https://user-images.githubusercontent.com/122415320/235342956-e6048d32-58a0-4d14-be72-4f6e91dc242f.jpg)
<br>


## 배운점 & 아쉬운 점
<br>
  
 * 데이터셋 파일을 구글 드라이브에 저장하지 않은 이유는 15,000장의 파일을 구글 드라이브를 통해 읽어오면 저장하는 속도가 매우 느리기 때문이라는 사실을 알게됨
  
 * 따라서, 구글 드라이브의 압축 파일 한 개를 읽어오는 시간은 오래 걸리지 않고, 압축을 풀면서 생성되는 15,000장의 파일을 코랩 환경에 저장하면 파일을 읽는 시간이 훨씬 단축됨
 
 * 함수형 API 사용
1. Sequential API 를 주로 사용했지만, 다양한 모델 구조를 구현할 수 있기에 공부를 위해 Functional API  사용

* 모델 성능
1. 오랜 모델 학습 시간 및 오류 장벽으로 모델 성능을 더 높이지 못해 아쉬움이 남음
2. epoch 을 더 늘리면 모델 성능이 개선될 여지가 남아 있음

<br>


## 이후의 계획
1. 클래스 단순화 => 모델 학습 시간 단축    
2. 모델 성능 개선에 대한 방법 공부   
3. 모델 테스트 및 웹 구현
<br>
