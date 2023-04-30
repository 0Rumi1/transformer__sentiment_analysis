# 알라미 앱 리뷰 데이터의 문장 요약 및 감성분석(긍정/부정) 키워드 분석

## 프로젝트 소개
* 알라미 앱 리뷰 데이터(리뷰내용 + 별점)을 활용해 다음 3가지 분석을 진행
1. 리뷰 속에 담긴 사람의 긍정 / 부정 감성을 파악하여 분류할 수 있는 감성 분류 예측 모델을 만듦
2. 만든 모델을 활용해 긍정 / 부정 키워드를 출력해, 유저가 느낀 알라미 앱의 장,단점을 파악함
3. 100문장 이상의 리뷰 내용을 문장 요약하여 유저의 리뷰 내용을 한눈에 파악함


* 멤버: 본인(1명)
* 역할: 데이터 수집, 데이터 분석, 감성분류 예측 모델 개발
* 개발 기간: 23.04.07-10
<br>

**개발 방향 및 목표는 아래와 같다.**
1. 긍정/부정 분류 예측 모델 개발
2. 긍정/부정 키워드 출력
3. Transformer 를 통한 100문장 이상의 리뷰 요약
4. 추후, 운영자를 위한 대시보드 개발
<br>

## 작업 순서
* 데이터 크롤링
* 데이터 분석
  * 5년간의 별점 추세 그래프
  * 별점 분포 분석 
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
* 2018.01 ~ 2023.04 의 리뷰 데이터 크롤링

|userName|content|score|thumbsUpCount|reviewCreatedVersion|at|replyContent|
|------|---|---|---|---|---|---|
|txt|txt|float|float|float|date|txt|


<br>

## 기술 스택
#### Environment
<img src="https://img.shields.io/badge/Google Colab-F9AB00?style=for-the-badge&logo=Google Colab&logoColor=white"/> <img src="https://img.shields.io/badge/windows-0078D6?style=for-the-badge&logo=windows&logoColor=white"/>


#### Development
<img src="https://img.shields.io/badge/tensorflow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"> <img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"> 
 
<br>
  
## Prerequisite

!pip install transformers

import pandas as pd   
import tensorflow as tf    
import matplotlib.pyplot as plt    
import seaborn as sns   
import matplotlib.pyplot as plt   
import numpy as np   
from datetime import datetime   
from transformers import PreTrainedTokenizerFast   
from transformers import BartForConditionalGeneration   
import torch   
from keras.models import load_model   
from sklearn.metrics import confusion_matrix   


<br>

---
<br>

## 목차
1. [사용법](#사용법)
3. [결과](#결과)
4. [배운점 & 아쉬운 점](#배운점-&-아쉬운-점)
5. [이후의 계획](#이후의-계획)
<br>

## 사용법

**이미지 zip 파일 압축 해제**   
압축 파일의 위치(구글 드라이브 - 내 드라이브 위치)   

``` 
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration
import pandas as pd
import torch

tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')
tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
```

```
def generate_summary(text):
        raw_input_ids = tokenizer.encode(text)
        input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]
        summary_ids = model.generate(torch.tensor([input_ids]), num_beams=5, max_length=1024, no_repeat_ngram_size=2)
        return tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
```


# confusion matrix
```
from sklearn.metrics import confusion_matrix

confu = confusion_matrix(y_true = y_test, y_pred = y_pred)

plt.figure(figsize=(4, 3))
sns.heatmap(confu, annot=True, annot_kws={'size':15}, cmap='OrRd', fmt='.10g')
plt.title('Confusion Matrix')
plt.show()
```

<br>

## 결과
1. 데이터 분석
* 월 평균 score 점수가 꾸준히 감소하는 추세
* 2021년 1월 이후부터 월 평균 score 이 4점 미만으로 유지되고 있음 => 해당 기간동안 리뷰 내용 파악

![image](https://user-images.githubusercontent.com/122415320/235353607-046240fe-2572-4c08-9b15-783bfb9a8822.png)

<br> 


2) 구글 앱 리뷰 18-21년 및 21-23년 별점 분포의 파이차트 비교
* 1 ~ 2 점대의 부정적 의견이 늘어남
* 해당 리뷰를 남긴 유저의 공통점 파악이 필요
* 리뷰 내용의 자연어 처리 요약 및 긍정/부정 키워드를 통해 issue 파악
긍정 키워드: 충성 고객으로 분류, 개선 및 의견 사항을 파악 
부정 키워드: 이탈 유저로 분류, 공통된 리뷰 내용 파악하여 개선을 통해 이탈 유저를 줄임

![image](https://user-images.githubusercontent.com/122415320/235353696-7ac6a0b7-a5d4-44ca-8e83-f6f6c7b0952e.png)


3) 별점별 추세 그래프 시각화
1점과 5점의 그래프 추세가 거의 비슷
![image](https://user-images.githubusercontent.com/122415320/235353839-9f24198e-9229-4ce1-8821-691a90ae56d0.png)

4) 모델 평가
모델이 지나치게 긍정(“1”)으로만 예측하는 경향이 있음   
따라서 긍정 리뷰를 잘 예측하지만, 부정 리뷰에 대한 예측 정확도가 매우 낮음    
이는 샘플데이터의 클래스 불균형으로 인한 문제로 보임 -> 따라서, 클래스 불균형 조정을 진행
![image](https://user-images.githubusercontent.com/122415320/235355500-4d11ddfb-9542-4727-8242-1ea04922e238.png)

5) 긍정/부정 키워드 분석
계수가 양인 경우는 단어가 긍정적인 영향을 미쳤다고 볼 수 있고, 반면에, 음인 경우는 부정적인 영향을 미쳤다고 볼 수 있음   
이 계수들을 크기순으로 정렬하면, 긍정 / 부정 키워드를 출력하는 지표가 됨
![image](https://user-images.githubusercontent.com/122415320/235355605-ae908106-5065-4661-b17a-e85f274696d3.png)


* 긍정적 키워드
앱을 사용하고 바로 일어날 수 있어 도움이 되었으며 학교에 지각하지 않고 늦잠에 효과적이라 만족하는 것으로 보임

* 부정적 키워드 
알람이 계속 울리고, 갑자기 오류/삭제/버그 로 인해 앱이 작동이 되지 않는 경우에 대한 개선이 필요해보인다.
카메라 관련 기능 개선이 필요해 보임 -> 해당 키워드를 포함한 리뷰만 뽑아 상세 분석하여 적절한 개선이 필요
프리미엄, 돈, 정기 결제 등의 불만 및 이에 대한 해제 문의가 있는 것으로 보임
정기적인 오류와 버그 현상을 점검하고 업데이트를 실시하는 것이 필요


## 배운점 & 아쉬운 점
<br>
  
* 단순히 별점을 기준으로 긍정과 부정을 분류하였기에 5점 및 4점에 포함된 부정적 내용은 잘 분류되지 않음

<br>


## 이후의 계획
* 긍정 키워드 내 부정적 내용을 분류하는 방법 중 리뷰 내용의 속성을 분류하고, 속성에 따른 서술어를 평가 기준으로 삼는 방법이 있음 [참고 링크](https://sieon-dev.tistory.com/15)
* 주요 키워드 추출
   * 단어 빈도수로 나열
   * 발화 의도 구분 
   * 시계열로 쪼개기 (시간별/계절별 주제 및 시간의 흐름에 따라 주제가 변화하는 패턴 파악)
* 추후, 알라미 앱 유저의 반응을 확인하기 위한 대시보드를 개발할 수 있음
<br>
