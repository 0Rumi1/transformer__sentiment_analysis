
# 앱 이름 추출 
from google_play_scraper import app
result = app(
    'droom.sleepIfUCan',            # 앱 id
    lang='ko', # efaults to 'en'    # 언어 
    country='kr' # defaults to 'us' # 국가
)

# 앱 리뷰
from google_play_scraper import Sort, reviews_all
result = reviews_all(
    'droom.sleepIfUCan',
    sleep_milliseconds=10, # 프로그램 실행 중지 시간(대기시간) : 대량의 요청으로 많은 트랙픽이 발생하므로
    lang='ko', # 언어
    country='kr', # 국가
    sort=Sort.MOST_RELEVANT, # 정렬(관련성, 최신 등 가능)
    filter_score_with=None # 별점 필터 None : 모든 별점을 뜻함
)



print(result)

# import pandas as pd

# result=pd.DataFrame(result)
# result.to_excel('딜라이트_구글앱리뷰.xlsx', engine='xlsxwriter')