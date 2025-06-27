
import pandas as pd
import os

# 파일 경로
file_attendance = "../data/original/독립기념관_관람객수(일별)_20250530.csv"
file_press = "../data/original/독립기념관_언론보도자료 현황_20240722.csv"
file_temp = "../data/original/extremum_기온.csv"
file_rain = "../data/original/extremum_강수량.csv"

# 관람객 수
df_att = pd.read_csv(file_attendance, encoding="cp949")
df_att['date'] = pd.to_datetime(df_att['일자'], errors='coerce')
df_att = df_att[['date', '관람객수']].rename(columns={'관람객수': 'attendences'})

# 기온
df_temp = pd.read_csv(file_temp, encoding="cp949")
df_temp.columns = df_temp.columns.str.strip()
df_temp['date'] = pd.to_datetime(df_temp['일시'], errors='coerce')
df_temp = df_temp[['date', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)']].rename(
    columns={'평균기온(℃)': 'avg_temp', '최고기온(℃)': 'max_temp', '최저기온(℃)': 'min_temp'})

# 강수량
df_rain = pd.read_csv(file_rain, encoding="cp949")
df_rain.columns = df_rain.columns.str.strip()
df_rain['date'] = pd.to_datetime(df_rain['일시'], errors='coerce')
df_rain = df_rain[['date', '강수량(mm)', '1시간최다강수량(mm)']].rename(
    columns={'강수량(mm)': 'precipitation', '1시간최다강수량(mm)': 'max_hourly_rain'})

# 병합
df_weather = df_temp.merge(df_rain, on='date', how='outer')
df = df_att.merge(df_weather, on='date', how='left')

# 기존 요일/월 컬럼 삭제 (혹시 남아 있을 경우 방지용)
weekday_cols = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
month_cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
df = df.drop(columns=[col for col in df.columns if col in weekday_cols + month_cols], errors='ignore')


# 날짜 파생 변수 생성
# 기존 요일/월 관련 컬럼들 모두 제거
df = df.drop(columns=[col for col in df.columns if col in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun',
                                                            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']], errors='ignore')

df['weekday'] = df['date'].dt.dayofweek

# 원핫 인코딩 단 한 번만 수행
df = pd.get_dummies(df, columns=['weekday'], prefix='', prefix_sep='', dtype=int)

# 컬럼명 명시적 변경
weekday_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
df = df.rename(columns={str(k): v for k, v in weekday_map.items() if str(k) in df.columns})

# 요일 인코딩 오류 검출용 (합계 1이 아닌 행 찾기)
weekday_check = df[['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']].sum(axis=1)
if not all(weekday_check == 1):
    print("❌ 요일 인코딩 오류 발견! 각 row에 요일이 하나만 True여야 합니다.")

df['month'] = df['date'].dt.month
df = pd.get_dummies(df, columns=['month'], prefix='', prefix_sep='', dtype=int)
month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
             7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
df = df.rename(columns={str(k): v for k, v in month_map.items() if str(k) in df.columns})

# 보도자료
df_press = pd.read_csv(file_press, encoding="cp949")
df_press['등록일'] = pd.to_datetime(df_press['등록일'], errors='coerce')
df_press = df_press.dropna(subset=['등록일'])
df_press['등록일'] = df_press['등록일'].dt.normalize()
df_press['title'] = df_press['제목']
df_press_grouped = df_press.groupby('등록일').agg(
    press_count=('title', 'count'),
    press_titles=('title', lambda x: ' | '.join(x))
).reset_index().rename(columns={'등록일': 'date'})
df = df.merge(df_press_grouped, on='date', how='left')

# 결측치 처리
df['press_count'] = df['press_count'].fillna(0)
df['press_titles'] = df['press_titles'].fillna('')
df = df.fillna(0)


# 날짜 공통 구간 결정
start_dates = [
    df_att['date'].min(),
    df_temp['date'].min(),
    df_rain['date'].min(),
    df_press_grouped['date'].min()
]
end_dates = [
    df_att['date'].max(),
    df_temp['date'].max(),
    df_rain['date'].max(),
    df_press_grouped['date'].max()
]
start_date = max(start_dates)
end_date = min(end_dates)
print(f"📅 사용할 날짜 범위: {start_date.date()} ~ {end_date.date()}")

# 날짜 필터링
df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]


# 저장
save_path = "../data/refined/dataset.csv"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
df.to_csv(save_path, index=False, encoding="utf-8-sig")
if not all(weekday_check == 1):
    raise ValueError('❌ 요일 인코딩 오류: 한 row에 두 개 이상의 요일이 True입니다.')
print(f"✅ 전처리 완료: {save_path} 로 저장됨.")
