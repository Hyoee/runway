"""원자료(버스 노선별·정류장별·일별 승하차, OA-12912 원본)를 모델용 요약 파일로 만든다.

    python scripts/build_route_data.py [data/BUS_STATION_BOARDING_MONTH_202610.csv]

1) 사용일자로 날의 유형을 판단한다 — 평일 / 토요일 / 일요일 / 휴일
   휴일은 날짜를 외워 넣지 않고 데이터로 판별: 평일인데 하루 승차가 평일 중앙값의 80% 미만이고
   운행 노선 수가 주말 수준으로 줄어든 날 (2026-10: 10/5 개천절 대체공휴일, 10/9 한글날)
2) 노선×정류장별로 유형별 하루 평균 승차·하차를 계산한다
   → data/route_stop_daytype_202610.csv  (모델이 쓰는 파일, 수 MB — 원본 144MB는 저장소에 올리지 않음)
   → data/daytype_calendar_202610.csv    (날짜별 판별 결과 — 근거 자료)
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding='utf-8')
RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'data' / 'BUS_STATION_BOARDING_MONTH_202610.csv'
OUT_ROUTE = ROOT / 'data' / 'route_stop_daytype_202610.csv'
OUT_CAL = ROOT / 'data' / 'daytype_calendar_202610.csv'
DAYTYPES = ['weekday', 'saturday', 'sunday', 'holiday']


def load_raw(path: Path) -> pd.DataFrame:
    for enc in ('cp949', 'utf-8-sig'):
        try:
            df = pd.read_csv(path, encoding=enc, dtype={'사용일자': str, '노선번호': str, '표준버스정류장ID': str, '버스정류장ARS번호': str})
            break
        except UnicodeDecodeError:
            continue
    df = df[df['버스정류장ARS번호'].str.fullmatch(r'\d{5}')]          # '~' = 차고지 가상 정류장 제외
    df['date'] = pd.to_datetime(df['사용일자'], format='%Y%m%d')
    return df


def classify_days(df: pd.DataFrame) -> pd.DataFrame:
    cal = df.groupby('date').agg(승차=('승차총승객수', 'sum'), 운행노선=('노선번호', 'nunique')).reset_index()
    cal['요일'] = cal['date'].dt.dayofweek.map(dict(enumerate('월화수목금토일')))
    dow = cal['date'].dt.dayofweek
    wk = cal[dow < 5]
    med_riders, med_routes = wk['승차'].median(), wk['운행노선'].median()
    low = (dow < 5) & (cal['승차'] < 0.8 * med_riders) & (cal['운행노선'] < med_routes)
    cal['유형'] = 'weekday'
    cal.loc[dow == 5, '유형'] = 'saturday'
    cal.loc[dow == 6, '유형'] = 'sunday'
    cal.loc[low, '유형'] = 'holiday'
    cal['판단근거'] = ''
    cal.loc[low, '판단근거'] = (f'평일인데 승차가 평일 중앙값({med_riders/1e4:,.0f}만)의 80% 미만, '
                              f'운행 노선도 평일({med_routes:.0f}개)보다 적음 → 휴일 운행')
    return cal


def main():
    df = load_raw(RAW)
    cal = classify_days(df)
    df = df.merge(cal[['date', '유형']], on='date')
    ndays = cal['유형'].value_counts()
    rows = df.groupby(['노선번호', '버스정류장ARS번호', '유형'])[['승차총승객수', '하차총승객수']].sum().reset_index()
    rows['n'] = rows['유형'].map(ndays)
    rows['board'] = rows['승차총승객수'] / rows['n']                  # 그 유형의 하루 평균 (운행 안 한 날은 0으로 평균)
    rows['alight'] = rows['하차총승객수'] / rows['n']
    wide = rows.pivot_table(index=['노선번호', '버스정류장ARS번호'], columns='유형', values=['board', 'alight'], fill_value=0.0)
    wide.columns = [f'{v}_{t}' for v, t in wide.columns]
    for v in ('board', 'alight'):
        for t in DAYTYPES:
            wide[f'{v}_{t}'] = wide.get(f'{v}_{t}', 0.0)
    wide = wide[[f'{v}_{t}' for v in ('board', 'alight') for t in DAYTYPES]].round(2).reset_index()
    wide.columns = ['route', 'ars'] + list(wide.columns[2:])
    wide.to_csv(OUT_ROUTE, index=False, encoding='utf-8-sig')
    cal.assign(date=cal['date'].dt.strftime('%Y-%m-%d')).to_csv(OUT_CAL, index=False, encoding='utf-8-sig')

    print(cal.to_string(index=False))
    print('\n유형별 일수:', ndays.to_dict(), '| 원자료에 없는 날:',
          sorted(set(pd.date_range(cal['date'].min(), cal['date'].max())) - set(cal['date'])))
    print(f'저장: {OUT_ROUTE.name} {len(wide):,}행 ({OUT_ROUTE.stat().st_size/1e6:.1f}MB), {OUT_CAL.name}')
    for t in DAYTYPES:
        print(f'  {t}: 하루 평균 승차 {wide[f"board_{t}"].sum()/1e4:,.0f}만 명')


if __name__ == '__main__':
    main()
