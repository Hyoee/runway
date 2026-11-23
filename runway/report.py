"""결과 → 근거 자료(facts)와 다운로드 파일.

- facts: LLM 브리핑·비교 요약에 넘기는 '숫자의 원천'. LLM은 이 안의 숫자만 인용할 수 있다.
- 다운로드: 구간별 통제 시각표 CSV, 영향 노선 CSV, 코스 GeoJSON
- 규칙 기반 브리핑: AI가 연결되지 않았을 때 쓰는 기본 협의 자료 (같은 facts로 만든다)
"""
from __future__ import annotations

import io
import json

import pandas as pd

from .engine import describe_params
from .events import event_of
from .params import Params

KPI_NAMES = {'riders': '못 타는 승객(명)', 'stops': '통제 정류장(곳)', 'routes': '영향 버스 노선(개)', 'turns': '방향 전환 45°↑(회)',
             'sharp': '급한 방향 전환 100°↑(회)', 'junctions': '통제 필요 교차로(곳)', 'closure_kmh': '도로 통제 부담(km×시간)',
             'late_km': '10시 이후까지 막히는 도로(km)', 'length_km': '코스 길이(km)'}


def pct(a, b):
    return None if not a else round((b - a) / a * 100)


def facts(res: dict, asis: dict | None) -> dict:
    """한 TO-BE 결과의 근거 자료 (AS-IS가 없는 종목은 비교 항목 없음)"""
    p = Params.model_validate(res['params'])
    ev = event_of(p)
    k = res['kpi']
    f = dict(종목=ev.label, 조건=describe_params(p), 출발=p.START_NAME, 도착=p.FINISH_NAME,
             출발시각='%02d:%02d' % (int(p.T0), round(p.T0 % 1 * 60)),
             TO_BE={KPI_NAMES[x]: k[x] for x in KPI_NAMES},
             피해_큰_도로=[dict(구간_시작_km=r['km'], 도로=r['도로'], 길이_km=r['길이km'], 못타는승객=r['피해'], 통제=r['통제'])
                       for r in sorted(res['roads'], key=lambda r: -r['피해'])[:8]],
             통제_시각표=[dict(구간_시작_km=r['km'], 도로=r['도로'], 길이_km=r['길이km'], 통제=r['통제']) for r in res['roads']],
             영향_노선_수=len(res['route_list']))
    if asis:
        a = asis['kpi']
        f['AS_IS'] = {KPI_NAMES[x]: a[x] for x in KPI_NAMES}
        f['AS_IS대비_변화율_퍼센트'] = {KPI_NAMES[x]: pct(a[x], k[x]) for x in ('riders', 'stops', 'routes', 'turns', 'junctions')}
        f['AS_IS_문제점'] = asis['violations']
        f['AS_IS_출처'] = asis.get('source', '')
    else:
        f['비교'] = '이 종목은 기존 코스(AS-IS) 데이터가 없어 비교하지 않음'
    return f


def timetable(res: dict) -> pd.DataFrame:
    df = pd.DataFrame(res['roads']).rename(columns={'km': '구간 시작(km)', '도로': '도로', '길이km': '길이(km)', '정류장': '버스 정류장(곳)',
                                                    '피해': '못 타는 승객(명)', '통제': '통제 시각'})
    cols = ['구간 시작(km)', '도로', '길이(km)', '버스 정류장(곳)', '못 타는 승객(명)', '통제 시각']
    return df[[c for c in cols if c in df.columns]]


def routes_df(res: dict) -> pd.DataFrame:
    rd = res.get('route_detour') or {}
    return pd.DataFrame([{'노선': r, '우회 시간 중 이용객(참고)': rd.get(r, '')} for r in res['route_list']])


def csv_bytes(df: pd.DataFrame) -> bytes:
    b = io.StringIO(); df.to_csv(b, index=False)
    return b.getvalue().encode('utf-8-sig')            # 엑셀에서 한글이 깨지지 않게 BOM


def geojson_bytes(res: dict) -> bytes:
    return json.dumps(dict(type='FeatureCollection', features=[res['geojson']]), ensure_ascii=False).encode('utf-8')


def rule_briefing(f: dict) -> dict:
    """AI 미연결 시 기본 브리핑 — facts의 숫자만 옮겨 적는다 (Briefing 스키마와 같은 모양)"""
    t = f['TO_BE']
    summary = [f"{f['종목']} {f['출발']}→{f['도착']}, {f['출발시각']} 출발 기준 최적 코스: 길이 {t['코스 길이(km)']}km, "
               f"통제 시간에 버스를 못 타는 승객 {t['못 타는 승객(명)']:,}명, 통제 정류장 {t['통제 정류장(곳)']}곳, 영향 노선 {t['영향 버스 노선(개)']}개"]
    if 'AS_IS' in f:
        d = f['AS_IS대비_변화율_퍼센트']
        summary.append(f"기존 코스 대비 못 타는 승객 {d['못 타는 승객(명)']:+d}%, 영향 노선 {d['영향 버스 노선(개)']:+d}%, 방향 전환 {d['방향 전환 45°↑(회)']:+d}%")
    else:
        summary.append(f['비교'])
    top = f['피해_큰_도로'][:3]
    return dict(
        title=f"{f['종목']} 코스 협의 브리핑",
        summary=summary,
        for_seoul=[f"피해가 큰 구간: " + ', '.join(f"{r['도로']}({r['구간_시작_km']}km~, {r['못타는승객']:,}명)" for r in top),
                   f"영향 버스 노선 {f['영향_노선_수']}개 — 노선 목록 CSV 참조"],
        for_police=[f"통제 필요 교차로 {t['통제 필요 교차로(곳)']}곳, 구간별 통제 시각은 통제 시각표 CSV 참조"],
        for_bus=[f"코스 위 정류장 {t['통제 정류장(곳)']}곳이 통제 시간 동안 이용 불가 — 임시 정류장 안내 필요"],
        risks=['버스 승객만 고려 — 일반 차량(승용차·택시·화물차)의 통행·우회 지체는 반영하지 못함', '일요일 시간대별 원자료가 없어 정류장별 월 시간대 비중을 사용', '코스 위 정류장이 있는 노선만 영향 노선으로 집계'],
    )


def briefing_markdown(b: dict, f: dict) -> str:
    sec = lambda h, xs: f"## {h}\n" + ''.join(f"- {x}\n" for x in xs) + '\n'
    md = f"# {b['title']}\n\n조건: {f['조건']}\n\n" + sec('요약', b['summary']) + sec('서울시(교통·버스정책)', b['for_seoul']) \
        + sec('경찰(교통 통제)', b['for_police']) + sec('버스 운수사', b['for_bus']) + sec('한계·주의', b['risks'])
    md += '## 구간별 통제 시각\n\n| 구간 시작(km) | 도로 | 길이(km) | 통제 |\n|---:|---|---:|---|\n'
    md += ''.join(f"| {r['구간_시작_km']} | {r['도로']} | {r['길이_km']} | {r['통제']} |\n" for r in f['통제_시각표'])
    return md
