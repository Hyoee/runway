"""종목 프리셋 — 풀코스 / 하프 / 10K 설정을 한 곳에서 관리한다.

화면·에이전트·사전 계산은 종목 이름이 아니라 이 프리셋의 값(특히 asis_file 있음/없음)으로 동작을 정한다.

- 거리 허용 오차: 풀코스 기준 비율(42,600 / 42,195)을 종목 거리에 비례 적용
- 랜드마크(롯데타워) 조건: 모든 종목에서 쓰지 않음 (IDENTITY = ()) — 풀코스도 이미 제외된 상태 그대로
- 통제 해제 시각: 후미 속도 = 종목 거리 / 컷오프 시간 (+ 해제 여유 POST 30분)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .params import BBOX, DEFAULT, Params

FULL_TOL = 42600.0 / 42195.0          # 풀코스 LMAX / LMIN


@dataclass(frozen=True)
class Event:
    key: str
    label: str
    distance_m: float
    start: tuple
    start_name: str
    finish: tuple
    finish_name: str
    t0: float                          # 출발 시각(시)
    cutoff_h: float                    # 후미(컷오프) 시간 — 통제 해제 시각 계산
    area: tuple                        # (위도 최소, 위도 최대, 경도 최소, 경도 최대) — 정류장 연결·코스 허용 영역
    osm: tuple                         # 도로망 파일 (data/)
    asis_file: str | None              # 고정된 AS-IS 코스 파일 (없으면 AS-IS 비교 안 함)
    asis_source: str = ''
    note: str = ''
    synonyms: tuple = field(default=())

    @property
    def has_asis(self) -> bool:
        return bool(self.asis_file)


_AREA_FULL = (BBOX['lat_min'], BBOX['lat_max'], BBOX['lon_min'], BBOX['lon_max'])

EVENTS: dict[str, Event] = {
    'full': Event(
        key='full', label='풀코스', distance_m=42195.0,
        start=DEFAULT.START, start_name=DEFAULT.START_NAME, finish=DEFAULT.FINISH, finish_name=DEFAULT.FINISH_NAME,
        t0=DEFAULT.T0, cutoff_h=5.0, area=_AREA_FULL, osm=('osm_roads.json',),
        asis_file='asis_2026_official.json',
        asis_source='2026 JTBC 서울마라톤 대회 책자 p28~29 풀코스 안내 (2026년 공개 코스 기준)',
        note='07:30 출발, 13:00 종료 (책자 p17)', synonyms=('풀', '풀코스', '마라톤', '42.195', 'full')),
    'half': Event(
        key='half', label='하프', distance_m=21097.5,
        start=DEFAULT.START, start_name=DEFAULT.START_NAME, finish=DEFAULT.FINISH, finish_name=DEFAULT.FINISH_NAME,
        t0=DEFAULT.T0, cutoff_h=3.0, area=_AREA_FULL, osm=('osm_roads.json',),
        asis_file=None,
        note='공식 하프 코스가 없어 출발·도착은 풀코스와 같은 기본값, 컷오프 3시간 가정', synonyms=('하프', '하프코스', 'half', '21km')),
    '10k': Event(
        key='10k', label='10K', distance_m=10000.0,
        start=(37.566668, 126.896326), start_name='상암 월드컵공원', finish=(37.525767, 126.924223), finish_name='여의도공원',
        t0=7.5, cutoff_h=2.5, area=(37.505, 37.598, 126.855, 126.972), osm=('osm_roads_west.json',),
        asis_file='asis_10k_2026_official.json',
        asis_source='2026 JTBC 서울마라톤 대회 책자 p34 10K 코스 안내 (2026년 공개 코스 기준)',
        note='07:30 출발, 10:30 종료 (책자 p17) → 컷오프 2.5시간 + 해제 여유 30분',
        synonyms=('10k', '10K', '10킬로', '10키로', '10km', '텐케이')),
}


def event_of(p: Params) -> Event:
    return EVENTS[p.EVENT]


def params_for(event: str, base: Params = DEFAULT) -> Params:
    """종목 기본 조건. 풀코스는 DEFAULT와 완전히 같다 (테스트로 확인)."""
    ev = EVENTS[event]
    d = ev.distance_m
    return base.with_(EVENT=ev.key, LMIN=d, LMAX=d * 42600.0 / 42195.0, START=ev.start, FINISH=ev.finish,
                      START_NAME=ev.start_name, FINISH_NAME=ev.finish_name, T0=ev.t0, V_SWEEP=d / 1000 / ev.cutoff_h,
                      AREA=ev.area, OSM=ev.osm, KEEP_POINTS=(ev.start, ev.finish), IDENTITY=())


def parse_event(text: str) -> str | None:
    """'하프로', '10킬로', '10K 코스' 같은 표현 → 종목 키 (없으면 None)"""
    t = str(text).strip().lower().replace(' ', '')
    for k in ('10k', 'half', 'full'):              # '하프 마라톤'이 풀코스로 잡히지 않게 짧은 종목부터
        ev = EVENTS[k]
        if t == k or any(s.lower().replace(' ', '') in t for s in ev.synonyms):
            return k
    return None
