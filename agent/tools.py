"""LLM이 부르는 도구. 원칙: LLM은 조건(Params)만 정하고, 숫자는 전부 엔진이 만든다.

도구 함수는 세션 상태(현재 조건·결과)를 바꾸고, 결과를 JSON 문자열로 돌려준다.
같은 함수를 UI 버튼에서도 써서, 채팅과 화면 조작이 같은 경로로 계산된다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from anthropic import beta_tool

from runway import DEFAULT, Engine, Forbid, Params
from runway.engine import describe_params
from runway.events import EVENTS, event_of, params_for, parse_event

INTERACTIVE_TIME_LIMIT = 120     # 채팅 중 새 조건 풀이 시간 제한(초)
INTERACTIVE_GAP = 0.02


@dataclass
class Session:
    engine: Engine
    params: Params = DEFAULT
    result: dict | None = None           # 현재 TO-BE 결과 (engine.package 형식)
    asis: dict | None = None
    log: list = field(default_factory=list)          # 도구 호출 기록 (숫자 검증·화면 표시용)
    progress: Callable[[str], None] = lambda m: None

    def ensure_asis(self):
        """현재 종목의 AS-IS — 종목 프리셋에 AS-IS가 없으면 None (비교하지 않음)"""
        key = (self.params.EVENT, self.params.T0)
        if getattr(self, '_asis_key', None) != key:
            self.asis, self._asis_key = self.engine.asis(self.params), key
        return self.asis


KEY_KPI = ['riders', 'riders_per_km', 'stops', 'routes', 'turns', 'sharp', 'junctions', 'closure_kmh', 'late_km', 'length_km']
KPI_MEANING = {   # LLM이 지표를 바르게 설명하도록 결과에 함께 넣는다
    'riders': '정류장이 마라톤 코스에 들어가 통제 시간 동안 버스를 못 타는 승객(명, 일요일) — 최적화가 줄이는 핵심 값',
    'stops': '코스 위에 있어 통제 시간 동안 쓸 수 없는 정류장 수',
    'routes': '코스 위에 정류장이 있는 버스 노선 수',
    'turns': '45° 이상 방향 전환 횟수 (적을수록 단순한 코스)',
}


def _pct(a, b):
    return None if not a else round((b - a) / a * 100)


def summarize(s: Session, res: dict) -> dict:
    """LLM에게 돌려줄 요약 — AS-IS가 있는 종목만 AS-IS 대비 변화를 넣는다 (없는 종목은 비교 키 자체가 없음)"""
    A = s.ensure_asis()
    k = res['kpi']
    p = Params.model_validate(res['params'])
    out = dict(종목=event_of(p).label, 조건=describe_params(p), 캐시결과=res.get('cached', False), KPI={x: k[x] for x in KEY_KPI})
    if A:
        a = A['kpi']
        out.update(AS_IS_KPI={x: a[x] for x in KEY_KPI},
                   AS_IS대비_변화율_퍼센트={x: _pct(a[x], k[x]) for x in ('riders', 'stops', 'routes', 'turns', 'junctions')},
                   AS_IS_출처=A.get('source', ''))
    else:
        out['비교'] = '이 종목은 기존 코스(AS-IS) 데이터가 없어 비교하지 않음 — AS-IS 대비 표현을 쓰지 말 것'
    out.update(
        지표_뜻=KPI_MEANING,
        주요도로=[f"{r['km']}km {r['도로']} {r['길이km']}km" for r in res['roads'][:12]],
        출발도착_교차로까지_거리m=res.get('snap_m'),
        솔버=dict((x, res['solver'].get(x)) for x in ('solver', '목적함수', '해 종류', '초', 'gap') if x in res['solver']),
    )
    return out


def _resolve(s: Session, query: str, area=None):
    """장소 이름 → (이름, 좌표) 또는 후보 목록(애매할 때). area: 종목의 분석 영역"""
    area = area or s.params.AREA
    c = [h for h in s.engine.places(query, area=area) if h['in_area']]
    if not c:
        out = s.engine.places(query, area=area)
        return None, dict(error=f"'{query}'를 이 종목의 분석 영역 안에서 찾지 못했습니다.",
                          영역밖_후보=[h['name'] for h in out[:5]])
    if c[0]['match'] == 'exact' or len(c) == 1:
        return c[0], None
    return None, dict(error=f"'{query}' 후보가 여러 곳입니다. 사용자에게 고르게 하거나 정확한 이름으로 다시 호출하세요.",
                      후보=[h['name'] for h in c])


def build_params(s: Session, event=None, start_place=None, finish_place=None, start_hour=None, avoid_places=(), radius=150.0,
                 clear_forbidden=False, simplicity=None, reset=False):
    """조건표 → Params (채팅 도구와 ① 목표 입력 화면이 같은 경로로 검증). 반환: (Params, None) 또는 (None, 오류 dict)
    장소는 정류소 위치 데이터에서만 찾는다 — 없거나 애매하면 오류와 후보를 돌려준다."""
    ev_key = s.params.EVENT
    if event:
        ev_key = event if event in EVENTS else parse_event(event)
        if ev_key is None:
            return None, dict(error=f"종목 '{event}'를 알 수 없습니다. 풀코스·하프·10K 중에서 고르세요.")
    p = params_for(ev_key) if (reset or ev_key != s.params.EVENT) else s.params
    upd = {}
    for key, nm_key, q in (('START', 'START_NAME', start_place), ('FINISH', 'FINISH_NAME', finish_place)):
        if q and q != getattr(p, nm_key):
            hit, err = _resolve(s, q, p.AREA)
            if err:
                return None, dict(항목='출발지' if key == 'START' else '도착지', **err)
            upd[key], upd[nm_key] = hit['coord'], hit['name']
    if start_hour is not None:
        upd['T0'] = float(start_hour)
    forb = () if clear_forbidden else tuple(p.FORBID)
    for q in avoid_places or ():
        hit, err = _resolve(s, q, p.AREA)
        if err:
            return None, dict(항목=f'피할 장소({q})', **err)
        if all(z.name != hit['name'] for z in forb):
            forb += (Forbid(name=hit['name'], coord=hit['coord'], r=radius),)
    if clear_forbidden or avoid_places:
        upd['FORBID'] = forb
    if simplicity is not None:
        upd['TURN'] = tuple(t * float(simplicity) for t in DEFAULT.TURN)
    try:
        return p.with_(**upd), None
    except Exception as e:                         # 스키마 검증 실패 (범위 밖 값 등)
        return None, dict(error=f'조건 값이 허용 범위를 벗어났습니다: {e}')


def make_tools(s: Session) -> list:
    def record(name, args, out):
        s.log.append(dict(tool=name, args=args, result=out))
        return json.dumps(out, ensure_ascii=False, default=str)

    @beta_tool
    def search_place(query: str) -> str:
        """서울 버스정류소 위치 데이터에서 장소를 찾아 후보와 좌표를 돌려준다. 출발지·도착지·금지 구간을 정하기 전에 쓴다.

        Args:
            query: 장소 이름 (예: '서울시청', '잠실종합운동장', '건대입구역')
        """
        hits = s.engine.places(query, 8, area=s.params.AREA)      # 분석영역안 = 현재 종목 영역 기준
        return record('search_place', dict(query=query), dict(후보=[dict(이름=h['name'], 좌표=h['coord'], 분석영역안=h['in_area']) for h in hits]))

    @beta_tool
    def optimize_course(event: str | None = None, start_place: str | None = None, finish_place: str | None = None,
                        start_hour: float | None = None, add_forbidden_place: str | None = None, forbidden_radius_m: float = 150.0,
                        clear_forbidden: bool = False, simplicity: float | None = None, reset: bool = False) -> str:
        """현재 조건에서 바꿀 값만 받아 PuLP MILP로 최적 코스를 계산한다. 같은 조건을 이전에 풀었으면 즉시 돌려준다.
        새 조건이면 최대 2분 걸린다. 결과는 AS-IS(2026 JTBC 코스) 대비 변화율을 포함한다.

        Args:
            event: 종목 — 'full'(풀코스), 'half'(하프), '10k'(10K). "하프로", "10킬로", "10K 코스" 같은 말이면 지정. 바꾸면 그 종목의 기본 조건에서 시작
            start_place: 새 출발지 이름 (장소 데이터에서 정확히 찾을 수 있는 이름)
            finish_place: 새 도착지 이름
            start_hour: 출발 시각(시, 예: 7.0 = 07:00, 7.5 = 07:30). 5~12 사이
            add_forbidden_place: 코스에서 피할 장소 이름 (병원 앞 등). 반경 forbidden_radius_m 안의 도로를 쓰지 않는다
            forbidden_radius_m: 금지 반경(m), 기본 150
            clear_forbidden: True면 기존 금지 구간을 모두 해제
            simplicity: 방향 전환 벌점 배율. 1이 기본, 2면 꺾임을 두 배로 싫어함(더 단순한 코스), 0.5면 덜 신경 씀
            reset: True면 현재 종목의 기본 조건으로 되돌린 뒤 나머지 값을 적용
        """
        args = {k: v for k, v in dict(event=event, start_place=start_place, finish_place=finish_place, start_hour=start_hour,
                                      add_forbidden_place=add_forbidden_place, clear_forbidden=clear_forbidden,
                                      simplicity=simplicity, reset=reset).items() if v not in (None, False)}
        newp, err = build_params(s, event=event, start_place=start_place, finish_place=finish_place, start_hour=start_hour,
                                 avoid_places=[add_forbidden_place] if add_forbidden_place else [], radius=forbidden_radius_m,
                                 clear_forbidden=clear_forbidden, simplicity=simplicity, reset=reset)
        if err:
            return record('optimize_course', args, dict(ok=False, **err))
        s.progress(f'조건: {describe_params(newp)}')
        res = s.engine.solve(newp, time_limit=INTERACTIVE_TIME_LIMIT, gap=INTERACTIVE_GAP, progress=s.progress)
        if not res['ok']:
            hint = ['금지 구간 해제', '출발·도착을 분석 영역 중심 쪽으로', '코스 단순성(simplicity) 낮추기']
            return record('optimize_course', args, dict(ok=False, 조건=describe_params(newp), 이유=res['reason'], 완화_후보=hint))
        s.params, s.result = newp, res
        return record('optimize_course', args, dict(ok=True, **summarize(s, res)))

    @beta_tool
    def get_current_result() -> str:
        """지금 화면에 있는 TO-BE 코스의 조건과 KPI, AS-IS 대비 변화를 돌려준다."""
        if s.result is None:
            return record('get_current_result', {}, dict(ok=False, error='아직 계산된 코스가 없습니다'))
        return record('get_current_result', {}, dict(ok=True, **summarize(s, s.result)))

    @beta_tool
    def get_asis_problems() -> str:
        """현재 종목의 기존 코스(AS-IS, 2026 JTBC 서울마라톤 공식 코스)의 KPI와 TO-BE 규칙(공인 거리·회귀 금지 등)을 어긴 점.
        AS-IS가 없는 종목이면 비교할 수 없다고 돌려준다."""
        a = s.ensure_asis()
        if a is None:
            return record('get_asis_problems', {}, dict(ok=False, 비교='이 종목은 기존 코스(AS-IS) 데이터가 없어 비교하지 않습니다.'))
        return record('get_asis_problems', {}, dict(KPI={x: a['kpi'][x] for x in KEY_KPI}, 규칙_위반=a['violations'],
                                                   공식_km표지_대조=a['km_check'], 주요도로=[f"{r['km']}km {r['도로']} 못 타는 승객 {r['피해']}명" for r in a['roads']]))

    @beta_tool
    def get_affected_bus_routes(course: str = 'TO-BE') -> str:
        """코스 위에 정류장이 있어 통제 시간에 영향을 받는 버스 노선 목록.

        Args:
            course: 'TO-BE' 또는 'AS-IS'
        """
        r = s.ensure_asis() if course.upper() == 'AS-IS' else s.result
        if r is None:
            msg = '이 종목은 기존 코스(AS-IS) 데이터가 없습니다' if course.upper() == 'AS-IS' else '코스 없음'
            return record('get_affected_bus_routes', dict(course=course), dict(ok=False, error=msg))
        return record('get_affected_bus_routes', dict(course=course), dict(ok=True, 노선수=len(r['route_list']), 노선=r['route_list']))

    @beta_tool
    def list_precomputed_scenarios() -> str:
        """미리 계산해 둔 시나리오(출발 시각별·코스 단순성별)의 KPI 표. 조건을 바꾸면 어떻게 달라지는지 질문에 쓴다."""
        lib = [x for x in s.engine.library() if x['params'].EVENT == s.params.EVENT]     # 현재 종목만
        return record('list_precomputed_scenarios', {}, dict(종목=event_of(s.params).label, 지표_뜻=KPI_MEANING, 시나리오=[dict(
            조건=x['조건'], 못타는승객=x['kpi']['riders'], 통제정류장=x['kpi']['stops'], 영향노선=x['kpi']['routes'],
            방향전환=x['kpi']['turns'], 길이km=x['kpi']['length_km']) for x in lib]))

    return [search_place, optimize_course, get_current_result, get_asis_problems, get_affected_bus_routes, list_precomputed_scenarios]
