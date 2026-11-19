"""서비스용 진입점 — Streamlit 앱과 LLM 도구는 이 클래스만 쓴다.

- 네트워크는 한 번만 로드
- 같은 문제(조건)의 결과는 scenarios/에 저장해 두고 바로 돌려준다 (사전 계산·재방문 즉시 응답)
- 더 좋은 해(목적함수가 작은 해)를 찾으면 그것으로 교체 — 솔버 gap 때문에 결과가 흔들리지 않게
- 결과는 저장 전에 validate_course로 독립 검사
"""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from pathlib import Path

from .asis import build_asis
from .evaluate import evaluate, kpis, objective, road_table, to_geojson_feature, validate_course
from .network import load_network
from .params import DEFAULT, MODEL_VERSION, NETWORK_FIELDS, Params
from .places import search_places
from .scenario import Scenario, ScenarioError
from .solve import optimize

# 결과(문제 정의)에 영향을 주지 않는 필드 — 캐시 키에서 뺀다
_NON_PROBLEM = {'TIME_LIMIT', 'GAP', 'START_NAME', 'FINISH_NAME'}


def problem_key(p: Params) -> str:
    return MODEL_VERSION + '-' + p.key(*[f for f in Params.model_fields if f not in _NON_PROBLEM])


def describe_params(p: Params) -> str:
    """기본값과 다른 조건만 사람이 읽는 문장으로 — UI 라벨과 LLM 맥락에 쓴다"""
    d, out = DEFAULT, []
    if p.START != d.START or p.START_NAME != d.START_NAME:
        out.append(f'출발 {p.START_NAME}')
    if p.FINISH != d.FINISH or p.FINISH_NAME != d.FINISH_NAME:
        out.append(f'도착 {p.FINISH_NAME}')
    if p.T0 != d.T0:
        out.append('출발 %02d:%02d' % (int(p.T0), round(p.T0 % 1 * 60)))
    if p.IDENTITY != d.IDENTITY:
        out += [f'{lm.name} 조망≥{lm.L/1000:g}km' for lm in p.IDENTITY] or ['정체성 조건 없음']
    if p.FORBID != d.FORBID:
        out += [f'금지: {z.name}(반경 {z.r:g}m)' for z in p.FORBID if z not in d.FORBID]
        out += [f'금지 해제: {z.name}' for z in d.FORBID if z not in p.FORBID]
    if p.TURN != d.TURN:
        r = p.TURN[0] / d.TURN[0]
        out.append(f"{'승객 우선' if r < 1 else '단순성 우선'} (꺾임 벌점 ×{r:g})")
    if p.ALPHA != d.ALPHA:
        out.append(f'노선당 고정 벌점 {p.ALPHA:g}')
    if p.DETOUR_W != d.DETOUR_W:
        out.append('우회 영향 미반영 (못 타는 승객만)' if p.DETOUR_W == 0 else f'우회 가중치 δ={p.DETOUR_W:g}')
    if p.DAYTYPE != d.DAYTYPE:
        out.append({'weekend': '수요: 토·일 평균', 'all': '수요: 전체 일평균', 'sunday': '수요: 일요일 평균'}[p.DAYTYPE])
    return ', '.join(out) or '기본 조건 (광화문→올림픽공원, 일요일 08:00)'


class Engine:
    def __init__(self, root: Path | str = '.'):
        self.root = Path(root)
        self.data_dir = self.root / 'data'
        self.store = self.root / 'scenarios'
        self.store.mkdir(exist_ok=True)
        self._nets: dict = {}
        self.net = self.network(DEFAULT)
        self._sc: OrderedDict[str, Scenario] = OrderedDict()
        self._lock = threading.Lock()

    def network(self, p: Params):
        """수요 기준(DAYTYPE 등)별 도로망 — 필요할 때 한 번 만든다"""
        k = p.key(*NETWORK_FIELDS)
        if k not in self._nets:
            self._nets[k] = load_network(self.data_dir, p)
        return self._nets[k]

    # ── 조건 → Scenario (최근 8개 메모리 캐시) ──
    def scenario(self, p: Params) -> Scenario:
        k = problem_key(p)
        with self._lock:
            if k in self._sc:
                self._sc.move_to_end(k)
                return self._sc[k]
        sc = Scenario(self.network(p), p)            # ScenarioError는 호출자에게 그대로
        with self._lock:
            self._sc[k] = sc
            while len(self._sc) > 8:
                self._sc.popitem(last=False)
        return sc

    def places(self, query: str, limit: int = 8):
        return search_places(self.data_dir, query, limit)

    # ── 결과 묶음 (UI·LLM이 쓰는 형태) ──
    def package(self, sc: Scenario, order, name: str, info: dict | None = None) -> dict:
        ev = evaluate(sc, order)
        return dict(name=name, params=sc.p.model_dump(mode='json'), kpi=kpis(ev), route_list=ev['route_list'],
                    route_detour=ev['route_detour'],
                    roads=road_table(ev, 300).to_dict('records'), geojson=to_geojson_feature(sc, order, name, ev),
                    order=[int(a) for a in order],
                    violations=validate_course(sc, order, check_identity=name != 'AS-IS', check_endpoints=name != 'AS-IS'),
                    objective=round(objective(sc, order)),                      # MILP 식 그대로 (솔버 값과 일치)
                    objective_exact=round(objective(sc, order, exact=True)),    # 승객 항목 정밀 평가 — 교체 판단 기준
                    snap_m=sc.snap_m, solver=info or {})

    def asis(self, p: Params = DEFAULT) -> dict:
        """AS-IS(2026 JTBC 코스)를 조건 p(출발 시각 등)로 평가. 출발·도착은 공식 코스 그대로."""
        sc = self.scenario(p.with_(START=DEFAULT.START, FINISH=DEFAULT.FINISH))
        order, chk = build_asis(sc)
        out = self.package(sc, order, 'AS-IS')
        out['km_check'] = chk
        return out

    # ── 캐시 ──
    def _path(self, p: Params) -> Path:
        return self.store / f'{problem_key(p)}.json'

    def cached(self, p: Params) -> dict | None:
        f = self._path(p)
        if f.exists():
            return json.load(open(f, encoding='utf-8'))
        return None

    def _save_if_better(self, p: Params, res: dict) -> dict:
        """정밀 목적함수(화면 KPI와 같은 승객 평가 + 같은 벌점)가 더 작을 때만 교체 — 발표 숫자가 나빠지지 않게"""
        old = self.cached(p)
        if old and old.get('objective_exact') is not None and old['objective_exact'] <= res['objective_exact']:
            return old
        with open(self._path(p), 'w', encoding='utf-8') as fh:
            json.dump(res, fh, ensure_ascii=False, default=float)
        return res

    def library(self) -> list[dict]:
        """미리 계산·저장된 시나리오 목록 (조건 설명 + 핵심 KPI)"""
        out = []
        for f in sorted(self.store.glob('*.json')):
            try:
                r = json.load(open(f, encoding='utf-8'))
                p = Params.model_validate(r['params'])
            except (OSError, ValueError, KeyError):       # 다른 프로세스가 쓰는 중이거나 옛 형식
                continue
            out.append(dict(조건=describe_params(p), params=p, kpi=r['kpi'], 목적함수=r.get('objective_exact', r.get('objective')), order=r['order']))
        return out

    def refresh_store(self) -> int:
        """저장된 결과를 현재 코드로 다시 평가·검사해 덮어쓴다 (평가 코드가 바뀐 뒤 실행)"""
        n = 0
        for f in sorted(self.store.glob('*.json')):
            r = json.load(open(f, encoding='utf-8'))
            p = Params.model_validate(r['params'])
            res = self.package(self.scenario(p), r['order'], r.get('name', 'TO-BE'), r.get('solver'))
            if res['violations']:
                f.unlink()                                   # 규칙을 어기는 해는 버린다
                continue
            with open(f, 'w', encoding='utf-8') as fh:
                json.dump(res, fh, ensure_ascii=False, default=float)
            n += 1
        return n

    def cross_seed(self) -> int:
        """저장된 해를 서로의 조건에 대입해 본다 — 더 엄격한 조건에서 찾은 해가 느슨한 조건에서도 더 좋을 수 있다
        (노트북 Step 8의 '찾은 해 중 조건을 만족하는 최선' 규칙). 교체된 개수를 돌려준다."""
        lib, n = self.library(), 0
        for tgt in lib:
            for src in lib:
                if src is tgt:
                    continue
                try:
                    sc = self.scenario(tgt['params'])
                    res = self.package(sc, src['order'], 'TO-BE', dict(solver='cross-seed', 출처=src['조건']))
                except ScenarioError:
                    continue
                if not res['violations'] and self._save_if_better(tgt['params'], res) is res:
                    n += 1
        return n

    def seed(self, p: Params, order, info: dict, name='TO-BE') -> dict:
        """다른 곳에서 구한 해(노트북 결과 등)를 캐시에 넣는다 — 제약 검사를 통과해야만"""
        sc = self.scenario(p)
        res = self.package(sc, order, name, info)
        if res['violations']:
            raise ScenarioError(f'제약 위반으로 저장하지 않음: {res["violations"]}')
        return self._save_if_better(p, res)

    # ── 최적화 ──
    def solve(self, p: Params, time_limit=None, gap=None, progress=None, use_cache=True) -> dict:
        """캐시에 있으면 즉시, 없으면 풀어서 저장. 반환 dict의 'cached'로 구분.
        해가 없으면 {'ok': False, 'reason': ...} (에이전트가 완화 후보를 제시하는 데 씀)"""
        if use_cache and (hit := self.cached(p)):
            return dict(hit, ok=True, cached=True)
        try:
            sc = self.scenario(p)
            t0 = time.time()
            prev = self.cached(p)
            order, info, _ = optimize(sc, time_limit=time_limit, gap=gap, progress=progress,
                                      extra=prev['order'] if prev else ())
        except ScenarioError as e:
            return dict(ok=False, reason=str(e))
        if order is None:
            return dict(ok=False, reason=f"{info.get('해 종류', '해 없음')} (LP 상태 {info.get('LP상태')}, {time.time() - t0:.0f}초)", solver=info)
        res = self.package(sc, order, 'TO-BE', info)
        if res['violations']:
            return dict(ok=False, reason=f'솔버 해가 제약 검사를 통과하지 못함: {res["violations"]}', solver=info)
        return dict(self._save_if_better(p, res), ok=True, cached=False)
