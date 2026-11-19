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
from .evaluate import evaluate, kpis, road_table, to_geojson_feature, validate_course
from .network import load_network
from .params import DEFAULT, Params
from .places import search_places
from .scenario import Scenario, ScenarioError
from .solve import optimize

# 결과(문제 정의)에 영향을 주지 않는 필드 — 캐시 키에서 뺀다
_NON_PROBLEM = {'TIME_LIMIT', 'GAP', 'START_NAME', 'FINISH_NAME'}


def problem_key(p: Params) -> str:
    return p.key(*[f for f in Params.model_fields if f not in _NON_PROBLEM])


class Engine:
    def __init__(self, root: Path | str = '.'):
        self.root = Path(root)
        self.data_dir = self.root / 'data'
        self.store = self.root / 'scenarios'
        self.store.mkdir(exist_ok=True)
        self.net = load_network(self.data_dir)
        self._sc: OrderedDict[str, Scenario] = OrderedDict()
        self._lock = threading.Lock()

    # ── 조건 → Scenario (최근 8개 메모리 캐시) ──
    def scenario(self, p: Params) -> Scenario:
        k = problem_key(p)
        with self._lock:
            if k in self._sc:
                self._sc.move_to_end(k)
                return self._sc[k]
        sc = Scenario(self.net, p)                   # ScenarioError는 호출자에게 그대로
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
                    roads=road_table(ev, 300).to_dict('records'), geojson=to_geojson_feature(sc, order, name, ev),
                    order=[int(a) for a in order], violations=validate_course(sc, order, check_identity=name != 'AS-IS'),
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
        old = self.cached(p)
        obj_new, obj_old = res['solver'].get('목적함수'), (old or {}).get('solver', {}).get('목적함수')
        if old and obj_old is not None and (obj_new is None or obj_old <= obj_new):
            return old
        with open(self._path(p), 'w', encoding='utf-8') as fh:
            json.dump(res, fh, ensure_ascii=False, default=float)
        return res

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
