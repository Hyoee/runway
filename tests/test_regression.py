"""회귀 테스트 — runway 패키지가 노트북(RunWay_코스최적화_StepByStep_PuLP.ipynb)과 같은 숫자를 내는지.

기준값은 노트북 저장 출력과 out/result.json.
    pytest -q                   # 빠른 테스트 (~1분, 네트워크 첫 생성 포함)
    RUNWAY_SLOW=1 pytest -q     # MILP 전체 최적화까지 (수 분)
"""
import json
import os
from pathlib import Path

import pytest

from runway import Scenario, ScenarioError, build_asis, evaluate, kpis, load_network, optimize, solve_milp, validate_course
from runway.params import NOTEBOOK as DEFAULT          # 노트북(v1) 목적식으로 회귀 비교
from runway.evaluate import objective

ROOT = Path(__file__).resolve().parents[1]
RESULT = json.load(open(ROOT / 'out' / 'result.json', encoding='utf-8'))


@pytest.fixture(scope='session')
def net():
    return load_network(ROOT / 'data')


@pytest.fixture(scope='session')
def sc(net):
    return Scenario(net, DEFAULT)


def test_network_matches_notebook(net):          # Step 2·4 출력
    s = net.stats
    assert (s['교차로'], s['구간'], s['총km'], s['대상정류장'], s['연결정류장'], s['arc']) == (485, 741, 550, 4158, 2504, 1482)
    assert (s['되돌아오기쌍'], s['재방문쌍']) == (326, 1247)


def test_scenario_matches_notebook(sc):          # Step 4·5-0 출력
    assert int((sc.VK[0] > 50).sum()) == 62
    assert (int(sc.FORB.sum()), int(sc.near_forbid.sum()), int(sc.narrow.sum())) == (12, 10, 2)
    assert int(sc.OK_A.sum()) == 1470
    assert sc.net.nodes[sc.S] == pytest.approx((37.5700896, 126.9771588))
    assert sc.net.nodes[sc.F] == pytest.approx((37.524592, 127.117051))


def test_asis_kpis(sc):                          # Step 6 출력
    order, chk = build_asis(sc)
    assert order == RESULT['asis']['order']
    expect = {'length_km': 43.163, 'riders': 32110, 'routes': 156, 'stops': 157, 'view_m': 1168, 'late_km': 29.6,
              'closure_kmh': 91, 'junctions': 57, 'turns': 21, 'sharp': 6, 'backtrack_km': 5.9}
    got = kpis(evaluate(sc, order))
    assert {k: got[k] for k in expect} == expect
    assert [c['모델_km'] for c in chk][:3] == [5.0, 6.1, 8.4]


def test_saved_tobe_kpis_and_valid(sc):          # Step 9 TO-BE (저장된 코스를 같은 기준으로 재평가)
    order = RESULT['tobe']['order']
    got = kpis(evaluate(sc, order))
    expect = {k: v for k, v in RESULT['tobe']['eval'].items() if k not in ('rows', 'route_list')}
    assert {k: got[k] for k in expect} == expect
    assert validate_course(sc, order) == []


def test_asis_violates_rules(sc):                # AS-IS는 TO-BE의 규칙(길이·왕복·금지 구간)을 어긴다 — 문제점 지적의 근거
    errs = validate_course(sc, build_asis(sc)[0], check_identity=False, check_endpoints=False)
    assert any('길이' in e for e in errs) and any('왕복' in e for e in errs) and any('건국대병원' in e for e in errs)


def test_objective_recomputation_matches_solver(sc):   # 고정 코스 목적함수 = 솔버가 보고한 값(38,336)
    assert round(objective(sc, RESULT['tobe']['order'])) == pytest.approx(38336, abs=2)


@pytest.mark.parametrize('version', ['v1', 'v2'])
def test_milp_objective_equals_recomputation(net, version):
    """경로가 하나뿐인 회랑(저장된 코스의 arc만)에서 MILP를 풀면, 솔버 목적함수 = objective() 재계산이어야 한다.
    → 모델 수식과 재계산 함수가 같은 식임을 v1(노트북)·v2(우회 승객) 모두에서 확인"""
    from runway.params import DEFAULT as V2
    sc_ = Scenario(net, DEFAULT if version == 'v1' else V2)
    order = RESULT['tobe']['order']
    corridor = sorted(set(order) | {int(net.REV[a]) for a in order})   # 반대 방향 포함(쌍 벌점 판정용) — 경로는 여전히 하나
    out, info = solve_milp(sc_, corridor, time_limit=60)
    assert out == order
    assert info['목적함수'] == pytest.approx(objective(sc_, order), abs=2)


def test_route_volumes_sane(net):
    """노선별 이용객 합 = 주간 노선이 있는 정류장들의 승차 합 (균등 배분이 승객을 만들거나 잃지 않음)"""
    import numpy as np
    from runway.network import load_data
    agg = load_data(ROOT / 'data', 'weekend')[0]
    day = agg['routes'].astype(str).map(lambda s: any(not r.startswith('N') for r in s.split('|')))
    tot = (agg.loc[day, [f'h{h}' for h in range(6, 15)]].to_numpy(float) * agg.loc[day, 'f'].to_numpy(float)[:, None]).sum()
    assert sum(v.sum() for v in net.route_vol.values()) == pytest.approx(tot, rel=1e-9)
    assert not any(r.startswith('N') for r in net.route_vol)
    assert all(np.all(v >= 0) for v in net.route_vol.values())


def test_lp_relaxation_bound(sc):                # Step 7 ① LP 완화 하한 24,239 — 모델 수식이 같은지
    A_all = [a for a in range(sc.net.NA) if sc.OK_A[a]]
    _, info = solve_milp(sc, A_all, relax=True)
    assert info['목적함수'] == 24239


def test_params_immutable_and_scenarios_independent(net):
    p7 = DEFAULT.with_(T0=7.0)
    assert DEFAULT.T0 == 8.0 and p7.T0 == 7.0
    with pytest.raises(Exception):
        DEFAULT.T0 = 9.0                           # frozen
    a, b = Scenario(net, DEFAULT), Scenario(net, p7)
    assert (a.RID != b.RID).any()


def test_bad_places_rejected(net):
    with pytest.raises(ScenarioError):
        Scenario(net, DEFAULT.with_(START=(37.5257, 126.9227)))      # 여의도 — 분석 영역 밖
    with pytest.raises(ScenarioError):
        Scenario(net, DEFAULT.with_(FINISH=DEFAULT.START))


@pytest.mark.skipif(not os.environ.get('RUNWAY_SLOW'), reason='MILP 전체 실행은 RUNWAY_SLOW=1 일 때만')
def test_full_optimize(sc):
    order, info, _ = optimize(sc)
    assert order is not None and validate_course(sc, order) == []
    assert info['목적함수'] <= 38336 * (1 + DEFAULT.GAP) + 1          # 노트북 해와 gap 1% 이내
