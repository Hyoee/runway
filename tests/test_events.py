"""종목(풀코스/하프/10K) 테스트 — 풀코스 불변, 하프 AS-IS 비교 없음, 하프·10K 랜드마크 미반영, 10K AS-IS 고정"""
import json
from pathlib import Path

import pytest

from runway import DEFAULT, Engine, evaluate, validate_course
from runway.asis import FROZEN_10K, frozen_asis, load_frozen
from runway.events import EVENTS, params_for, parse_event

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def eng():
    return Engine(ROOT)


def test_full_preset_is_default():
    assert params_for('full') == DEFAULT                         # 풀코스 설정은 기존과 완전히 같다


def test_full_results_unchanged(eng):
    """AS-IS는 스냅숏과 완전히 같고, 기본 조건(코스 단순성 ×2 고정)은 이전 '단순성 우선(×2)' 결과와 같다"""
    snap = json.load(open(ROOT / 'tests' / 'snapshots' / 'full_before_events.json', encoding='utf-8'))
    x2 = next(v for k, v in snap.items() if '×2' in k)
    for got, v, label in ((eng.asis(DEFAULT), snap['AS-IS'], 'AS-IS'), (eng.cached(DEFAULT), x2, '기본(×2)')):
        assert got['order'] == v['order'], label
        assert {k: got['kpi'][k] for k in v['kpi']} == v['kpi'], label


def test_tolerance_scales_with_distance():
    for k, ev in EVENTS.items():
        p = params_for(k)
        assert p.LMIN == ev.distance_m and p.LMAX == pytest.approx(ev.distance_m * 42600 / 42195)


@pytest.mark.parametrize('ev', ['half', '10k'])
def test_no_landmark_for_half_and_10k(eng, ev):
    p = params_for(ev)
    assert p.IDENTITY == ()                                       # 제약·보상에 쓰는 랜드마크 목록이 비어 있음
    assert float(eng.scenario(p).VK.sum()) == 0.0                 # 조망 길이 행렬도 0 → 목적함수·제약에 반영될 수 없음


def test_half_has_no_asis(eng):
    assert not EVENTS['half'].has_asis
    assert eng.asis(params_for('half')) is None


def test_10k_asis_frozen(eng):
    fz = load_frozen(ROOT / 'data', FROZEN_10K)
    assert fz['official_waypoints'] == ['상암 월드컵공원(출발)', '합정역', '양화대교 남단', '여의2교', '여의하류IC',
                                        'KBS 앞 교차로', '마포대교 남단', '여의도공원(도착)']
    assert '2026년 공개 코스 기준' in fz['sources'][0]
    p = params_for('10k')
    order, _ = frozen_asis(eng.scenario(p), ROOT / 'data', FROZEN_10K)
    assert 9.5 < evaluate(eng.scenario(p), order)['length_km'] < 11.0          # 10km에서 크게 벗어나지 않음


@pytest.mark.parametrize('ev', ['half', '10k'])
def test_event_tobe_valid(eng, ev):
    p = params_for(ev)
    r = eng.cached(p)
    assert r is not None, f'{ev} 사전 계산 결과 없음 — scripts/precompute.py --only 하프 10K'
    assert r['violations'] == []
    assert p.LMIN / 1000 - 1e-6 <= r['kpi']['length_km'] <= p.LMAX / 1000 + 1e-3


def test_agent_half_has_no_asis_comparison(eng):
    """하프: 에이전트 도구 결과에 AS-IS 비교 항목이 없고, '비교 안 함' 안내가 있다"""
    from agent import Session, make_tools
    s = Session(engine=eng)
    tools = {t.name: t for t in make_tools(s)}
    out = json.loads(tools['optimize_course'].call({'event': 'half'}))
    assert out['ok'] and out['종목'] == '하프'
    assert not any(k.startswith('AS_IS') for k in out) and '비교' in out
    assert json.loads(tools['get_asis_problems'].call({}))['ok'] is False
    txt = json.dumps(out, ensure_ascii=False)
    assert '롯데' not in txt and '조망' not in txt


def test_agent_10k_compares_with_asis(eng):
    from agent import Session, make_tools
    s = Session(engine=eng)
    out = json.loads({t.name: t for t in make_tools(s)}['optimize_course'].call({'event': '10킬로'}))
    assert out['ok'] and out['종목'] == '10K' and 'AS_IS대비_변화율_퍼센트' in out


def test_parse_event():
    assert [parse_event(t) for t in ['하프로', '10킬로', '10K 코스', '하프 마라톤', '풀코스로']] == ['half', '10k', '10k', 'half', 'full']
