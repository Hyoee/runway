"""대표 시나리오를 미리 풀어 scenarios/에 저장한다 — 배포 앱은 이 결과를 즉시 보여 준다.

    python scripts/precompute.py              # 전체
    python scripts/precompute.py --seed       # 알려진 코스(노트북·v1 결과)만 각 시나리오에 후보로 등록
    python scripts/precompute.py --only 승객  # 이름 일부가 맞는 시나리오만
    python scripts/precompute.py --refresh    # 풀지 않고 저장 결과 재평가 + 서로 대입

결과가 이미 있으면 그 해를 회랑에 넣고 다시 풀어, 더 좋은 해(목적함수 재계산 기준)를 찾을 때만 교체한다.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding='utf-8')

from runway import DEFAULT, Engine, ScenarioError  # noqa: E402


def lotte(L):
    return (DEFAULT.IDENTITY[0].model_copy(update={'L': L}),)


PASSENGER_FIRST = DEFAULT.with_(TURN=tuple(t * 0.5 for t in DEFAULT.TURN))

SCENARIOS = {                                   # 이름: 조건 (데모·파레토·일정·가중치 질문에 바로 답하기 위한 것)
    '기본(일요일 08:00, 조망≥2km, 못 타는 승객 최소)': DEFAULT,
    '조망 하한 0km': DEFAULT.with_(IDENTITY=lotte(0.0)),
    '조망 하한 1km': DEFAULT.with_(IDENTITY=lotte(1000.0)),
    '조망 하한 3km': DEFAULT.with_(IDENTITY=lotte(3000.0)),
    '출발 07:00': DEFAULT.with_(T0=7.0),
    '출발 09:00': DEFAULT.with_(T0=9.0),
    '꺾임 덜 중시(벌점 ×0.5)': PASSENGER_FIRST,
    '참고: 우회 노선 영향 반영(δ=0.2)': DEFAULT.with_(DETOUR_W=0.2),
}


def known_orders():
    """지금까지 찾은 코스 — 노트북 결과 + v1 사전 계산 결과 (새 조건에서 출발 후보로 쓴다)"""
    out = [json.load(open(ROOT / 'out' / 'result.json', encoding='utf-8'))['tobe']['order']]
    for f in sorted((ROOT / 'scenarios_v1').glob('*.json')):
        out.append(json.load(open(f, encoding='utf-8'))['order'])
    return out


def seed_all(eng):
    orders = known_orders()
    for name, p in SCENARIOS.items():
        best = None
        for o in orders:
            try:
                res = eng.seed(p, o, dict(solver='seed(알려진 코스)'))
                best = res
            except ScenarioError:                 # 이 조건의 제약을 어기는 코스 (예: 조망 3km 미달)
                pass
        if best:
            print(f"seed [{name}] 시민영향 {best['kpi']['citizen_impact']:,} 못탐 {best['kpi']['riders']:,} 노선 {best['kpi']['routes']} 목적함수 {best['objective']:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', action='store_true')
    ap.add_argument('--time-limit', type=float, default=600)
    ap.add_argument('--gap', type=float, default=0.005)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--refresh', action='store_true')
    a = ap.parse_args()
    eng = Engine(ROOT)
    if a.refresh:
        print('재평가:', eng.refresh_store(), '개 | cross-seed 교체:', eng.cross_seed(), '개')
    else:
        seed_all(eng)
        if not a.seed:
            for name, p in SCENARIOS.items():
                if a.only and not any(o in name for o in a.only):
                    continue
                t0 = time.time()
                out = eng.solve(p, time_limit=a.time_limit, gap=a.gap, use_cache=False, progress=lambda m: print('   ', m, flush=True))
                k = out.get('kpi', {})
                print(f"[{name}] ok={out['ok']} {time.time()-t0:.0f}초 | 시민영향 {k.get('citizen_impact')} 못탐 {k.get('riders')} "
                      f"노선 {k.get('routes')} 목적함수 {out.get('objective')} {out.get('reason', '')}", flush=True)
            print('cross-seed 교체:', eng.cross_seed(), '개')
    for x in eng.library():
        k = x['kpi']
        print(f"  {x['조건']}: 시민영향 {k['citizen_impact']:,} 못탐 {k['riders']:,} 우회이용객 {k['detour_riders']:,} "
              f"노선 {k['routes']} 전환 {k['turns']} 조망 {k['view_km']}km 목적함수 {x['목적함수']:,}")


if __name__ == '__main__':
    main()
