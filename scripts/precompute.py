"""대표 시나리오를 미리 풀어 scenarios/에 저장한다 — 배포 앱은 이 결과를 즉시 보여 준다.

    python scripts/precompute.py            # 전체
    python scripts/precompute.py --seed     # 노트북 결과(out/result.json)만 기본 시나리오로 등록

결과가 이미 있으면 그 해를 회랑에 넣고 다시 풀어, 더 좋은 해를 찾을 때만 교체한다.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding='utf-8')

from runway import DEFAULT, Engine  # noqa: E402


def lotte(L):
    return (DEFAULT.IDENTITY[0].model_copy(update={'L': L}),)


SCENARIOS = {                                   # 이름: 조건 (데모·파레토·일정 질문에 바로 답하기 위한 것)
    '기본(08:00, 조망≥2km)': DEFAULT,
    '조망 하한 0km': DEFAULT.with_(IDENTITY=lotte(0.0)),
    '조망 하한 1km': DEFAULT.with_(IDENTITY=lotte(1000.0)),
    '조망 하한 3km': DEFAULT.with_(IDENTITY=lotte(3000.0)),
    '출발 07:00': DEFAULT.with_(T0=7.0),
    '출발 09:00': DEFAULT.with_(T0=9.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', action='store_true')
    ap.add_argument('--time-limit', type=float, default=600)
    ap.add_argument('--gap', type=float, default=0.005)
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    eng = Engine(ROOT)

    r = json.load(open(ROOT / 'out' / 'result.json', encoding='utf-8'))
    res = eng.seed(DEFAULT, r['tobe']['order'], r['solver'])
    print('seed 기본 시나리오:', res['kpi']['riders'], '명,', res['kpi']['routes'], '노선, 목적함수', res['solver'].get('목적함수'))
    if a.seed:
        return
    for name, p in SCENARIOS.items():
        if a.only and name not in a.only:
            continue
        t0 = time.time()
        out = eng.solve(p, time_limit=a.time_limit, gap=a.gap, use_cache=False, progress=lambda m: print('   ', m, flush=True))
        k = out.get('kpi', {})
        print(f"[{name}] ok={out['ok']} {time.time()-t0:.0f}초 | 피해 {k.get('riders')} 노선 {k.get('routes')} "
              f"조망 {k.get('view_km')}km 목적함수 {out.get('solver', {}).get('목적함수')} {out.get('reason', '')}", flush=True)


if __name__ == '__main__':
    main()
