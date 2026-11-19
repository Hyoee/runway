"""공식 AS-IS(2026 JTBC 서울마라톤 풀코스)를 data/asis_2026_official.json에 고정한다.

    python scripts/freeze_asis.py

AS-IS는 고정 자료다 — 서비스·테스트는 이 파일만 읽는다. 경유지(runway/asis.py ASIS_WP)를 의도적으로 바꿀 때만 다시 실행한다.
"""
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding='utf-8')

from runway import DEFAULT, Engine  # noqa: E402
from runway.asis import ASIS_WP, FROZEN, build_asis, network_signature  # noqa: E402


def main():
    eng = Engine(ROOT)
    sc = eng.scenario(DEFAULT)
    order, chk = build_asis(sc)
    out = dict(
        name='2026 JTBC 서울마라톤 풀코스 (AS-IS, 공식 경유지)',
        sources=['2026 JTBC 서울마라톤 대회 책자 p28~29 풀코스 안내, p17 경기일정(풀코스 07:30 출발·13:00 종료)',
                 'https://marathon.jtbc.com/49/?bmode=view&idx=175312670 (청계천 비우당교 반환, 둔촌사거리 반환 왕복)'],
        official_waypoints=[n for n, _, _ in ASIS_WP if not n.startswith('(경로)')],
        waypoints=[dict(name=n, coord=list(ll), official_km=k) for n, ll, k in ASIS_WP],
        order=[int(a) for a in order],
        coords=[[round(la, 6), round(lo, 6)] for la, lo in sc.net.coords(order)],
        km_check=[{k: (float(v) if not isinstance(v, str) else v) for k, v in c.items()} for c in chk],
        length_km=round(float(sum(sc.net.Lm[a] for a in order)) / 1000, 3),
        network_signature=network_signature(sc.net),
        frozen_on=str(date.today()),
    )
    f = ROOT / 'data' / FROZEN
    json.dump(out, open(f, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'고정: {f.name} — arc {len(order)}개, {out["length_km"]}km, 경유지 {len(out["official_waypoints"])}곳')


if __name__ == '__main__':
    main()
