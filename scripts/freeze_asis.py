"""공식 AS-IS를 data/에 고정한다 — 풀코스(asis_2026_official.json), 10K(asis_10k_2026_official.json)

    python scripts/freeze_asis.py          # 풀코스
    python scripts/freeze_asis.py 10k      # 10K

AS-IS는 고정 자료다 — 서비스·테스트는 이 파일만 읽는다. 경유지(runway/asis.py ASIS_WP)를 의도적으로 바꿀 때만 다시 실행한다.
"""
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding='utf-8')

from runway import Engine  # noqa: E402
from runway.asis import ASIS_BY_FILE, build_asis, network_signature  # noqa: E402
from runway.events import EVENTS, params_for  # noqa: E402


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else 'full'
    ev = EVENTS[key]
    WP = ASIS_BY_FILE[ev.asis_file]
    eng = Engine(ROOT)
    sc = eng.scenario(params_for(key))
    order, chk = build_asis(sc, WP)
    srcs = {'full': ['2026 JTBC 서울마라톤 대회 책자 p28~29 풀코스 안내, p17 경기일정(풀코스 07:30 출발·13:00 종료)',
                     'https://marathon.jtbc.com/49/?bmode=view&idx=175312670 (청계천 비우당교 반환, 둔촌사거리 반환 왕복)'],
            '10k': ['2026 JTBC 서울마라톤 대회 책자 p34 10K 코스 안내 (2026년 공개 코스 기준), p17 경기일정(10K 07:30 출발·10:30 종료)']}
    out = dict(
        name=f'2026 JTBC 서울마라톤 {ev.label} (AS-IS, 공식 경유지)',
        sources=srcs[key],
        official_waypoints=[n for n, _, _ in WP if not n.startswith('(경로)')],
        waypoints=[dict(name=n, coord=list(ll), official_km=k) for n, ll, k in WP],
        order=[int(a) for a in order],
        coords=[[round(la, 6), round(lo, 6)] for la, lo in sc.net.coords(order)],
        km_check=[{k: (float(v) if not isinstance(v, str) else v) for k, v in c.items()} for c in chk],
        length_km=round(float(sum(sc.net.Lm[a] for a in order)) / 1000, 3),
        network_signature=network_signature(sc.net),
        frozen_on=str(date.today()),
    )
    f = ROOT / 'data' / ev.asis_file
    json.dump(out, open(f, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'고정: {f.name} — arc {len(order)}개, {out["length_km"]}km, 경유지 {len(out["official_waypoints"])}곳')


if __name__ == '__main__':
    main()
