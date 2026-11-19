"""Step 6 — AS-IS: 2026 JTBC 서울마라톤 공식 풀코스.

출처: 2026 JTBC 서울마라톤 대회 책자 p28~29 '풀코스 안내'(경유지 순서·km 표지·반환 구간), p17 경기일정(풀코스 07:30 출발, 13:00 종료),
      공지 https://marathon.jtbc.com/49/?bmode=view&idx=175312670 ("청계천 반환 지점(고산자교→비우당교)과
      피니시 직전 둔촌사거리 반환 왕복 구간")
공식 경유지: 광화문광장 → 숭례문 오거리 → 한국은행 앞 사거리 → 시청 삼거리 → 을지로1가 → 대림상가 → 을지로2가 → 비우당교
            → 모전교 → 종로구청입구 교차로 → 신설동 오거리 → 군자역 → 건대입구역 → 자양사거리 → 석촌호수사거리
            → 학원사거리 → 학여울역 → 수서IC → (둔촌사거리 반환) → 올림픽공원

경유지를 도로 그래프 최단 경로로 이어 만든 arc 순서는 data/asis_2026_official.json에 **고정 저장**하고, 서비스는 그 파일만 읽는다.
(경유지를 바꾸려면 의도적으로 scripts/freeze_asis.py를 다시 실행해야 하며, 테스트가 고정본과의 일치를 확인한다)
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from .scenario import Scenario

FROZEN = 'asis_2026_official.json'

ASIS_WP = [   # (이름, 좌표, 공식 약도의 km 표지 — 없으면 None). '(경로)'는 공식 경유지 사이 도로를 따라가게 하는 보조점
    ('광화문광장(출발)', (37.5753, 126.9771), 0.0), ('숭례문 오거리', (37.5600, 126.9755), None), ('한국은행 앞 사거리', (37.5622, 126.9805), None),
    ('시청 삼거리', (37.5655, 126.9778), None), ('을지로1가', (37.5660, 126.9822), None), ('대림상가(반환)', (37.5663, 126.9958), None),
    ('을지로2가', (37.5661, 126.9865), 5.0), ('(경로) 청계2가', (37.5687, 126.9870), None), ('(경로) 청계4가', (37.5687, 126.9978), 7.5),
    ('(경로) 청계7가', (37.5701, 127.0150), None), ('비우당교(반환)', (37.5718, 127.0247), 10.0), ('(경로) 청계7가', (37.5701, 127.0150), None),
    ('(경로) 청계4가', (37.5687, 126.9978), 12.5), ('모전교', (37.5690, 126.9790), None), ('종로구청입구 교차로', (37.5703, 126.9800), None),
    ('(경로) 종로5가', (37.5708, 127.0020), None), ('신설동 오거리', (37.5756, 127.0246), 17.5), ('군자역', (37.5571, 127.0794), 22.5),
    ('건대입구역', (37.5404, 127.0702), 25.0), ('자양사거리', (37.5368, 127.0823), None), ('석촌호수사거리', (37.5054, 127.1069), None),
    ('학원사거리', (37.5076, 127.0898), None), ('학여울역(반환)', (37.4967, 127.0713), 32.7), ('수서IC', (37.4896, 127.1092), 35.3),
    ('(경로) 양재대로', (37.4944, 127.1042), None), ('둔촌사거리(반환)', (37.5277, 127.1365), 41.0), ('올림픽공원(도착)', (37.5215, 127.1339), 42.195),
]


def load_frozen(data_dir) -> dict | None:
    f = Path(data_dir) / FROZEN
    return json.load(open(f, encoding='utf-8')) if f.exists() else None


def network_signature(net) -> str:
    """도로 그래프의 arc 구성 해시 — 고정 AS-IS의 arc 번호가 이 그래프와 맞는지 확인"""
    import hashlib
    return hashlib.sha1(json.dumps([(int(a[1]), int(a[2])) for a in net.arcs]).encode()).hexdigest()[:16]


def frozen_asis(sc: Scenario, data_dir):
    """고정된 공식 AS-IS (arc 순서, km 표지 대조). 파일이 없거나 도로 그래프가 달라졌으면 오류 — 조용히 다시 만들지 않는다."""
    fz = load_frozen(data_dir)
    if fz is None:
        raise FileNotFoundError(f'{FROZEN}가 없습니다. scripts/freeze_asis.py로 공식 AS-IS를 고정하세요.')
    if fz['network_signature'] != network_signature(sc.net):
        raise RuntimeError('도로 그래프가 고정 AS-IS를 만들 때와 다릅니다. AS-IS를 바꾸지 않으려면 그래프 설정을 되돌리세요.')
    return fz['order'], fz['km_check']


def build_asis(sc: Scenario, waypoints=ASIS_WP):
    """경유지 사이를 최단 경로로 이어 arc 순서를 만든다. 반환: (arc 순서, 공식 km 표지 검증표)"""
    net = sc.net
    DG = nx.DiGraph()
    for a in range(net.NA):
        DG.add_edge(net.arcs[a][1], net.arcs[a][2], w=net.Lm[a], arc=a)
    asis, chk, prev = [], [], net.nearest(waypoints[0][1])[0]
    for name, ll, official in waypoints[1:]:
        n = net.nearest(ll)[0]
        if n != prev:
            p = nx.shortest_path(DG, prev, n, weight='w'); asis += [DG[x][y]['arc'] for x, y in zip(p[:-1], p[1:])]
        km_now = sum(net.Lm[a] for a in asis) / 1000
        if official is not None:
            chk.append(dict(지점=name, 공식_km=official, 모델_km=round(km_now, 1), 차이_km=round(km_now - official, 1)))
        prev = n
    return asis, chk


# 노트북 v1이 쓴 경유지 (자양사거리·학원사거리 대신 잠실역·삼전역) — 노트북 결과 재현 테스트 전용. 서비스는 쓰지 않는다.
ASIS_WP_V1 = [(n, ll, k) for n, ll, k in ASIS_WP if n not in ('자양사거리', '학원사거리')]
ASIS_WP_V1.insert([n for n, _, _ in ASIS_WP_V1].index('석촌호수사거리'), ('잠실역', (37.5133, 127.1001), None))
ASIS_WP_V1.insert([n for n, _, _ in ASIS_WP_V1].index('학여울역(반환)'), ('삼전역', (37.5046, 127.0880), None))
