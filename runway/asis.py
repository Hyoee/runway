"""Step 6 — AS-IS: 2026 JTBC 서울마라톤 공식 경유지를 도로 그래프 최단 경로로 이은 코스."""
from __future__ import annotations

import networkx as nx

from .scenario import Scenario

ASIS_WP = [   # (이름, 좌표, 공식 약도의 km 표지 — 없으면 None)
    ('광화문광장(출발)', (37.5753, 126.9771), 0.0), ('숭례문 오거리', (37.5600, 126.9755), None), ('한국은행 앞 사거리', (37.5622, 126.9805), None),
    ('시청 삼거리', (37.5655, 126.9778), None), ('을지로1가', (37.5660, 126.9822), None), ('대림상가(반환)', (37.5663, 126.9958), None),
    ('을지로2가', (37.5661, 126.9865), 5.0), ('청계2가', (37.5687, 126.9870), None), ('청계4가', (37.5687, 126.9978), 7.5),
    ('청계7가', (37.5701, 127.0150), None), ('비우당교(반환)', (37.5718, 127.0247), 10.0), ('청계7가', (37.5701, 127.0150), None),
    ('청계4가', (37.5687, 126.9978), 12.5), ('모전교', (37.5690, 126.9790), None), ('종로구청입구 교차로', (37.5703, 126.9800), None),
    ('종로5가', (37.5708, 127.0020), None), ('신설동 오거리', (37.5756, 127.0246), 17.5), ('군자역', (37.5571, 127.0794), 22.5),
    ('건대입구역', (37.5404, 127.0702), 25.0), ('잠실역', (37.5133, 127.1001), None), ('석촌호수사거리(석촌역)', (37.5054, 127.1069), None),
    ('삼전역', (37.5046, 127.0880), None), ('학여울역(반환)', (37.4967, 127.0713), 32.7), ('수서IC', (37.4896, 127.1092), 35.3), ('양재대로 진입', (37.4944, 127.1042), None),
    ('둔촌사거리(반환)', (37.5277, 127.1365), 41.0), ('올림픽공원 남2문(도착)', (37.5215, 127.1339), 42.195),
]


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
