"""KPI 평가 — AS-IS와 TO-BE에 같은 기준. 노트북 evaluate()/road_table()과 같은 계산.

추가: validate_course() — 솔버 결과든 사람이 그린 코스든, 제약을 지키는지 독립적으로 검사한다.
"""
from __future__ import annotations

import pandas as pd

from .scenario import Scenario

KPI_LABELS = [  # 화면·보고에 쓰는 핵심 KPI (표시 이름, 키, 낮을수록 좋은가)
    ('못 타는 승객 (명/일요일)', 'riders', True),
    ('km당 못 타는 승객 (명/km)', 'riders_per_km', True),
    ('통제되는 정류장 (곳)', 'stops', True),
    ('영향 받는 버스 노선 (개)', 'routes', True),
    ('방향 전환 45°↑ (회)', 'turns', True),
    ('급한 방향 전환 100°↑ (회)', 'sharp', True),
    ('통제 필요 교차로 (곳)', 'junctions', True),
    ('도로 통제 부담 (km×시간)', 'closure_kmh', True),
    ('10시 이후까지 막히는 도로 (km)', 'late_km', True),
    ('코스 길이 (km)', 'length_km', None),
]


def _hm(t):
    return '%02d:%02d' % (int(t), round(t % 1 * 60) % 60)


def evaluate(sc: Scenario, order) -> dict:
    net = sc.net
    km, visits, rows, view = 0.0, {}, [], 0.0
    lm0 = sc.p.IDENTITY[0] if sc.p.IDENTITY else None
    for a in order:
        s = net.segs[net.SEG[a]]; w = sc.window((km + s['length'] / 2) / 1000)
        v = visits.setdefault(int(net.SEG[a]), [w[0], w[1]]); v[0] = min(v[0], w[0]); v[1] = max(v[1], w[1])
        if lm0 is not None and lm0.a <= km <= lm0.b:
            view += sc.VK[0, a]
        rows.append(dict(km=round(km / 1000, 2), 도로=s['name'] or '(이름 없음)', 길이m=round(s['length']), seg=int(net.SEG[a])))
        km += s['length']
    tot, rset, nstop, late, kmh = 0.0, set(), 0, 0.0, 0.0
    seg_r = {}
    for sid, w in visits.items():
        s = net.segs[sid]; seg_r[sid] = sc.riders(s, w); tot += seg_r[sid]; nstop += len(s['stops'])
        for stp in s['stops']:
            rset.update(stp['routes'])
        if w[1] > 10.0:
            late += s['length']
        kmh += s['length'] / 1000 * (w[1] - max(w[0], 6.0))
    done = set()
    for r in rows:
        w = visits[r['seg']]; first = r['seg'] not in done
        r['피해'] = round(seg_r[r['seg']]) if first else 0
        r['정류장'] = len(net.segs[r['seg']]['stops']) if first else 0     # 같은 구간을 다시 지나면 중복 집계하지 않음 (KPI 'stops'와 같은 기준)
        done.add(r['seg'])
        r['통제'] = f'{_hm(w[0])}–{_hm(w[1])}'
    # v2: 우회 노선 이용객 — 노선별 통제 창 = 그 노선 정류장이 있는 코스 구간들의 통제 창 합집합(첫 시작~마지막 해제)
    rwin = {}
    for sid, w in visits.items():
        for stp in net.segs[sid]['stops']:
            for r in stp['routes']:
                o = rwin.setdefault(r, [w[0], w[1]]); o[0] = min(o[0], w[0]); o[1] = max(o[1], w[1])
    detour = {r: sc.route_detour_riders(r, w) for r, w in rwin.items()}
    ta = [net.turn_angle(a, b) for a, b in zip(order, order[1:])]
    back, seen_a = 0.0, set()
    for b in order:
        back += max([net.TWINS[b].get(x, 0) for x in seen_a] + [0]); seen_a.add(b)
    L = km / 1000
    return dict(length_km=round(L, 3), riders=round(tot), routes=len(rset), stops=nstop, view_m=round(view),
                view_km=round(view / 1000, 1), late_km=round(late / 1000, 1), closure_kmh=round(kmh),
                junctions=len({x for a in order for x in (net.U[a], net.V[a])}),
                turns=int(sum(t >= 45 for t in ta)), sharp=int(sum(t >= 100 for t in ta)), backtrack_km=round(back / 1000, 1),
                riders_per_km=round(tot / L) if L else 0,
                detour_riders=round(sum(detour.values())),                                 # 우회 노선 이용객 (가중치 없음)
                citizen_impact=round(tot + sc.p.DETOUR_W * sum(detour.values())),          # 시민 영향 = 못 타는 승객 + δ×우회 이용객
                route_list=sorted(rset),
                route_detour={r: round(v) for r, v in sorted(detour.items(), key=lambda t: -t[1])}, rows=rows)


def kpis(ev: dict) -> dict:
    """rows·route_list 등 목록을 뺀 숫자 KPI만"""
    return {k: v for k, v in ev.items() if k not in ('rows', 'route_list', 'route_detour')}


def road_table(ev, min_len=300) -> pd.DataFrame:
    seq = []
    for r in ev['rows']:
        if not seq or seq[-1]['도로'] != r['도로']:
            seq.append(dict(km=r['km'], 도로=r['도로'], 길이km=0.0, 정류장=0, 피해=0, 통제=r['통제']))
        seq[-1]['길이km'] += r['길이m'] / 1000; seq[-1]['피해'] += r['피해']; seq[-1]['정류장'] += r.get('정류장', 0)
    df = pd.DataFrame([q for q in seq if q['길이km'] * 1000 >= min_len])
    if len(df):
        df['길이km'] = df['길이km'].round(1)
    return df


def compare(ev_a: dict, ev_b: dict, names=('AS-IS', 'TO-BE')) -> pd.DataFrame:
    rows = []
    for label, k, lower in KPI_LABELS:
        a, b = ev_a[k], ev_b[k]
        rows.append({'지표': label, names[0]: a, names[1]: b, '변화': f'{(b - a) / a * 100:+.0f}%' if a else '–'})
    return pd.DataFrame(rows)


def objective(sc: Scenario, order, exact: bool = False) -> float:
    """고정된 코스의 목적함수 값(5-4식)을 직접 계산 — 다른 조건에서 찾은 해를 같은 기준으로 비교할 때 쓴다.

    exact=False: MILP와 똑같은 식 (승객 피해를 3km 구간대 평균으로 근사) — 솔버 결과 검증용
    exact=True : 승객 항목만 화면 KPI와 같은 정밀 평가(10m 단위 통제 시간)로 바꾼 값 — '더 좋은 코스' 판단에 쓴다
                 (구간대 근사 때문에 MILP 값은 더 작아도 실제 못 타는 승객은 더 많은 코스가 생길 수 있어서)"""
    from .model import tau
    net, P = sc.net, sc.p
    B = P.BAND_KM * 1000
    used = {int(net.SEG[a]) for a in order}
    km, f = 0.0, 0.0
    if exact:
        ev = evaluate(sc, order)
        f += ev['riders'] + P.DETOUR_W * ev['detour_riders']                         # ①② 정밀 평가
    for a in order:                                                                  # ① 승객 피해 (구간대 평균) ⑥ 정체성 보상
        b = int(km // B)
        lo, hi = b * B, min((b + 1) * B, P.LMAX)
        if net.vol[a].sum() > 0 and not exact:
            f += sc.band_cost(a, lo, hi)
        for k, lm in enumerate(P.IDENTITY):
            if lo >= lm.a and hi <= lm.b:
                f -= lm.w * sc.VK[k, a]
        f += P.EPS_LEN * net.hw_w[a] * net.Lm[a]                                     # ⑤ 거리
        km += net.Lm[a]
    f += P.ALPHA * len({r for a in order for r in net.seg_routes[a]})                # ② 영향 노선 (v1)
    if P.DETOUR_W > 0 and not exact:                                                 # ② 우회 노선 이용객 (v2, MILP 식 10'과 동일)
        dmax, km = {}, 0.0
        for a in order:
            if net.vol[a].sum() > 0 or sc.VK[:, a].max() > 1:                     # MILP에서 y_{e,b}가 있는 arc
                b = int(km // B); lo, hi = b * B, min((b + 1) * B, P.LMAX)
                for r in net.seg_routes[a]:
                    dmax[r] = max(dmax.get(r, 0.0), sc.route_band_cost(r, lo, hi))
            km += net.Lm[a]
        f += sum(dmax.values())
    f += sum(tau(P, net.turn_angle(a, c)) for a, c in zip(order, order[1:]))          # ③ 방향 전환
    pairs = {}                                                                       # ④ 되돌아오기·재방문
    for s1 in used:
        for a in (2 * s1, 2 * s1 + 1):
            for b_, l in net.TWINS[a].items():
                s2 = int(net.SEG[b_])
                if s2 in used and s2 != s1:
                    k = tuple(sorted((s1, s2)))
                    pairs[k] = max(pairs.get(k, (0, 0))[0], l), P.PEN_TWIN
    for (s1, s2), l in net.NEAR.items():
        if s1 in used and s2 in used and (s1, s2) not in pairs:
            pairs[(s1, s2)] = (l, P.PEN_NEAR)
    f += sum(l * pen for l, pen in pairs.values())
    return float(f)


def validate_course(sc: Scenario, order, check_identity=True, check_endpoints=True) -> list[str]:
    """제약 위반 목록(비어 있으면 통과). 솔버 출력을 믿지 않고 독립적으로 다시 확인한다."""
    net, p, errs = sc.net, sc.p, []
    if not order:
        return ['코스가 비어 있습니다']
    if check_endpoints and int(net.U[order[0]]) != sc.s0:
        errs.append('출발 교차로에서 시작하지 않음')
    if check_endpoints and int(net.V[order[-1]]) != sc.f0:
        errs.append('도착 교차로에서 끝나지 않음')
    for a, b in zip(order, order[1:]):
        if net.V[a] != net.U[b]:
            errs.append(f'arc {a}→{b} 끊김'); break
    visited = [int(net.U[order[0]])] + [int(net.V[a]) for a in order]
    if len(set(visited)) != len(visited):
        errs.append('같은 교차로를 두 번 지남')
    segs_used = [int(net.SEG[a]) for a in order]
    if len(set(segs_used)) != len(segs_used):
        errs.append('같은 도로를 두 번 사용(왕복)')
    L = float(sum(net.Lm[a] for a in order))
    if not (p.LMIN - 1e-6 <= L <= p.LMAX + 1e-6):
        errs.append(f'길이 {L/1000:.3f}km가 허용 범위 밖')
    from .geo import hav
    for z in p.FORBID:
        if any(hav(net.allnodes[n], z.coord) < z.r for a in order for n in net.path_of(a)):
            errs.append(f'금지 구간 통과: {z.name} (반경 {z.r:g}m)')
    if any(sc.narrow[a] for a in order):
        errs.append('차로 수 부족 도로 사용')
    if p.NO_RETURN:
        used = set(order)
        inv = {x for (a, b) in net.RET if a in used and b in used for x in (a, b)}       # 회귀 쌍에 들어간 arc
        back = sum(net.Lm[x] for x in inv) / 2                                             # 갔다가 돌아온 길이 ≈ 관련 도로 길이의 절반
        if back:
            errs.append(f'회귀 구간 {back / 1000:.1f}km (반환 왕복·반대 방향으로 되돌아오기)')
        pos, km = {}, 0.0
        for a in order:
            pos[int(net.SEG[a])] = km; km += net.Lm[a]
        rv = [(s1, s2) for s1, s2 in net.CLOSE if s1 in pos and s2 in pos and abs(pos[s1] - pos[s2]) > p.REVISIT_GAP_KM * 1000 + 1e-6]
        if rv:
            errs.append(f'재방문 {len(rv)}곳 (코스상 {p.REVISIT_GAP_KM:g}km 넘게 떨어진 두 지점이 {p.REVISIT_R:g}m 안)')
    if p.AREA_LIMIT and not all(sc.inside[a] for a in order):
        errs.append('정류장 데이터 영역 밖 도로 사용')
    if check_identity:
        for k, lm in enumerate(p.IDENTITY):
            km, v = 0.0, 0.0
            for a in order:                     # MILP 식 (7)과 같이: 시작 지점이 [a_k, b_k] 안인 arc의 조망 길이
                if lm.a <= km <= lm.b:
                    v += sc.VK[k, a]
                km += net.Lm[a]
            if v + 1e-6 < lm.L:
                errs.append(f"'{lm.name}' 조망 {v:.0f}m < 하한 {lm.L:.0f}m")
    return errs


def to_geojson_feature(sc: Scenario, order, name, ev) -> dict:
    return dict(type='Feature', properties=dict(name=name, **kpis(ev)),
                geometry=dict(type='LineString', coordinates=[[round(lo, 6), round(la, 6)] for la, lo in sc.net.coords(order)]))
