"""Step 1·2·4(3) — 데이터 로드, 도로 그래프, 정류장 연결, 형상 특징(방향·되돌아오기·재방문).

조건(출발 시각·금지 구간·정체성 등)과 무관한 부분만 여기서 만든다. 한 번 만들어 pickle로 캐시한다.
노트북 코드를 순서·자료구조까지 그대로 옮겨서, arc 번호가 노트북과 같다(회귀 테스트로 확인).
"""
from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geo import hav
from .params import DEFAULT, NETWORK_FIELDS, Params

HOURS = list(range(6, 15))
EXCLUDE = ['강변북로', '올림픽대로', '동부간선', '내부순환', '북부간선', '서부간선', '분당수서', '강남순환', '고가', '터널']
KEEP = {'trunk', 'primary', 'secondary'}
KX, KY = 88200.0, 111000.0
CACHE_VERSION = 1


@dataclass
class Network:
    allnodes: dict
    segs: list
    nodes: dict
    arcs: list
    U: np.ndarray
    V: np.ndarray
    SEG: np.ndarray
    REV: np.ndarray
    Lm: np.ndarray
    vol: np.ndarray            # (NA, len(HOURS)) arc 위 정류장 시간대별 승차(주말 하루)
    seg_routes: list           # arc별 노선 집합
    hw_w: np.ndarray           # 도로 등급 계수
    HOUT: np.ndarray
    HIN: np.ndarray
    TWINS: dict
    NEAR: dict
    base: list                 # 배경 지도용 OSM 도로 좌표열
    stats: dict = field(default_factory=dict)

    @property
    def NA(self) -> int:
        return len(self.arcs)

    def path_of(self, a: int) -> list:
        sid, u, v = self.arcs[a]
        p = self.segs[sid]['path']
        return p if u == self.segs[sid]['u'] else p[::-1]

    def coords(self, order) -> list:
        pts = []
        for a in order:
            c = [self.allnodes[n] for n in self.path_of(a)]
            pts += c if not pts else c[1:]
        return pts

    def turn_angle(self, a: int, b: int) -> float:
        return abs((self.HOUT[b] - self.HIN[a] + 180) % 360 - 180)

    def nearest(self, ll) -> tuple:
        """가장 가까운 교차로 (노드 id, 거리 m)"""
        n = min(self.nodes, key=lambda n: hav(self.nodes[n], ll))
        return n, hav(self.nodes[n], ll)


# ───────────────────────── Step 1. 데이터 ─────────────────────────
def load_data(data_dir: Path, daytype: str):
    agg = pd.read_csv(data_dir / 'bus_stop_agg_202610.csv', dtype={'ars': str}, encoding='utf-8-sig')
    agg['ars'] = agg['ars'].str.zfill(5)
    xy = pd.read_excel(data_dir / 'bus_stop_xy_20260902.xlsx')
    xy['ars'] = xy['ARS_ID'].astype(int).astype(str).str.zfill(5)
    osm = json.load(open(data_dir / 'osm_roads.json', encoding='utf-8'))
    tunnel = set(int(x) for x in open(data_dir / 'tunnel_way_ids.txt').read().strip().split(','))

    dt = pd.read_csv(data_dir / 'bus_stop_daytype_202610.csv', dtype={'ars': str}, encoding='utf-8-sig')
    dt['ars'] = dt['ars'].str.zfill(5); dt = dt.groupby('ars', as_index=False).sum()
    T = dt['month_sum'] * 31 / 30
    dt['f_weekend'] = (dt['sat_sum'] + dt['sun_sum']) / 9 / T
    dt['f_sunday'] = dt['sun_sum'] / 4 / T
    dt['f_all'] = 1 / 31
    city = {k: float((dt['sat_sum'].sum() * (k != 'sunday') + dt['sun_sum'].sum()) / (9 if k == 'weekend' else 4) / T.sum())
            for k in ('weekend', 'sunday')}
    agg = agg.merge(dt[['ars', 'f_weekend', 'f_sunday', 'f_all']], on='ars', how='left')
    agg['f'] = agg['f_' + daytype].fillna(city.get(daytype, 1 / 31))
    return agg, xy, osm, tunnel


# ───────────────────────── Step 2. 도로 그래프 ─────────────────────────
def _build_segments(osm, tunnel, allnodes, junction_r):
    G = nx.Graph()
    for w in osm['ways']:
        if w['hw'] not in KEEP or any(x in w['name'] for x in EXCLUDE):
            continue
        ns = [n for n in w['nodes'] if n in allnodes]
        if w['id'] in tunnel and sum(hav(allnodes[a], allnodes[b]) for a, b in zip(ns[:-1], ns[1:])) > 400:
            continue
        ln = int(w['lanes']) if str(w['lanes']).isdigit() else None
        lanes = None if ln is None else (ln if w['oneway'] == 'yes' else ln / 2)
        for a, b in zip(ns[:-1], ns[1:]):
            if a != b and not G.has_edge(a, b):
                G.add_edge(a, b, length=hav(allnodes[a], allnodes[b]), name=w['name'], hw=w['hw'], lanes=lanes)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    # 2-2) 차수 2 노드 축약
    keep = {n for n in G.nodes if G.degree(n) != 2}
    segs, seen = [], set()
    for u in keep:
        for v in G.neighbors(u):
            if (u, v) in seen:
                continue
            path, prev, cur = [u, v], u, v
            while cur not in keep:
                nxt = [x for x in G.neighbors(cur) if x != prev][0]
                path.append(nxt); prev, cur = cur, nxt
            ed = list(zip(path[:-1], path[1:]))
            for a, b in ed:
                seen.add((a, b)); seen.add((b, a))
            names = [G[a][b]['name'] for a, b in ed if G[a][b]['name']]
            lk = sum(G[a][b]['length'] for a, b in ed if G[a][b]['lanes'] is not None)
            segs.append(dict(u=path[0], v=path[-1], path=path, length=sum(G[a][b]['length'] for a, b in ed),
                             name=max(set(names), key=names.count) if names else '', hw=G[path[0]][path[1]]['hw'],
                             lanes=sum(G[a][b]['lanes'] * G[a][b]['length'] for a, b in ed if G[a][b]['lanes'] is not None) / lk if lk else None,
                             lk=lk))
    best = {}
    for s in segs:
        if s['u'] != s['v']:
            k = tuple(sorted((s['u'], s['v'])))
            if k not in best or s['length'] < best[k]['length']:
                best[k] = s
    segs = list(best.values())

    # 2-3) 교차로 묶기 + 다시 축약
    ends = sorted({s['u'] for s in segs} | {s['v'] for s in segs})
    exy = np.array([(allnodes[n][1] * 88200, allnodes[n][0] * 111000) for n in ends])
    root = list(range(len(ends)))

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]; i = root[i]
        return i
    for i, j in cKDTree(exy).query_pairs(junction_r):
        root[find(i)] = find(j)
    rep_of = {n: ends[find(i)] for i, n in enumerate(ends)}
    members = {}
    for n, r in rep_of.items():
        members.setdefault(r, []).append(n)
    nodes = {r: tuple(float(c) for c in np.mean([allnodes[m] for m in ms], axis=0)) for r, ms in members.items()}
    best = {}
    for s in segs:
        u, v = rep_of[s['u']], rep_of[s['v']]
        if u == v:
            continue
        k = (min(u, v), max(u, v)); s = dict(s, u=u, v=v)
        if k not in best or s['length'] < best[k]['length']:
            best[k] = s
    segs = list(best.values())
    # 기본 출발·도착 교차로는 축약하지 않는다(노트북과 동일). 다른 출발·도착은 가장 가까운 교차로를 쓴다.
    keep_node = {rep_of.get(n) for n in [min(nodes, key=lambda n: hav(nodes[n], DEFAULT.START)),
                                          min(nodes, key=lambda n: hav(nodes[n], DEFAULT.FINISH))]}
    while True:
        inc = {}
        for i, s in enumerate(segs):
            inc.setdefault(s['u'], []).append(i); inc.setdefault(s['v'], []).append(i)
        n2 = None
        for n, l in inc.items():
            if len(l) != 2 or n in keep_node:
                continue
            a, b = segs[l[0]], segs[l[1]]
            ua = a['u'] if a['v'] == n else a['v']; vb = b['v'] if b['u'] == n else b['u']
            if ua != vb:
                n2 = n; break
        if n2 is None:
            break
        i, j = inc[n2]; a, b = segs[i], segs[j]
        pa = a['path'] if a['v'] == n2 else a['path'][::-1]; ua = a['u'] if a['v'] == n2 else a['v']
        pb = b['path'] if b['u'] == n2 else b['path'][::-1]; vb = b['v'] if b['u'] == n2 else b['u']
        main = a if a['length'] >= b['length'] else b
        lk = a['lk'] + b['lk']
        lanes = ((a['lanes'] or 0) * a['lk'] + (b['lanes'] or 0) * b['lk']) / lk if lk else None
        segs = [s for k, s in enumerate(segs) if k not in (i, j)] + [dict(u=ua, v=vb, path=pa + pb, length=a['length'] + b['length'],
                                                                         name=main['name'], hw=main['hw'], lanes=lanes, lk=lk)]
    for i, s in enumerate(segs):
        s['id'] = i; s['stops'] = []
    nodes = {n: nodes[n] for n in {s['u'] for s in segs} | {s['v'] for s in segs}}
    return segs, nodes


def _snap_stops(segs, agg, xy, allnodes):
    """2-4) 정류장 → 구간 (35m 이내)"""
    st = agg.merge(xy[['ars', 'X좌표', 'Y좌표']], on='ars', how='inner')
    st = st[st['Y좌표'].between(37.470, 37.600) & st['X좌표'].between(126.955, 127.150)].copy()
    kx, ky = 111320 * math.cos(math.radians(37.55)), 110540
    A, B, SID = [], [], []
    for s in segs:
        for a, b in zip(s['path'][:-1], s['path'][1:]):
            A.append((allnodes[a][1] * kx, allnodes[a][0] * ky)); B.append((allnodes[b][1] * kx, allnodes[b][0] * ky)); SID.append(s['id'])
    A, B, SID = np.array(A), np.array(B), np.array(SID)
    AB = B - A; ab2 = (AB ** 2).sum(1) + 1e-9
    n_snap = 0
    for _, r in st.iterrows():
        p = np.array((r['X좌표'] * kx, r['Y좌표'] * ky))
        t = np.clip(((p - A) * AB).sum(1) / ab2, 0, 1)
        d = np.sqrt(((A + AB * t[:, None] - p) ** 2).sum(1)); j = d.argmin()
        if d[j] <= 35:
            n_snap += 1
            segs[SID[j]]['stops'].append(dict(ars=r['ars'], name=r['name'], vol={h: r[f'h{h}'] * r['f'] for h in HOURS},
                                              routes=str(r['routes']).split('|')))
    return len(st), n_snap


# ───────────────────────── Step 4(3). 형상 특징 ─────────────────────────
def _bearing(p, q):
    return math.degrees(math.atan2(q[0] - p[0], q[1] - p[1])) % 360


def _head(pts, from_start, dist=40.0):
    seq = pts if from_start else pts[::-1]; acc, j = 0.0, 1
    while j < len(seq) - 1 and acc < dist:
        acc += np.hypot(*(seq[j] - seq[j - 1])); j += 1
    return _bearing(seq[0], seq[j]) if from_start else _bearing(seq[j], seq[0])


def _shape_features(net: Network, near_r):
    NA, SEG, REV, Lm, segs = net.NA, net.SEG, net.REV, net.Lm, net.segs
    xy_of = lambda a: np.array([(net.allnodes[n][1] * KX, net.allnodes[n][0] * KY) for n in net.path_of(a)])
    HOUT = np.zeros(NA); HIN = np.zeros(NA)
    PTS, PH, PA = [], [], []
    for a in range(NA):
        pts = xy_of(a); HOUT[a], HIN[a] = _head(pts, True), _head(pts, False)
        for i in range(1, len(pts)):
            L = float(np.hypot(*(pts[i] - pts[i - 1])))
            if L < 1e-6:
                continue
            h = _bearing(pts[i - 1], pts[i])
            for t in np.arange(0, L, 20.0):
                PTS.append(pts[i - 1] + (pts[i] - pts[i - 1]) * (t / L)); PH.append(h); PA.append(a)
    PTS, PH, PA = np.array(PTS), np.array(PH), np.array(PA)

    cnt = {}
    for i, nb in enumerate(cKDTree(PTS).query_ball_point(PTS, 50.0)):
        seen_a = set()
        for j in nb:
            b = int(PA[j])
            if SEG[b] == SEG[PA[i]] or b in seen_a:
                continue
            if abs((PH[i] - PH[j] + 180) % 360 - 180) > 150:
                seen_a.add(b); cnt[(int(PA[i]), b)] = cnt.get((int(PA[i]), b), 0) + 1
    TWINS = {a: {int(REV[a]): float(Lm[a])} for a in range(NA)}
    for (a, b), c in cnt.items():
        if c * 20.0 >= 120.0:
            TWINS[a][b] = max(TWINS[a].get(b, 0), c * 20.0); TWINS[b][a] = max(TWINS[b].get(a, 0), c * 20.0)

    def near_pairs(R, min_overlap=200.0):
        R_same, R_anti = R
        even = PA % 2 == 0
        pts, hd, sg = PTS[even], PH[even], SEG[PA[even]]
        ends = {int(s['id']): {s['u'], s['v']} for s in segs}
        out = {}
        for i, nb in enumerate(cKDTree(pts).query_ball_point(pts, R_anti)):
            si, seen = int(sg[i]), set()
            for j in nb:
                sj = int(sg[j])
                if sj <= si or sj in seen or ends[si] & ends[sj]:
                    continue
                d = abs((hd[i] - hd[j] + 180) % 360 - 180)
                if 30 < d < 150:
                    continue
                hx, hy = math.sin(math.radians(hd[i])), math.cos(math.radians(hd[i]))
                dx, dy = pts[j] - pts[i]
                if d <= 30 and (abs(dx * hy - dy * hx) < 60 or math.hypot(dx, dy) > R_same):
                    continue
                seen.add(sj); out[(si, sj)] = out.get((si, sj), 0) + 20.0
        return {k: v for k, v in out.items() if v >= min_overlap}
    return HOUT, HIN, TWINS, near_pairs(tuple(near_r))


# ───────────────────────── 조립 + 캐시 ─────────────────────────
def build_network(data_dir: Path, p: Params = DEFAULT) -> Network:
    agg, xy, osm, tunnel = load_data(data_dir, p.DAYTYPE)
    allnodes = {int(k): tuple(v) for k, v in osm['nodes'].items()}
    segs, nodes = _build_segments(osm, tunnel, allnodes, p.JUNCTION_R)
    n_area, n_snap = _snap_stops(segs, agg, xy, allnodes)

    arcs = []
    for s in segs:
        arcs += [(s['id'], s['u'], s['v']), (s['id'], s['v'], s['u'])]
    NA = len(arcs)
    nidx = {n: i for i, n in enumerate(nodes)}
    U = np.array([nidx[a[1]] for a in arcs], dtype=np.int32); V = np.array([nidx[a[2]] for a in arcs], dtype=np.int32)
    SEG = np.array([a[0] for a in arcs], dtype=np.int32)
    REV = np.array([a ^ 1 for a in range(NA)], dtype=np.int32)
    Lm = np.array([segs[a[0]]['length'] for a in arcs])
    vol = np.zeros((NA, len(HOURS)))
    for a in range(NA):
        for stp in segs[SEG[a]]['stops']:
            vol[a] += [stp['vol'][h] for h in HOURS]
    seg_routes = [set(r for stp in segs[SEG[a]]['stops'] for r in stp['routes']) for a in range(NA)]
    hw_w = np.array([1.0 if segs[SEG[a]]['hw'] in ('trunk', 'primary') else 1.6 for a in range(NA)])
    base = [[allnodes[n] for n in w['nodes'] if n in allnodes] for w in osm['ways']]

    net = Network(allnodes=allnodes, segs=segs, nodes=nodes, arcs=arcs, U=U, V=V, SEG=SEG, REV=REV, Lm=Lm, vol=vol,
                  seg_routes=seg_routes, hw_w=hw_w, HOUT=np.zeros(NA), HIN=np.zeros(NA), TWINS={}, NEAR={}, base=base)
    net.HOUT, net.HIN, net.TWINS, net.NEAR = _shape_features(net, p.NEAR_R)
    net.stats = dict(교차로=len(nodes), 구간=len(segs), 총km=round(sum(s['length'] for s in segs) / 1000),
                     대상정류장=n_area, 연결정류장=n_snap, arc=NA,
                     되돌아오기쌍=sum(len(d) - 1 for d in net.TWINS.values()) // 2, 재방문쌍=len(net.NEAR))
    return net


def load_network(data_dir: Path, p: Params = DEFAULT, cache_dir: Path | None = None) -> Network:
    """캐시가 있으면 읽고, 없으면 만들어 저장한다. 데이터 파일이 바뀌면 키가 달라져 다시 만든다."""
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir or data_dir.parent / 'cache')
    stamp = '-'.join(str(int(f.stat().st_mtime)) + str(f.stat().st_size) for f in sorted(data_dir.iterdir()) if f.is_file())
    import hashlib
    key = hashlib.sha1(f'{CACHE_VERSION}|{p.key(*NETWORK_FIELDS)}|{stamp}'.encode()).hexdigest()[:16]
    f = cache_dir / f'network_{key}.pkl'
    if f.exists():
        with open(f, 'rb') as fh:
            return pickle.load(fh)
    net = build_network(data_dir, p)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(f, 'wb') as fh:
        pickle.dump(net, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return net
