"""Step 3·4(1)(2)·5-0 — 조건(Params)에 따라 달라지는 값들.

노트북에서는 전역 변수(RID, VK, FORB, lo_a, hi_a, S, F)였고, 조건을 바꾸면 해당 셀을 다시 실행해야 했다.
여기서는 `Scenario(net, params)` 하나가 이 값들을 모두 갖고, 조건이 다르면 새 Scenario를 만든다.
"""
from __future__ import annotations

import math
from functools import cached_property

import networkx as nx
import numpy as np

from .geo import hav
from .network import HOURS, Network
from .params import Params, in_bbox

MAX_SNAP_M = 1000.0         # 출발·도착 좌표에서 가장 가까운 교차로가 이보다 멀면 거절 (올림픽공원 기본값이 578m)


class ScenarioError(ValueError):
    """조건이 데이터·도로망과 맞지 않을 때 (에이전트가 사용자에게 설명할 메시지)"""


class Scenario:
    def __init__(self, net: Network, p: Params):
        self.net, self.p = net, p
        if getattr(net, 'daytype', p.DAYTYPE) != p.DAYTYPE:
            raise ValueError(f'네트워크 수요 기준({net.daytype})과 조건(DAYTYPE={p.DAYTYPE})이 다릅니다 — load_network(data, p)로 만드세요')
        for nm, ll in (('출발지', p.START), ('도착지', p.FINISH)):
            if not in_bbox(ll, p.AREA):
                raise ScenarioError(f'{nm} {ll}가 이 종목의 분석 영역(위도 {p.AREA[0]}~{p.AREA[1]}, 경도 {p.AREA[2]}~{p.AREA[3]}) 밖입니다.')
        self.S, dS = net.nearest(p.START)
        self.F, dF = net.nearest(p.FINISH)
        if dS > MAX_SNAP_M or dF > MAX_SNAP_M:
            raise ScenarioError(f'출발지/도착지 근처 {MAX_SNAP_M:.0f}m 안에 간선도로 교차로가 없습니다 (출발 {dS:.0f}m, 도착 {dF:.0f}m).')
        self.snap_m = (round(dS), round(dF))   # 사용자에게 "실제 출발 교차로는 ○m 떨어진 곳"이라고 알려 줄 값
        if self.S == self.F:
            raise ScenarioError('출발지와 도착지가 같은 교차로입니다. 이 모델은 출발≠도착(편도) 코스만 다룹니다.')
        nidx = {n: i for i, n in enumerate(net.nodes)}
        self.nidx = nidx
        self.s0, self.f0 = int(nidx[self.S]), int(nidx[self.F])
        self.K = int(p.LMAX / p.STEP)
        self.ks = np.arange(self.K + 1) * p.STEP

    # ── Step 3. 통제 시간 ──
    def window(self, km: float):
        p = self.p
        a, b = p.T0 + km / p.V_LEAD - p.PRE, p.T0 + km / p.V_SWEEP + p.POST
        if km * 1000 < p.STAGE[0]:
            a, b = min(a, p.STAGE[1]), max(b, p.STAGE[2])
        return (a, b)

    @staticmethod
    def riders(seg, win):
        a, b = win
        return sum(max(0.0, min(b, h + 1) - max(a, h)) * v for stp in seg['stops'] for h, v in stp['vol'].items())

    @cached_property
    def RID(self) -> np.ndarray:
        """RID[a, k] = arc a를 누적거리 k×STEP m에서 시작할 때의 피해 승객"""
        p, net, ks = self.p, self.net, self.ks
        HA = np.array(HOURS)
        R = np.zeros((net.NA, self.K + 1))
        for a in np.where(net.vol.sum(1) > 0)[0]:
            mid = (ks + net.Lm[a] / 2) / 1000
            w0 = p.T0 + mid / p.V_LEAD - p.PRE; w1 = p.T0 + mid / p.V_SWEEP + p.POST
            stage = mid * 1000 < p.STAGE[0]
            w0 = np.where(stage, np.minimum(w0, p.STAGE[1]), w0); w1 = np.where(stage, np.maximum(w1, p.STAGE[2]), w1)
            R[a] = np.clip(np.minimum(w1[:, None], HA[None, :] + 1) - np.maximum(w0[:, None], HA[None, :]), 0, None) @ net.vol[a]
        return R

    # ── Step 4(1). 정체성 v[k, e] ──
    @staticmethod
    def _facing(p, q, lm):
        mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        if hav(mid, lm.coord) > lm.dist:
            return False
        c = math.cos(math.radians(37.55))
        hv = ((q[1] - p[1]) * c, q[0] - p[0]); tv = ((lm.coord[1] - mid[1]) * c, lm.coord[0] - mid[0])
        n1, n2 = math.hypot(*hv), math.hypot(*tv)
        if n1 < 1e-12 or n2 < 1e-12:
            return False
        return math.degrees(math.acos(max(-1, min(1, (hv[0] * tv[0] + hv[1] * tv[1]) / (n1 * n2))))) <= lm.angle

    @cached_property
    def VK(self) -> np.ndarray:
        net = self.net
        VK = np.zeros((max(1, len(self.p.IDENTITY)), net.NA))
        for k, lm in enumerate(self.p.IDENTITY):
            for a in range(net.NA):
                pth = net.path_of(a)
                VK[k, a] = sum(hav(net.allnodes[x], net.allnodes[y]) for x, y in zip(pth[:-1], pth[1:])
                               if self._facing(net.allnodes[x], net.allnodes[y], lm))
        return VK

    # ── Step 4(2). 금지 구간 ──
    @cached_property
    def near_forbid(self) -> np.ndarray:
        net = self.net
        return np.array([any(hav(net.allnodes[n], z.coord) < z.r for z in self.p.FORBID for n in net.segs[net.SEG[a]]['path'])
                         for a in range(net.NA)], dtype=bool)

    @cached_property
    def narrow(self) -> np.ndarray:
        net = self.net
        return np.array([net.segs[net.SEG[a]]['lanes'] is not None and net.segs[net.SEG[a]]['lanes'] < self.p.MIN_LANES
                         for a in range(net.NA)], dtype=bool)

    @property
    def FORB(self) -> np.ndarray:
        return self.near_forbid | self.narrow

    # ── 5-0. arc 시작 누적거리 범위 ──
    @cached_property
    def _bounds(self):
        net, p = self.net, self.p
        DGL = nx.DiGraph()
        for a in range(net.NA):
            DGL.add_edge(int(net.U[a]), int(net.V[a]), w=float(net.Lm[a]))
        dS = nx.single_source_dijkstra_path_length(DGL, self.s0, weight='w')
        dF = nx.single_source_dijkstra_path_length(DGL.reverse(), self.f0, weight='w')
        lo = np.array([dS.get(int(net.U[a]), 1e9) for a in range(net.NA)])
        hi = np.array([p.LMAX - net.Lm[a] - dF.get(int(net.V[a]), 1e9) for a in range(net.NA)])
        return lo, hi, dS.get(self.f0)

    @property
    def lo_a(self):
        return self._bounds[0]

    @property
    def hi_a(self):
        return self._bounds[1]

    @property
    def OK_A(self) -> np.ndarray:
        ok = (self.lo_a <= self.hi_a) & ~self.FORB
        ok = ok & self.inside if self.p.AREA_LIMIT else ok
        ex = getattr(self, 'extra_forbid', None)          # 반사실 분석: 특정 도로를 빼고 다시 풀 때
        if ex:
            ok = ok & ~np.isin(self.net.SEG, list(ex))
        return ok

    @cached_property
    def inside(self) -> np.ndarray:
        """arc의 모든 점이 정류장 데이터 영역(BBOX) 안인가"""
        net = self.net
        return np.array([all(in_bbox(net.allnodes[n], self.p.AREA) for n in net.segs[net.SEG[a]]['path']) for a in range(net.NA)], dtype=bool)

    def check_feasible_basics(self):
        """솔버를 돌리기 전에 확실히 안 되는 조건을 빠르게 거른다"""
        d = self._bounds[2]
        if d is None:
            raise ScenarioError('출발지에서 도착지로 가는 간선도로 경로가 없습니다.')
        if d > self.p.LMAX:
            raise ScenarioError(f'출발→도착 최단거리 {d/1000:.1f}km가 코스 최대 길이 {self.p.LMAX/1000:.1f}km보다 깁니다.')
        for lm in self.p.IDENTITY:
            if not in_bbox(lm.coord, self.p.AREA):
                raise ScenarioError(f"랜드마크 '{lm.name}'가 분석 영역 밖입니다.")

    # ── 우회 노선 영향 (v2) ──
    def hour_overlap(self, win) -> np.ndarray:
        """통제 창 win이 각 시간대(06~14시)와 겹치는 시간(h)"""
        HA = np.array(HOURS)
        return np.clip(np.minimum(win[1], HA + 1) - np.maximum(win[0], HA), 0, None)

    def band_hours(self, lo, hi) -> np.ndarray:
        """구간대 [lo, hi) m 안에서 통제될 때 시간대별 평균 통제 시간 — 노선 우회 비용에 쓴다"""
        key = (round(lo), round(hi))
        c = self.__dict__.setdefault('_bh', {})
        if key not in c:
            kms = np.arange(lo, max(hi, lo + 1), 250.0) / 1000
            c[key] = np.mean([self.hour_overlap(self.window(k)) for k in kms], axis=0)
        return c[key]

    def route_band_cost(self, r, lo, hi) -> float:
        """노선 r이 구간대 [lo, hi)에서 통제될 때의 우회 비용 = DETOUR_W × 그 시간 동안의 노선 전체 이용객"""
        q = self.net.route_vol.get(r)
        return 0.0 if q is None else float(self.p.DETOUR_W * self.band_hours(lo, hi) @ q)

    def route_detour_riders(self, r, win) -> float:
        """노선 r이 통제 창 win 동안 우회할 때 영향 받는 노선 이용객 (가중치 없이)"""
        q = self.net.route_vol.get(r)
        return 0.0 if q is None else float(self.hour_overlap(win) @ q)

    def band_cost(self, a, lo, hi):
        st = self.p.STEP
        k0, k1 = int(max(lo, self.lo_a[a]) // st), int(min(hi, self.hi_a[a], self.p.LMAX) // st)
        k0 = min(max(k0, 0), self.K)                     # 이 종목 거리 범위 밖이면 끝 칸 (빈 구간 평균 NaN 방지)
        return float(self.RID[a, k0:max(k1, k0) + 1].mean())
