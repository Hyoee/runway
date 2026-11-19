"""Step 5 — MILP 수식(5-4 목적함수, 5-5 제약식)을 PuLP로. 노트북 build_milp와 같은 식.

달라진 점: 전역 변수 대신 Scenario를 받고, 정체성 하한은 `identity_L`로 넘긴다(P를 바꾸지 않음).
"""
from __future__ import annotations

import math

import pulp

from .scenario import Scenario

# ── PuLP 3.x/4.x 호환 ──
PULP_V4 = hasattr(pulp.LpProblem, 'add_variable') and not hasattr(pulp.LpVariable, 'dicts')


def new_var(prob, name, lb=None, ub=None, cat='Continuous'):
    if hasattr(prob, 'add_variable'):
        return prob.add_variable(name, lowBound=lb, upBound=ub, cat=cat)
    return pulp.LpVariable(name, lowBound=lb, upBound=ub, cat=cat)


def new_vars(prob, name, keys, lb=None, ub=None, cat='Continuous'):
    return {k: new_var(prob, f'{name}_{i}', lb, ub, cat) for i, k in enumerate(keys)}


def tau(p, ang):
    return 0.0 if ang < 45 else p.TURN[0] if ang < 100 else p.TURN[1] if ang < 150 else p.TURN[2]


def build_milp(sc: Scenario, A, relax=False, identity_L=None):
    """arc 집합 A 위의 모델. relax=True면 LP 완화. identity_L: 랜드마크별 하한 덮어쓰기(파레토용)"""
    net, P = sc.net, sc.p
    U, V, SEG, REV, Lm, VK = net.U, net.V, net.SEG, net.REV, net.Lm, sc.VK
    lo_a, hi_a, s0, f0 = sc.lo_a, sc.hi_a, sc.s0, sc.f0
    IDENT = list(P.IDENTITY)
    Ls = [lm.L for lm in IDENT] if identity_L is None else list(identity_L)

    A = [int(a) for a in A]; Aset = set(A); cat = 'Continuous' if relax else 'Binary'
    Nn = sorted({int(U[a]) for a in A} | {int(V[a]) for a in A})
    out_, in_ = {v: [] for v in Nn}, {v: [] for v in Nn}
    for a in A:
        out_[int(U[a])].append(a); in_[int(V[a])].append(a)
    prob = pulp.LpProblem('RunWay', pulp.LpMinimize)

    # ── 결정변수 ──
    x = new_vars(prob, 'x', A, 0, 1, cat)
    p = {a: new_var(prob, f'p_{i}', 0, float(hi_a[a])) for i, a in enumerate(A)}
    B = P.BAND_KM * 1000; nb = int(math.ceil(P.LMAX / B)); Y = {}
    for a in A:
        if net.vol[a].sum() == 0 and VK[:, a].max() <= 1:
            continue
        for b in range(nb):
            lo, hi = b * B, min((b + 1) * B, P.LMAX)
            if hi > lo_a[a] and lo <= hi_a[a]:
                Y[(a, b)] = (lo, hi)
    y = new_vars(prob, 'y', list(Y), 0, 1, cat)
    T = [(a, c) for v in Nn for a in in_[v] for c in out_[v] if c != REV[a]]
    z = new_vars(prob, 'z', T, 0, 1)
    rmap = {}
    for a in A:
        for r in net.seg_routes[a]:
            rmap.setdefault(r, []).append(a)
    u = new_vars(prob, 'u', list(rmap), 0, 1)
    segA = {}
    for a in A:
        segA.setdefault(int(SEG[a]), []).append(a)
    Xs = {s_: pulp.lpSum(x[a] for a in As) for s_, As in segA.items()}
    pairs = {}
    for a in A:
        for b_, l in net.TWINS[a].items():
            s1, s2 = sorted((int(SEG[a]), int(SEG[b_])))
            if b_ in Aset and s1 != s2:
                pairs[(s1, s2)] = (max(pairs.get((s1, s2), (0, 0))[0], l), P.PEN_TWIN)
    for (s1, s2), l in net.NEAR.items():
        if s1 in segA and s2 in segA and (s1, s2) not in pairs:
            pairs[(s1, s2)] = (l, P.PEN_NEAR)
    q = new_vars(prob, 'q', list(pairs), 0)

    # ── 목적함수 (5-4) ──
    ta = {k: net.turn_angle(*k) for k in T}
    prob += (pulp.lpSum(sc.band_cost(a, *Y[(a, b)]) * y[(a, b)] for (a, b) in Y)                               # ① 승객 피해
             + P.ALPHA * pulp.lpSum(u.values())                                                               # ② 영향 노선
             + pulp.lpSum(tau(P, ta[k]) * z[k] for k in T if ta[k] >= 45)                                    # ③ 방향 전환
             + pulp.lpSum(float(pen * l) * q[k] for k, (l, pen) in pairs.items())                             # ④ 되돌아오기·재방문
             + pulp.lpSum(float(P.EPS_LEN * net.hw_w[a] * Lm[a]) * x[a] for a in A)                           # ⑤ 거리
             - pulp.lpSum(float(lm.w * VK[k, a]) * y[(a, b)] for k, lm in enumerate(IDENT)                    # ⑥ 정체성 보상
                          for (a, b) in Y if Y[(a, b)][0] >= lm.a and Y[(a, b)][1] <= lm.b))

    # ── 제약식 (5-5) ──
    for v in Nn:
        net_ = 1 if v == s0 else (-1 if v == f0 else 0)
        prob += pulp.lpSum(x[a] for a in out_[v]) - pulp.lpSum(x[a] for a in in_[v]) == net_                 # (1)
        prob += pulp.lpSum(x[a] for a in in_[v]) <= (0 if v == s0 else 1)                                    # (2)
        if v == s0:
            prob += pulp.lpSum(p[a] for a in out_[v]) == 0                                                   # (5)
        elif v != f0:
            prob += pulp.lpSum(p[a] for a in out_[v]) == pulp.lpSum(p[a] + float(Lm[a]) * x[a] for a in in_[v])
    zin, zout = {a: [] for a in A}, {a: [] for a in A}
    for (a, c) in T:
        zout[a].append(z[(a, c)]); zin[c].append(z[(a, c)])
    for a in A:
        if int(U[a]) != s0:
            prob += pulp.lpSum(zin[a]) == x[a]                                                               # (3)
        if int(V[a]) != f0:
            prob += pulp.lpSum(zout[a]) == x[a]
        prob += p[a] >= float(lo_a[a]) * x[a]                                                                # (5)
        prob += p[a] <= float(hi_a[a]) * x[a]
        r_ = int(REV[a])
        if r_ in Aset and a < r_:
            prob += x[a] + x[r_] <= 1                                                                        # (8)
    Ltot = pulp.lpSum(float(Lm[a]) * x[a] for a in A)
    prob += Ltot >= P.LMIN; prob += Ltot <= P.LMAX                                                           # (4)
    byA = {}
    for k_ in Y:
        byA.setdefault(k_[0], []).append(k_)
    for a, ks_ in byA.items():
        prob += pulp.lpSum(y[k_] for k_ in ks_) == x[a]                                                      # (6)
        prob += p[a] >= pulp.lpSum(Y[k_][0] * y[k_] for k_ in ks_)
        prob += p[a] <= pulp.lpSum(Y[k_][1] * y[k_] for k_ in ks_)
    for k, lm in enumerate(IDENT):
        prob += pulp.lpSum(float(VK[k, a]) * y[(a, b)] for (a, b) in Y
                           if Y[(a, b)][0] >= lm.a and Y[(a, b)][1] <= lm.b) >= Ls[k]                        # (7)
    for r, As in rmap.items():
        for a in As:
            prob += u[r] >= x[a]                                                                             # (10)
    for (s1, s2) in pairs:
        prob += q[(s1, s2)] >= Xs[s1] + Xs[s2] - 1                                                           # (11)
    nvar = len(x) + len(p) + len(y) + len(z) + len(u) + len(q)
    return prob, x, nvar
