"""Step 7 — LP 완화로 후보 회랑을 고르고, 회랑 위 MILP로 최적 코스를 구한다.

노트북 optimize()와 같은 2단계. 차이:
- 정체성 하한 없는 LP를 풀 때 P를 바꿨다 되돌리는 대신 identity_L=0을 넘긴다(동시 실행 안전).
- progress 콜백으로 단계별 진행 상황을 UI에 알린다.
"""
from __future__ import annotations

import re
import time
from typing import Callable

import pulp

from .model import build_milp
from .scenario import Scenario

Progress = Callable[[str], None]


def make_solver(time_limit, gap):
    avail = pulp.listSolvers(onlyAvailable=True)
    if 'HiGHS' in avail:
        return 'HiGHS', pulp.HiGHS(msg=False, timeLimit=time_limit, gapRel=gap)
    if 'PULP_CBC_CMD' in avail:
        return 'CBC', pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, gapRel=gap, logPath='cbc.log')
    if 'COIN_CMD' in avail:
        return 'CBC', pulp.COIN_CMD(msg=False, timeLimit=time_limit, gapRel=gap, logPath='cbc.log')
    raise RuntimeError(f'사용할 수 있는 솔버가 없습니다({avail}). `pip install "pulp[highs]"`를 실행하세요.')


def run_solver(prob, name, solver):
    res = prob.solve(solver)
    if hasattr(res, 'status_str'):                                 # PuLP 4.x
        st = str(res.status_str)
        has = bool(getattr(res, 'has_solution', st in ('Optimal', 'GapLimit', 'TimeLimit')))
        return st, has, st in ('Optimal', 'GapLimit'), getattr(res, 'gap_rel', None)
    st = pulp.LpStatus[prob.status]                                # PuLP 3.x
    sol = getattr(prob, 'sol_status', 1 if prob.status == 1 else 0)
    has, proved = prob.status == 1 and sol in (1, 2), sol == 1
    gap = None
    try:
        if name == 'CBC':
            txt = open('cbc.log', encoding='utf-8', errors='ignore').read()
            ob = float(re.findall(r'Objective value:\s+([-\d.eE+]+)', txt)[-1]); lb = float(re.findall(r'Lower bound:\s+([-\d.eE+]+)', txt)[-1])
            gap = abs(ob - lb) / max(1.0, abs(ob))
        else:
            gap = float(prob.solverModel.getInfo().mip_gap)
    except Exception:
        gap = getattr(prob, 'mip_gap', None)
    return st, has, proved, gap


def _n_constraints(prob):
    c = prob.constraints
    return len(c() if callable(c) else c)


def solve_milp(sc: Scenario, A, relax=False, time_limit=None, gap=None, identity_L=None):
    t0 = time.time()
    prob, x, nvar = build_milp(sc, A, relax, identity_L)
    name, solver = make_solver(time_limit or sc.p.TIME_LIMIT, gap or sc.p.GAP)
    st, has, proved, g = run_solver(prob, name, solver)
    info = dict(solver=name, 변수=nvar, 제약=_n_constraints(prob), 상태=st,
                목적함수=round(pulp.value(prob.objective) or 0) if has else None, 초=round(time.time() - t0, 1))
    if relax:
        return {a: (pulp.value(x[a]) or 0.0) for a in x}, info
    info['해 종류'] = '최적해(허용 gap 이내)' if proved else ('시간 제한 — 가능해' if has else '해 없음')
    info['gap'] = g
    if not has or any(pulp.value(x[a]) is None for a in x):
        return None, info
    U, V = sc.net.U, sc.net.V
    nxt = {int(U[a]): a for a in x if pulp.value(x[a]) > 0.5}
    order, v = [], sc.s0
    while v != sc.f0 and v in nxt and len(order) < 2000:
        order.append(nxt[v]); v = int(V[nxt[v]])
    return (order if v == sc.f0 else None), info


def optimize(sc: Scenario, time_limit=None, gap=None, extra=(), progress: Progress | None = None):
    """① LP 완화(현재 제약 + 정체성 하한 없이) → 회랑 ② 회랑 위 MILP. 반환: (arc 순서 또는 None, 풀이 정보, 회랑)"""
    say = progress or (lambda m: None)
    sc.check_feasible_basics()
    net, P = sc.net, sc.p
    A_all = [a for a in range(net.NA) if sc.OK_A[a]]
    say(f'① LP 완화로 후보 도로 고르는 중 (arc {len(A_all):,}개)')
    xv, info_lp = solve_milp(sc, A_all, relax=True)
    keep = {a for a in A_all if xv.get(a, 0) > P.LP_EPS}
    if P.IDENTITY:
        xv0, _ = solve_milp(sc, A_all, relax=True, identity_L=[0.0] * len(P.IDENTITY))
        keep |= {a for a in A_all if xv0.get(a, 0) > P.LP_EPS}
    keep |= {int(a) for a in extra}
    corr = sorted({b for a in keep for b in (a, int(net.REV[a])) if sc.OK_A[b]})
    say(f'② 후보 회랑 {len(corr)}개 arc 위에서 MILP 푸는 중 (최대 {time_limit or P.TIME_LIMIT:.0f}초)')
    order, info = solve_milp(sc, corr, time_limit=time_limit, gap=gap)
    info.update(LP하한=info_lp['목적함수'], LP초=info_lp['초'], LP상태=info_lp['상태'], 회랑arc=len(corr), 전체arc=len(A_all))
    say(f"완료: {info['해 종류']}, {info['초']}초")
    return order, info, corr
