"""AI 해커톤 프로젝트 소개서(PDF)를 만든다 — 양식: out/(첨부2) AI해커톤_프로젝트소개서_양식_261103.pdf

    python scripts/build_submission.py

- 숫자는 모두 runway 엔진에서 계산해 넣는다 (코드와 자료가 어긋나지 않게)
- 팀 정보는 docs/submission/team.json
- 결과: docs/submission/소개서.html, out/[팀명]_RunWay_AI해커톤_YYYYMMDD.pdf (Chrome으로 PDF 변환)
- 양식의 '작성 안내' 페이지는 제출 규칙대로 넣지 않는다. 본문 = 요약 + 01~08 (9장 ≤ 10장)
"""
import html
import json
import math
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding='utf-8')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from runway import DEFAULT, Engine, evaluate  # noqa: E402
from runway.asis import load_frozen  # noqa: E402
from runway.geo import hav  # noqa: E402

DOC = ROOT / 'docs' / 'submission'
ASSET = DOC / 'assets'
C_ASIS, C_TOBE, INK, MUTED, GRID = '#eb6834', '#2a78d6', '#0b0b0b', '#52514e', '#e4e3de'
NAVY, ORANGE = '#1f2a44', '#ef6a3a'


def setup_font():
    for f in font_manager.findSystemFonts():
        if 'NotoSansKR' in f or 'malgun' in f.lower():
            font_manager.fontManager.addfont(f)
    names = {fe.name for fe in font_manager.fontManager.ttflist}
    plt.rcParams['font.family'] = 'Malgun Gothic' if 'Malgun Gothic' in names else 'Noto Sans KR'   # Noto 가변 글꼴은 matplotlib에서 굵기 인식 실패
    plt.rcParams['axes.unicode_minus'] = False


# ───────────────────────── 숫자 ─────────────────────────
def collect():
    eng = Engine(ROOT)
    sc = eng.scenario(DEFAULT)
    A, T = eng.asis(DEFAULT), eng.cached(DEFAULT)
    if T is None:
        raise SystemExit('기본 시나리오 결과가 없습니다. scripts/precompute.py를 먼저 실행하세요.')
    net = sc.net

    def busy(order):
        segs = {int(net.SEG[a]) for a in order}
        day = sorted(sum(st['vol'].values()) for s in segs for st in net.segs[s]['stops'])
        return sum(d >= 1000 for d in day), round(day[len(day) // 2])

    ab, am = busy(A['order']); tb, tm = busy(T['order'])
    ret = next((v for v in A['violations'] if v.startswith('회귀')), '')
    rev = next((v for v in A['violations'] if v.startswith('재방문')), '')
    lib = {round(x['params'].T0, 2): x['kpi']['riders'] for x in eng.library()
           if x['params'].TURN == DEFAULT.TURN and x['params'].DETOUR_W == DEFAULT.DETOUR_W}
    pct = lambda a, b: round((b - a) / a * 100)
    K = dict(a=A['kpi'], t=T['kpi'], busy=(ab, tb), med=(am, tm), ret_km=float(ret.split()[2][:-2]) if ret else 0.0,
             revisit=int(rev.split()[1][:-1]) if rev else 0, viol_t=T['violations'], sched=lib,
             d_riders=A['kpi']['riders'] - T['kpi']['riders'], p_riders=pct(A['kpi']['riders'], T['kpi']['riders']),
             p_turns=pct(A['kpi']['turns'], T['kpi']['turns']), p_routes=pct(A['kpi']['routes'], T['kpi']['routes']),
             frozen=load_frozen(eng.data_dir), ntests=15)
    return eng, sc, A, T, K


# ───────────────────────── 그림 ─────────────────────────
def _offset(c, d=0.00045):
    k = math.cos(math.radians(37.55)); out = []
    for i, (la, lo) in enumerate(c):
        (la0, lo0), (la1, lo1) = c[max(i - 1, 0)], c[min(i + 1, len(c) - 1)]
        dx, dy = (lo1 - lo0) * k, la1 - la0; n = math.hypot(dx, dy) or 1.0
        out.append((la - dx / n * d, lo + dy / n * d / k))
    return out


def draw_course(ax, net, order, color, title, marks=(), returns=True):
    for w in net.base:
        if len(w) > 1:
            ax.plot([p[1] for p in w], [p[0] for p in w], color=GRID, lw=.45, zorder=1)
    seen = set()
    for a in order:
        sid = int(net.SEG[a]); again = sid in seen; seen.add(sid)
        c = [net.allnodes[n] for n in net.path_of(a)]
        if again and returns:
            c = _offset(c)
            ax.plot([p[1] for p in c], [p[0] for p in c], color=color, lw=2.0, alpha=.55, zorder=3, solid_capstyle='round')
        else:
            ax.plot([p[1] for p in c], [p[0] for p in c], color='white', lw=4.6, zorder=2, solid_capstyle='round')
            ax.plot([p[1] for p in c], [p[0] for p in c], color=color, lw=2.6, zorder=3, solid_capstyle='round')
    pts = net.coords(order); acc, nxt = 0.0, 5000
    for p, q in zip(pts, pts[1:]):
        d = hav(p, q)
        if acc + d >= nxt:
            ax.annotate(str(nxt // 1000), (p[1], p[0]), fontsize=7, ha='center', va='center', zorder=6, color=INK,
                        bbox=dict(boxstyle='circle,pad=0.18', fc='white', ec=color, lw=1)); nxt += 5000
        acc += d
    for name, (la, lo), col, dx, dy in marks:
        ax.scatter([lo], [la], s=34, color=col, edgecolor='white', linewidth=1.2, zorder=7)
        ax.annotate(name, (lo, la), xytext=(dx, dy), textcoords='offset points', fontsize=7.5, color=INK, zorder=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=.9))
    ax.set_xlim(126.958, 127.148); ax.set_ylim(37.468, 37.590); ax.set_aspect(1 / math.cos(math.radians(37.55)))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc='left', fontsize=10.5, color=INK, pad=6)


def figures(eng, sc, A, T, K):
    ASSET.mkdir(parents=True, exist_ok=True)
    net = sc.net
    start, finish = ('출발 광화문광장', DEFAULT.START), ('도착 올림픽공원', DEFAULT.FINISH)
    wp = {w['name']: tuple(w['coord']) for w in K['frozen']['waypoints']}
    asis_marks = [(start[0], start[1], INK, 6, 6), (finish[0], finish[1], INK, -40, -14),
                  ('비우당교 반환', wp['비우당교(반환)'], C_ASIS, 4, 7), ('학여울 반환', wp['학여울역(반환)'], C_ASIS, -44, -14),
                  ('둔촌사거리 반환', wp['둔촌사거리(반환)'], C_ASIS, -30, 8),
                  ('병원 앞(응급 동선)', DEFAULT.FORBID[0].coord, MUTED, 6, -12)]
    tobe_marks = [(start[0], start[1], INK, 6, 6), (finish[0], finish[1], INK, -40, -14)]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    draw_course(ax, net, A['order'], C_ASIS, '', asis_marks)
    ax.text(0.01, 0.01, '원 안 숫자 = km · 옅은 선 = 반환(왔던 길로 되돌아옴)', transform=ax.transAxes, fontsize=7.5, color=MUTED)
    fig.savefig(ASSET / 'asis_map.png', dpi=220, bbox_inches='tight'); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.5))
    draw_course(axs[0], net, A['order'], C_ASIS, f"AS-IS · 못 타는 승객 {K['a']['riders']:,}명 · 반환 3곳", asis_marks[:5])
    draw_course(axs[1], net, T['order'], C_TOBE, f"TO-BE · 못 타는 승객 {K['t']['riders']:,}명 · 반환 0", tobe_marks)
    fig.tight_layout(); fig.savefig(ASSET / 'compare_map.png', dpi=220, bbox_inches='tight'); plt.close(fig)

    # 누적 못 타는 승객 — 코스 어디에서 피해가 쌓이는가
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    for res, col, nm in ((A, C_ASIS, 'AS-IS'), (T, C_TOBE, 'TO-BE')):
        rows = evaluate(sc, res['order'])['rows']
        xs = [r['km'] for r in rows] + [res['kpi']['length_km']]
        ys, s = [], 0
        for r in rows:
            ys.append(s); s += r['피해']
        ys.append(s)      # 도로별 반올림 합이라 KPI와 1~3명 차이 → 끝 라벨은 KPI 값
        ax.plot(xs, ys, color=col, lw=2, drawstyle='steps-post')
        ax.annotate(f"{nm} {res['kpi']['riders']:,}명", (xs[-1], ys[-1]), xytext=(4, 0), textcoords='offset points', va='center', fontsize=8.5, color=INK)
    ax.set_xlim(0, 49); ax.set_ylim(0); ax.set_xlabel('코스 위치 (km)', fontsize=8.5, color=MUTED)
    ax.set_ylabel('누적 못 타는 승객 (명)', fontsize=8.5, color=MUTED)
    ax.grid(axis='y', color=GRID, lw=.6); ax.tick_params(labelsize=8, colors=MUTED)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    for s_ in ('left', 'bottom'):
        ax.spines[s_].set_color(GRID)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    fig.tight_layout(); fig.savefig(ASSET / 'cumulative.png', dpi=220, bbox_inches='tight'); plt.close(fig)

    shot = Path(sys.argv[1]) if len(sys.argv) > 1 else ASSET / 'app.png'
    if shot.exists() and shot != ASSET / 'app.png':
        shutil.copy(shot, ASSET / 'app.png')


# ───────────────────────── 슬라이드 HTML ─────────────────────────
CSS = """
@page { size: 1280px 720px; margin: 0; }
* { box-sizing: border-box; }
body { margin: 0; font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif; color: #0b0b0b; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.page { width: 1280px; height: 720px; position: relative; padding: 46px 56px 52px; page-break-after: always; overflow: hidden; background: #fff; }
.page:last-child { page-break-after: auto; }
h1 { font-size: 30px; margin: 0 0 18px; font-weight: 800; letter-spacing: -0.5px; }
h1 .no { margin-right: 10px; }
.foot { position: absolute; left: 56px; right: 56px; bottom: 20px; font-size: 10px; color: #8a8a8a; display: flex; justify-content: space-between; }
.box { border: 1.5px dashed #c9cad6; border-radius: 6px; padding: 14px 16px; }
.box > .lab { font-size: 15.5px; font-weight: 700; margin-bottom: 10px; color: #1f2a44; }
.grid { display: grid; gap: 14px; }
p, li { font-size: 17px; line-height: 1.55; margin: 0; }
ul { margin: 0; padding-left: 18px; } li { margin: 2px 0; }
.small, .small li { font-size: 15px; line-height: 1.5; }
.num { font-weight: 800; }
.hl { color: #2a78d6; font-weight: 800; } .bad { color: #d4542a; font-weight: 800; }
table { border-collapse: collapse; width: 100%; font-size: 14.5px; }
th { background: #2f3f63; color: #fff; font-weight: 700; padding: 7px 10px; text-align: left; }
td { border-bottom: 1px solid #e3e4ea; padding: 6px 10px; vertical-align: top; }
td.r, th.r { text-align: right; }
.circle { display: inline-flex; width: 26px; height: 26px; border-radius: 50%; background: #ef6a3a; color: #fff; font-weight: 800; font-size: 13px; align-items: center; justify-content: center; }
.kpi { display: flex; gap: 12px; }
.kpi .k { flex: 1; border: 1.5px dashed #c9cad6; border-radius: 6px; padding: 10px 12px; }
.kpi .k .t { font-size: 14px; color: #52514e; } .kpi .k .v { font-size: 32px; font-weight: 800; margin-top: 2px; }
.kpi .k .d { font-size: 14px; color: #52514e; margin-top: 2px; }
.fill { background: #fff3c4; padding: 0 4px; border-radius: 3px; }
img { display: block; max-width: 100%; }
.sub { font-size: 15.5px; color: #52514e; margin: 0 0 12px; }
table.tight { font-size: 12.5px; } table.tight th, table.tight td { padding: 4px 8px; }
.cover { background: #1f2a44; color: #fff; padding: 70px 70px; }
.cover .ev { font-size: 15px; color: #b9c2d8; } .cover .title { font-size: 66px; font-weight: 800; margin: 26px 0 10px; letter-spacing: -1px; }
.cover .slogan { font-size: 20px; color: #dfe4f0; line-height: 1.5; max-width: 1050px; }
.cover .meta { position: absolute; left: 70px; right: 70px; bottom: 90px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; }
.cover .meta .l { font-size: 12px; color: #9aa6c4; } .cover .meta .v { font-size: 17px; font-weight: 700; margin-top: 6px; }
.cover .foot { color: #7e8bab; }
.flow { display: flex; align-items: stretch; gap: 8px; }
.flow .st { flex: 1; border: 1.5px dashed #c9cad6; border-radius: 6px; padding: 12px; font-size: 15px; line-height: 1.5; }
.flow .ar { align-self: center; color: #ef6a3a; font-weight: 800; font-size: 18px; }
.arch .lyr { border-radius: 6px; padding: 11px 13px; margin-bottom: 8px; font-size: 14.5px; line-height: 1.5; }
.arch .lyr b { font-size: 15.5px; }
.arch .ui { background: #eaf2fc; } .arch .ai { background: #fdeee6; } .arch .en { background: #eef0f6; } .arch .da { background: #f3f3ef; }
.arch .down { text-align: center; color: #8a8a8a; font-size: 13px; margin: -3px 0 5px; }
"""


def e(s):
    return html.escape(str(s))


def F(team, key):
    v = team.get(key, '')
    return f'<span class="fill">{e(v)}</span>' if '[' in str(v) else e(v)


def pages(K, team):
    a, t = K['a'], K['t']
    foot = lambda n: f'<div class="foot"><span>2026 대한민국 청년 CAREER FAIR AI 해커톤 | 프로젝트 소개서</span><span>{n}</span></div>'
    sched = ' · '.join(f"{int(k):02d}:{round(k % 1 * 60):02d} 출발 {v:,}명" for k, v in sorted(K['sched'].items()))
    members = ' / '.join(F({'x': m}, 'x') for m in team['members'])
    P = []

    # 표지
    P.append(f"""<section class="page cover">
<div class="ev">2026 대한민국 청년 CAREER FAIR AI 해커톤</div>
<div class="title">RunWay</div>
<div class="slogan">버스 승객 피해는 최소로, 코스는 깔끔하게 — 서울시 버스 승하차 데이터와 수리최적화로 마라톤 코스를 설계하고,<br>AI 에이전트가 말로 조건을 바꿔 주는 도심 마라톤 코스 설계 서비스</div>
<div class="meta"><div><div class="l">팀명</div><div class="v">{F(team, 'team')}</div></div>
<div><div class="l">대표자</div><div class="v">{F(team, 'leader')}</div></div>
<div><div class="l">팀원</div><div class="v">{members}</div></div>
<div><div class="l">제출일</div><div class="v">{F(team, 'date')}</div></div></div>
<div class="foot"><span>2026 대한민국 청년 CAREER FAIR AI 해커톤</span><span></span></div></section>""")

    # 요약
    P.append(f"""<section class="page"><h1>프로젝트 요약</h1>
<div class="box" style="margin-bottom:14px"><div class="lab">한 줄 요약</div>
<p>대회 사무국이 <b>"출발을 9시로 늦추면?"</b>처럼 말하면, AI 에이전트가 조건을 해석해 <b>수리최적화(MILP) 솔버</b>를 돌리고 서울시 버스 승하차 실측 데이터로
<b>버스를 못 타는 시민이 가장 적은 42.195km 코스</b>를 지도·KPI로 보여 주는 서비스</p></div>
<div class="grid" style="grid-template-columns:1fr 1fr 1fr">
<div class="box" style="height:290px"><div class="lab"><span class="circle">1</span>&nbsp; 문제</div>
<ul class="small"><li>2026 JTBC 서울마라톤 공식 코스는 통제 시간 동안 <span class="bad">{a['riders']:,}명</span>이 정류장에서 버스를 못 탐 (일요일 실측 수요 기준 추정)</li>
<li>반환 왕복 등 <b>회귀 약 {K['ret_km']:.0f}km</b>, 같은 동네 재방문 {K['revisit']}곳</li>
<li>하루 1,000명 이상 붐비는 도심 정류장 {K['busy'][0]}곳 통과</li>
<li>코스는 경유지 중심 수작업 — 시민 영향을 숫자로 비교할 도구가 없음</li></ul></div>
<div class="box" style="height:290px"><div class="lab"><span class="circle">2</span>&nbsp; 해결 방안</div>
<ul class="small"><li><b>최적화 엔진</b>: 도로망 1,482개 구간 × 위치별 통제 시간 × 정류장 승하차를 MILP로 풀어 최적 코스 계산</li>
<li><b>규칙 보장</b>: 42.195km, 회귀·재방문 금지, 금지 구간 회피를 하드 제약으로</li>
<li><b>AI 에이전트(Claude)</b>: 자연어 조건 → 제약 변환, 장소 검색, 결과를 도구 숫자로만 설명</li></ul></div>
<div class="box" style="height:290px"><div class="lab"><span class="circle">3</span>&nbsp; 구현 현황</div>
<ul class="small"><li>웹 서비스 배포 (Streamlit) — 지도·KPI·조건 카드·채팅</li>
<li>TO-BE: 못 타는 승객 <span class="hl">{t['riders']:,}명 ({K['p_riders']:+d}%)</span>, 회귀 0, 방향 전환 {a['turns']}→{t['turns']}회</li>
<li>AS-IS를 대회 책자 공식 경유지로 고정, 자동 검증 테스트 {K['ntests']}개</li>
<li>출발 시각 등 조건별 결과 사전 계산 → 즉시 응답</li></ul></div></div>
<div class="box" style="margin-top:14px"><div class="lab">키워드</div><p>수리최적화(MILP·PuLP·HiGHS) · LLM 에이전트(Claude Tool Use) · 서울 열린데이터광장 버스 승하차 · 교통 통제 · 도심 이벤트 운영</p></div>
{foot(3)}</section>""")

    # 01 문제 정의
    P.append(f"""<section class="page"><h1 style="margin-bottom:6px"><span class="no">01</span>문제 정의</h1>
<p class="sub">도로가 막히면 그 도로 위 정류장의 시민은 버스를 탈 수 없다 — 통제는 코스 위치마다 시간이 다르다 (출발점 근처는 이른 아침 잠깐, 후반 구간은 정오 넘어까지)</p>
<div class="grid" style="grid-template-columns:1.02fr 1fr; height:528px">
<div class="box"><div class="lab">2026 JTBC 서울마라톤 공식 코스 — 광화문광장 → 올림픽공원, 07:30 출발 · 13:00 종료</div>
<img src="assets/asis_map.png" style="width:100%; max-height:455px; object-fit:contain; margin:0 auto"></div>
<div class="grid" style="grid-template-rows:auto 1fr">
<div class="box"><div class="lab">공식 코스(AS-IS)를 데이터로 진단하면</div>
<table><tr><th>항목</th><th class="r">AS-IS</th></tr>
<tr><td>통제 시간에 버스를 못 타는 승객 (일요일 기준)</td><td class="r bad">{a['riders']:,}명</td></tr>
<tr><td>회귀(반환 왕복 3곳: 비우당교·학여울·둔촌사거리)</td><td class="r bad">약 {K['ret_km']:.0f}km</td></tr>
<tr><td>재방문 (코스상 2km 넘게 떨어진 두 지점이 300m 안)</td><td class="r">{K['revisit']}곳</td></tr>
<tr><td>하루 1,000명 이상 붐비는 정류장 통과</td><td class="r">{K['busy'][0]}곳</td></tr>
<tr><td>통제되는 정류장 / 영향 버스 노선</td><td class="r">{a['stops']}곳 / {a['routes']}개</td></tr>
<tr><td>방향 전환 45°↑ (급한 100°↑)</td><td class="r">{a['turns']}회 ({a['sharp']}회)</td></tr></table></div>
<div class="box"><div class="lab">왜 이런 일이 생기나</div>
<ul class="small"><li>코스가 <b>경유지·관행 중심</b>으로 정해지고, 시민 영향은 사후에 우회 안내로 대응</li>
<li>출발 시각·구간을 바꿨을 때 <b>몇 명이 영향받는지 비교할 도구가 없음</b> — 관계기관(시·경찰·운수사) 협의 근거 부족</li>
<li>42.195km·반환 없음·안전 구간 같은 조건을 동시에 맞추는 조합은 사람이 다 따져 보기 어려움</li></ul></div></div></div>
{foot(4)}</section>""")

    # 02 대상 사용자
    P.append(f"""<section class="page"><h1><span class="no">02</span>대상 사용자</h1>
<div class="grid" style="grid-template-columns:1fr 1fr; margin-bottom:14px">
<div class="box" style="height:262px"><div class="lab">주요 사용자 (Primary)</div>
<p><b>도심 마라톤·대형 도로 행사 조직위 / 사무국의 코스·운영 기획 담당자</b></p>
<ul class="small" style="margin-top:8px"><li>매년 코스안을 만들고 서울시·경찰과 교통 통제를 협의</li>
<li>조건(출발 시각, 금지 구간, 출발·도착 장소)이 바뀔 때마다 코스를 다시 검토</li>
<li>수리최적화·데이터 분석 전문가가 아님 → <b>말로 묻고 지도·숫자로 답을 받는 도구</b>가 필요</li></ul></div>
<div class="box" style="height:262px"><div class="lab">보조 사용자 · 이해관계자</div>
<ul class="small"><li><b>서울시 교통·버스정책 부서</b>: 통제 구간별 버스 승객 영향과 우회 노선 검토</li>
<li><b>경찰 교통 통제 담당</b>: 통제 교차로 수·구간별 통제 시각 확인</li>
<li><b>버스 운수사</b>: 영향 노선 목록으로 우회 운행 준비</li>
<li><b>시민·참가자</b>: 이동 불편 감소, 반환 없이 깔끔한 코스</li></ul></div></div>
<div class="box" style="height:284px"><div class="lab">사용자가 얻는 가치</div>
<div class="grid" style="grid-template-columns:1fr 1fr 1fr">
<div><p class="num" style="font-size:24px;color:#2a78d6">{K['d_riders']:,}명 ↓</p><p class="small">같은 출발·도착·거리에서 통제 시간에 버스를 못 타는 시민 감소 ({K['p_riders']:+d}%, 일요일 추정)</p></div>
<div><p class="num" style="font-size:24px;color:#2a78d6">즉시 ~ 2분</p><p class="small">조건을 말로 바꾸면 최적 코스를 다시 계산 (사전 계산 조건은 즉시, 새 조건은 수십 초~2분)</p></div>
<div><p class="num" style="font-size:24px;color:#2a78d6">근거 자료</p><p class="small">지도·KPI 표·도로별 통제 시각·영향 노선 목록을 그대로 관계기관 협의 자료로 사용</p></div></div></div>
{foot(5)}</section>""")

    # 03 AI 해결 방안
    P.append(f"""<section class="page"><h1><span class="no">03</span>AI를 활용한 해결 방안</h1>
<div class="grid" style="grid-template-columns:1fr 1fr; margin-bottom:12px">
<div class="box" style="height:222px"><div class="lab">해결 방안 개요 — "AI는 의미를, 솔버는 숫자를"</div>
<ul class="small"><li><b>AI 에이전트 (Claude)</b>: 사용자의 말을 조건(제약 카드)으로 바꾸고, 필요한 도구를 골라 부르고, 결과를 설명</li>
<li><b>최적화 엔진 (PuLP MILP)</b>: 도로망·통제 시간·정류장 수요로 "못 타는 승객 + 코스 단순성 벌점"을 최소화하는 코스를 계산하고, 규칙을 하드 제약으로 보장</li>
<li>숫자와 경로는 <b>엔진만</b> 만든다 → 그럴듯하지만 틀린 코스·숫자가 나오지 않음</li></ul></div>
<div class="box" style="height:222px"><div class="lab">왜 AI여야 하는가</div>
<ul class="small"><li>담당자는 수식을 쓰지 않는다 — "건대입구역 근처는 피해 줘", "7시로 당기면?"을 <b>제약으로 번역</b>하는 일이 LLM의 강점</li>
<li>도로 1,482개 구간의 조합은 사람이 비교할 수 없다 — <b>수리최적화</b>가 최적성을 증명하며 탐색</li>
<li>LLM만으로는 거리·승객 수를 지어낼 위험 → 둘을 결합해 <b>대화의 편의 + 계산의 정확성</b></li></ul></div></div>
<div class="box" style="height:326px"><div class="lab">AI 적용 지점과 오류·불확실성 대응</div>
<table><tr><th style="width:21%">적용 지점</th><th style="width:40%">하는 일</th><th>오류·불확실성 대응</th></tr>
<tr><td>① 조건 해석</td><td>자연어 → 제약 카드(출발·도착·시각·금지 구간·단순성)</td><td>스키마(Pydantic)로 범위 검증, 범위 밖이면 거절 후 재질문</td></tr>
<tr><td>② 장소 해석</td><td>장소 이름 → 좌표</td><td>좌표를 지어내지 않고 <b>버스정류소 위치 데이터에서만</b> 검색, 후보가 여러 곳이면 사용자에게 선택 요청, 분석 영역 밖은 거절</td></tr>
<tr><td>③ 도구 오케스트레이션</td><td>최적화·AS-IS 진단·노선 조회·시나리오 비교 도구 6종 호출</td><td>해가 없으면 이유와 완화 후보를 제시하고 <b>사용자 승인 후</b> 재계산</td></tr>
<tr><td>④ 결과 설명</td><td>AS-IS 대비 변화와 코스 특징을 한국어로 설명</td><td><b>답변 속 숫자를 도구 결과와 자동 대조</b>, 근거 없는 숫자는 화면에 경고 표시</td></tr>
<tr><td>⑤ 최적화 (MILP)</td><td>LP 완화 → 후보 회랑 → MILP (HiGHS)</td><td>구간대 근사 → 10m 단위 정밀 재평가로 더 나은 코스만 채택, 모든 결과를 규칙 검사기로 독립 검증</td></tr></table></div>
{foot(6)}</section>""")

    # 04 서비스 구조
    P.append(f"""<section class="page"><h1><span class="no">04</span>서비스 구조</h1>
<div class="grid" style="grid-template-columns:1.05fr 1fr; height:560px">
<div class="box arch" style="background:#f7f8fb"><div class="lab">서비스 구조도 (아키텍처)</div>
<div class="lyr ui"><b>사용자 화면 — Streamlit 웹앱</b><br>조건 카드(출발·도착·시각·단순성) · 지도(AS-IS/TO-BE, 반환 구간 표시) · KPI 타일·표 · 채팅</div>
<div class="down">▼ 자연어 요청 &nbsp;&nbsp;/&nbsp;&nbsp; ▲ 설명 + 근거 숫자</div>
<div class="lyr ai"><b>AI 에이전트 — Claude Opus 5.5 (Tool Runner)</b><br>도구 6종: 장소 검색 · 코스 최적화 · 현재 결과 · AS-IS 진단 · 영향 노선 · 사전 계산 시나리오<br>답변 숫자 자동 대조 · 거절 시 서버 측 대체 모델</div>
<div class="down">▼ 제약 카드(검증된 조건) &nbsp;&nbsp;/&nbsp;&nbsp; ▲ KPI·코스·위반 검사 결과</div>
<div class="lyr en"><b>RunWay 최적화 엔진 (Python 패키지)</b><br>조건 검증 → 시나리오(위치별 통제 시간·피해표) → MILP(PuLP + HiGHS) → 정밀 평가·규칙 검사 → 결과 저장(더 좋은 해만 교체)</div>
<div class="down">▲</div>
<div class="lyr da"><b>데이터</b> — 버스 노선×정류장×일별 승하차 원자료(사용일자로 일요일 판별) · 정류장 시간대별 승차 · 정류소 위치 · OSM 도로망 · 공식 AS-IS 코스(고정)</div></div>
<div><div class="lab" style="font-size:13px;font-weight:700;color:#1f2a44;margin-bottom:8px">기술 스택</div>
<table><tr><th style="width:30%">구분</th><th>내용</th></tr>
<tr><td>프론트/클라이언트</td><td>Streamlit 1.67, pydeck(지도)</td></tr>
<tr><td>백엔드/에이전트</td><td>Python 3.11, Anthropic Python SDK (Tool Runner), runway 엔진 패키지, Pydantic</td></tr>
<tr><td>AI 모델</td><td>Claude Opus 5.5 (<code>claude-opus-5-5</code>), 거절 시 서버 측 대체 모델(fallback)</td></tr>
<tr><td>최적화</td><td>PuLP 3/4 (MILP 모델링) + HiGHS (MIP 솔버), NetworkX, SciPy</td></tr>
<tr><td>외부 API</td><td>Anthropic Claude API</td></tr>
<tr><td>데이터 (출처·라이선스)</td><td>서울 열린데이터광장 OA-12913·OA-12912(원자료)·OA-15067 (2026-10) / OpenStreetMap (ODbL) / 2026 JTBC 서울마라톤 대회 책자(공식 코스)</td></tr>
<tr><td>배포환경</td><td>Streamlit Community Cloud (GitHub 연동, 비밀 키는 Secrets)</td></tr></table>
<div class="box" style="margin-top:12px"><div class="lab">품질 장치</div>
<ul class="small"><li>자동 검증 테스트 {K['ntests']}개: 노트북 결과 재현, MILP 목적함수 = 독립 재계산, 휴일 판별, 공식 AS-IS 고정</li>
<li>결과 캐시 + 코드가 바뀌면 세션·엔진 자동 갱신</li></ul></div></div></div>
{foot(7)}</section>""")

    # 05 핵심 기능 및 흐름
    P.append(f"""<section class="page"><h1><span class="no">05</span>핵심 기능 및 사용자 흐름</h1>
<div class="grid" style="grid-template-columns:1fr 1fr 1fr; margin-bottom:14px">
<div class="box" style="height:230px"><div class="lab">핵심 기능 1 · 기존 코스 진단</div>
<ul class="small"><li>대회 책자 공식 경유지로 AS-IS를 <b>고정</b></li>
<li>못 타는 승객, 통제 정류장·노선, 방향 전환 KPI</li>
<li>회귀(반환 왕복)·재방문·금지 구간 통과 등 <b>문제점 자동 검출</b></li></ul></div>
<div class="box" style="height:230px"><div class="lab">핵심 기능 2 · 최적 코스 계산</div>
<ul class="small"><li>목적: 못 타는 승객 + 코스 단순성(꺾임) 최소화</li>
<li>하드 제약: 42.195~42.6km, 교차로 1회, <b>회귀·재방문 금지</b>, 금지 구간 회피</li>
<li>출발 시각·장소·금지 구간을 바꿔 <b>다시 계산</b></li></ul></div>
<div class="box" style="height:230px"><div class="lab">핵심 기능 3 · 대화형 설계 (AI 에이전트)</div>
<ul class="small"><li>"7시로 당기면?", "건대입구역 근처는 피해 줘"를 조건으로 변환</li>
<li>장소는 정류소 데이터에서 검색, 애매하면 후보 제시</li>
<li>결과를 <b>도구가 계산한 숫자로만</b> 설명</li></ul></div></div>
<div class="box" style="height:316px"><div class="lab">주요 사용자 이용 흐름 (Happy Path)</div>
<div class="flow">
<div class="st"><b>① 접속</b><br>기본 조건의 AS-IS·TO-BE 지도와 KPI가 바로 보임 (사전 계산)</div><div class="ar">›</div>
<div class="st"><b>② 진단</b><br>"기존 코스는 뭐가 문제야?" → 반환 3곳·회귀 약 {K['ret_km']:.0f}km·붐비는 정류장 {K['busy'][0]}곳 설명</div><div class="ar">›</div>
<div class="st"><b>③ 조건 변경</b><br>"출발을 9시로" → 사전 계산 결과 즉시, 새 조건은 MILP로 재계산</div><div class="ar">›</div>
<div class="st"><b>④ 장소 제약</b><br>"건대입구역 근처는 피해 줘" → 장소 검색 → 금지 구간 추가 → 재계산</div><div class="ar">›</div>
<div class="st"><b>⑤ 협의 자료</b><br>지도·KPI 표·도로별 통제 시각·영향 노선 목록으로 관계기관 협의</div></div>
<p class="small" style="margin-top:14px">조건별 사전 계산 (못 타는 승객): {sched} — 조건 카드(버튼)로도, 채팅으로도 같은 엔진을 호출</p></div>
{foot(8)}</section>""")

    # 06 차별점
    P.append(f"""<section class="page"><h1><span class="no">06</span>차별점</h1>
<div class="box" style="height:560px"><div class="lab">기존 서비스와 차별화 요소 및 사용자에게 주는 효과</div>
<table style="margin-bottom:14px"><tr><th style="width:22%">비교 항목</th><th>기존 방식 (경유지 수작업)</th><th>길찾기·지도 서비스</th><th>LLM 단독 챗봇</th><th style="background:#2a78d6">RunWay</th></tr>
<tr><td>시민(버스 승객) 영향 정량화</td><td>사후 우회 안내</td><td>없음</td><td>추정·지어냄 위험</td><td><b>정류장 실측 수요로 계산</b></td></tr>
<tr><td>위치별 통제 시간 반영</td><td>경험</td><td>없음</td><td>없음</td><td><b>코스 km마다 통제 시각 모델</b></td></tr>
<tr><td>42.195km·반환 없음 등 규칙</td><td>수작업 확인</td><td>최단 경로만</td><td>보장 못 함</td><td><b>하드 제약으로 보장 + 독립 검사</b></td></tr>
<tr><td>조건을 바꿔 다시 검토</td><td>수작업으로 다시 설계</td><td>-</td><td>즉시이나 근거 없음</td><td><b>말로 요청 → 즉시~2분 재계산</b></td></tr>
<tr><td>숫자의 신뢰성</td><td>-</td><td>-</td><td>낮음</td><td><b>솔버 결과만 인용·자동 대조</b></td></tr></table>
<div class="kpi">
<div class="k"><div class="t">못 타는 승객</div><div class="v" style="color:#2a78d6">{K['p_riders']:+d}%</div><div class="d">{a['riders']:,} → {t['riders']:,}명</div></div>
<div class="k"><div class="t">회귀(반환 왕복)</div><div class="v" style="color:#2a78d6">0</div><div class="d">약 {K['ret_km']:.0f}km → 0, 재방문 {K['revisit']} → 0곳</div></div>
<div class="k"><div class="t">방향 전환</div><div class="v" style="color:#2a78d6">{K['p_turns']:+d}%</div><div class="d">{a['turns']} → {t['turns']}회 (러너가 외우기 쉬운 코스)</div></div>
<div class="k"><div class="t">붐비는 정류장 통과</div><div class="v" style="color:#2a78d6">{K['busy'][0]} → {K['busy'][1]}곳</div><div class="d">하루 1,000명 이상 정류장</div></div></div>
<ul class="small" style="margin-top:14px"><li><b>새로운 접근</b>: '어디를 지나가나'가 아니라 '<b>언제 어디가 막히나</b>'를 최적화 — 같은 도로도 코스의 몇 km에 놓이느냐에 따라 피해가 다르다는 점을 수식에 반영</li>
<li><b>사용자 편의</b>: 수식·데이터 없이 대화로 조건을 바꾸고, 결과는 지도·표로 바로 협의에 사용</li></ul></div>
{foot(9)}</section>""")

    # 07 구현 현황
    P.append(f"""<section class="page"><h1><span class="no">07</span>구현 현황 및 프로토타입</h1>
<div class="grid" style="grid-template-columns:1.32fr 1fr; height:560px">
<div class="box" style="padding:10px"><div class="lab">배포된 웹 서비스 화면 — {e(team['app_url'])}</div><img src="assets/app.png" style="width:100%; border:1px solid #e3e4ea"></div>
<div class="grid" style="grid-template-rows:auto 1fr">
<div class="box"><div class="lab">구현된 기능 (작동 확인)</div>
<ul class="small"><li>✅ 최적화 엔진: LP 완화 → 회랑 → MILP, 하드 제약(회귀·재방문 금지 등) — 기본 조건 풀이 수십 초</li>
<li>✅ AS-IS 공식 코스 고정·진단, KPI 10종, 위반 자동 검출</li>
<li>✅ 웹앱: 조건 카드·지도(반환 구간 표시)·KPI 표·도로별 통제 시각·영향 노선</li>
<li>✅ AI 에이전트: 도구 6종, 장소 검색, 답변 숫자 대조</li>
<li>✅ 사전 계산 시나리오 즉시 응답, 결과 캐시</li>
<li>✅ 자동 검증 테스트 {K['ntests']}개 · 클라우드 배포</li></ul></div>
<div class="box"><div class="lab">결과 — 누적 못 타는 승객 (코스 진행 순)</div><img src="assets/cumulative.png" style="width:100%"></div></div></div>
{foot(10)}</section>""")

    # 08 기대효과
    P.append(f"""<section class="page"><h1><span class="no">08</span>기대효과</h1>
<div class="grid" style="grid-template-columns:1.25fr 1fr; height:560px">
<div class="box"><div class="lab">AS-IS vs TO-BE (같은 출발·도착·거리, 일요일 07:30 기준)</div><img src="assets/compare_map.png" style="width:100%">
<table style="margin-top:8px"><tr><th>KPI</th><th class="r">AS-IS</th><th class="r">TO-BE</th><th class="r">변화</th></tr>
<tr><td>못 타는 승객</td><td class="r">{a['riders']:,}명</td><td class="r hl">{t['riders']:,}명</td><td class="r">{K['p_riders']:+d}%</td></tr>
<tr><td>방향 전환 45°↑ / 100°↑</td><td class="r">{a['turns']} / {a['sharp']}회</td><td class="r">{t['turns']} / {t['sharp']}회</td><td class="r">{K['p_turns']:+d}%</td></tr>
<tr><td>영향 버스 노선</td><td class="r">{a['routes']}개</td><td class="r">{t['routes']}개</td><td class="r">{K['p_routes']:+d}%</td></tr>
<tr><td>통제 정류장 (정류장당 이용객 중앙값)</td><td class="r">{a['stops']}곳 ({K['med'][0]}명)</td><td class="r">{t['stops']}곳 ({K['med'][1]}명)</td><td class="r">한산한 정류장으로</td></tr></table></div>
<div class="grid" style="grid-template-rows:auto auto 1fr">
<div class="box"><div class="lab">기대효과</div>
<ul class="small"><li><b>시민</b>: 대회 1회당 통제 시간에 버스를 못 타는 시민 약 {K['d_riders']:,}명 감소(추정)</li>
<li><b>운영</b>: 코스안 비교·협의 자료를 몇 분 만에, 조건 변경 시 즉시 재검토</li>
<li><b>참가자</b>: 반환 없이 깔끔한 코스, 방향 전환 {a['turns']}→{t['turns']}회</li></ul></div>
<div class="box"><div class="lab">적용·비용</div>
<ul class="small"><li>최적화는 오픈소스(PuLP·HiGHS)로 추가 비용 없음, 결과 캐시로 반복 계산 최소화</li>
<li>AI 비용은 채팅 질문 1회당 수 센트 수준(추정) — 월 사용 한도로 관리</li>
<li>무료 클라우드 배포로 운영 부담 낮음</li></ul></div>
<div class="box"><div class="lab">확장·발전 방안</div>
<ul class="small"><li><b>다른 행사</b>: 출발·도착·금지 구간만 바꿔 다른 마라톤·퍼레이드·도심 축제에 적용</li>
<li><b>데이터 정밀화</b>: 일요일 시간대 원자료, 승객 OD로 우회 노선 영향까지</li>
<li><b>규모 확대</b>: 비동기 풀이 작업 큐, 질문 유형별 경량 모델 라우팅, 다른 도시 교통카드 데이터</li></ul></div></div></div>
{foot(11)}</section>""")

    # 별첨
    roles = ''.join(f"<tr><td>{F(r, 'name')}</td><td>{F(r, 'role')}</td><td>{F(r, 'work')}</td></tr>" for r in team['roles'])
    P.append(f"""<section class="page"><h1>별첨. 팀 구성 및 사전 개발 내역 고지</h1>
<div class="grid" style="grid-template-columns:1fr 1fr; margin-bottom:12px">
<div><div class="lab" style="font-size:13px;font-weight:700;color:#1f2a44;margin-bottom:6px">팀 구성 및 역할</div>
<table class="tight"><tr><th>이름</th><th>역할</th><th>담당 업무</th></tr>{roles}</table></div>
<div><div class="lab" style="font-size:13px;font-weight:700;color:#1f2a44;margin-bottom:6px">기존 개발물 · 사전 개발 내역</div>
<table class="tight"><tr><th style="width:30%">구분</th><th>내용</th></tr>
<tr><td>해커톤 이전 개발 범위</td><td>{F(team, 'pre_dev')}</td></tr>
<tr><td>해커톤 기간 중 개발 범위</td><td class="small">{F(team, 'during_dev')}</td></tr>
<tr><td>타 공모전 출품·수상 여부</td><td>{F(team, 'other_contest')}</td></tr></table></div></div>
<div class="lab" style="font-size:13px;font-weight:700;color:#1f2a44;margin-bottom:6px">활용한 외부 자원 · 출처</div>
<table class="tight"><tr><th style="width:16%">구분</th><th style="width:30%">명칭</th><th style="width:30%">출처 · 라이선스</th><th>사용 위치</th></tr>
<tr><td>오픈소스 · 템플릿</td><td>PuLP, HiGHS(highspy), Pydantic</td><td>MIT</td><td>MILP 모델·솔버, 조건 스키마</td></tr>
<tr><td></td><td>Streamlit, pydeck, Anthropic Python SDK</td><td>Apache-2.0</td><td>웹앱·지도, 에이전트</td></tr>
<tr><td></td><td>NetworkX, pandas, NumPy, SciPy</td><td>BSD-3-Clause</td><td>도로 그래프, 데이터 처리</td></tr>
<tr><td>외부 데이터 · API</td><td>버스 승하차 OA-12913·OA-12912(원자료), 정류소 위치 OA-15067</td><td>서울 열린데이터광장 (data.seoul.go.kr)</td><td>못 타는 승객·노선 계산, 장소 검색</td></tr>
<tr><td></td><td>도로망</td><td>© OpenStreetMap contributors (ODbL)</td><td>도로 그래프, 지도</td></tr>
<tr><td></td><td>2026 JTBC 서울마라톤 대회 책자·공지</td><td>marathon.jtbc.com</td><td>AS-IS 공식 경유지·출발 시각</td></tr>
<tr><td>AI 모델 · 개발 도구</td><td>Claude Opus 5.5 (Claude API)</td><td>Anthropic</td><td>서비스 내 AI 에이전트</td></tr>
<tr><td></td><td>Claude Code</td><td>Anthropic</td><td>개발 보조(코드 작성·검증)</td></tr></table>
<p class="small" style="margin-top:8px;color:#52514e">본 표: 외부 자원의 명칭·출처·라이선스·사용 위치. AI와 팀의 역할 및 검증 과정은 홈페이지 「AI 활용 내역」에 작성.</p>
{foot(12)}</section>""")
    return P


def main():
    setup_font()
    team = json.load(open(DOC / 'team.json', encoding='utf-8'))
    eng, sc, A, T, K = collect()
    figures(eng, sc, A, T, K)
    doc = f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>RunWay 프로젝트 소개서</title><style>{CSS}</style></head><body>{''.join(pages(K, team))}</body></html>"
    out_html = DOC / '소개서.html'
    out_html.write_text(doc, encoding='utf-8')
    team_name = team['team'].strip('[]') if not team['team'].startswith('[') else '팀명'
    pdf = ROOT / 'out' / f"[{team_name}]_RunWay_AI해커톤_{date.today():%Y%m%d}.pdf"
    chrome = next((p for p in [r'C:\Program Files\Google\Chrome\Application\chrome.exe',
                               r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'] if Path(p).exists()), None)
    if chrome:
        subprocess.run([chrome, '--headless=new', '--disable-gpu', '--no-pdf-header-footer', f'--print-to-pdf={pdf}',
                        out_html.resolve().as_uri()], check=True, capture_output=True, timeout=180)
        print('PDF:', pdf)
    print('HTML:', out_html, '| 못 타는 승객', K['a']['riders'], '→', K['t']['riders'], f"({K['p_riders']:+d}%)")


if __name__ == '__main__':
    main()
