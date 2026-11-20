"""RunWay — 마라톤 코스 설계 서비스 (Streamlit)

    streamlit run app.py

흐름: ① 목표 입력(LLM: 말 → 조건) → ② 결과 비교(최적화 엔진: 조건 → 코스·숫자) → ③ 협의 자료(LLM: 숫자 → 기관별 브리핑)
      + AI 어시스턴트(질의응답) + 작동 원리(왜 LLM, 왜 최적화 엔진, 구조도, 플로우차트)
숫자는 모두 runway 엔진(PuLP MILP)이 계산하고, LLM은 조건 해석과 설명만 한다. AI가 연결되지 않아도 흐름은 끊기지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parent


def code_version() -> str:
    """엔진·에이전트 소스의 해시 — 새 코드가 배포되면 값이 바뀐다"""
    h = hashlib.sha1()
    for f in sorted([*(ROOT / 'runway').glob('*.py'), *(ROOT / 'agent').glob('*.py')]):
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


VERSION = code_version()
# Streamlit Cloud는 git push 뒤 app.py만 다시 실행하고, 이미 import한 패키지는 메모리에 옛 버전으로 남겨 둔다
# → 소스가 바뀌었으면 지우고 다시 import (옛 Params로 AttributeError가 나던 문제)
_loaded = sys.modules.get('runway')
if _loaded is not None and getattr(_loaded, '__app_version__', None) != VERSION:
    for name in [m for m in sys.modules if m.split('.')[0] in ('runway', 'agent')]:
        del sys.modules[name]

from agent import MODEL, Session, make_tools, run_turn, unverified_numbers  # noqa: E402
from agent import llm  # noqa: E402
from agent.tools import build_params  # noqa: E402
import runway  # noqa: E402
from runway import DEFAULT, Engine, in_bbox  # noqa: E402
from runway.engine import describe_params  # noqa: E402
from runway.evaluate import KPI_LABELS  # noqa: E402
from runway.events import EVENTS, event_of, params_for  # noqa: E402
from runway import report  # noqa: E402

runway.__app_version__ = VERSION

C_ASIS, C_TOBE, C_ALT = '#eb6834', '#2a78d6', '#1baf7a'      # 검증된 기본 팔레트 1·2·3번
st.set_page_config(page_title='RunWay — 마라톤 코스 설계', page_icon='🏃', layout='wide')


# ───────────────────────── 공용 자원 ─────────────────────────
@st.cache_resource(show_spinner='도로망·버스 데이터 불러오는 중 (첫 실행 약 20초)…')
def get_engine(version: str) -> Engine:             # version이 바뀌면(새 배포) 엔진을 새로 만든다
    eng = Engine(ROOT)
    if eng.cached(DEFAULT) is None:                 # 기본 시나리오 캐시가 없으면 노트북 결과로 채운다 (시작 시 긴 풀이 방지)
        r = json.load(open(ROOT / 'out' / 'result.json', encoding='utf-8'))
        try:
            eng.seed(DEFAULT, r['tobe']['order'], r['solver'])
        except Exception:                           # 노트북 코스가 현재 규칙(회귀 금지 등)을 어기면 시드하지 않음
            pass
    return eng


def api_key() -> str | None:
    try:
        k = st.secrets.get('ANTHROPIC_API_KEY')
    except Exception:
        k = None
    return k or os.environ.get('ANTHROPIC_API_KEY')


@st.cache_resource
def get_client(key: str):
    import anthropic
    return anthropic.Anthropic(api_key=key)


eng = get_engine(VERSION)
if st.session_state.get('version') != VERSION:     # 새 코드 배포 → 옛 세션 버림
    for k in ('sess', 'history', 'chat', 'ai_card', 'briefing', 'tradeoff', 'cf', 'cf_ai'):
        st.session_state.pop(k, None)
    st.session_state['version'] = VERSION
if 'sess' not in st.session_state:
    s0 = Session(engine=eng)
    s0.result = eng.solve(DEFAULT, time_limit=120, gap=0.02)
    st.session_state.update(sess=s0, history=[], chat=[])
S: Session = st.session_state.sess
TOOLS = {t.name: t for t in make_tools(S)}
KEY = api_key()


def hm(t):
    return '%02d:%02d' % (int(t), round(t % 1 * 60))


def llm_call(fn, *args):
    """LLM 호출 + 오류를 사용자 문장으로"""
    import anthropic
    try:
        return fn(get_client(KEY), *args), None
    except anthropic.RateLimitError:
        return None, '요청이 많아 잠시 제한되었습니다. 30초 뒤 다시 시도해 주세요.'
    except anthropic.APIStatusError as e:
        return None, f'Claude API 오류({e.status_code}). 잠시 뒤 다시 시도해 주세요.'
    except anthropic.APIConnectionError:
        return None, 'Claude API에 연결하지 못했습니다.'
    except RuntimeError as e:
        return None, str(e)


def role_strip(llm_part: str, engine_part: str):
    """이 화면에서 LLM과 최적화 엔진이 각각 무엇을 하는지 — 모든 단계 위에 표시"""
    c1, c2 = st.columns(2)
    c1.markdown(f"<div style='background:#fdeee6;border-radius:8px;padding:10px 14px'><b>🧠 LLM (Claude)</b> — {llm_part}</div>",
                unsafe_allow_html=True)
    c2.markdown(f"<div style='background:#eaf2fc;border-radius:8px;padding:10px 14px'><b>⚙️ 최적화 엔진 (PuLP MILP)</b> — {engine_part}</div>",
                unsafe_allow_html=True)
    st.write('')


def run_solve(p, title='최적 코스 계산'):
    """엔진 풀이 + 단계별 진행 표시 (P1-8). 사전 계산된 조건이면 즉시."""
    if (hit := eng.cached(p)):
        return dict(hit, ok=True, cached=True)
    t0 = time.time()
    with st.status(f'{title} — 보통 10초~2분 (LP 완화 → 후보 회랑 → MILP)', expanded=True) as box:
        log, area = [], st.empty()

        def prog(m):
            log.append(f'{time.time() - t0:5.1f}초 · {m}')
            area.markdown('\n'.join(f'- {x}' for x in log))
        res = eng.solve(p, time_limit=120, gap=0.02, progress=prog)
        box.update(label=f"{'완료' if res.get('ok') else '해 없음'} ({time.time() - t0:.0f}초)",
                   state='complete' if res.get('ok') else 'error', expanded=not res.get('ok'))
    return res


# ───────────────────────── 지도·KPI 그리기 ─────────────────────────
def _offset(c, d=0.00035):
    k = math.cos(math.radians(37.55)); out = []
    for i, (lo, la) in enumerate(c):
        (lo0, la0), (lo1, la1) = c[max(i - 1, 0)], c[min(i + 1, len(c) - 1)]
        dx, dy = (lo1 - lo0) * k, la1 - la0; n = math.hypot(dx, dy) or 1.0
        out.append([lo + dy / n * d / k, la - dx / n * d])
    return out


def path_rows(net, res, name, color):
    """'처음 달리는 구간'과 '이미 달린 도로로 되돌아오는(반환) 구간'을 나눠 그린다"""
    rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    info = f"{name}: {res['kpi']['length_km']}km, 못 타는 승객 {res['kpi']['riders']:,}명"
    rows, seen, cur = [], set(), None
    for a in res['order']:
        sid = int(net.SEG[a]); ret = sid in seen; seen.add(sid)
        c = [[net.allnodes[n][1], net.allnodes[n][0]] for n in net.path_of(a)]
        if ret:
            c = _offset(c)
        if cur is not None and cur['ret'] == ret:
            cur['path'] += c[1:]
        else:
            cur = dict(ret=ret, path=c, name=name, color=rgb + ([170] if ret else [255]), info=info + (' · 반환' if ret else ''))
            rows.append(cur)
    return rows


def course_map(courses, p, key, height=480, turn_file=None):
    """courses: [(결과, 이름, 색)]"""
    net, ev = eng.network(p), event_of(p)
    paths = [r for res, nm, col in courses for r in path_rows(net, res, nm, col)]
    pts = [dict(name=f'출발 · {p.START_NAME}', pos=[p.START[1], p.START[0]], color=[11, 11, 11]),
           dict(name=f'도착 · {p.FINISH_NAME}', pos=[p.FINISH[1], p.FINISH[0]], color=[11, 11, 11])]
    if turn_file:
        from runway.asis import load_frozen
        pts += [dict(name=f"↩ {w['name'].replace('(반환)', '')} 반환", pos=[w['coord'][1], w['coord'][0]], color=[235, 104, 52])
                for w in (load_frozen(eng.data_dir, turn_file) or {}).get('waypoints', []) if '반환' in w['name'] and '대림상가' not in w['name']]
    zones = [dict(name=f'금지 · {z.name}', pos=[z.coord[1], z.coord[0]], r=z.r) for z in p.FORBID if in_bbox(z.coord, p.AREA)]
    for x in pts + zones:
        x['info'] = x['name']
    layers = [pdk.Layer('PathLayer', paths, get_path='path', get_color=[255, 255, 255], width_min_pixels=7),
              pdk.Layer('PathLayer', paths, get_path='path', get_color='color', width_min_pixels=4, pickable=True),
              pdk.Layer('ScatterplotLayer', zones, get_position='pos', get_radius='r', get_fill_color=[82, 81, 78, 50],
                        get_line_color=[82, 81, 78], stroked=True, line_width_min_pixels=1, pickable=True),
              pdk.Layer('ScatterplotLayer', pts, get_position='pos', get_fill_color='color', get_radius=70, radius_min_pixels=6,
                        get_line_color=[255, 255, 255], stroked=True, line_width_min_pixels=2, pickable=True)]
    st.pydeck_chart(pdk.Deck(layers=layers, map_provider='carto', map_style='light',
                             initial_view_state=pdk.ViewState(latitude=(ev.area[0] + ev.area[1]) / 2, longitude=(ev.area[2] + ev.area[3]) / 2,
                                                              zoom=11.2 if ev.distance_m > 15000 else 12.0),
                             tooltip={'text': '{info}'}), height=height, key=key)


TILES = [('못 타는 승객 (명)', 'riders', '정류장이 마라톤 코스에 들어가 통제 시간 동안 버스를 못 타는 승객 (일요일 하루 기준)'),
         ('통제되는 정류장', 'stops', '코스 위에 있어 통제 시간 동안 쓸 수 없는 버스 정류장 수'),
         ('영향 버스 노선', 'routes', '코스 위에 정류장이 있는 버스 노선 수'),
         ('방향 전환 45°↑', 'turns', '45° 이상 방향을 바꾸는 횟수'),
         ('통제 필요 교차로', 'junctions', '코스가 지나가 통제 인력이 필요한 교차로 수')]


def kpi_tiles(k, a):
    for c, (lab, key_, hlp) in zip(st.columns(5), TILES):
        if a is None:
            c.metric(lab, f'{k[key_]:,}', help=hlp)
        else:
            d = k[key_] - a[key_]
            c.metric(lab, f'{k[key_]:,}', f"{d:+,} ({d / a[key_] * 100:+.0f}%)" if d and a[key_] else '변화 없음', delta_color='inverse', help=hlp)


def show_tradeoff(t, warn):
    st.markdown(f"**{t['headline']}**")
    c1, c2 = st.columns(2)
    c1.markdown('**얻는 것**\n' + ''.join(f'\n- {x}' for x in t['gains']))
    c2.markdown('**잃는 것·대가**\n' + ''.join(f'\n- {x}' for x in t['costs']))
    st.markdown(f"**추천** — {t['recommendation']}")
    if warn:
        st.caption('⚠️ 엔진 결과에서 확인되지 않은 숫자: ' + ', '.join(warn))


# ───────────────────────── 사이드바 ─────────────────────────
with st.sidebar:
    st.markdown('### 🏃 RunWay')
    ev_keys = list(EVENTS)
    pick = st.radio('종목', ev_keys, index=ev_keys.index(S.params.EVENT), format_func=lambda k: EVENTS[k].label, horizontal=True)
    if pick != S.params.EVENT:
        with st.spinner(f'{EVENTS[pick].label} 코스 불러오는 중…'):
            TOOLS['optimize_course'].call(dict(event=pick))
        for k in ('briefing', 'tradeoff', 'cf', 'cf_ai', 'ai_card'):
            st.session_state.pop(k, None)
        st.rerun()
    st.caption(f'현재 조건: {describe_params(S.params)}')
    st.caption(f"현재 코스: {S.result['kpi']['length_km']}km · 못 타는 승객 {S.result['kpi']['riders']:,}명")
    st.divider()
    st.caption(f'LLM: {MODEL} ' + ('● 연결됨' if KEY else '○ 미연결 — AI 단계는 수동 입력·규칙 기반 자료로 대신합니다'))
    st.caption('데이터: 서울 열린데이터광장 버스 승하차(2026-10)·정류소 위치, © OpenStreetMap contributors')


# ───────────────────────── ① 목표 입력 ─────────────────────────
def page_goal():
    st.title('① 목표 입력')
    role_strip('담당자의 <b>말로 된 목표</b>를 엔진이 읽을 수 있는 조건으로 바꾸고, 반영할 수 없는 요청은 이유와 함께 알려 줍니다',
               '바뀐 조건으로 <b>최적 코스를 계산</b>합니다 — 공인 거리·회귀 금지·재방문 금지·금지 구간은 항상 지켜집니다')
    examples = ['건대입구역 근처는 피해 줘', '출발을 8시로 늦춰 줘', '10K로 바꾸고 합정역 근처는 피해 줘', '하프 코스로 바꿔 줘']
    st.markdown('**어떤 코스를 원하세요?** 문장으로 적으면 AI가 조건을 읽고 엔진이 바로 계산합니다.')
    for c, ex in zip(st.columns(len(examples)), examples):
        if c.button(ex, key=f'ex_{ex}', width='stretch'):
            st.session_state['goal_text'] = ex
    goal = st.text_area('목표', key='goal_text', height=90, label_visibility='collapsed',
                        placeholder='예) 건대입구역 근처는 피하고 출발을 7시로 당겨 줘.')
    if not KEY:
        st.info('AI가 연결되지 않았습니다 (Streamlit secrets에 ANTHROPIC_API_KEY). 종목은 왼쪽 사이드바에서 바꿀 수 있습니다.')
    if not st.button('🧠 AI로 코스 계산', type='primary', disabled=not (KEY and goal.strip())):
        return
    with st.spinner('AI가 목표를 읽는 중…'):
        card, err = llm_call(llm.interpret_goal, goal, describe_params(S.params))
    if err:
        st.error(err); return
    changed = card.event or card.start_place or card.finish_place or card.start_hour is not None or card.avoid_places
    if not changed:
        for u in card.unsupported:
            st.warning(f'반영할 수 없는 요청: {u}')
        st.info(card.question or '바꿀 조건을 찾지 못했습니다. 종목·출발지·도착지·출발 시각·피할 장소 중 무엇을 바꿀지 적어 주세요.')
        return
    newp, err = build_params(S, event=card.event, start_place=card.start_place, finish_place=card.finish_place,
                             start_hour=card.start_hour, avoid_places=card.avoid_places)
    if err:
        st.error(f"{err.get('항목', '')} {err.get('error', '')}")
        if err.get('후보'):
            st.info('이 중 하나로 다시 적어 주세요: ' + ', '.join(err['후보']))
        return
    res = run_solve(newp)
    if not res.get('ok'):
        st.error(f"조건을 만족하는 코스가 없습니다: {res.get('reason')} — 피할 장소를 줄이거나 출발·도착을 바꿔 보세요.")
        return
    S.params, S.result = newp, res
    for k in ('briefing', 'tradeoff', 'cf', 'cf_ai'):
        st.session_state.pop(k, None)
    st.session_state['ai_note'] = dict(goal=goal, unsupported=card.unsupported, question=card.question)
    st.switch_page(PAGES['result'])


# ───────────────────────── ② 결과 비교 ─────────────────────────
def page_result():
    st.title('② 결과 비교')
    role_strip('여러 안의 <b>장단점을 말로 정리</b>하고, 엔진이 고른 도로가 <b>왜 최선인지</b> 설명합니다',
               '도로 1,000여 개 구간 조합에서 <b>못 타는 승객</b>이 최소인 코스를 찾고, 도로를 빼며 <b>다시 계산</b>해 근거를 만듭니다')
    p, R, A = S.params, S.result, S.ensure_asis()
    ev = event_of(p)
    st.caption(f'현재 조건: {describe_params(p)}')
    if (note := st.session_state.pop('ai_note', None)):
        st.success(f"“{note['goal']}” → AI가 조건을 읽고 엔진이 다시 계산했습니다.")
        for u in note['unsupported']:
            st.warning(f'반영하지 못한 요청: {u}')
    kpi_tiles(R['kpi'], A['kpi'] if A else None)
    st.caption(f"변화는 AS-IS({A.get('source', '')}) 대비" if A else '이 종목은 기존 코스 데이터가 없어 비교하지 않습니다.')
    course_map(([(A, 'AS-IS', C_ASIS)] if A else []) + [(R, 'TO-BE', C_TOBE)], p, key='main_map', turn_file=ev.asis_file if A else None)
    st.markdown((f"<span style='color:{C_ASIS}'>━━</span> AS-IS (2026 JTBC 공식 코스, 옅은 선 = 반환 구간) &nbsp; " if A else '')
                + f"<span style='color:{C_TOBE}'>━━</span> TO-BE (RunWay 최적 코스) &nbsp; ● 출발·도착 &nbsp; 점·선에 마우스를 올리면 이름과 KPI",
                unsafe_allow_html=True)

    # P1-6 여러 안 나란히
    st.divider()
    st.subheader('여러 안 나란히 비교')
    lib = [x for x in eng.library() if x['params'].EVENT == p.EVENT]
    opts = {'현재 코스 (TO-BE)': R} | {x['조건']: dict(eng.cached(x['params']), params=x['params'].model_dump(mode='json')) for x in lib}
    if A:
        opts = {'AS-IS (2026 공식 코스)': A} | opts
    chosen = st.multiselect('비교할 안 (최대 3개)', list(opts), default=list(opts)[:2], max_selections=3)
    if chosen:
        cols = st.columns(len(chosen))
        for i, (col, nm) in enumerate(zip(cols, chosen)):
            with col:
                st.markdown(f'**{nm}**')
                course_map([(opts[nm], nm, [C_ASIS, C_TOBE, C_ALT][i] if not nm.startswith('AS-IS') else C_ASIS)], p, key=f'cmp_{i}', height=300)
        st.dataframe(pd.DataFrame([{'지표': lab, **{nm: opts[nm]['kpi'][k] for nm in chosen}} for lab, k, _ in KPI_LABELS]),
                     hide_index=True, width='stretch')
        if len(chosen) >= 2:
            fx = {nm: report.facts(opts[nm] if 'order' in opts[nm] else R, None)['TO_BE'] | {'조건': nm} for nm in chosen}
            if st.button('🧠 AI로 장단점 요약', disabled=not KEY):
                with st.spinner('AI가 안들을 비교하는 중…'):
                    t, err = llm_call(llm.compare_options, fx)
                st.session_state['tradeoff'] = dict(err=err) if err else dict(t=t.model_dump(), warn=llm.check_numbers(t, fx), names=chosen)
            tr = st.session_state.get('tradeoff')
            if tr and tr.get('names') == chosen:
                show_tradeoff(tr['t'], tr['warn'])
            elif tr and tr.get('err'):
                st.error(tr['err'])
            if not KEY:
                best = min(chosen, key=lambda nm: opts[nm]['kpi']['riders']); simp = min(chosen, key=lambda nm: opts[nm]['kpi']['turns'])
                st.info(f"규칙 기반 요약 (AI 미연결): 못 타는 승객이 가장 적은 안은 **{best}** ({opts[best]['kpi']['riders']:,}명), "
                        f"방향 전환이 가장 적은 안은 **{simp}** ({opts[simp]['kpi']['turns']}회).")

    # P1-7 왜 이 길인가
    st.divider()
    st.subheader('왜 이 길인가? — 도로를 빼고 다시 계산')
    st.caption('엔진이 고른 도로를 하나 빼고 다시 풀어, 그 도로가 없으면 무엇이 나빠지는지 숫자로 보여 줍니다 (처음 계산은 1~2분, 이후 즉시).')
    roads = sorted({r['도로']: r['피해'] for r in R['roads'] if r['도로'] != '(이름 없음)'}.items(), key=lambda kv: -kv[1])
    road = st.selectbox('뺄 도로 (못 타는 승객이 많은 순)', [r for r, _ in roads], format_func=lambda r: f"{r} — 이 도로 구간에서 {dict(roads)[r]:,}명")
    if st.button('⚙️ 이 도로를 빼고 다시 계산'):
        t0 = time.time()
        with st.status(f"'{road}' 없이 다시 계산 중 — 보통 1~2분", expanded=True) as box:
            area, log = st.empty(), []

            def prog(m):
                log.append(f'{time.time() - t0:5.1f}초 · {m}'); area.markdown('\n'.join(f'- {x}' for x in log))
            cf = eng.counterfactual(p, road, R, progress=prog)
            box.update(label='완료' if cf.get('ok') else '해 없음', state='complete' if cf.get('ok') else 'error', expanded=False)
        st.session_state['cf'] = cf; st.session_state.pop('cf_ai', None)
    cf = st.session_state.get('cf')
    if cf and cf.get('road') == road:
        if not cf.get('ok'):
            st.warning(cf.get('reason'))
        else:
            d = cf['delta']
            m = st.columns(4)
            for col, (lab, k) in zip(m, [('못 타는 승객', 'riders'), ('통제 정류장', 'stops'), ('영향 노선', 'routes'), ('방향 전환', 'turns')]):
                col.metric(f'{road} 빼면 — {lab}', f"{cf['kpi'][k]:,}", f'{d[k]:+,}', delta_color='inverse')
            course_map([(R, '현재 TO-BE', C_TOBE), (cf, f'{road} 제외', C_ALT)], p, key='cf_map', height=360)
            verdict = '나빠집니다 — 엔진이 이 도로를 고른 이유' if d['riders'] + 1000 * d['turns'] > 0 else '좋아지거나 비슷합니다'
            st.markdown(f"**{road}**(약 {cf['removed_km']}km)를 빼면 못 타는 승객 {d['riders']:+,}명, 방향 전환 {d['turns']:+d}회 → 전체로 보면 {verdict}.")
            fx = dict(도로=road, 뺀_길이_km=cf['removed_km'], 현재=report.facts(R, None)['TO_BE'], 도로_제외=report.facts(cf, None)['TO_BE'], 변화=d)
            if st.button('🧠 AI 설명', disabled=not KEY):
                with st.spinner('AI가 설명을 쓰는 중…'):
                    t, err = llm_call(llm.explain_counterfactual, fx)
                st.session_state['cf_ai'] = dict(err=err) if err else dict(t=t.model_dump(), warn=llm.check_numbers(t, fx))
            ca = st.session_state.get('cf_ai')
            if ca:
                st.error(ca['err']) if ca.get('err') else show_tradeoff(ca['t'], ca['warn'])

    st.divider()
    tabs = st.tabs(['KPI 표', '코스 도로별 통제 시각', '영향 버스 노선'] + (['AS-IS 문제점'] if A else []))
    with tabs[0]:
        rows = []
        for lab, k, lower in KPI_LABELS:
            if A is None:
                rows.append({'지표': lab, 'TO-BE': R['kpi'][k]}); continue
            av, tv = A['kpi'][k], R['kpi'][k]
            better = None if lower is None or av == tv else ((tv < av) == lower)
            rows.append({'지표': lab, 'AS-IS': av, 'TO-BE': tv, '변화': f'{(tv - av) / av * 100:+.0f}%' if av else '–',
                         '판정': '' if better is None else ('개선' if better else '악화')})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    with tabs[1]:
        st.dataframe(report.timetable(R), hide_index=True, width='stretch')
    with tabs[2]:
        cols3 = st.columns(2 if A else 1)
        for col, res, nm in zip(cols3, (R, A), ('TO-BE', 'AS-IS')):
            col.markdown(f"**{nm} · {len(res['route_list'])}개 노선**"); col.write(', '.join(res['route_list']))
    if A:
        with tabs[3]:
            for v in A['violations']:
                st.markdown(f'- {v}')
            st.caption(f"출처: {A.get('source', '')}. 모델상 길이 {A['kpi']['length_km']}km (공식 경유지를 간선 도로망 위에 옮김)")


# ───────────────────────── ③ 협의 자료 ─────────────────────────
def page_docs():
    st.title('③ 협의 자료')
    role_strip('엔진의 숫자를 <b>서울시·경찰·버스 운수사별 브리핑</b>으로 씁니다. 숫자는 엔진 결과만 인용하고 자동 대조합니다',
               '<b>구간별 통제 시각표, 영향 노선 목록, 코스 파일</b>을 그대로 만들어 냅니다')
    R, A = S.result, S.ensure_asis()
    f = report.facts(R, A)
    st.caption(f'대상 코스: {describe_params(S.params)}')
    c1, c2 = st.columns([1, 3])
    if c1.button('🧠 AI 브리핑 작성', type='primary', disabled=not KEY):
        with st.spinner('AI가 브리핑을 쓰는 중…'):
            b, err = llm_call(llm.write_briefing, f)
        st.session_state['briefing'] = dict(err=err) if err else dict(b=b.model_dump(), warn=llm.check_numbers(b, f), src='ai', cond=f['조건'])
    br = st.session_state.get('briefing')
    if not br or br.get('err') or br.get('cond') != f['조건']:
        if br and br.get('err'):
            st.error(br['err'])
        br = dict(b=report.rule_briefing(f), warn=[], src='rule', cond=f['조건'])
        c2.caption('AI 브리핑을 작성하기 전이거나 AI가 연결되지 않아 규칙 기반 기본 브리핑을 보여 줍니다 (같은 엔진 숫자).')
    b = br['b']
    with st.container(border=True):
        st.markdown(f"### {b['title']}" + (' &nbsp;<span style="font-size:13px;color:#d4542a">🧠 AI 작성</span>' if br['src'] == 'ai' else ''),
                    unsafe_allow_html=True)
        st.markdown('\n'.join(f'- {x}' for x in b['summary']))
        g = st.columns(3)
        for col, (h, k) in zip(g, [('서울시 (교통·버스정책)', 'for_seoul'), ('경찰 (교통 통제)', 'for_police'), ('버스 운수사', 'for_bus')]):
            col.markdown(f'**{h}**\n' + ''.join(f'\n- {x}' for x in b[k]))
        st.markdown('**한계·주의**\n' + ''.join(f'\n- {x}' for x in b['risks']))
        if br['warn']:
            st.caption('⚠️ 엔진 결과에서 확인되지 않은 숫자: ' + ', '.join(br['warn']))
    st.subheader('내려받기')
    d = st.columns(4)
    d[0].download_button('브리핑 (.md)', report.briefing_markdown(b, f).encode('utf-8'), 'RunWay_브리핑.md', 'text/markdown', width='stretch')
    d[1].download_button('구간별 통제 시각표 (.csv)', report.csv_bytes(report.timetable(R)), 'RunWay_통제시각표.csv', 'text/csv', width='stretch')
    d[2].download_button('영향 노선 (.csv)', report.csv_bytes(report.routes_df(R)), 'RunWay_영향노선.csv', 'text/csv', width='stretch')
    d[3].download_button('코스 (.geojson)', report.geojson_bytes(R), 'RunWay_코스.geojson', 'application/geo+json', width='stretch')
    st.dataframe(report.timetable(R), hide_index=True, width='stretch')


# ───────────────────────── AI 어시스턴트 ─────────────────────────
def page_chat():
    st.title('AI 어시스턴트')
    role_strip('질문을 이해하고 <b>어떤 도구를 부를지</b> 정해 답합니다 (조건 변경·기존 코스 진단·노선 조회·시나리오 비교)',
               '도구가 부르면 계산하고, 답변 속 숫자의 <b>원천</b>이 됩니다 — 근거 없는 숫자는 경고')
    box = st.container(height=560)
    with box:
        if not st.session_state.chat:
            st.info('예) "기존 코스는 뭐가 문제야?" · "출발을 8시로 늦추면?" · "하프로 바꿔 줘" · "10K 코스 보여 줘" · "건대입구역 근처는 피해 줘"')
        for m in st.session_state.chat:
            with st.chat_message(m['role']):
                st.markdown(m['content'])
                for t in m.get('tools', []):
                    st.caption(f'🔧 {t}')
                if m.get('warn'):
                    st.caption('⚠️ 도구 결과에서 확인되지 않은 숫자: ' + ', '.join(m['warn']))
    prompt = st.chat_input('질문하거나 조건을 말로 바꿔 보세요', disabled=not KEY)
    if not KEY:
        st.caption('Claude API 키가 설정되면 채팅이 켜집니다 (Streamlit secrets: ANTHROPIC_API_KEY).')
    if prompt and KEY:
        import anthropic
        st.session_state.chat.append(dict(role='user', content=prompt))
        tools_used = []
        with box:
            with st.chat_message('user'):
                st.markdown(prompt)
            with st.chat_message('assistant'):
                with st.status('생각 중…', expanded=False) as stt:
                    S.progress = stt.write
                    S.log.clear()

                    def on_event(kind, data):
                        tools_used.append(f"{data['name']}({', '.join(f'{k}={v}' for k, v in data['input'].items())})")
                        stt.update(label=f"도구 실행: {data['name']}")
                    try:
                        answer = run_turn(get_client(KEY), S, st.session_state.history, prompt, on_event)
                        stt.update(label='완료', state='complete')
                    except anthropic.RateLimitError:
                        answer = '요청이 많아 잠시 제한되었습니다. 30초 뒤 다시 시도해 주세요.'; stt.update(state='error')
                    except anthropic.APIStatusError as e:
                        answer = f'Claude API 오류({e.status_code}). 잠시 뒤 다시 시도해 주세요.'; stt.update(state='error')
                    except anthropic.APIConnectionError:
                        answer = 'Claude API에 연결하지 못했습니다. 네트워크를 확인해 주세요.'; stt.update(state='error')
                    finally:
                        S.progress = lambda m: None
                warn = unverified_numbers(answer, S, [m['content'] for m in st.session_state.chat if m['role'] == 'user'])
        st.session_state.history = (st.session_state.history + [dict(role='user', content=prompt), dict(role='assistant', content=answer)])[-20:]
        st.session_state.chat.append(dict(role='assistant', content=answer, tools=tools_used, warn=warn))
        st.rerun()


# ───────────────────────── 작동 원리 ─────────────────────────
def _boxes(items, bg, border):
    return ''.join(f"<div style='flex:1;background:{bg};border:1.5px solid {border};border-radius:8px;padding:8px 10px;"
                   f"font-size:13px;line-height:1.45;text-align:center'>{t}</div>" for t in items)


def _layer(title, items, bg, border, arrow=False):
    sep = "<div style='align-self:center;color:#8a8a8a;font-size:18px'>→</div>"
    inner = sep.join(f"<div style='flex:1;display:flex'>{_boxes([t], bg, border)}</div>" for t in items) if arrow else _boxes(items, bg, border)
    return (f"<div style='border:1.5px solid {border};border-radius:10px;padding:10px 12px;margin:2px 0'>"
            f"<div style='font-weight:700;font-size:14px;margin-bottom:8px'>{title}</div>"
            f"<div style='display:flex;gap:8px;align-items:stretch'>{inner}</div></div>")


def _flow(down, up):
    return (f"<div style='display:flex;justify-content:center;gap:60px;color:#52514e;font-size:12.5px;margin:4px 0'>"
            + (f"<span>▼ {down}</span>" if down else '') + (f"<span>▲ {up}</span>" if up else '') + "</div>")


ARCH_HTML = (
    _layer('① 사용자 화면 — Streamlit', ['<b>① 목표 입력</b><br>말로 목표 입력 → 바로 계산', '<b>② 결과 비교</b><br>지도·KPI·여러 안·도로 빼고 재계산',
                                         '<b>③ 협의 자료</b><br>기관별 브리핑·통제 시각표·노선·코스 파일', '<b>AI 어시스턴트</b><br>질문·조건 변경'], '#eaf2fc', '#9ec5f0')
    + _flow('자연어 목표 · 질문', '설명 · 브리핑 · 코스 · KPI')
    + _layer('② LLM 계층 — Claude Opus 5.5 · 말 ↔ 조건, 숫자 → 말 (숫자는 만들지 않음)',
             ['<b>목표 → 엔진 조건</b><br>구조화 출력 · 반영 불가 요청 알림', '<b>도구 호출 에이전트</b><br>Tool Runner · 도구 6종',
              '<b>장단점 요약 · 이유 설명</b><br>여러 안 비교 · 반사실 결과 해석', '<b>기관별 브리핑</b><br>서울시 · 경찰 · 운수사',
              '<b>숫자 대조 검사</b><br>엔진에 없는 숫자 경고'], '#fdeee6', '#f3b79b')
    + _flow('엔진 조건 · 도구 호출', 'facts — 엔진이 계산한 숫자만')
    + _layer('③ 최적화 엔진 — runway · 조건 → 코스·숫자', ['<b>조건 검증</b><br>Pydantic · 장소는 정류소 데이터에서만',
                                                         '<b>시나리오</b><br>위치별 통제 시간 × 정류장 수요', '<b>MILP</b><br>PuLP + HiGHS<br>LP 완화 → 회랑 → MILP',
                                                         '<b>정밀 평가 · 규칙 검사</b><br>10m 단위 · 독립 검증', '<b>결과 저장</b><br>더 좋은 해만 교체 · 반사실 재계산'],
             '#eef0f6', '#9aa6c4', arrow=True)
    + _flow('', '수요 · 도로망 · 공식 코스')
    + _layer('④ 데이터', ['<b>버스 노선×정류장×일별 승하차</b><br>사용일자로 일요일·휴일 판별', '<b>정류소 위치 · 시간대별 승차</b><br>서울 열린데이터광장',
                         '<b>OSM 도로망</b><br>풀·하프 / 10K(서부)', '<b>공식 AS-IS 코스</b><br>2026 대회 책자, 고정 파일'], '#f3f3ef', '#c9c8c2'))

_KIND = {'h': ('#eaf2fc', '#9ec5f0', '👤 사람'), 'l': ('#fdeee6', '#f3b79b', '🧠 LLM'), 'e': ('#eef0f6', '#9aa6c4', '⚙️ 엔진'),
         'q': ('#fff6d6', '#e8c65a', '◇ 판단')}


def _steps(title, steps):
    """steps: [(종류, 제목, 설명, 분기 설명 or '')] — 한 줄 흐름"""
    out = []
    for kind, head, body, branch in steps:
        bg, bd, tag = _KIND[kind]
        out.append(f"<div style='flex:1;background:{bg};border:1.5px solid {bd};border-radius:{14 if kind == 'q' else 8}px;padding:8px 10px;"
                   f"font-size:13px;line-height:1.45;text-align:center'><div style='font-size:11px;color:#52514e'>{tag}</div><b>{head}</b><br>{body}"
                   + (f"<div style='margin-top:4px;font-size:11.5px;color:#9a4a1f'>{branch}</div>" if branch else '') + "</div>")
    arrow = "<div style='align-self:center;color:#8a8a8a;font-size:18px'>→</div>"
    return (f"<div style='font-weight:700;font-size:13.5px;margin:10px 0 6px'>{title}</div>"
            f"<div style='display:flex;gap:6px;align-items:stretch'>{arrow.join(out)}</div>")


FLOW_HTML = (
    _steps('1단계 · 목표를 조건으로 (① 목표 입력)', [
        ('h', '목표를 말로 입력', '“건대입구역 근처는 피해 줘”', ''),
        ('l', '조건 해석', '종목·출발·도착·시각·피할 장소 · 반영 불가 요청 알림', ''),
        ('q', '장소가 데이터에 있나?', '정류소 위치에서 검색', '아니오·여러 곳 → 후보 제시, 사람이 선택'),
        ('e', '조건 검증', '허용 범위 · 데이터 영역 확인', '')])
    + _steps('2단계 · 조건을 코스로 (② 결과 비교)', [
        ('e', 'MILP 풀이', '사전 계산이면 즉시, 아니면 10초~2분', ''),
        ('q', '해가 있나?', '규칙을 모두 지키는 코스', '아니오 → 이유 + 완화 후보 → 1단계로'),
        ('e', '규칙 검사 · 정밀 평가', '거리 · 회귀 · 재방문 · 금지 구간', ''),
        ('e', '여러 안 비교 · 도로 빼고 재계산', '“왜 이 길인가”의 숫자 근거', '')])
    + _steps('3단계 · 숫자를 협의 자료로 (③ 협의 자료)', [
        ('l', '장단점 · 이유 · 기관별 브리핑', '엔진 facts만 인용', ''),
        ('q', '숫자가 엔진 결과와 일치?', '자동 대조', '아니오 → 근거 없는 숫자 경고 표시'),
        ('h', '협의 자료 내려받기', '브리핑 · 통제 시각표 · 노선 · 코스', '')]))


def page_how():
    st.title('작동 원리 — 왜 LLM이고, 왜 최적화 엔진인가')
    st.markdown('**한 줄 요약: LLM은 "말 ↔ 조건"과 "숫자 → 말"을, 최적화 엔진은 "조건 → 코스·숫자"를 맡는다. 서로가 못 하는 일을 맡는다.**')
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown('#### 🧠 왜 LLM이 필요한가')
        st.markdown("""
- **목표가 말로 온다.** "10K로 바꾸고 합정역 근처는 피해 줘"처럼 담당자의 요구는 조건 여러 개가 섞인 문장이다. 슬라이더로는 *피할 장소* 같은 열린 조건과 조합을 입력하기 어렵다 → **엔진 조건으로 번역**
- **반영할 수 없는 요구를 걸러 낸다.** "반드시 ○○로를 지나기"처럼 엔진이 다루지 못하는 요청은 이유와 함께 표시하고, 애매하면 **되묻는다**
- **숫자를 기관의 언어로 바꾼다.** 같은 결과라도 서울시(노선), 경찰(교차로·통제 시각), 운수사(정류장)가 알아야 할 것이 다르다 → **기관별 브리핑**
- **트레이드오프를 설명한다.** 엔진은 '최적'만 알려 준다. 여러 안 사이에서 *무엇을 얻고 잃는지*는 말로 풀어야 결정할 수 있다""")
        st.markdown('**LLM에게 맡기지 않는 것**: 코스 경로와 모든 숫자. 그럴듯한 숫자를 지어낼 수 있어서 → 숫자는 엔진만, 답변 숫자는 자동 대조.')
    with c2, st.container(border=True):
        st.markdown('#### ⚙️ 왜 최적화 엔진이 필요한가')
        net = eng.network(S.params)
        sv = S.result.get('solver', {})
        size = f"변수 {sv['변수']:,}개 · 제약 {sv['제약']:,}개 · {sv.get('초', '-')}초" if sv.get('변수') else '변수 수천 개 · 제약 수천~수만 개'
        st.markdown(f"""
- **조합이 사람 손을 넘는다.** 이 종목 도로망 {net.stats['구간']:,}개 구간({net.NA:,}개 방향) 중에서 정확히 {S.params.LMIN / 1000:g}km가 되는 길을 골라야 한다 — 모델 크기: {size}
- **같은 도로도 '언제' 지나느냐에 따라 피해가 다르다.** 출발점 근처는 이른 아침 잠깐, 후반 구간은 정오 넘어까지 막힌다 → 위치×시간×정류장 수요가 얽힌 비용은 수식으로만 일관되게 비교된다
- **규칙을 보장한다.** 공인 거리, 회귀·재방문 금지, 금지 구간을 **하드 제약**으로 넣어 어긴 코스는 나오지 않는다 (사람이나 LLM은 보장 못 함)
- **최적성의 근거가 있다.** LP 하한과 비교한 gap으로 '이보다 크게 좋은 코스는 없다'를 수치로 보인다""")
        cf_any = next((json.load(open(f, encoding='utf-8')) for f in sorted((eng.store / 'counterfactual').glob('*.json'))
                       if json.load(open(f, encoding='utf-8')).get('params', {}).get('EVENT') == S.params.EVENT), None) \
            if (eng.store / 'counterfactual').exists() else None
        if cf_any:
            d = cf_any['delta']
            st.info(f"**직관과 다른 최적 선택 (실제 계산)**: 현재 코스에서 못 타는 승객이 가장 많은 도로인 **{cf_any['road']}**를 빼고 다시 풀면 "
                    f"못 타는 승객 {d['riders']:+,}명, 방향 전환 {d['turns']:+d}회로 **오히려 나빠진다**. 도로 하나씩 보면 나빠 보여도 코스 전체로는 최선 — "
                    f"이런 판단은 전체 조합을 푸는 엔진만 할 수 있다. (② 결과 비교 → '왜 이 길인가'에서 직접 해 볼 수 있음)")
    st.markdown('#### 함께일 때만 되는 것')
    st.dataframe(pd.DataFrame([
        {'능력': '말로 된 목표 이해', 'LLM 단독': '✅', '최적화 엔진 단독': '❌ (수식 입력 필요)', 'RunWay': '✅ LLM이 엔진 조건으로'},
        {'능력': '42.195km·회귀 금지 등 규칙 보장', 'LLM 단독': '❌', '최적화 엔진 단독': '✅', 'RunWay': '✅ 하드 제약 + 독립 검사'},
        {'능력': '1,000여 개 도로 조합의 최적 코스', 'LLM 단독': '❌', '최적화 엔진 단독': '✅', 'RunWay': '✅'},
        {'능력': '숫자의 정확성', 'LLM 단독': '❌ 지어낼 위험', '최적화 엔진 단독': '✅', 'RunWay': '✅ 엔진 숫자만 인용·자동 대조'},
        {'능력': '기관별 설명·장단점 정리', 'LLM 단독': '✅ (근거 없음)', '최적화 엔진 단독': '❌', 'RunWay': '✅ 엔진 숫자로 근거'},
        {'능력': '조건 바꿔 다시 검토', 'LLM 단독': '즉시이나 근거 없음', '최적화 엔진 단독': '전문가만', 'RunWay': '✅ 말로 요청 → 재계산'}]),
        hide_index=True, width='stretch')
    st.markdown('#### 서비스 구조도')
    st.markdown(ARCH_HTML, unsafe_allow_html=True)
    st.markdown('#### 사용자 흐름 (플로우차트)')
    st.markdown(FLOW_HTML, unsafe_allow_html=True)
    st.markdown('#### 최적화 모델 (MILP) 요약')
    st.latex(r'\min\ \sum_{e,b} c_{e,b}\,y_{e,b} \;+\; \sum_{(e,g)} \tau_{eg}\,z_{eg} \;+\; \sum \rho\,\omega\,q \;+\; \varepsilon\sum_e \ell_e x_e')
    st.markdown("""
- $c_{e,b}$: 도로 $e$를 코스의 $b$번째 3km 구간대에서 달릴 때 **통제 시간 동안 버스를 못 타는 승객** (위치마다 통제 시각이 달라 값이 다름)
- $\\tau$: 방향 전환 벌점(고정값), $\\rho\\omega q$: 되돌아오기·재방문 벌점, $\\varepsilon$: 미세 거리 비용
- 제약: 출발→도착 하나의 경로 · 같은 교차로 1회 · 누적거리 전파(고리 금지) · 공인 거리 · 같은 도로 왕복 금지 · **회귀 금지**(1km 안 반대 방향 나란히 달리기) · **재방문 금지**(코스상 2km 넘게 떨어진 두 지점이 300m 안) · 금지 구간 · 데이터 영역""")
    st.markdown('#### 검증과 한계')
    c3, c4 = st.columns(2)
    c3.markdown("""**검증**
- 자동 테스트 27개: 노트북 결과 재현, MILP 목적함수 = 독립 재계산, 휴일 판별, 공식 AS-IS 고정, 종목별 규칙
- 모든 결과를 규칙 검사기로 다시 확인, 더 좋은 해(10m 단위 정밀 평가)만 교체""")
    c4.markdown("""**한계**
- **버스 승객만 고려 — 일반 차량(승용차·택시·화물차)의 통행·우회 지체는 반영하지 못했다**
- 일요일 시간대별 원자료가 없어 정류장별 월 시간대 비중을 사용
- 코스 위에 정류장이 있는 노선만 영향 노선으로 집계 (가로지르기만 하는 노선 제외)
- AS-IS는 공식 경유지를 간선 도로망 위에 옮긴 근사 (모델상 길이가 조금 다름)
- 최적성은 '후보 회랑 안에서 허용 gap 이내' (회랑에 해가 없으면 전체 후보로 재풀이)""")


PAGES = {'goal': st.Page(page_goal, title='① 목표 입력', icon='🧠', default=True),
         'result': st.Page(page_result, title='② 결과 비교', icon='⚙️'),
         'docs': st.Page(page_docs, title='③ 협의 자료', icon='📄'),
         'chat': st.Page(page_chat, title='AI 어시스턴트', icon='💬'),
         'how': st.Page(page_how, title='작동 원리', icon='🔍')}
st.navigation({'코스 설계': [PAGES['goal'], PAGES['result'], PAGES['docs']], '더 보기': [PAGES['chat'], PAGES['how']]}).run()
