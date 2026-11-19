"""RunWay — 마라톤 코스 최적화 챗봇 (Streamlit)

    streamlit run app.py

화면: 왼쪽 = 지도·KPI·표, 오른쪽 = 채팅. 사이드바 = 조건 카드(LLM 없이도 직접 조작 가능).
숫자는 모두 runway 엔진(PuLP MILP)이 계산하고, Claude는 조건 해석과 설명만 한다.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

from agent import MODEL, Session, make_tools, run_turn, unverified_numbers
from runway import DEFAULT, Engine
from runway.engine import describe_params
from runway.evaluate import KPI_LABELS

ROOT = Path(__file__).resolve().parent
C_ASIS, C_TOBE = '#eb6834', '#2a78d6'          # 검증된 기본 팔레트 1·2번 (노트북과 동일)

st.set_page_config(page_title='RunWay — 마라톤 코스 최적화', page_icon='🏃', layout='wide')


# ───────────────────────── 공용 자원 ─────────────────────────
@st.cache_resource(show_spinner='도로망·버스 데이터 불러오는 중 (첫 실행 약 20초)…')
def get_engine() -> Engine:
    eng = Engine(ROOT)
    if eng.cached(DEFAULT) is None:                 # 기본 시나리오 캐시가 없으면 노트북 결과로 채운다 (시작 시 긴 풀이 방지)
        import json
        r = json.load(open(ROOT / 'out' / 'result.json', encoding='utf-8'))
        eng.seed(DEFAULT, r['tobe']['order'], r['solver'])
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


eng = get_engine()
if 'sess' not in st.session_state:
    s = Session(engine=eng)
    s.result = eng.solve(DEFAULT)                   # 기본 시나리오는 사전 계산되어 즉시
    st.session_state.update(sess=s, history=[], chat=[])
S: Session = st.session_state.sess
TOOLS = {t.name: t for t in make_tools(S)}


def hm(t):
    return '%02d:%02d' % (int(t), round(t % 1 * 60))


# ───────────────────────── 사이드바: 조건 카드 ─────────────────────────
with st.sidebar:
    st.header('조건 카드')
    st.caption('채팅으로 바꿔도 되고, 여기서 직접 바꿔도 됩니다. 같은 엔진이 계산합니다.')
    p = S.params
    with st.form('cond'):
        start = st.text_input('출발지', p.START_NAME)
        finish = st.text_input('도착지', p.FINISH_NAME)
        t0 = st.slider('출발 시각', 6.0, 10.0, float(p.T0), 0.5, format='%.1f시')
        view = st.slider('30km 이후 롯데월드타워 조망 하한 (km)', 0.0, 4.0,
                         float(p.IDENTITY[0].L / 1000) if p.IDENTITY else 0.0, 0.5)
        simple = st.select_slider('코스 단순성 (꺾임 벌점 배율)', [0.5, 1.0, 2.0, 3.0], value=round(p.TURN[0] / DEFAULT.TURN[0], 1)
                                  if round(p.TURN[0] / DEFAULT.TURN[0], 1) in (0.5, 1.0, 2.0, 3.0) else 1.0)
        go = st.form_submit_button('최적 코스 계산', type='primary', width='stretch')
    if go:
        kw = dict(start_hour=t0, view_min_km=view, simplicity=simple)
        if start != p.START_NAME:
            kw['start_place'] = start
        if finish != p.FINISH_NAME:
            kw['finish_place'] = finish
        with st.status('계산 중…', expanded=True) as box:
            S.progress = box.write
            out = TOOLS['optimize_course'].call(kw)
            S.progress = lambda m: None
            import json as _j
            o = _j.loads(out)
            if o.get('ok'):
                box.update(label='완료' + (' (저장된 결과)' if o.get('캐시결과') else ''), state='complete', expanded=False)
            else:
                box.update(label='계산하지 못했습니다', state='error')
                st.error(o.get('error') or o.get('이유'))
                if o.get('후보'):
                    st.info('후보: ' + ', '.join(o['후보']))
    if st.button('기본 조건으로', width='stretch'):
        TOOLS['optimize_course'].call(dict(reset=True))
        st.rerun()

    st.divider()
    st.subheader('미리 계산된 시나리오')
    lib = eng.library()
    for i, x in enumerate(lib):
        if st.button(f"{x['조건']} — 피해 {x['kpi']['riders']:,}명", key=f'lib{i}', width='stretch'):
            S.params, S.result = x['params'], dict(eng.cached(x['params']), ok=True, cached=True)
            st.rerun()
    st.divider()
    key = api_key()
    st.caption(f'LLM: {MODEL} ' + ('● 연결됨' if key else '○ API 키 없음 (조건 카드로만 사용)'))
    st.caption('데이터: 서울 열린데이터광장 버스 승하차(2026-10)·정류소 위치, © OpenStreetMap contributors')


# ───────────────────────── 본문 ─────────────────────────
st.title('RunWay')
st.markdown('**버스 승객 피해는 최소로, 코스는 단순하게.** 2026 JTBC 서울마라톤 코스(AS-IS)와 PuLP 최적화 코스(TO-BE) 비교')

R, A = S.result, S.ensure_asis()
left, right = st.columns([3, 2], gap='large')

with left:
    st.caption(f"현재 조건: {describe_params(S.params)}")
    k, a = R['kpi'], A['kpi']
    cols = st.columns(4)
    for c, (lab, key_) in zip(cols, [('승객 피해 (명/주말 하루)', 'riders'), ('영향 버스 노선', 'routes'),
                                     ('방향 전환 45°↑', 'turns'), ('왕복 회귀 (km)', 'backtrack_km')]):
        d = k[key_] - a[key_]
        c.metric(lab, f'{k[key_]:,}', f'{d:+,} vs AS-IS' if d else '변화 없음', delta_color='inverse')

    # 지도
    def path_rows(res, name, color):
        coords = res['geojson']['geometry']['coordinates']
        rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
        return dict(name=name, path=coords, color=rgb, info=f"{name}: {res['kpi']['length_km']}km, 승객 피해 {res['kpi']['riders']:,}명")
    show = st.radio('지도에 표시', ['둘 다', 'TO-BE만', 'AS-IS만'], horizontal=True, label_visibility='collapsed')
    paths = ([path_rows(A, 'AS-IS', C_ASIS)] if show != 'TO-BE만' else []) + ([path_rows(R, 'TO-BE', C_TOBE)] if show != 'AS-IS만' else [])
    pts = [dict(name=f'출발 · {S.params.START_NAME}', pos=[S.params.START[1], S.params.START[0]], color=[11, 11, 11]),
           dict(name=f'도착 · {S.params.FINISH_NAME}', pos=[S.params.FINISH[1], S.params.FINISH[0]], color=[11, 11, 11])]
    pts += [dict(name=f'★ {lm.name}', pos=[lm.coord[1], lm.coord[0]], color=[82, 81, 78]) for lm in S.params.IDENTITY]
    zones = [dict(name=f'금지 · {z.name}', pos=[z.coord[1], z.coord[0]], r=z.r) for z in S.params.FORBID]
    layers = [
        pdk.Layer('PathLayer', paths, get_path='path', get_color=[255, 255, 255], width_min_pixels=7),     # 흰 테두리 (겹침 구분)
        pdk.Layer('PathLayer', paths, get_path='path', get_color='color', width_min_pixels=4, pickable=True),
        pdk.Layer('ScatterplotLayer', zones, get_position='pos', get_radius='r', get_fill_color=[82, 81, 78, 50],
                  get_line_color=[82, 81, 78], stroked=True, line_width_min_pixels=1, pickable=True),
        pdk.Layer('ScatterplotLayer', pts, get_position='pos', get_fill_color='color', get_radius=60, radius_min_pixels=5,
                  get_line_color=[255, 255, 255], stroked=True, line_width_min_pixels=2, pickable=True),
        pdk.Layer('TextLayer', pts + zones, get_position='pos', get_text='name', get_size=13, get_color=[11, 11, 11],
                  get_pixel_offset=[0, -16], character_set='auto', font_family='"Malgun Gothic", "Noto Sans KR", sans-serif',
                  background=True, get_background_color=[255, 255, 255, 220]),
    ]
    st.pydeck_chart(pdk.Deck(layers=layers, map_provider='carto', map_style='light',
                             initial_view_state=pdk.ViewState(latitude=37.535, longitude=127.05, zoom=11.2),
                             tooltip={'text': '{info}{name}'}), height=520)
    st.markdown(f"<span style='color:{C_ASIS}'>━━</span> AS-IS (2026 JTBC 공식 경유지)&nbsp;&nbsp;&nbsp;"
                f"<span style='color:{C_TOBE}'>━━</span> TO-BE (RunWay 최적 코스)", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(['KPI 비교', '코스 도로별', '영향 버스 노선', 'AS-IS 문제점'])
    with t1:
        rows = []
        for lab, key_, lower in KPI_LABELS:
            av, tv = a[key_], k[key_]
            better = None if lower is None or av == tv else ((tv < av) == lower)
            rows.append({'지표': lab, 'AS-IS': av, 'TO-BE': tv, '변화': f'{(tv - av) / av * 100:+.0f}%' if av else '–',
                         '판정': '' if better is None else ('개선' if better else '악화')})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        sv = R.get('solver', {})
        obj = R.get('objective')
        st.caption(f"TO-BE 풀이: {sv.get('solver', '-')} · {sv.get('해 종류', sv.get('상태', '-'))} · 목적함수 {'-' if obj is None else f'{obj:,}'}"
                   + (' · 저장된 결과' if R.get('cached') else ''))
    with t2:
        st.dataframe(pd.DataFrame(R['roads']), hide_index=True, width='stretch')
        st.caption('피해 = 해당 도로 통제 시간 동안 그 도로 정류장에서 버스를 타지 못하는 승객(주말 하루 기준)')
    with t3:
        c1, c2 = st.columns(2)
        c1.markdown(f"**TO-BE {len(R['route_list'])}개**"); c1.write(', '.join(R['route_list']))
        c2.markdown(f"**AS-IS {len(A['route_list'])}개**"); c2.write(', '.join(A['route_list']))
    with t4:
        for v in A['violations']:
            st.markdown(f'- {v}')
        st.dataframe(pd.DataFrame(A['km_check']), hide_index=True, width='stretch')
        st.caption('AS-IS는 공식 경유지를 간선 도로망 위에 옮긴 것이라 모델상 길이가 실제(42.195km)보다 깁니다. km당 피해로도 비교하세요.')

# ───────────────────────── 채팅 ─────────────────────────
with right:
    st.subheader('RunWay 어시스턴트')
    box = st.container(height=640)
    with box:
        if not st.session_state.chat:
            st.info('예) "출발을 7시로 당기면?" · "건대입구역 근처는 피해 줘" · "기존 코스는 뭐가 문제야?" · '
                    '"도착을 잠실종합운동장으로 바꾸고 조망 조건은 빼 줘"')
        for m in st.session_state.chat:
            with st.chat_message(m['role']):
                st.markdown(m['content'])
                for t in m.get('tools', []):
                    st.caption(f"🔧 {t}")
                if m.get('warn'):
                    st.caption('⚠️ 도구 결과에서 확인되지 않은 숫자: ' + ', '.join(m['warn']))
    prompt = st.chat_input('조건을 말로 바꿔 보세요', disabled=not key)
    if not key:
        st.caption('Claude API 키가 설정되면 채팅이 켜집니다 (Streamlit secrets: ANTHROPIC_API_KEY).')
    if prompt and key:
        st.session_state.chat.append(dict(role='user', content=prompt))
        with box:
            with st.chat_message('user'):
                st.markdown(prompt)
            with st.chat_message('assistant'):
                tools_used = []
                with st.status('생각 중…', expanded=False) as stt:
                    S.progress = stt.write
                    S.log.clear()

                    def on_event(kind, data):
                        tools_used.append(f"{data['name']}({', '.join(f'{k}={v}' for k, v in data['input'].items())})")
                        stt.update(label=f"도구 실행: {data['name']}")
                    import anthropic
                    try:
                        answer = run_turn(get_client(key), S, st.session_state.history, prompt, on_event)
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
        st.session_state.history += [dict(role='user', content=prompt), dict(role='assistant', content=answer)]
        st.session_state.history = st.session_state.history[-20:]
        st.session_state.chat.append(dict(role='assistant', content=answer, tools=tools_used, warn=warn))
        st.rerun()
