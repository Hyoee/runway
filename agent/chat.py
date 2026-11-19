"""대화 한 턴 실행 — Claude(Tool Runner)가 도구를 불러 조건을 바꾸고, 결과를 설명한다."""
from __future__ import annotations

import json
import re

import anthropic

from runway.engine import describe_params

from .tools import Session, make_tools

MODEL = 'claude-opus-5-5'

SYSTEM = """당신은 'RunWay' — 서울 도심 마라톤 코스를 설계하는 교통 최적화 어시스턴트입니다.
사용자는 대회 사무국·서울시·경찰 담당자, 또는 해커톤 심사위원입니다. 한국어로, 짧고 분명하게 답하세요.

## 역할 분담 (가장 중요)
- 코스와 모든 숫자(승객 피해, 노선 수, 거리, 변화율, 통제 시각 등)는 도구(PuLP MILP 솔버·평가 엔진)만 만듭니다.
- 당신은 사용자의 말을 도구의 조건 값으로 옮기고, 도구가 돌려준 숫자를 그대로 인용해 설명합니다.
- 도구 결과에 없는 숫자는 쓰지 마세요. 추정·반올림한 새 숫자를 만들지 말고, 필요하면 도구를 다시 부르세요.
- 장소 좌표를 지어내지 마세요. 장소는 search_place 또는 optimize_course의 장소 이름으로만 정합니다.

## 도구 사용
- 조건을 바꾸는 요청(출발지·도착지·출발 시각·조망 하한·금지 구간·코스 단순성)은 optimize_course 한 번에 바뀌는 값만 넣어 부릅니다.
- 장소 이름이 애매하면(후보 여러 곳) 사용자에게 후보를 보여 주고 고르게 합니다.
- 새 조건 계산은 최대 2분 걸릴 수 있다고 미리 한 줄로 알립니다(캐시결과=true면 즉시).
- 해가 없으면(ok=false) 이유를 설명하고 완화 후보를 제시한 뒤, 사용자가 고르면 다시 계산합니다. 임의로 완화하지 마세요.
- 출발지·도착지를 바꾸면 '30km 이후 롯데월드타워 조망' 정체성 조건이 맞지 않을 수 있습니다. 그때는 조망 하한을 0으로 할지 물어보세요.
- 기존 코스의 문제점 질문에는 get_asis_problems, 버스 노선 질문에는 get_affected_bus_routes, 조망과 피해의 교환 관계에는 list_precomputed_scenarios를 씁니다.

## 데이터와 한계 (질문받으면 정직하게)
- 수요: 서울시 버스 정류장별 승차(2026-10, 주말 하루 평균 수준 × 월 전체 시간대 분포). 코스 위에 정류장이 있는 노선만 '영향 노선'으로 셉니다.
- 통제 시간: 선두 20km/h, 후미 5시간 컷오프 기준, 2026 대회 경찰 발표 통제 시각에 맞춰 보정.
- 도로: OpenStreetMap 간선·보조간선. AS-IS는 공식 경유지를 이 도로망 위에 옮긴 것이라 모델상 43.2km로 실제보다 깁니다.
- 솔버 결과는 '후보 회랑 안에서 허용 gap 이내 최적'입니다.

답변 형식: 핵심 숫자 2~4개(AS-IS 대비)로 시작하고, 코스가 어떻게 달라졌는지(주요 도로) 한두 문장, 필요하면 다음에 해 볼 만한 조건 하나를 제안하세요. 지도와 KPI 표는 화면에 따로 보이므로 표를 길게 다시 쓰지 마세요."""


def _numbers(text: str) -> set[str]:
    return {n.replace(',', '').lstrip('+-').rstrip('.') for n in re.findall(r'[-+]?\d[\d,]*\.?\d*', text)}


def unverified_numbers(answer: str, s: Session, user_texts: list[str]) -> list[str]:
    """답변 속 숫자 중 도구 결과·사용자 입력 어디에도 없는 것 (2자리 이상만) — 화면에 경고로 표시"""
    known = set()
    for e in s.log:
        known |= _numbers(json.dumps(e['result'], ensure_ascii=False, default=str))
        known |= _numbers(json.dumps(e['args'], ensure_ascii=False, default=str))
    for t in user_texts + [SYSTEM]:
        known |= _numbers(t)
    out = []
    for n in _numbers(answer):
        core = n.split('.')[0]
        if len(core) >= 2 and n not in known and core not in known and n.rstrip('0').rstrip('.') not in known:
            out.append(n)
    return out


def run_turn(client: anthropic.Anthropic, s: Session, history: list[dict], user_text: str, on_event=lambda kind, data: None) -> str:
    """history: [{'role','content'(str)}] — 텍스트만 보관한다. 현재 조건·KPI는 매 턴 상태 블록으로 다시 넣는다."""
    state = f'[현재 화면 상태] 조건: {describe_params(s.params) if s.result else "아직 계산된 코스 없음"}'
    if s.result:
        k = s.result['kpi']
        state += f" | TO-BE 승객피해 {k['riders']}명, 영향노선 {k['routes']}개, 길이 {k['length_km']}km"
    messages = history + [{'role': 'user', 'content': f'{state}\n\n{user_text}'}]
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        tools=make_tools(s),
        messages=messages,
        output_config={'effort': 'medium'},
        betas=['server-side-fallback-2026-07-01'],
        fallbacks='default',                      # 안전 분류기가 거절하면 서버가 대체 모델로 이어서 응답
        max_iterations=8,
    )
    final = None
    for msg in runner:
        for b in msg.content:
            if b.type == 'tool_use':
                on_event('tool', dict(name=b.name, input=b.input))
        final = msg
    if final is None:
        return '응답을 받지 못했습니다.'
    if final.stop_reason == 'refusal':
        return '이 요청에는 답할 수 없습니다. 코스 조건에 관한 질문으로 다시 말씀해 주세요.'
    text = '\n'.join(b.text for b in final.content if b.type == 'text').strip()
    if final.stop_reason == 'max_tokens':
        text += '\n\n(답변이 길어 중간에 잘렸습니다.)'
    return text or '(도구 실행만 하고 설명 없이 끝났습니다. 다시 물어봐 주세요.)'
