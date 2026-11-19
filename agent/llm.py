"""LLM이 맡는 일 — '말 ↔ 조건', '숫자 → 말'. 코스와 숫자는 만들지 않는다.

1) interpret_goal : 자유로운 목표 문장 → 조건표(ConditionCard). 장소는 이름만 뽑고 좌표는 엔진이 데이터에서 찾는다.
2) write_briefing : 엔진 결과(facts) → 관계기관별 협의 브리핑(Briefing). facts에 없는 숫자는 쓰지 않는다.
3) compare_options: 여러 안의 facts → 무엇을 얻고 잃는지(Tradeoff)
4) explain_counterfactual: '이 도로를 빼면?' 재계산 결과 → 왜 그 도로를 골랐는지 설명

모두 구조화 출력(Pydantic 스키마)으로 받아 화면에 그대로 그린다. 응답 속 숫자는 check_numbers로 facts와 대조한다.
"""
from __future__ import annotations

import json
import re
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

MODEL = 'claude-opus-5-5'
_COMMON = dict(model=MODEL, max_tokens=8000, betas=['server-side-fallback-2026-07-01'], fallbacks='default',
               output_config={'effort': 'medium'})


class ConditionCard(BaseModel):
    event: Literal['full', 'half', '10k'] | None = Field(None, description='종목. 언급 없으면 null')
    start_place: str | None = Field(None, description='출발지 장소 이름 (바꾸자는 말이 있을 때만)')
    finish_place: str | None = Field(None, description='도착지 장소 이름 (바꾸자는 말이 있을 때만)')
    start_hour: float | None = Field(None, description='출발 시각(시). 07:30 → 7.5. 언급 없으면 null')
    avoid_places: list[str] = Field(default_factory=list, description='코스가 피해야 할 장소 이름들 (병원, 학교, 특정 역 등)')
    simplicity: Literal[0.5, 1.0, 2.0, 3.0] | None = Field(None, description='코스 단순성 배율. 단순하게=2, 아주 단순=3, 덜 신경=0.5')
    goals_understood: list[str] = Field(description='사용자의 목표를 어떻게 이해했는지, 조건 하나당 한 줄 (한국어)')
    unsupported: list[str] = Field(default_factory=list, description='이 서비스로 반영할 수 없는 요청과 그 이유')
    question: str | None = Field(None, description='애매해서 사용자에게 확인할 질문 (없으면 null)')


class Briefing(BaseModel):
    title: str
    summary: list[str] = Field(description='핵심 3~4줄. 숫자는 facts에 있는 값만')
    for_seoul: list[str] = Field(description='서울시 교통·버스정책 부서에 전달할 내용')
    for_police: list[str] = Field(description='경찰 교통 통제에 전달할 내용')
    for_bus: list[str] = Field(description='버스 운수사에 전달할 내용')
    risks: list[str] = Field(description='한계·주의 사항')


class Tradeoff(BaseModel):
    headline: str = Field(description='한 줄 결론')
    gains: list[str] = Field(description='얻는 것 (숫자는 facts 값만)')
    costs: list[str] = Field(description='잃는 것·대가')
    recommendation: str = Field(description='어떤 상황에 어느 안이 맞는지')


SYS = """당신은 서울 도심 마라톤 코스를 설계하는 RunWay의 AI입니다. 사용자는 대회 사무국·서울시·경찰 담당자입니다.
역할 분담: 당신은 '사람의 말 ↔ 조건'과 '엔진의 숫자 → 사람의 말'을 맡고, 코스와 숫자는 최적화 엔진(PuLP MILP)이 만듭니다.
- 숫자는 주어진 facts(JSON)에 있는 값만 그대로 인용하세요. 계산·추정·반올림한 새 숫자를 만들지 마세요.
- facts에 '비교' 항목(AS-IS 데이터 없음)이 있으면 기존 코스와 비교하는 표현을 쓰지 마세요.
- 장소 좌표를 지어내지 마세요. 장소는 이름만 적습니다.
- 랜드마크 조망 조건은 없습니다. 한국어로, 담당자가 바로 쓸 수 있게 짧고 구체적으로."""

GOAL_GUIDE = """서비스가 반영할 수 있는 조건: 종목(풀코스·하프·10K), 출발지, 도착지, 출발 시각(06:00~10:00, 30분 단위), 피할 장소(반경 150m 도로 금지),
코스 단순성(꺾임 벌점 배율 0.5/1/2/3). 항상 지켜지는 규칙: 공인 거리, 회귀·재방문 금지, 같은 교차로 1회.
최적화 목표는 '정류장이 코스에 들어가 통제 시간 동안 버스를 못 타는 승객' 최소화 + 코스 단순성입니다.
이 밖의 요청(예: 특정 도로를 반드시 지나기, 날씨, 참가자 수, 예산)은 unsupported에 이유와 함께 적으세요."""


def _parse(client: anthropic.Anthropic, schema, system: str, user: str):
    r = client.beta.messages.parse(system=system, messages=[{'role': 'user', 'content': user}], output_format=schema, **_COMMON)
    if r.stop_reason == 'refusal':
        raise RuntimeError('AI가 이 요청에 답하지 않았습니다. 표현을 바꿔 다시 시도해 주세요.')
    if r.parsed_output is None:
        raise RuntimeError(f'AI 응답을 조건표로 읽지 못했습니다 (stop_reason={r.stop_reason})')
    return r.parsed_output


def interpret_goal(client, text: str, current: str) -> ConditionCard:
    return _parse(client, ConditionCard, SYS + '\n\n' + GOAL_GUIDE,
                  f'현재 조건: {current}\n\n사용자의 목표:\n"""{text}"""\n\n이 목표를 조건표로 바꾸세요. 바꾸자는 말이 없는 항목은 null로 두세요.')


def write_briefing(client, f: dict) -> Briefing:
    return _parse(client, Briefing, SYS, 'facts:\n' + json.dumps(f, ensure_ascii=False) +
                  '\n\n이 결과로 관계기관 협의용 브리핑을 쓰세요. 기관마다 무엇을 준비해야 하는지 구체적으로.')


def compare_options(client, options: dict) -> Tradeoff:
    return _parse(client, Tradeoff, SYS, '여러 안의 facts:\n' + json.dumps(options, ensure_ascii=False) +
                  '\n\n각 안이 무엇을 얻고 무엇을 잃는지 비교하고, 어떤 상황에 어느 안이 맞는지 정리하세요.')


def explain_counterfactual(client, f: dict) -> Tradeoff:
    return _parse(client, Tradeoff, SYS, '"이 도로를 빼고 다시 계산" 결과 facts:\n' + json.dumps(f, ensure_ascii=False) +
                  '\n\n최적화 엔진이 왜 그 도로를 골랐는지, 빼면 무엇이 나빠지는지 설명하세요.')


def _nums(text: str) -> set[str]:
    return {n.replace(',', '').lstrip('+-').rstrip('.') for n in re.findall(r'[-+]?\d[\d,]*\.?\d*', text)}


def check_numbers(obj, *sources) -> list[str]:
    """AI 응답 속 숫자 중 근거 자료(facts·사용자 입력) 어디에도 없는 것 (2자리 이상)"""
    known = set()
    for src in sources:
        known |= _nums(src if isinstance(src, str) else json.dumps(src, ensure_ascii=False, default=str))
    text = obj.model_dump_json() if hasattr(obj, 'model_dump_json') else json.dumps(obj, ensure_ascii=False)
    out = []
    for n in _nums(text):
        core = n.split('.')[0]
        if len(core) >= 2 and n not in known and core not in known:
            out.append(n)
    return sorted(set(out))
