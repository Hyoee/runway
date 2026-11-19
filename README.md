# RunWay — 서울 도심 마라톤 코스 최적화 (PuLP MILP + LLM 챗봇)

> **버스 승객 피해는 최소로, 코스는 단순하게.**
> 대회 기획자가 말로 조건을 바꾸면, 수리최적화(MILP) 솔버가 서울시 버스 승하차 공공데이터로
> 시민 피해가 가장 적은 42.195km 코스를 계산하고, LLM이 결과를 설명한다. **LLM은 의미를, 솔버는 숫자를.**

## 결과 요약 — 2026 JTBC 서울마라톤 코스(AS-IS) vs RunWay(TO-BE)

| 지표 | AS-IS | TO-BE | 변화 |
|---|---:|---:|---:|
| 통제 시간대 승객 피해 (명/주말 하루) | 32,110 | 10,815 | −66% |
| 영향 받는 버스 노선 | 156 | 84 | −46% |
| 방향 전환 45° 이상 | 21 | 15 | −29% |
| 급한 방향 전환 100° 이상 | 6 | 1 | −83% |
| 왕복 회귀 구간 (km) | 5.9 | 0.0 | −100% |
| 롯데월드타워 조망, 30km 이후 (km) | 1.2 | 2.5 | +108% |

AS-IS의 문제점(모델 검사): 같은 도로 왕복 사용, 공인 거리 범위 초과(모델상 43.2km), 건국대병원 앞(응급 동선) 통과.
TO-BE는 출발(광화문)·도착(올림픽공원) 고정, 42.195~42.6km, 교차로 1회 통과, 금지 구간 회피, 30km 이후 롯데월드타워 조망 ≥ 2km를 모두 지킨다.

> 솔버는 '최적해와 1% 이내'면 멈추므로 다시 풀면 비슷한 품질의 다른 코스가 나올 수 있다(재실행 시 −60% 수준).
> 그래서 찾은 해는 저장해 두고, 같은 조건에서 목적함수가 더 작은 해가 나올 때만 교체한다.

## 구조

```
runway/            최적화 엔진 (LLM 없이 동작, 테스트됨)
  params.py        제약 카드 = 검증 가능한 스키마 (LLM이 채우는 값)
  network.py       도로 그래프·정류장 연결·형상 특징 (캐시)
  scenario.py      조건별 값: 통제 시간 피해표, 조망, 금지 구간, 출발·도착 검증
  model.py         MILP 수식 (목적함수 5-4, 제약식 5-5)
  solve.py         LP 완화 → 후보 회랑 → MILP
  evaluate.py      KPI·제약 검사·목적함수 재계산
  engine.py        서비스 진입점: 결과 캐시, 더 좋은 해만 교체
  places.py        장소 이름 → 좌표 (버스정류소 위치 데이터에서만 — LLM이 좌표를 지어내지 않음)
agent/             Claude 도구 계층 (Tool Runner) + 숫자 검증
app.py             Streamlit: 조건 카드 · 지도 · KPI · 채팅
scenarios/         미리 계산한 결과 (즉시 응답용)
scripts/precompute.py
tests/             노트북 결과와 같은 숫자를 내는지 회귀 테스트
RunWay_코스최적화_StepByStep_PuLP.ipynb   수식·과정 설명 노트북
```

## 로컬 실행

```bash
pip install -r requirements.txt
pytest -q                                   # 회귀 테스트 (~40초)
streamlit run app.py                        # http://localhost:8501
```
채팅을 쓰려면 Claude API 키를 환경변수로: PowerShell `$env:ANTHROPIC_API_KEY="..."` (키는 커밋하지 않는다).
키가 없어도 사이드바 조건 카드로 모든 계산 기능을 쓸 수 있다.

## 배포 (Streamlit Community Cloud)

1. https://share.streamlit.io → GitHub 로그인 → **Create app**
2. Repository `Hyoee/runway`, Branch `main`, Main file `app.py`
3. **Advanced settings** → Python **3.11** (테스트한 버전), **Secrets**에:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
4. Deploy → 첫 실행 때 도로망을 만드는 데 20~30초.
5. Anthropic 콘솔에서 **월 사용 한도(spend limit)** 를 걸어 둔다 — 공개 URL은 누구나 채팅할 수 있다.

미리 계산된 시나리오는 즉시 뜨고, 새 조건은 서버에서 최대 2분(시간 제한 120초, gap 2%) 풀이한다.

## 데이터 출처

- 서울 열린데이터광장: 버스노선별 정류장별 시간대별 승하차(OA-12913), 일별 승하차(OA-12912), 버스정류소 위치정보(OA-15067) — 2026년 10월
- 도로망: © OpenStreetMap contributors (ODbL)

## 한계

- 수요는 정류장별 주말 하루 평균 × 월 전체 시간대 분포로 추정 (주말 시간대 원자료 없음)
- 코스 위에 정류장이 있는 노선만 '영향 노선'으로 셈 (코스를 가로지르는 노선의 우회는 미반영)
- AS-IS는 공식 경유지를 간선 도로망 위에 옮긴 것 — 주최 측 GPX가 있으면 교체 가능
- 최적성은 'LP 완화가 고른 후보 회랑 안에서 허용 gap 이내'
