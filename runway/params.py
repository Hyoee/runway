"""제약 카드 — 노트북 Step 0의 `P`를 검증 가능한 스키마로 옮긴 것.

LLM 에이전트는 이 스키마의 값만 바꾼다. 필드 이름은 노트북과 같게 두어 수식·문서와 바로 대응된다.
`Params`는 불변(frozen)이라 여러 사용자가 동시에 써도 서로의 조건이 섞이지 않는다.
바꿀 때는 `p.model_copy(update={...})` 또는 `p.with_(...)`.
"""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

LatLon = tuple[float, float]

# 도로 그래프(OSM)와 정류장 데이터가 덮는 영역 — 출발·도착·랜드마크는 이 안에 있어야 한다
BBOX = dict(lat_min=37.470, lat_max=37.600, lon_min=126.955, lon_max=127.150)


def in_bbox(ll: LatLon) -> bool:
    return BBOX['lat_min'] <= ll[0] <= BBOX['lat_max'] and BBOX['lon_min'] <= ll[1] <= BBOX['lon_max']


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')


class Landmark(_Frozen):
    """대회 정체성 (소개서 스키마 {L_k, a_k, b_k, w_k})"""
    name: str
    type: Literal['view'] = 'view'
    coord: LatLon
    angle: float = Field(30.0, gt=0, le=90, description='정면 ±각도(°) 안에 보이면 조망')
    dist: float = Field(4000.0, gt=0, description='조망 최대 거리(m)')
    L: float = Field(2000.0, ge=0, description='조망 길이 하한(m)')
    a: float = Field(30000.0, ge=0, description='적용 구간 시작(m, 출발점 기준 누적거리)')
    b: float = Field(42600.0, ge=0, description='적용 구간 끝(m)')
    w: float = Field(0.0, ge=0, description='조망 1m당 보상 가중치(0이면 하한만)')


class Forbid(_Frozen):
    """금지 구간: 반경 r(m) 안의 도로 진입 금지"""
    name: str
    coord: LatLon
    r: float = Field(150.0, gt=0, le=2000)


class Params(_Frozen):
    # 일정·통제 시간
    T0: float = Field(8.0, ge=5.0, le=12.0, description='출발 시각(시)')
    V_LEAD: float = 20.0
    V_SWEEP: float = 42.195 / 5.0
    PRE: float = 0.25
    POST: float = 0.5
    STAGE: tuple[float, float, float] = (600.0, 4.0, 10.0)
    # 공인 코스 조건
    LMIN: float = 42195.0
    LMAX: float = 42600.0
    START: LatLon = (37.5716, 126.9768)          # 광화문광장
    FINISH: LatLon = (37.5207, 127.1214)         # 올림픽공원
    START_NAME: str = '광화문광장'
    FINISH_NAME: str = '올림픽공원'
    # 대회 정체성(랜드마크 조망 하한) — 서비스 기본은 없음. 노트북 v1은 롯데월드타워(LOTTE)
    IDENTITY: tuple[Landmark, ...] = ()
    FORBID: tuple[Forbid, ...] = (
        Forbid(name='건국대병원 앞(응급 동선)', coord=(37.5404, 127.0718), r=150),
    )
    MIN_LANES: float = 1.0
    # 목적식 가중치
    ALPHA: float = Field(0.0, ge=0, description='영향 노선 1개당 고정 벌점 (노트북 v1은 60). 우회 영향은 DETOUR_W로 승객 단위 계산')
    DETOUR_W: float = Field(0.0, ge=0, le=1, description='우회 노선 이용객 1명의 가중치 (못 타는 승객 1명 = 1). 기본 0 = 못 타는 승객만. '
                                                         '통제 시간 동안 우회하는 노선의 전체 이용객 중 지연을 겪는 비율로 해석')
    TURN: tuple[float, float, float] = (1000.0, 2000.0, 5000.0)
    PEN_TWIN: float = 3.0
    PEN_NEAR: float = 2.0
    NEAR_R: tuple[float, float] = (300.0, 800.0)
    EPS_LEN: float = 0.02
    # 계산 설정
    STEP: float = 10.0
    JUNCTION_R: float = 60.0
    BAND_KM: float = 3.0
    LP_EPS: float = 1e-3
    TIME_LIMIT: float = 600
    GAP: float = 0.01
    DAYTYPE: Literal['weekend', 'sunday', 'all'] = Field(
        'sunday', description="수요 기준일: 'sunday'(대회일=일요일 평균, 기본) | 'weekend'(토·일, 노트북 v1) | 'all'(전체 일평균)")

    # ── 도우미 ──
    def with_(self, **kw) -> 'Params':
        """일부 값만 바꾼 새 Params (검증 포함)"""
        return Params.model_validate({**self.model_dump(), **kw})

    def key(self, *fields: str) -> str:
        """캐시 키: 지정 필드(없으면 전체)의 해시"""
        d = self.model_dump(mode='json')
        if fields:
            d = {f: d[f] for f in fields}
        return hashlib.sha1(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


# 도로 그래프에 영향을 주는 값 — 바뀌면 네트워크를 다시 만든다(드묾)
NETWORK_FIELDS = ('DAYTYPE', 'JUNCTION_R', 'NEAR_R')
# 피해 행렬 RID에 영향을 주는 값
SCHEDULE_FIELDS = ('T0', 'V_LEAD', 'V_SWEEP', 'PRE', 'POST', 'STAGE', 'LMAX', 'STEP')

DEFAULT = Params()
# 노트북(v1)의 목적식: 노선 1개 = 60점, 우회 승객 미반영 — 회귀 테스트와 비교용
LOTTE = Landmark(name='롯데월드타워', coord=(37.5126, 127.1025))       # 30km 이후 조망 ≥ 2km (노트북 v1의 정체성 조건)
NOTEBOOK = Params(ALPHA=60.0, DETOUR_W=0.0, DAYTYPE='weekend', IDENTITY=(LOTTE,))
MODEL_VERSION = 'v3'    # 목적식·데이터가 바뀌면 올린다 → 저장된 결과 캐시 키가 달라져 섞이지 않음
