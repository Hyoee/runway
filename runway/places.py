"""장소 이름 → 좌표. LLM이 좌표를 지어내지 않도록, 서울시 버스정류소 위치 데이터(OA-15067)에서만 찾는다.

같은 이름의 정류소(길 양쪽)는 평균 좌표 하나로 묶고, 일치 정도 순으로 후보를 돌려준다.
후보가 여러 곳이면 에이전트가 사용자에게 고르게 한다.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .geo import hav
from .params import in_bbox

_SUFFIX = re.compile(r'\(.*?\)|\.|\s')


def _norm(s: str) -> str:
    return _SUFFIX.sub('', str(s))


@lru_cache(maxsize=4)
def _table(data_dir: str) -> pd.DataFrame:
    xy = pd.read_excel(Path(data_dir) / 'bus_stop_xy_20260902.xlsx')
    xy = xy[xy['정류소타입'] != '한강선착장'].copy()
    xy['key'] = xy['정류소명'].map(_norm)
    g = xy.groupby('정류소명').agg(lat=('Y좌표', 'mean'), lon=('X좌표', 'mean'), n=('ARS_ID', 'size'), key=('key', 'first')).reset_index()
    return g


def search_places(data_dir, query: str, limit: int = 8, area=None) -> list[dict]:
    """후보 목록 [{name, coord, in_area, match}] — match: exact > prefix > contains"""
    q = _norm(query)
    if not q:
        return []
    g = _table(str(data_dir))
    rank = {'exact': 0, 'prefix': 1, 'contains': 2}
    hits = []
    for r in g.itertuples():
        m = 'exact' if r.key == q else 'prefix' if r.key.startswith(q) else 'contains' if q in r.key else None
        if m:
            hits.append(dict(name=r.정류소명, coord=(round(r.lat, 6), round(r.lon, 6)), in_area=in_bbox((r.lat, r.lon), area), match=m))
    hits.sort(key=lambda h: (rank[h['match']], not h['in_area'], len(h['name'])))
    # 서로 300m 안의 후보는 하나로 (같은 장소의 여러 정류소 이름)
    out = []
    for h in hits:
        if all(hav(h['coord'], o['coord']) > 300 for o in out):
            out.append(h)
        if len(out) >= limit:
            break
    return out
