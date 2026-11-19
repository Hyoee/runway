"""10K 코스(상암 → 여의도) 지역의 OpenStreetMap 도로를 data/osm_roads_west.json 형식으로 만든다.

기존 data/osm_roads.json은 경도 126.93 동쪽만 담고 있어 10K 코스(경도 126.89~126.94)를 만들 수 없다.
풀코스 결과가 바뀌지 않도록 기존 파일은 그대로 두고, 10K 도로망만 이 파일을 함께 쓴다.

    python scripts/fetch_osm_west.py            # Overpass API에서 내려받아 변환
    python scripts/fetch_osm_west.py raw.json   # 이미 받은 Overpass 응답을 변환

Overpass 쿼리 (bbox: 위도 37.50~37.60, 경도 126.85~126.975):
  [out:json];(way["highway"~"^(trunk|trunk_link|primary|primary_link|secondary|tertiary)$"](37.50,126.85,37.60,126.975););
  out body;>;out skel qt;
데이터: © OpenStreetMap contributors, ODbL
"""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUERY = ('[out:json][timeout:120];(way["highway"~"^(trunk|trunk_link|primary|primary_link|secondary|tertiary)$"]'
         '(37.50,126.85,37.60,126.975););out body;>;out skel qt;')


def main():
    if len(sys.argv) > 1:
        raw = json.load(open(sys.argv[1], encoding='utf-8'))
    else:
        req = urllib.request.Request('https://overpass-api.de/api/interpreter',
                                     data=urllib.parse.urlencode({'data': QUERY}).encode(), headers={'User-Agent': 'RunWay/1.0'})
        raw = json.load(urllib.request.urlopen(req, timeout=180))
    nodes, ways, tunnels = {}, [], []
    for el in raw['elements']:
        if el['type'] == 'node':
            nodes[str(el['id'])] = [round(el['lat'], 7), round(el['lon'], 7)]
        elif el['type'] == 'way':
            tg = el.get('tags', {})
            ways.append(dict(id=el['id'], hw=tg.get('highway', ''), name=tg.get('name', ''), lanes=tg.get('lanes', ''),
                             oneway=tg.get('oneway', ''), bridge=tg.get('bridge', ''), nodes=el['nodes']))
            if tg.get('tunnel') and tg.get('tunnel') != 'no':
                tunnels.append(str(el['id']))
    json.dump(dict(nodes=nodes, ways=ways), open(ROOT / 'data' / 'osm_roads_west.json', 'w', encoding='utf-8'), ensure_ascii=False)
    (ROOT / 'data' / 'tunnel_way_ids_west.txt').write_text(','.join(tunnels) or '0')
    print(f'도로 way {len(ways):,}개, 노드 {len(nodes):,}개, 터널 way {len(tunnels)}개 → data/osm_roads_west.json')


if __name__ == '__main__':
    main()
