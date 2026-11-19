"""RunWay — 마라톤 코스 최적화 엔진 (PuLP MILP).

    from runway import load_network, Scenario, DEFAULT, optimize, evaluate, build_asis
    net = load_network('data')
    sc = Scenario(net, DEFAULT.with_(T0=7.0))
    order, info, corridor = optimize(sc)
    ev = evaluate(sc, order)

LLM은 Params 값만 바꾸고, 숫자는 모두 이 패키지가 계산한다.
"""
from .asis import ASIS_WP, build_asis
from .evaluate import compare, evaluate, kpis, road_table, to_geojson_feature, validate_course
from .network import Network, build_network, load_network
from .params import BBOX, DEFAULT, Forbid, Landmark, Params, in_bbox
from .scenario import Scenario, ScenarioError
from .solve import optimize, solve_milp

__all__ = ['ASIS_WP', 'BBOX', 'DEFAULT', 'Forbid', 'Landmark', 'Network', 'Params', 'Scenario', 'ScenarioError',
           'build_asis', 'build_network', 'compare', 'evaluate', 'in_bbox', 'kpis', 'load_network', 'optimize',
           'road_table', 'solve_milp', 'to_geojson_feature', 'validate_course']
