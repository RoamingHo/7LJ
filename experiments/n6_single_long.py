"""Generate/replay an exact spectral exclusion of all 18 single-long N6 classes.

Generation uses NumPy only to propose rational eigenvalue upper bounds.
Verification uses exact Fraction arithmetic and Sylvester inertia, never NumPy.
"""
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
import argparse
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.n6_reduction import (B, CUT, PAIRS, TARGET, audit,
                                                     bernstein_coefficients, graph_cover, require)


def polynomial_positive(poly, lo, hi, depth=0):
    if min(bernstein_coefficients(poly, lo, hi)) >= 0:
        return 1
    require(depth < 24, 'scalar support positivity not established')
    mid = (lo+hi)/2
    return polynomial_positive(poly, lo, mid, depth+1)+polynomial_positive(poly, mid, hi, depth+1)


@lru_cache(maxsize=None)
def scalar_check(curvature, lam, slope, offset, lo, hi):
    require(curvature > 0 and lam > -1 and 0 < lo < hi, 'invalid scalar parameters')
    poly = (1+lam, F(0), F(0), -(2+lam), F(0), F(0), -offset, -slope, -curvature)
    return polynomial_positive(poly, lo, hi)


def spectral_matrix(record):
    curvature = F(record['curvature'])
    lines = [tuple(map(F, line)) for line in record['lines']]
    dual = list(map(F, record['dual']))
    require(len(lines) == 15 and len(dual) == 7, 'wrong N6 spectral dimension')
    v, t = dual[:6], dual[6]
    M = [[F(0) for _ in range(6)] for _ in range(6)]
    for (i, j), (slope, _) in zip(PAIRS, lines):
        M[i][j] = M[j][i] = -slope/(2*curvature)
    row = [sum(r)/6 for r in M]
    mean = sum(row)/6
    G0 = [[-(M[i][j]-row[i]-row[j]+mean)/2 for j in range(6)] for i in range(6)]
    C = [[6*v[i]*(i == j)-v[i]-v[j]+sum(v)/6+t*(F(i == j)-F(1, 6))
          for j in range(6)] for i in range(6)]
    H = [[G0[i][j]-C[i][j]/2 for j in range(6)] for i in range(6)]
    base = (2*sum(x*x for row in G0 for x in row)-
            2*sum(C[i][j]*G0[i][j] for i in range(6) for j in range(6))-
            6*sum(x*x for x in v)-t*t)
    constant = sum(offset for _, offset in lines)-curvature*sum(M[i][j]**2 for i, j in PAIRS)
    return H, base, constant


def positive_inertia(matrix, upper):
    work = [[x-(upper if i == j else 0) for j, x in enumerate(row)]
            for i, row in enumerate(matrix)]
    count = 0
    for k in range(6):
        pivot = work[k][k]
        require(pivot != 0, 'zero exact inertia pivot')
        count += pivot > 0
        for i in range(k+1, 6):
            for j in range(i, 6):
                work[i][j] -= work[i][k]*work[k][j]/pivot
                work[j][i] = work[i][j]
    return count


def verify_leaf(record, auxiliary_mask):
    curvature, lam = F(record['curvature']), F(record['lambda'])
    lo, hi = map(F, record['long_interval'])
    require(B <= lo < hi <= 4*B, 'long interval outside proved matching root')
    subdivisions = 0
    for k, (slope, offset) in enumerate(record['lines']):
        domain = ((lo, hi) if k == 0 else
                  ((CUT, B) if auxiliary_mask >> k & 1 else (F(289, 400), CUT)))
        subdivisions += scalar_check(curvature, lam, F(slope), F(offset), *domain)
    H, base, constant = spectral_matrix(record)
    upper = list(map(F, record['eigenvalue_upper_bounds']))
    require(len(upper) == 3 and all(x > 0 for x in upper), 'positive top-three bounds required')
    for k, endpoint in enumerate(upper):
        require(positive_inertia(H, endpoint) <= k, 'exact spectral upper bound failed')
    lower = curvature*(base-2*sum(x*x for x in upper))+constant
    require(lower > TARGET, 'single-long spectral bound does not exclude target')
    return lower, subdivisions


def verify(data, verify_reduction=True):
    require(data['schema'] == 'lj6.exact-single-long-spectral.v1' and
            data['target_exact'] == str(TARGET) and data['requires_stationarity'] is True and
            data['global_N6_optimality_proved'] is False, 'wrong spectral certificate scope')
    # The pair bound gives r>.85 on E<=TARGET, while a shared third vertex gives
    # r_long<2R when the true-long graph is a matching.
    collision_lower = F(289, 400)**-6-2*F(289, 400)**-3-14
    require(collision_lower > TARGET, 'collision cutoff is not justified')
    if verify_reduction:
        audit()
    required = graph_cover(1)['representative_auxiliary_masks']
    require(sorted(row['auxiliary_mask'] for row in data['graphs']) == required,
            'single-long auxiliary graph coverage differs')
    lower = None
    leaves = subdivisions = 0
    for graph in data['graphs']:
        records = sorted(graph['leaves'], key=lambda row: F(row['long_interval'][0]))
        endpoint = B
        for record in records:
            lo, hi = map(F, record['long_interval'])
            require(lo == endpoint, 'single-long interval cover has a gap or overlap')
            endpoint = hi
            bound, nodes = verify_leaf(record, graph['auxiliary_mask'])
            lower = bound if lower is None else min(lower, bound)
            leaves += 1
            subdivisions += nodes
        require(endpoint == 4*B, 'single-long root interval is not covered')
    return {'verified': True, 'graphs': len(data['graphs']), 'leaves': leaves,
            'minimum_energy_lower_exact': str(lower), 'margin_display': float(lower-TARGET),
            'scalar_subinterval_checks': subdivisions, 'requires_stationarity': True,
            'fresh_matching_reduction_checked': verify_reduction,
            'single_long_family_excluded': True, 'global_N6_optimality_proved': False}


def generate(numerical):
    import numpy as np
    graphs = []
    for graph in numerical['rows']:
        require(graph['matching_edges'] == 1 and not graph['pending'] and not graph['terminal'],
                'numerical graph is incomplete')
        leaves = []
        for proposed in graph['numerical_leaves']:
            endpoints = []
            for endpoint in proposed['intervals'][0]:
                value = F(endpoint)
                if abs(value-B) < F(1, 10**12):
                    value = B
                if abs(value-4*B) < F(1, 10**12):
                    value = 4*B
                endpoints.append(str(value))
            row = {'curvature': str(F(str(proposed['curvature']))),
                   'lambda': str(F(str(proposed['lambda']))), 'long_interval': endpoints,
                   'lines': [[str(F(round(slope*10**8), 10**8)),
                              str(F(math.floor(offset*10**8)-50, 10**8))]
                             for slope, offset in proposed['lines']],
                   'dual': [str(F(round(x*10**8), 10**8)) for x in proposed['dual']]}
            H, _, _ = spectral_matrix(row)
            eigenvalues = np.linalg.eigvalsh(np.array(H, dtype=float))[-3:][::-1]
            row['eigenvalue_upper_bounds'] = [str(F(math.ceil(max(x, 0)*10**8)+100, 10**8))
                                                for x in eigenvalues]
            verify_leaf(row, graph['auxiliary_mask'])
            leaves.append(row)
        graphs.append({'auxiliary_mask': graph['auxiliary_mask'], 'leaves': leaves})
    return {'schema': 'lj6.exact-single-long-spectral.v1', 'target_exact': str(TARGET),
            'requires_stationarity': True, 'global_N6_optimality_proved': False, 'graphs': graphs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('certificate', type=Path)
    parser.add_argument('--generate-from', type=Path)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if args.generate_from:
        require(not args.certificate.exists(), 'refusing to overwrite certificate')
        data = generate(json.loads(args.generate_from.read_text()))
        args.certificate.write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')
    raw = args.certificate.read_bytes()
    result = verify(json.loads(raw))
    if args.receipt:
        require(not args.receipt.exists(), 'refusing to overwrite receipt')
        args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
