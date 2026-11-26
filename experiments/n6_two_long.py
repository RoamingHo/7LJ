"""Exact exclusion of the three nonmatching-auxiliary two-long N6 classes."""
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import argparse
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.n6_reduction import B, CUT, TARGET, audit, graph_cover, require
from experiments.n6_single_long import scalar_check, spectral_matrix, positive_inertia

EXPECTED_MASKS = (513, 515, 521)


def check_rectangle_cover(boxes, root):
    total = F(0)
    for box in boxes:
        require(len(box) == len(root), 'wrong rectangle dimension')
        volume = F(1)
        for (lo, hi), (a, b) in zip(box, root):
            require(a <= lo < hi <= b, 'rectangle outside root')
            volume *= hi-lo
        total += volume
    for left, right in combinations(boxes, 2):
        require(any(a[1] <= b[0] or b[1] <= a[0] for a, b in zip(left, right)),
                'rectangle interiors overlap')
    root_volume = F(1)
    for lo, hi in root:
        root_volume *= hi-lo
    require(total == root_volume, 'rectangle root has an uncovered region')


def verify_leaf(record, mask):
    curvature, lam = F(record['curvature']), F(record['lambda'])
    intervals = [tuple(map(F, interval)) for interval in record['long_intervals']]
    require(len(intervals) == 2, 'two long intervals required')
    for lo, hi in intervals:
        require(B <= lo < hi <= 4*B, 'long interval outside matching root')
    long = dict(zip((0, 9), intervals))
    subdivisions = 0
    for k, (slope, offset) in enumerate(record['lines']):
        domain = long.get(k, (CUT, B) if mask >> k & 1 else (F(289, 400), CUT))
        subdivisions += scalar_check(curvature, lam, F(slope), F(offset), *domain)
    H, base, constant = spectral_matrix(record)
    upper = list(map(F, record['eigenvalue_upper_bounds']))
    require(len(upper) == 3 and all(x > 0 for x in upper), 'positive top-three bounds required')
    for k, endpoint in enumerate(upper):
        require(positive_inertia(H, endpoint) <= k, 'exact spectral upper bound failed')
    lower = curvature*(base-2*sum(x*x for x in upper))+constant
    require(lower > TARGET, 'two-long spectral bound does not exclude target')
    return lower, subdivisions


def verify(data, verify_reduction=True):
    require(data['schema'] == 'lj6.exact-two-long-nonmatching-spectral.v1' and
            data['target_exact'] == str(TARGET) and data['requires_stationarity'] is True and
            data['global_N6_optimality_proved'] is False, 'wrong two-long theorem scope')
    require(F(289, 400)**-6-2*F(289, 400)**-3-14 > TARGET,
            'collision cutoff is not justified')
    if verify_reduction:
        audit()
    cover = graph_cover(2)['representative_auxiliary_masks']
    require([mask for mask in cover if mask != 16897] == list(EXPECTED_MASKS),
            'expected two-long reduction changed')
    require(sorted(row['auxiliary_mask'] for row in data['graphs']) == list(EXPECTED_MASKS),
            'two-long auxiliary graph coverage differs')
    minimum = None
    leaves = subdivisions = 0
    for graph in data['graphs']:
        boxes = [[tuple(map(F, pair)) for pair in record['long_intervals']]
                 for record in graph['leaves']]
        check_rectangle_cover(boxes, [(B, 4*B)]*2)
        for record in graph['leaves']:
            lower, nodes = verify_leaf(record, graph['auxiliary_mask'])
            minimum = lower if minimum is None else min(minimum, lower)
            leaves += 1
            subdivisions += nodes
    return {'verified': True, 'graphs': 3, 'leaves': leaves,
            'minimum_energy_lower_exact': str(minimum), 'margin_display': float(minimum-TARGET),
            'scalar_subinterval_checks': subdivisions, 'requires_stationarity': True,
            'fresh_matching_reduction_checked': verify_reduction,
            'two_long_nonmatching_auxiliary_family_excluded': True,
            'remaining_auxiliary_mask': 16897, 'global_N6_optimality_proved': False}


def generate(numerical):
    import numpy as np
    graphs = []
    for graph in numerical['rows']:
        require(graph['matching_edges'] == 2 and not graph['pending'] and not graph['terminal'],
                'numerical two-long graph is incomplete')
        leaves = []
        for proposed in graph['numerical_leaves']:
            intervals = []
            for interval in proposed['intervals']:
                endpoints = []
                for endpoint in interval:
                    value = F(endpoint)
                    if abs(value-B) < F(1, 10**12):
                        value = B
                    if abs(value-4*B) < F(1, 10**12):
                        value = 4*B
                    endpoints.append(str(value))
                intervals.append(endpoints)
            row = {'curvature': str(F(str(proposed['curvature']))),
                   'lambda': str(F(str(proposed['lambda']))), 'long_intervals': intervals,
                   'lines': [[str(F(round(slope*10**9), 10**9)),
                              str(F(math.floor(offset*10**9)-10, 10**9))]
                             for slope, offset in proposed['lines']],
                   'dual': [str(F(round(x*10**9), 10**9)) for x in proposed['dual']]}
            H, _, _ = spectral_matrix(row)
            eigenvalues = np.linalg.eigvalsh(np.array(H, dtype=float))[-3:][::-1]
            row['eigenvalue_upper_bounds'] = [str(F(math.ceil(max(x, 0)*10**9)+10, 10**9))
                                                for x in eigenvalues]
            verify_leaf(row, graph['auxiliary_mask'])
            leaves.append(row)
        graphs.append({'auxiliary_mask': graph['auxiliary_mask'], 'leaves': leaves})
    return {'schema': 'lj6.exact-two-long-nonmatching-spectral.v1', 'target_exact': str(TARGET),
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
