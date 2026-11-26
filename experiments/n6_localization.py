"""Exact Gram-radius to N6 octahedral local-uniqueness endpoint.

This proves a conditional endpoint only; global graph/cell coverage is separate.
"""
from fractions import Fraction as F
from math import isqrt
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.certificate import (Interval, evaluate, interval_matmul,
                                           transpose, active_coordinates)
from experiments.n6_reduction import require
from experiments.n6_gap import QS_LO, QS_HI

SCALE = 1 << 160
TEMPLATE = (2, 1, 1, 1, 0, 1, 1, -1, 0, 1, 0, -1)


def sqrt_interval(value):
    require(value.lo > 0, 'strictly positive squared reconstruction denominator required')
    lo = isqrt(value.lo.numerator*SCALE*SCALE//value.lo.denominator)
    hi = isqrt(value.hi.numerator*SCALE*SCALE//value.hi.denominator)
    if F(hi*hi, SCALE*SCALE) < value.hi:
        hi += 1
    return Interval(F(lo, SCALE), F(hi, SCALE))


def recover(eta):
    """Relabel old (0,1,2,3,4,5) to (0,1,2,4,3,5), then align/refelect."""
    require(eta > 0, 'positive Gram radius required')
    q = [[Interval(0) for _ in range(6)] for _ in range(6)]
    opposite = {(0, 1), (2, 4), (3, 5)}
    for i in range(6):
        for j in range(i+1, 6):
            multiple = 2 if (i, j) in opposite else 1
            q[i][j] = q[j][i] = Interval(multiple*QS_LO-2*eta, multiple*QS_HI+2*eta)
    s = lambda i, j: (q[0][i]+q[0][j]-q[i][j])*F(1, 2)
    d = sqrt_interval(q[0][1])
    x = {i: s(1, i)*d**-1 for i in range(2, 6)}
    y2 = sqrt_interval(q[0][2]-x[2]**2)
    y = {2: y2, **{i: (s(2, i)-x[2]*x[i])*y2**-1 for i in range(3, 6)}}
    z3 = sqrt_interval(q[0][3]-x[3]**2-y[3]**2)
    z = {3: z3, **{i: (s(3, i)-x[3]*x[i]-y[3]*y[i])*z3**-1 for i in range(4, 6)}}
    points = [[Interval(0) for _ in range(3)] for _ in range(6)]
    points[1][0] = d
    for i in range(2, 6):
        points[i][0] = x[i]
        points[i][1] = y[i]
        if i >= 3:
            points[i][2] = z[i]
    box = [points[i][axis] for i, axis in active_coordinates(6)]
    return box, {'d': str(d.lo), 'y2': str(y2.lo), 'z3': str(z3.lo)}


def record_fraction(record):
    return F(record['fraction'])


def record_interval(record):
    return Interval(record_fraction(record['lower']), record_fraction(record['upper']))


def verify(eta=F(1, 100000)):
    raw = (ROOT/'results/n6_candidate_boxes.json').read_bytes()
    artifact = json.loads(raw)
    require(artifact['particles'] == 6 and artifact['dimension'] == 12,
            'local candidate artifact has wrong N6 dimensions')
    matches = [box for box in artifact['boxes'] if tuple(box['unit_circumradius_template']) == TEMPLATE]
    require(len(matches) == 1, 'octahedral coordinate template missing or duplicated')
    saved = matches[0]['local_certificate']
    require(saved['particles'] == 6 and saved['dimension'] == 12 and
            len(saved['center']) == len(saved['box']) == len(TEMPLATE) == 12,
            'local candidate center/box has wrong dimensions')
    center = list(map(record_fraction, saved['center']))
    local_box = list(map(record_interval, saved['box']))
    P = [list(map(F, row)) for row in saved['rational_preconditioner_P']]
    require(len(P) == 12 and all(len(row) == 12 for row in P) and
            all(P[i][i] > 0 and all(P[i][j] == 0 for j in range(i)) for i in range(12)),
            'local preconditioner is not nonsingular upper triangular')
    require(QS_LO**3 < F(257, 264) < QS_HI**3, 'candidate scale root is not enclosed')
    a = sqrt_interval(Interval(QS_LO/2, QS_HI/2))
    candidate = [a*entry for entry in TEMPLATE]
    require(all(B.lo < C.lo <= C.hi < B.hi for B, C in zip(local_box, candidate)),
            'exact candidate not strictly inside local box')
    require((F(4)+F(2, 128))/(F(4)+F(2, 16)) == F(257, 264),
            'full octahedral Cartesian-force identity failed')
    _, _, Hlocal, _ = evaluate(local_box)
    transformed = interval_matmul(transpose(P), interval_matmul(Hlocal, P))
    margin = min(transformed[i][i].lo-sum(transformed[i][j].abs_upper()
                                           for j in range(12) if i != j) for i in range(12))
    require(margin > 0, 'fresh local Hessian positive definiteness failed')
    C = [[sum(P[i][k]*P[j][k] for k in range(12)) for j in range(12)] for i in range(12)]
    _, gradient, _, _ = evaluate([Interval(x) for x in center])
    correction = [row[0] for row in interval_matmul(C, [[entry] for entry in gradient])]
    box, positive = recover(eta)
    widths = [max(entry.hi-entry.lo for entry in box)]
    iterations = 0
    while not all(A.lo <= B.lo <= B.hi <= A.hi for A, B in zip(local_box, box)):
        require(iterations < 8, 'Gram enclosure has not contracted into local uniqueness box')
        require(all(B.lo <= value <= B.hi for B, value in zip(box, center)),
                'Newton expansion center is outside current box')
        _, _, H, _ = evaluate(box)
        CH = interval_matmul(C, H)
        remainder = [[int(i == j)-CH[i][j] for j in range(12)] for i in range(12)]
        image = interval_matmul(remainder, [[entry-value] for entry, value in zip(box, center)])
        new_box = []
        for old, value, shift, row in zip(box, center, correction, image):
            image_entry = value-shift+row[0]
            lower, upper = max(old.lo, image_entry.lo), min(old.hi, image_entry.hi)
            require(lower <= upper, 'candidate-containing Newton box unexpectedly empty')
            new_box.append(Interval(lower, upper))
        box = new_box
        require(all(B.lo <= A.lo <= A.hi <= B.hi for A, B in zip(candidate, box)),
                'exact candidate left a Newton enclosure')
        iterations += 1
        widths.append(max(entry.hi-entry.lo for entry in box))
    return {'schema': 'lj6.exact-gram-localization-endpoint.v1', 'verified': True,
            'assumed_gram_frobenius_radius': str(eta), 'candidate_energy_exact': '-3267/257',
            'unit_circumradius_template': list(TEMPLATE), 'positive_denominator_lower_bounds': positive,
            'fresh_local_hessian_gershgorin_margin': str(margin),
            'newton_contractions': iterations, 'maximum_box_widths': list(map(str, widths)),
            'maximum_box_widths_display': list(map(float, widths)),
            'localization_implies_unique_stationary_octahedron': True,
            'requires_stationarity_for_final_candidate_identification': True,
            'global_N6_optimality_proved': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eta', type=F, default=F(1, 100000))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(args.eta)
    if args.output:
        require(not args.output.exists(), 'refusing to overwrite endpoint receipt')
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
