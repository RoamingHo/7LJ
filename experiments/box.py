#!/usr/bin/env python3
"""Enumerate every canonical copy of the exact LJ7 candidate and certify boxes.

All permutation/reflection decisions use rational interval separation, retaining
ties.  The exported dyadic endpoints are exactly representable in binary64.
Run with the existing Python environment containing SymPy.
"""

from fractions import Fraction as F
from itertools import permutations
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.tools import (
    add, sub, neg, mul, scale, point, divide, positive_power, positive_root,
    root_count, sturm_chain, polynomial_value, record, interval_record, require,
)
from experiments.certificate import active_coordinates, certify

SCALE_BITS = 50
DENOMINATOR = 1 << SCALE_BITS
RADIUS_NUMERATOR = 1 << (SCALE_BITS-17)
RADIUS = F(RADIUS_NUMERATOR, DENOMINATOR)
EXPECTED_SHAPE_POLYNOMIAL = [-150, -550, -900, 7576, 3157, 11571, 19985,
                             -635225, 1118016, 1109952, 369216]


def read_fraction(value):
    return F(int(value['numerator']), int(value['denominator']))


def read_interval(value):
    return read_fraction(value['lower']), read_fraction(value['upper'])


def square(a):
    candidates = [a[0]*a[0], a[1]*a[1]]
    if a[0] <= 0 <= a[1]:
        candidates.append(F(0))
    return min(candidates), max(candidates)


def dot(a, b):
    result = point(0)
    for x, y in zip(a, b):
        result = add(result, mul(x, y))
    return result


def cross(a, b):
    return [sub(mul(a[1], b[2]), mul(a[2], b[1])),
            sub(mul(a[2], b[0]), mul(a[0], b[2])),
            sub(mul(a[0], b[1]), mul(a[1], b[0]))]


def exact_candidate_enclosures():
    source = json.loads((ROOT/'results/exact_candidate.json').read_text())
    require([int(x) for x in source['shape_polynomial_coefficients_ascending']] ==
            EXPECTED_SHAPE_POLYNOMIAL, "Unexpected exact-candidate shape polynomial")
    t = read_interval(source['shape_root_interval'])
    chain = sturm_chain([F(x) for x in EXPECTED_SHAPE_POLYNOMIAL])
    require(0 < t[0] < t[1] and root_count(chain, F(0), '+infinity') == 1 and
            root_count(chain, *t) == 1, "Shape root isolation did not verify")
    require(polynomial_value(EXPECTED_SHAPE_POLYNOMIAL, t[0]) < 0 <
            polynomial_value(EXPECTED_SHAPE_POLYNOMIAL, t[1]), "Shape bracket signs failed")
    one_plus_t = add(point(1), t)
    A = add(add(point(F(18, 25)), scale(positive_power(one_plus_t, -6), 10)),
            scale(positive_power(t, -6), F(1, 4096)))
    B = add(add(point(2), scale(positive_power(one_plus_t, -3), 10)),
            scale(positive_power(t, -3), F(1, 64)))
    energy = neg(divide(positive_power(B, 2), A))
    require(energy == read_interval(source['energy_root_interval']), "Exact energy enclosure changed")
    R = positive_root(divide(A, B), 6)
    H = mul(R, positive_root(t, 2))
    s = positive_root(point(5), 2)
    u = positive_root(scale(add(point(5), s), F(1, 8)), 2)
    v = positive_root(scale(sub(point(5), s), F(1, 8)), 2)
    ru = mul(R, u)
    ring_y = neg(scale(mul(R, s), F(1, 2)))
    pole_y = neg(scale(mul(R, sub(s, point(1))), F(1, 4)))
    points = [[point(0), point(0), point(0)], [scale(ru, 2), point(0), point(0)],
              [ru, scale(mul(R, sub(point(5), s)), F(1, 4)), point(0)],
              [mul(R, add(u, v)), ring_y, point(0)],
              [mul(R, sub(u, v)), ring_y, point(0)],
              [ru, pole_y, neg(H)], [ru, pole_y, H]]
    return points, energy


def pair_distances(points):
    distances = {}
    for i in range(7):
        distances[i, i] = point(0)
        for j in range(i):
            q = point(0)
            for a, b in zip(points[i], points[j]):
                q = add(q, square(sub(a, b)))
            require(q[0] > 0, "Candidate pair collision not excluded")
            distances[i, j] = distances[j, i] = q
    return distances


def map_gauge(points, distances, anchors):
    i0, i1, i2 = anchors
    d = positive_root(distances[i0, i1], 2)
    x = {i: divide(sub(add(distances[i0, i], distances[i0, i1]), distances[i1, i]), scale(d, 2))
         for i in range(7)}
    y2_squared = sub(distances[i0, i2], square(x[i2]))
    require(y2_squared[0] > 0, "Cannot prove this candidate gauge is nonsingular; do not discard it")
    y2 = positive_root(y2_squared, 2)
    v1 = [sub(a, b) for a, b in zip(points[i1], points[i0])]
    v2 = [sub(a, b) for a, b in zip(points[i2], points[i0])]
    normal = cross(v1, v2)
    result = {}
    for i in range(7):
        y = divide(sub(sub(add(distances[i0, i], distances[i0, i2]), distances[i2, i]),
                       scale(mul(x[i], x[i2]), 2)), scale(y2, 2))
        vi = [sub(a, b) for a, b in zip(points[i], points[i0])]
        z = divide(dot(normal, vi), mul(d, y2))
        result[i] = [x[i], y, z]
    # These coordinates hold exactly by the definition of the gauge.
    result[i0] = [point(0), point(0), point(0)]
    result[i1] = [d, point(0), point(0)]
    result[i2] = [x[i2], y2, point(0)]
    return result


def nearest_grid_integer(value):
    shifted = value*DENOMINATOR+F(1, 2)
    return shifted.numerator//shifted.denominator


def stable_center(coordinates):
    key = []
    for lower, upper in coordinates:
        lo = nearest_grid_integer(lower)
        hi = nearest_grid_integer(upper)
        require(lo == hi, "Candidate interval crosses a quantization boundary; refine instead of guessing")
        key.append(lo)
    return tuple(key)


def enumerate_canonical(points):
    distances = pair_distances(points)
    gauges, groups = {}, {}
    counts = {"total_permutations": 0, "total_orientations": 0,
              "provably_non_diameter_orientations": 0,
              "provably_unsorted_x_orientations": 0,
              "provably_negative_z3_orientations": 0,
              "retained_orientations": 0, "retained_z3_zero_or_undecided": 0}
    active = [pair for pair, _ in sorted(active_coordinates(7).items(), key=lambda item: item[1])]
    for permutation in permutations(range(7)):
        counts['total_permutations'] += 1
        counts['total_orientations'] += 2
        chosen = distances[permutation[0], permutation[1]]
        if any(q[0] > chosen[1] for q in distances.values()):
            counts['provably_non_diameter_orientations'] += 2
            continue
        anchors = permutation[:3]
        if anchors not in gauges:
            gauges[anchors] = map_gauge(points, distances, anchors)
        mapped = gauges[anchors]
        if any(mapped[permutation[i]][0][0] > mapped[permutation[i+1]][0][1]
               for i in range(2, 6)):
            counts['provably_unsorted_x_orientations'] += 2
            continue
        for reflection in (1, -1):
            z3 = scale(mapped[permutation[3]][2], reflection)
            if z3[1] < 0:
                counts['provably_negative_z3_orientations'] += 1
                continue
            coordinates = [scale(mapped[permutation[atom]][axis], reflection if axis == 2 else 1)
                           for atom, axis in active]
            key = stable_center(coordinates)
            representation = {"permutation": list(permutation), "z_reflection": reflection}
            if key not in groups:
                groups[key] = {"enclosure": coordinates, "representations": []}
            else:
                groups[key]['enclosure'] = [(min(a[0], b[0]), max(a[1], b[1]))
                                            for a, b in zip(groups[key]['enclosure'], coordinates)]
            groups[key]['representations'].append(representation)
            counts['retained_orientations'] += 1
            if z3[0] <= 0:
                counts['retained_z3_zero_or_undecided'] += 1
    require(counts['total_permutations'] == 5040 and counts['total_orientations'] == 10080,
            "Permutation enumeration is incomplete")
    require(sum(counts[key] for key in ('provably_non_diameter_orientations',
            'provably_unsorted_x_orientations', 'provably_negative_z3_orientations',
            'retained_orientations')) == 10080, "Enumeration accounting failed")
    require(counts['retained_orientations'] == 80 and len(groups) == 4,
            "Expected candidate symmetry counts changed; no representations were silently dropped")
    return active, groups, counts, len(gauges)


def main():
    points, energy = exact_candidate_enclosures()
    active, groups, counts, gauge_count = enumerate_canonical(points)
    boxes = []
    for index, (numerators, group) in enumerate(sorted(groups.items())):
        center = [F(n, DENOMINATOR) for n in numerators]
        lower = [n-RADIUS_NUMERATOR for n in numerators]
        upper = [n+RADIUS_NUMERATOR for n in numerators]
        require(all(F(lo, DENOMINATOR) < interval[0] <= interval[1] < F(hi, DENOMINATOR)
                    for lo, hi, interval in zip(lower, upper, group['enclosure'])),
                "A retained exact candidate copy is outside its proposed box")
        require(all(abs(n) < 2**53 and F.from_float(float(F(n, DENOMINATOR))) == F(n, DENOMINATOR)
                    for n in lower+upper), "An exported endpoint is not exactly binary64 representable")
        certificate = certify(7, center, RADIUS)
        e = certificate['unique_local_minimum_energy_interval']
        require(F(e['lower']['fraction']) <= energy[0] <= energy[1] <= F(e['upper']['fraction']),
                "Candidate energy enclosure does not fit the new local certificate")
        require(len(group['representations']) == 20, "Unexpected template multiplicity")
        boxes.append({
            "id": index,
            "center_numerators": list(numerators),
            "lower_numerators": lower,
            "upper_numerators": upper,
            "radius_numerator": RADIUS_NUMERATOR,
            "representation_count": len(group['representations']),
            "representations": group['representations'],
            "exact_candidate_coordinate_hull": [interval_record(x) for x in group['enclosure']],
            "candidate_energy_interval": interval_record(energy),
            "local_certificate": certificate,
            "exact_identity_argument": "Every listed representation is an exact permutation and orthogonal gauge transform of the algebraic stationary candidate. Its interval lies strictly in this unique-critical-point box, so every listed copy is that box's unique local minimum with energy E_star.",
        })
    for i, first in enumerate(boxes):
        for second in boxes[:i]:
            require(any(a_hi < b_lo or b_hi < a_lo for a_lo, a_hi, b_lo, b_hi in
                        zip(first['lower_numerators'], first['upper_numerators'],
                            second['lower_numerators'], second['upper_numerators'])),
                    "Distinct exported template boxes unexpectedly overlap")
    result = {
        "schema": "lj7.canonical-candidate-boxes.v1",
        "particles": 7,
        "dimension": 15,
        "coordinate_order": [list(pair) for pair in active],
        "scale_bits": SCALE_BITS,
        "endpoint_rule": "endpoint = integer_numerator * 2^(-scale_bits)",
        "common_radius": record(RADIUS),
        "cpp_conversion": "std::ldexp(static_cast<double>(numerator), -50); all numerators have absolute value below 2^53",
        "cpp_use": "A complete search box may use this local certificate only when every coordinate interval is contained in one exported closed candidate box. Do not enlarge these boxes by outward rounding.",
        "canonical_rules": ["p0=0", "p1 lies on positive x and is a maximum-distance partner of p0",
                            "p2 is in the xy plane with positive y", "x2<=x3<=x4<=x5<=x6", "z3>=0"],
        "tie_policy": "Reject only a strictly separated interval violation; retain equality and undecidable overlap. Both reflections are retained when z3 contains zero.",
        "enumeration_counts": counts,
        "distinct_anchor_gauges_evaluated": gauge_count,
        "template_count": len(boxes),
        "boxes": boxes,
        "exact_candidate_source": "results/exact_candidate.json",
        "candidate_root_isolation_reverified": True,
        "candidate_coordinates_recomputed_from_algebraic_formulas": True,
        "claim": "Covers all canonicalized copies of this exact candidate and certifies each template locally. This is not coverage of all seven-point configurations and is not a global-optimality certificate.",
        "arithmetic": "Exact Fraction interval decisions and root enclosures; local certificates use their explicit outward 160-bit interval checks; no acceptance condition uses assert.",
        "all_exact_checks_passed": True,
    }
    output = ROOT/'results/canonical_candidate_boxes.json'
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print('Enumeration:', counts)
    for box in boxes:
        cert = box['local_certificate']
        print('template', box['id'], 'representations', box['representation_count'],
              'PD', cert['preconditioned_gershgorin_minimum_margin']['decimal_approximation'],
              'contraction', cert['contraction_infinity_norm_upper_bound']['decimal_approximation'])
    print('Certified four dyadic boxes; wrote', output)


if __name__ == '__main__':
    main()
