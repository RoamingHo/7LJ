"""Enumerate unsorted candidate charts, then certify their actual local boxes."""
from collections import Counter
from fractions import Fraction as F
from itertools import combinations, permutations
from pathlib import Path
import json
import math
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from experiments.box import (
    exact_candidate_enclosures, pair_distances, map_gauge, square, stable_center,
    DENOMINATOR, SCALE_BITS, EXPECTED_SHAPE_POLYNOMIAL,
)
from experiments.tools import add, sub, scale, divide, positive_root, require
from experiments.certificate import Interval, certify, active_coordinates, interval_record

PAIRS = tuple(combinations(range(7), 2))
ACTIVE = sorted(active_coordinates(7), key=active_coordinates(7).get)


def mask(edges):
    return sum(1 << i for i, edge in enumerate(PAIRS) if edge in edges)


def verify_stationarity_identity():
    import sympy as sp
    t = sp.symbols('t', positive=True)
    s = sp.sqrt(5)
    short, long = (5-s)/2, (5+s)/2
    require(sp.simplify(5*(short**-6+long**-6)-sp.Rational(18,25)) == 0,
            'equatorial repulsion sum')
    require(sp.simplify(5*(short**-3+long**-3)-2) == 0, 'equatorial attraction sum')
    A = sp.Rational(18,25)+10/(1+t)**6+sp.Rational(1,4096)/t**6
    B = 2+10/(1+t)**3+sp.Rational(1,64)/t**3
    P = sum(value*t**i for i,value in enumerate(EXPECTED_SHAPE_POLYNOMIAL))
    require(sp.cancel(2*sp.diff(B,t)*A-B*sp.diff(A,t)+3*P/(25600*t**7*(1+t)**7)) == 0,
            'algebraic full-stationarity shape identity')
    return True


def full_cartesian(variables):
    points = [[Interval(0) for _ in range(3)] for _ in range(7)]
    for (atom, axis), q in zip(ACTIVE, variables):
        points[atom][axis] = Interval(*q)
    energy = Interval(0)
    forces = [[Interval(0) for _ in range(3)] for _ in range(7)]
    diagonal = [[Interval(0) for _ in range(3)] for _ in range(7)]
    for i, j in PAIRS:
        delta = [points[i][k] - points[j][k] for k in range(3)]
        q = sum((x*x for x in delta), Interval(0))
        require(q.lo > 0, 'fixture collision')
        energy += q**-6 - 2*q**-3
        radial, outer = 12*(q**-4-q**-7), 168*q**-8-96*q**-5
        for k in range(3):
            force = radial*delta[k]
            forces[i][k] += force
            forces[j][k] -= force
            curvature = radial+outer*delta[k]**2
            diagonal[i][k] += curvature
            diagonal[j][k] += curvature
    require(all(g.lo <= 0 <= g.hi for row in forces for g in row), 'full Cartesian zero-force enclosure')
    return {'energy_interval': interval_record(energy),
            'full_gradient_intervals': [[interval_record(g) for g in row] for row in forces],
            'full_hessian_diagonal_intervals': [[interval_record(g) for g in row] for row in diagonal],
            'gradient_absolute_bound': str(max(g.abs_upper() for row in forces for g in row)),
            'zero_in_all_21_gradient_intervals': True,
            'caution': 'Zero inclusion is a consistency check; exact stationarity uses the algebraic symmetry argument.'}


def verify(expected=None):
    """Fresh exact isolation, unsorted enumeration and local uniqueness checks.

    Optionally compare a saved JSON object to this fresh deterministic result.
    This function does not write files or trust a saved hash as verification.
    """
    verify_stationarity_identity()
    points, energy = exact_candidate_enclosures()
    distances = pair_distances(points)
    threshold = F(73, 50)**2
    long_edges = set()
    for pair in PAIRS:
        q = distances[pair]
        require(q[1] < threshold or q[0] > threshold, 'candidate edge class unresolved')
        if q[0] > threshold:
            long_edges.add(pair)
    cover = json.loads((HERE/'chart_cover.json').read_text())
    c5 = next(g for g in cover['graphs'] if g['components'] == ['P1', 'P1', 'C5'])
    by_mask = {c['mask']: c for c in c5['charts']}
    counts = Counter(total_orientations=10080)
    axis_cache, gauge_cache, direct, raw = {}, {}, {}, []
    for order in permutations(range(7)):
        a, b, anchor = order[:3]
        diameter_q = distances[a, b]
        if any(q[0] > diameter_q[1] for q in distances.values()):
            counts['diameter_rejected'] += 2
            continue
        if (a, b) not in axis_cache:
            diameter = positive_root(diameter_q, 2)
            x = {i: divide(sub(add(distances[a, i], diameter_q), distances[b, i]), scale(diameter, 2))
                 for i in range(7)}
            transverse = {i: sub(distances[a, i], square(x[i])) for i in range(7)}
            axis_cache[a, b] = diameter, x, transverse
        diameter, x, transverse = axis_cache[a, b]
        if any(q[0] > transverse[anchor][1] for q in transverse.values()):
            counts['maximum_transverse_rejected'] += 2
            continue
        if x[anchor][0] > diameter[1]/2:
            counts['half_axis_rejected'] += 2
            continue
        if order[:3] not in gauge_cache:
            gauge_cache[order[:3]] = map_gauge(points, distances, order[:3])
        mapped = gauge_cache[order[:3]]
        edges = {(i, j) for i, j in PAIRS if tuple(sorted((order[i], order[j]))) in long_edges}
        for reflection in (1, -1):
            z3 = scale(mapped[order[3]][2], reflection)
            if z3[1] < 0:
                counts['z3_rejected'] += 1
                continue
            counts['retained_without_tail_sort'] += 1
            raw.append((order, reflection, edges))
            if mask(edges) not in by_mask:
                continue
            chart = by_mask[mask(edges)]
            coords = [scale(mapped[order[atom]][axis], reflection if axis == 2 else 1)
                      for atom, axis in ACTIVE]
            center = stable_center(coords)
            key = chart['id'], center
            if key not in direct:
                direct[key] = {'chart_id': chart['id'], 'center': center, 'enclosure': coords, 'representations': []}
            else:
                direct[key]['enclosure'] = [(min(a[0], b[0]), max(a[1], b[1]))
                                             for a, b in zip(direct[key]['enclosure'], coords)]
            direct[key]['representations'].append({'permutation': list(order), 'z_reflection': reflection})
    require(sum(counts[k] for k in ('diameter_rejected', 'maximum_transverse_rejected',
            'half_axis_rejected', 'z3_rejected', 'retained_without_tail_sort')) == 10080, 'enumeration accounting')
    # Every unsorted representation can be relabelled into a listed chart,
    # with any required z reflection applied after changing the tail labels.
    covered_raw = 0
    for order, reflection, edges in raw:
        covered = False
        for tail in permutations(range(3, 7)):
            slot_order = (0, 1, 2)+tail
            changed = {(i, j) for i, j in PAIRS if tuple(sorted((slot_order[i], slot_order[j]))) in edges}
            if mask(changed) not in by_mask:
                continue
            chart = by_mask[mask(changed)]
            reordered = tuple(order[i] for i in slot_order)
            mapped = gauge_cache[order[:3]]
            for sign in (1, -1):
                if scale(mapped[reordered[3]][2], sign)[1] < 0:
                    continue
                coords = [scale(mapped[reordered[atom]][axis], sign if axis == 2 else 1)
                          for atom, axis in ACTIVE]
                require((chart['id'], stable_center(coords)) in direct, 'canonical chart fixture missing')
                covered = True
        require(covered, 'unsorted candidate representation has no chart fixture')
        covered_raw += 1
    require(covered_raw == len(raw), 'raw candidate cover incomplete')
    fixtures = []
    for (chart_id, center), value in sorted(direct.items()):
        coords = value['enclosure']
        radius = F(1, 1 << 17)
        require(all(F(c, DENOMINATOR)-radius < q[0] <= q[1] < F(c, DENOMINATOR)+radius
                    for c, q in zip(center, coords)), 'algebraic candidate not strictly in box')
        # Hex fixtures use exact binary64 dyadics enclosing the algebraic intervals.
        lower = [q[0].numerator*DENOMINATOR//q[0].denominator for q in coords]
        upper = [-((-q[1].numerator*DENOMINATOR)//q[1].denominator) for q in coords]
        require(all(F(a,DENOMINATOR) <= q[0] <= q[1] <= F(b,DENOMINATOR)
                    for a,b,q in zip(lower,upper,coords)), 'dyadic fixture enclosure')
        fixture = {'graph_id': c5['id'], 'chart_id': chart_id, 'mask': c5['charts'][chart_id]['mask'],
                   'coordinate_order': cover['coordinate_order'], 'scale_bits': SCALE_BITS,
                   'center_numerators': list(center), 'radius': str(radius),
                   'exact_enclosures': [{'lower': str(a), 'upper': str(b)} for a,b in coords],
                   'fixture_lower_numerators': lower, 'fixture_upper_numerators': upper,
                   'fixture_hex': [[float(F(a,DENOMINATOR)).hex(),float(F(b,DENOMINATOR)).hex()]
                                   for a,b in zip(lower,upper)],
                   'local_box_lower_numerators': [c-int(radius*DENOMINATOR) for c in center],
                   'local_box_upper_numerators': [c+int(radius*DENOMINATOR) for c in center],
                   'representations': value['representations'], 'cartesian_check': full_cartesian(coords)}
        certificate = certify(7, [F(c,DENOMINATOR) for c in center], radius)
        fixture['local_certificate'] = certificate
        require(all(F(a, DENOMINATOR) == F(entry['lower']['fraction']) and
                    F(b, DENOMINATOR) == F(entry['upper']['fraction'])
                    for a, b, entry in zip(fixture['local_box_lower_numerators'],
                                           fixture['local_box_upper_numerators'], certificate['box'])),
                'exported local box does not match fresh certificate')
        local_energy = certificate['unique_local_minimum_energy_interval']
        require(F(local_energy['lower']['fraction']) <= energy[0] <= energy[1] <=
                F(local_energy['upper']['fraction']), 'exact identity outside local energy interval')
        require(all(type(n) is int and abs(n) < 1 << 53 for n in
                    fixture['local_box_lower_numerators']+fixture['local_box_upper_numerators']),
                'local box endpoints are not binary64-exact dyadics')
        fixtures.append(fixture)
    result = {'schema': 'lj7.exact-unsorted-chart-candidate-fixtures.v1',
              'enumeration_counts': dict(counts), 'raw_unsorted_representations_covered': covered_raw,
              'candidate_long_edges_in_source_labels': sorted(map(list,long_edges)),
              'candidate_chart_ids': sorted({f['chart_id'] for f in fixtures}), 'fixture_count': len(fixtures),
              'direct_chart_representations': sum(len(f['representations']) for f in fixtures),
              'candidate_energy_interval': {'lower': str(energy[0]), 'upper': str(energy[1])},
              'candidate_exact_stationarity': 'Algebraic scale and shape equations, D5h symmetry, and nonsingular gauge; see source exact-candidate certificate.',
              'family_pair_sums_and_shape_derivative_identity_freshly_verified': True,
              'all_raw_unsorted_representations_mapped_to_fixtures': True,
              'old_sorted_candidate_boxes_used': False, 'fixtures': fixtures,
              'all_fixture_boxes_locally_certified': True,
              'global_LJ7_optimality_proved': False}
    if expected is not None:
        require(result == expected, 'saved candidate fixture differs from fresh exact verification')
    return result


def serialized(data):
    return json.dumps(data, indent=2)+'\n'


def binary_endpoint(value, upper):
    number = float(value)
    if (upper and F(number) < value) or (not upper and F(number) > value):
        number = math.nextafter(number, math.inf if upper else -math.inf)
    require(F(number) >= value if upper else F(number) <= value, 'binary energy enclosure')
    return number.hex()


def main():
    result = verify()
    (HERE/'audit_candidate_charts.json').write_text(serialized(result))
    print(json.dumps({k: result[k] for k in ('enumeration_counts','raw_unsorted_representations_covered',
                         'candidate_chart_ids','fixture_count','direct_chart_representations','all_fixture_boxes_locally_certified')}), flush=True)


if __name__ == '__main__':
    main()
