"""Fresh exact candidate frames and a local certificate in the C5 apex-axis gauge."""
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import argparse
import json
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(1, str(HERE))
from experiments.box import (
    exact_candidate_enclosures, pair_distances, map_gauge, stable_center, DENOMINATOR,
)
from experiments.tools import require, scale
from experiments.certificate import active_coordinates, certify
from chart_audit import (
    verify_stationarity_identity, full_cartesian,
)

PAIRS = tuple(combinations(range(7), 2))
ACTIVE = sorted(active_coordinates(7), key=active_coordinates(7).get)
LONG = {(2, 4), (2, 5), (3, 5), (3, 6), (4, 6)}
RADIUS = F(1, 1 << 17)


def serialized(data):
    return json.dumps(data, sort_keys=True, indent=2)+'\n'


def verify(expected=None):
    verify_stationarity_identity()
    points, energy = exact_candidate_enclosures()
    distances = pair_distances(points)
    threshold = F(73, 50)**2
    long_pairs = set()
    for pair in PAIRS:
        q = distances[pair]
        require(q[1] < threshold or q[0] > threshold, 'candidate distance class unresolved')
        if q[0] > threshold:
            long_pairs.add(pair)
    apices = [i for i in range(7) if not any(i in pair for pair in long_pairs)]
    ring = [i for i in range(7) if i not in apices]
    require(len(apices) == 2 and len(ring) == 5 and len(long_pairs) == 5,
            'candidate does not have the required five-cycle and apices')
    neighbors = {i: [j for j in ring if j != i and tuple(sorted((i, j))) not in long_pairs]
                 for i in ring}
    require(all(len(row) == 2 for row in neighbors.values()), 'candidate short ring is not a cycle')
    groups = {}
    for first, second in (apices, apices[::-1]):
        for anchor in ring:
            mapped = map_gauge(points, distances, (first, second, anchor))
            for next_vertex in neighbors[anchor]:
                cycle = [anchor, next_vertex]
                while len(cycle) < 5:
                    choices = [i for i in neighbors[cycle[-1]] if i != cycle[-2]]
                    require(len(choices) == 1 and choices[0] not in cycle, 'short cycle walk repeated a vertex')
                    cycle.append(choices[0])
                require(anchor in neighbors[cycle[-1]], 'short cycle did not close')
                order = [first, second]+cycle
                edges = {(i, j) for i, j in PAIRS if tuple(sorted((order[i], order[j]))) in long_pairs}
                require(edges == LONG, 'candidate frame has the wrong C5 labels')
                z3 = mapped[order[3]][2]
                require(z3[0] > 0 or z3[1] < 0, 'candidate z3 orientation was not separated')
                reflection = 1 if z3[0] > 0 else -1
                coordinates = [scale(mapped[order[atom]][axis], reflection if axis == 2 else 1)
                               for atom, axis in ACTIVE]
                center = stable_center(coordinates)
                representation = {'permutation': order, 'z_reflection': reflection}
                if center not in groups:
                    groups[center] = {'hull': coordinates, 'representations': [representation]}
                else:
                    previous = groups[center]
                    previous['hull'] = [(min(a[0], b[0]), max(a[1], b[1]))
                                        for a, b in zip(previous['hull'], coordinates)]
                    previous['representations'].append(representation)
    require(len(groups) == 1, 'apex-axis candidate requires more than one local template')
    center, group = next(iter(groups.items()))
    require(len(group['representations']) == 20, 'candidate frame coverage is incomplete')
    hull = group['hull']
    low = [q[0].numerator*DENOMINATOR//q[0].denominator for q in hull]
    high = [-((-q[1].numerator*DENOMINATOR)//q[1].denominator) for q in hull]
    radius_numerator = int(RADIUS*DENOMINATOR)
    local_low, local_high = [c-radius_numerator for c in center], [c+radius_numerator for c in center]
    require(all(F(a, DENOMINATOR) <= q[0] <= q[1] <= F(b, DENOMINATOR)
                for a, b, q in zip(low, high, hull)), 'exact hull escaped its dyadic export')
    require(all(F(a, DENOMINATOR) < q[0] <= q[1] < F(b, DENOMINATOR)
                for a, b, q in zip(local_low, local_high, hull)), 'candidate is outside the local box')
    require(all(abs(n) < 1 << 53 and F(float(F(n, DENOMINATOR))) == F(n, DENOMINATOR)
                for n in low+high+local_low+local_high), 'candidate endpoint is not binary64 exact')
    local = certify(7, [F(c, DENOMINATOR) for c in center], RADIUS)
    require(all(F(a, DENOMINATOR) == F(row['lower']['fraction']) and
                F(b, DENOMINATOR) == F(row['upper']['fraction'])
                for a, b, row in zip(local_low, local_high, local['box'])), 'local box differs from its certificate')
    local_energy = local['unique_local_minimum_energy_interval']
    require(F(local_energy['lower']['fraction']) <= energy[0] <= energy[1] <=
            F(local_energy['upper']['fraction']), 'algebraic energy is outside the fresh local certificate')
    result = {
        'schema': 'lj7.c5-apex-axis-exact-candidate.v1', 'particles': 7, 'dimension': 15,
        'gauge': 'c5_apex_axis', 'template_count': 1, 'long_mask': 733184,
        'coordinate_order': [list(pair) for pair in ACTIVE], 'scale_bits': 50,
        'candidate_root_isolation_reverified': True,
        'family_stationarity_identity_freshly_verified': True,
        'representations': group['representations'], 'center_numerators': list(center),
        'exact_enclosures': [{'lower': str(q[0]), 'upper': str(q[1])} for q in hull],
        'fixture_lower_numerators': low, 'fixture_upper_numerators': high,
        'fixture_hex': [[float(F(a, DENOMINATOR)).hex(), float(F(b, DENOMINATOR)).hex()]
                        for a, b in zip(low, high)],
        'local_box_lower_numerators': local_low, 'local_box_upper_numerators': local_high,
        'local_radius': str(RADIUS), 'local_certificate': local,
        'candidate_energy_interval': {'lower': str(energy[0]), 'upper': str(energy[1])},
        'cartesian_consistency_check': full_cartesian(hull),
        'identity_argument': 'All twenty frames are exact permutations and orthogonal transforms of the algebraic stationary candidate. Every frame lies strictly inside the same freshly certified unique-critical-point box, so all frames equal its unique local minimum with the exact algebraic energy.',
        'tie_argument': 'The D5h candidate has equal ring transverse radii and x2=d/2. All five ring anchors and both apex orders are included; weak domain inequalities retain these equalities. Both short-cycle directions are included before the final z reflection.',
        'global_LJ7_optimality_proved': False,
        'scope': 'One new apex-axis local template for this exact candidate only. This does not exclude the rest of the C5 domain or prove global optimality.',
    }
    if expected is not None:
        require(result == expected, 'saved apex-axis candidate differs from fresh verification')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', type=Path, required=True)
    args = parser.parse_args()
    verify(json.loads(args.check.read_text()))
    print(json.dumps({'verified': True, 'global_LJ7_optimality_proved': False}))


if __name__ == '__main__':
    main()
