"""Exact N6 sublevel reduction, not an exact global-minimum certificate.

The N5 dependency is checked for integrity; its interval tree is not replayed.
All new scalar inequalities and finite graph coverage checks use Fraction/integers.
"""
from fractions import Fraction as F
from itertools import combinations, permutations
from math import comb
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parents[1]
TARGET = F(-3267, 257)
L5 = F(-71123847, 7812500)
R = F(27, 20)
B = R*R
CUT = F(3, 2)
PAIRS = tuple(combinations(range(6), 2))
INDEX = {pair: k for k, pair in enumerate(PAIRS)}
MATCHING = ((0, 1), (2, 3), (4, 5))
N5_BUNDLE = 'proofs/n5_tree'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def potential(q):
    return q**-6-2*q**-3


def loss(q):
    return (1-q**-3)**2


def bernstein_coefficients(poly, lo, hi):
    degree = len(poly)-1
    mono = [sum(poly[j]*comb(j, k)*lo**(j-k)*(hi-lo)**k
                for j in range(k, degree+1)) for k in range(degree+1)]
    return [sum(mono[k]*F(comb(i, k), comb(degree, k)) for k in range(i+1))
            for i in range(degree+1)]


def zero_long_certificate(curvature=F(3, 2), slope=F(-3381, 1000), offset=F(219, 250)):
    """f(q)>=Kq^2+bq+c for 0<q<=B, then the rank-three moment bound."""
    require(curvature > 0 and slope < 0, 'wrong scalar support signs')
    poly = [F(1), F(0), F(0), F(-2), F(0), F(0), -offset, -slope, -curvature]
    smallest = None
    subdivisions = 256
    for k in range(subdivisions):
        beta = bernstein_coefficients(poly, B*F(k, subdivisions), B*F(k+1, subdivisions))
        require(min(beta) >= 0, 'scalar polynomial is not certified nonnegative')
        smallest = min(beta) if smallest is None else min(smallest, *beta)
    # Centered rank<=3 Gram matrix: S2>=2*tr(G^2)+2*tr(G)^2
    # >=(8/3)*tr(G)^2=(2/27)*S1^2; S1=6*tr(G).
    lower = 15*offset-F(27, 8)*slope*slope/curvature
    require(lower > TARGET, 'zero-long lower bound misses target')
    return {'curvature': str(curvature), 'slope': str(slope), 'offset': str(offset),
            'support_domain': ['0', str(B)], 'bernstein_subdivisions': subdivisions,
            'minimum_bernstein_coefficient': str(smallest), 'energy_lower_exact': str(lower),
            'target_margin_exact': str(lower-TARGET), 'requires_stationarity': False,
            'rank_three_moment_inequality': 'sum(q_ij^2)>=(2/27)*sum(q_ij)^2'}


def n5_integrity(root):
    folder = root/N5_BUNDLE
    final = json.loads((folder/'receipt.json').read_text())
    require(final['full_n5_independent_proof_closed'] is True and
            final['full_n5_candidate_localization_verified'] is True and
            final['pending_tree_nodes'] == 0 and final['nonempty_pending_boxes'] == 0,
            'recorded N5 proof is not closed')
    candidate = json.loads((folder/'candidate.json').read_text())
    lower = candidate['energy_root_interval']['lower']
    require(L5 < F(int(lower['numerator']), int(lower['denominator'])),
            'N5 rational lower bound does not enclose the proved exact value')
    return {'L5_exact': str(L5), 'fresh_interval_replay': False}


def relabel(mask, weights):
    return sum(weights[k] for k in range(15) if mask >> k & 1)


def graph_cover(count):
    """All auxiliary masks extending the fixed matching, modulo its stabilizer."""
    require(count in (1, 2, 3), 'matching count must be 1, 2 or 3')
    true_mask = sum(1 << INDEX[pair] for pair in MATCHING[:count])
    stabilizer = []
    for permutation in permutations(range(6)):
        weights = [1 << INDEX[tuple(sorted((permutation[i], permutation[j])))] for i, j in PAIRS]
        if relabel(true_mask, weights) == true_mask:
            stabilizer.append(weights)
    labelled, representatives = set(), set()
    for mask in range(1 << 15):
        if mask & true_mask != true_mask:
            continue
        edges = tuple(k for k in range(15) if mask >> k & 1)
        if len(edges) > 4 or count*loss(B)+(len(edges)-count)*loss(CUT) > 15+TARGET:
            continue
        if max(sum(vertex in PAIRS[k] for k in edges) for vertex in range(6)) > 2:
            continue
        labelled.add(mask)
        representatives.add(min(relabel(mask, weights) for weights in stabilizer))
    recovered = {relabel(mask, weights) for mask in representatives for weights in stabilizer}
    require(recovered == labelled, 'finite auxiliary graph cover is incomplete')
    return {'true_long_count': count, 'true_long_mask': true_mask,
            'labelled_auxiliary_masks': len(labelled), 'stabilizer_size': len(stabilizer),
            'representative_auxiliary_masks': sorted(representatives)}


def audit(root=ROOT):
    dependency = n5_integrity(Path(root))
    degree_two_lower = L5-3+2*potential(B)
    auxiliary_degree_three_lower = L5-2+3*potential(CUT)
    five_auxiliary_lower = -15+5*loss(CUT)
    three_true_four_auxiliary_lower = -15+3*loss(B)+loss(CUT)
    require(1 < CUT < B and degree_two_lower > TARGET and
            auxiliary_degree_three_lower > TARGET and five_auxiliary_lower > TARGET and
            three_true_four_auxiliary_lower > TARGET, 'exact graph reduction inequality failed')
    graphs = [graph_cover(k) for k in (1, 2, 3)]
    require(graphs[-1]['representative_auxiliary_masks'] == [graphs[-1]['true_long_mask']],
            'three true-long edges did not reduce to the auxiliary matching')
    return {'schema': 'lj6.exact-matching-sublevel-reduction.v1', 'verified': True,
            'particles': 6, 'ambient_dimension': 3, 'target_exact': str(TARGET),
            'true_long_distance_threshold': str(R), 'true_long_squared_threshold': str(B),
            'auxiliary_squared_threshold': str(CUT), 'N5_dependency': dependency,
            'exact_lower_bounds': {'two_true_long_edges_at_one_vertex': str(degree_two_lower),
                'three_auxiliary_edges_at_one_vertex': str(auxiliary_degree_three_lower),
                'five_auxiliary_edges': str(five_auxiliary_lower),
                'three_true_long_and_four_auxiliary_edges': str(three_true_four_auxiliary_lower)},
            'zero_true_long_exclusion': zero_long_certificate(), 'remaining_graph_cover': graphs,
            'theorem': 'Any six-point R3 configuration with energy<=-3267/257 has 1, 2, or 3 true-long edges forming a matching. Its auxiliary graph has maximum degree two and at most four edges, and belongs to the listed complete graph cover. With three true-long edges, the auxiliary graph is exactly that matching, and all twelve other squared distances are<3/2.',
            'requires_stationarity': False, 'global_N6_optimality_proved': False,
            'remaining_obligation': 'Exclude or localize every remaining matching/auxiliary distance cell at the exact octahedral energy, then use the existing local uniqueness certificates.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit()
    if args.output:
        require(not args.output.exists(), 'refusing to overwrite '+str(args.output))
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
