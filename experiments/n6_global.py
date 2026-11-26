"""Exact N6 global octahedral minimum and equality-case verification.

New scalar, graph, spectral, gap-tree and local-box mathematics are replayed.
The reused N5 global bound enters through its recorded from-root closure
receipt; the interval tree itself is not rerun by this command.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import n6_reduction as reduction
from experiments import n6_single_long as single
from experiments import n6_two_long as two
from experiments import n6_gap as gap
from experiments import n6_localization as endpoint


def verify(root=ROOT):
    start = time.monotonic()
    root = Path(root)
    reduction.require(root.resolve() == ROOT.resolve(), 'proof verifier bound to its local artifact root')
    reduced = reduction.audit(root)
    paths = ('results/n6_single_long_certificate_v1.json',
             'results/n6_two_long_certificate_v1.json',
             'results/n6_matching_gap_v2.json')
    data = [json.loads((root/name).read_text()) for name in paths]
    single_result = single.verify(data[0], verify_reduction=False)
    two_result = two.verify(data[1], verify_reduction=False)
    gap_result = gap.verify_tree(data[2])
    local_result = endpoint.verify(F(gap_result['eta_exact']))
    target = reduction.TARGET
    # Explicit realization: ±a e_i, where q=2a² and q³=257/264.
    # Pair multiplicities and the completed scale square are exact identities.
    A, Bshape = F(771, 64), F(99, 8)
    reduction.require(A == F(12)+F(3, 64) and Bshape == F(12)+F(3, 8) and
                      -Bshape*Bshape/A == target and A/Bshape == F(257, 264),
                      'octahedral realization/scale identity failed')
    reduction.require(reduction.B > reduction.CUT > 1 and
                      F(289, 400)**-6-2*F(289, 400)**-3-14 > target,
                      'compact sublevel collision or distance conditions failed')
    reduction.require(single_result['single_long_family_excluded'] and
                      two_result['two_long_nonmatching_auxiliary_family_excluded'] and
                      two_result['remaining_auxiliary_mask'] == 16897 and
                      gap_result['matching_cube_covered'] and
                      local_result['localization_implies_unique_stationary_octahedron'],
                      'global theorem dependency did not establish its scope')
    return {'schema': 'lj6.exact-global-octahedron-proof.v1', 'verified': True,
            'particles': 6, 'ambient_dimension': 3,
            'energy_normalization': 'sum_{i<j}(r_ij^-12-2*r_ij^-6)',
            'minimum_energy_exact': str(target), 'minimum_energy_display': float(target),
            'minimizer': 'regular octahedron with short-edge squared distance q, q^3=257/264',
            'unordered_pair_multiplicities': {'q': 12, '2q': 3},
            'uniqueness': 'Every equality configuration is congruent to the scaled regular octahedron, up to atom permutation.',
            'proof_steps': [
                'The explicit octahedron attains -3267/257.',
                'The recorded N5 bound makes every E<=-3267/257 true-long graph a matching at R=27/20. Each pair has a third vertex at short distance from both endpoints, so the diameter is<2R. The direct pair bound gives rmin>17/20. Fixing one point therefore makes the nonempty sublevel compact, and an unrestricted minimizer is attained and stationary.',
                'The exact Gram moment inequality excludes zero true-long edges; finite graph reduction leaves 18,4,1 auxiliary classes for one,two,three true-long edges.',
                'Exact scalar Bernstein and spectral inertia certificates exclude all 18 single-long classes and the three nonmatching-auxiliary two-long classes at the same exact target.',
                'The only remaining auxiliary graph is a three-edge matching. The complete [3/2,4R^2]^3 tree either excludes a cell or localizes its centered Gram matrix within 1/100000 of the exact octahedral Gram matrix.',
                'Exact coordinate recovery and one full-box Newton contraction put every localized stationary point into the freshly verified strict local uniqueness box containing the exact octahedron.'],
            'N5_dependency': reduced['N5_dependency'],
            'fresh_new_mathematics': {'single_long': single_result, 'two_long_nonmatching': two_result,
                                      'matching_gap': gap_result, 'local_endpoint': local_result},
            'fresh_N5_interval_replay': False, 'global_N6_optimality_proved': True,
            'global_N6_uniqueness_proved': True, 'wall_seconds': time.monotonic()-start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify()
    if args.output:
        reduction.require(not args.output.exists(), 'refusing to overwrite global proof receipt')
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({key: result[key] for key in ('verified', 'minimum_energy_exact',
        'global_N6_optimality_proved', 'global_N6_uniqueness_proved', 'fresh_N5_interval_replay',
        'wall_seconds')}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
