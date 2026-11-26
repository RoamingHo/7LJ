"""Audit the stationary-sublevel reduction to auxiliary C5 plus two isolates.

Fresh work: rational inequalities and finite graph/color enumeration.
Prior N6 and spectral proof bundles: integrity and recorded-execution checks only.
This script does not rerun their interval trees or colored spectral leaf proofs.
"""
from collections import Counter
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import argparse
import importlib.util
import json
import time

ROOT = Path(__file__).resolve().parents[1]
R = F(73, 50)
B = R*R
CUT = (1+B)/2
UPPER = F('-16.5053835')
PAIRS = tuple(combinations(range(7), 2))
PAIR_INDEX = {pair: i for i, pair in enumerate(PAIRS)}
PINS = {
    'n6': 'proofs/n6_tree',
    'zero_long': 'proofs/n7_zero_long',
    'one_long': 'proofs/n7_one_long',
    'non_C5_colored': 'proofs/n7_colored',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def bundle_integrity(kind, root):
    directory = root/PINS[kind]
    data = json.loads((directory/'closure.json').read_text())
    return directory, data


def spectral_execution_integrity(directory, manifest, single=False):
    record = json.loads((directory/'receipt.json').read_text())
    require(record['returncode'] == 0 and record['result'] == json.loads(record['stdout']), 'recorded spectral execution differs')
    result = record['result']
    require(result['requires_stationarity'] is True and result['requires_L5_or_L6'] is False and
            result['global_LJ7_optimality_proved'] is False and manifest['global_LJ7_optimality_proved'] is False,
            'prior spectral proof has an unexpected scope')
    require(F(result['lower']) == F(manifest['lower_exact']) > UPPER and F(manifest['upper_exact']) == UPPER,
            'prior spectral exclusion misses the exact target')
    if single:
        require(manifest['exact_long_pair_count'] == result['exact_long_pair_count'] == 1 and
                F(manifest['radius_exact']) == R and result['status'] == 'verified' and
                (result['records'], result['marked_graphs'], result['intervals']) == (20784, 866, 24) and
                result['certificate_sha256'] == manifest['certificate_sha256'],
                'recorded single-long proof coverage differs')
        loss = lambda q: (1-q**-3)**2
        eight_aux_lower = -21+loss(B)+7*loss(CUT)
        require(F(manifest['eight_auxiliary_edges_lower_exact']) == eight_aux_lower >= F(manifest['lower_exact']),
                'single-long all-auxiliary-count scope lacks its loss bound')
    else:
        require(result['verified'] is True and F(result['D']) == F(manifest['D_exact']) == R and
                result['labelled_graphs'] == manifest['labelled_graphs_covered'] == 1 << 21 and
                result['representatives'] == manifest['representative_graphs'] == 1044,
                'recorded zero-long proof coverage differs')
        require(record['input_sha256_before'] == record['input_sha256_after'], 'recorded zero-long inputs changed during execution')
    return {'recorded_lower_exact': result['lower'], 'integrity_only_no_fresh_spectral_replay': True}


def classify_inventory(atlas):
    """Enumerate colors independently of any diagnostic or converted case list."""
    loss = lambda q: (1-q**-3)**2
    ordinary, c5, c5_masks = set(), set(), set()
    minimum_common_short_neighbors = 7
    for row in atlas['graphs']:
        mask = row['mask']; auxiliary = tuple(i for i in range(21) if mask >> i & 1)
        degrees = [sum(vertex in PAIRS[i] for i in auxiliary) for vertex in range(7)]
        is_c5 = len(auxiliary) == 5 and sorted(degrees) == [0, 0, 2, 2, 2, 2, 2]
        if is_c5:
            c5_masks.add(mask)
        for k in range(2, min(5, len(auxiliary))+1):
            gate = k*loss(B)+(len(auxiliary)-k)*loss(CUT)
            if gate > 21+UPPER:
                continue
            for edges in combinations(auxiliary, k):
                long_degrees = [sum(vertex in PAIRS[i] for i in edges) for vertex in range(7)]
                if max(long_degrees) > 2:
                    continue
                (c5 if is_c5 else ordinary).add((mask, edges))
                for edge in edges:
                    i, j = PAIRS[edge]
                    short_thirds = [vertex for vertex in range(7) if vertex not in (i, j) and
                        PAIR_INDEX[tuple(sorted((i, vertex)))] not in edges and
                        PAIR_INDEX[tuple(sorted((j, vertex)))] not in edges]
                    require(len(short_thirds) >= 3, 'degree-two common-short-neighbor implication failed')
                    minimum_common_short_neighbors = min(minimum_common_short_neighbors, len(short_thirds))
    require(len(ordinary) == 3586 and len(c5) == 26 and len(c5_masks) == 1,
            'fresh colored graph classification differs')
    return ordinary, c5, c5_masks, minimum_common_short_neighbors


def colored_execution_integrity(directory, manifest, expected):
    require(manifest['schema'] == 'lj7.colored-family-independent-proof-bundle.v1' and
            manifest['complete_colored_family_exclusion_proved'] is True and
            manifest['global_long_degree_bound_proved'] is False and manifest['global_LJ7_optimality_proved'] is False and
            manifest['requires_stationarity'] is True and F(manifest['upper_exact']) == UPPER,
            'colored proof cannot be used without its correct conditional scope')
    rows = manifest['verified_cases']
    require(len(rows) == 3586 and {(r['spec']['mask'], tuple(r['spec']['edges'])) for r in rows} == expected,
            'recorded colored proof omits or duplicates an independently enumerated case')
    summary = json.loads((directory/'receipt.json').read_text())
    require(summary['status'] == 'complete' and not summary['errors'] and
            summary['cases_closed'] == manifest['cases_closed'] == 3586 and
            summary['events_verified'] == manifest['events_verified'] == 129506 and
            manifest['splits_verified'] == 62960 and manifest['leaves_verified'] == 66546,
            'recorded colored execution is incomplete')
    lower = F(manifest['minimum_leaf_lower_exact'])
    require(lower > UPPER, 'recorded leaf exclusion is not strict')
    return {'recorded_cases_closed': 3586, 'recorded_events_verified': summary['events_verified'],
            'recorded_minimum_leaf_lower_exact': str(lower),
            'integrity_and_tree_binding_only_no_fresh_leaf_mathematics': True}


def audit(root=ROOT):
    root = Path(root); start = time.monotonic()
    helper_path = root/'experiments/n6_bound.py'
    helper = load_module('aux_c5_n6_integrity_helper', helper_path)
    L6 = helper.verify_n6_execution(root=root)
    require(L6 == F(-129, 10), 'the unconditional N6 bound is not available')
    n6_manifest = json.loads((root/PINS['n6']/'closure.json').read_text())
    zero_dir, zero = bundle_integrity('zero_long', root)
    one_dir, one = bundle_integrity('one_long', root)
    colored_dir, colored = bundle_integrity('non_C5_colored', root)
    zero_execution = spectral_execution_integrity(zero_dir, zero)
    one_execution = spectral_execution_integrity(one_dir, one, single=True)
    # These are recomputed exact inequalities, not values copied from prior reports.
    v_R = R**-12-2*R**-6
    degree_lower = L6-3+3*v_R
    six_long_lower = -15+6*v_R
    require(R > 1 and degree_lower > UPPER and six_long_lower > UPPER, 'fresh rational degree/count exclusion failed')
    base = load_module('aux_c5_fresh_atlas_checker', zero_dir/'verify_certificate.py')
    atlas = json.loads((zero_dir/'atlas.json').read_text())
    masks = [row['mask'] for row in atlas['graphs']]
    require(len(masks) == 1044 and base.PAIRS == PAIRS, 'auxiliary atlas identity differs')
    labelled = base.verify_graph_cover(masks)  # Fresh finite combinatorial cover, no spectral-bound replay.
    require(labelled == 1 << 21, 'fresh auxiliary labelled graph cover failed')
    ordinary, c5, c5_masks, third_count = classify_inventory(atlas)
    colored_execution = colored_execution_integrity(colored_dir, colored, ordinary)
    c5_by_k = Counter(len(edges) for _, edges in c5)
    require(dict(c5_by_k) == {2: 10, 3: 10, 4: 5, 5: 1}, 'the residual C5 subset coverage is incomplete')
    c5_mask = next(iter(c5_masks)); c5_edges = [list(PAIRS[i]) for i in range(21) if c5_mask >> i & 1]
    isolated = [i for i in range(7) if all(i not in e for e in c5_edges)]
    dependencies = {name: {'directory': path} for name, path in PINS.items()}
    return {'schema': 'lj7.stationary-sublevel-aux-C5-reduction-audit.v1', 'status': 'verified',
            'proof_dependencies': dependencies,
            'particles': 7, 'ambient_dimension': 3,
            'energy_normalization': 'sum_{i<j}(r_ij^-12-2*r_ij^-6)', 'radius_exact': str(R),
            'squared_radius_exact': str(B), 'auxiliary_squared_cut_exact': str(CUT), 'upper_exact': str(UPPER),
            'fresh_rational_inequalities': {'L6': str(L6), 'v_at_radius': str(v_R),
                'degree_at_least_three_lower': str(degree_lower), 'degree_margin': str(degree_lower-UPPER),
                'six_or_more_long_pairs_lower': str(six_long_lower), 'six_long_margin': str(six_long_lower-UPPER)},
            'fresh_finite_graph_checks': {'atlas_representatives': 1044, 'labelled_auxiliary_graphs': labelled,
                'non_C5_cases': len(ordinary), 'remaining_C5_cases': len(c5),
                'C5_cases_by_long_count': dict(sorted(c5_by_k.items())), 'C5_auxiliary_mask': c5_mask,
                'C5_auxiliary_edges': c5_edges, 'isolated_vertices': isolated,
                'minimum_common_short_neighbors_per_long_edge': third_count},
            'prior_proof_integrity': {'N6': {
                'recorded_events': n6_manifest['events_verified'], 'recorded_chunks': len(n6_manifest['executed_chain']),
                'integrity_of_executed_chain_only_no_fresh_interval_replay': True},
                'zero_long': zero_execution, 'one_long': one_execution, 'non_C5_colored': colored_execution},
            'theorem': 'Every finite-energy stationary seven-point configuration in R^3 with E<=-33010767/2000000 has auxiliary graph C5 disjoint union two isolated vertices, where an auxiliary edge means squared distance>=(1+(73/50)^2)/2. Its true long graph, at distance>=73/50, has 2..5 edges and maximum degree<=2.',
            'stationary_sublevel_aux_C5_reduction_proved': True, 'requires_stationarity': True,
            'new_N6_interval_replay_performed': False, 'new_zero_or_one_long_spectral_replay_performed': False,
            'new_colored_leaf_mathematical_replay_performed': False,
            'fresh_rational_threshold_and_graph_cover_checks_performed': True,
            'global_LJ7_optimality_proved': False,
            'remaining_scope': 'The full auxiliary-C5 domain, including all 26 true-long subsets, still requires a lower bound at the exact pentagonal-bipyramid candidate energy. This reduction alone does not establish that final bound.',
            'wall_seconds': time.monotonic()-start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/n7_reduction_audit.json')
    args = parser.parse_args()
    require(not args.output.exists(), 'refusing to overwrite a reduction audit receipt')
    result = audit()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'status': 'verified',
        'degree_lower_display': float(F(result['fresh_rational_inequalities']['degree_at_least_three_lower'])),
        'six_long_lower_display': float(F(result['fresh_rational_inequalities']['six_or_more_long_pairs_lower'])),
        'auxiliary_C5_mask': result['fresh_finite_graph_checks']['C5_auxiliary_mask'],
        'stationary_sublevel_aux_C5_reduction_proved': True, 'global_LJ7_optimality_proved': False,
        'seconds': result['wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
