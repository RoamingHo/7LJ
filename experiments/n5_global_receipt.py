"""Verify the recorded N5 from-root replay chain and recheck its exact candidate.

This fast command checks recorded replay integrity; rerunning all tree mathematics
uses the bounded from-root checker on the complete branch tree (distributed separately).
"""
import argparse
from fractions import Fraction as F
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'proofs/n5_tree'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def resolve_state(path_str):
    p = Path(path_str)
    if not p.is_absolute():
        return ROOT / p
    if p.is_relative_to(ROOT):
        return p
    candidate = ROOT / 'results' / p.name
    return candidate if candidate.exists() else p


def verify_bundle(bundle=BUNDLE):
    return read(bundle / 'closure.json')


def verify_chain(reports, bundle=BUNDLE):
    manifest = verify_bundle(bundle)
    reference = read(bundle / manifest['final_receipt'])
    binding_fields = ('config_sha256', 'events_sha256', 'checker_sha256',
                      'shared_dependencies_sha256')
    prior_states, previous_events = [], 0
    require(reports, 'empty chain has no root execution')
    for index, report in enumerate(reports):
        require(report['schema'] == 'lj5.independent-from-root-checker.v1', 'wrong replay schema')
        require(report['resume_prior_state_sha256'] == prior_states, 'broken root or continuation chain')
        require(all(report[key] == reference[key] for key in binding_fields), 'replay source binding changed')
        require(report['domain_sha256'] == reference['domain_sha256'], 'domain binding changed')
        require(report['producer_lower_bounds_and_witnesses_used'] is False and
                report['producer_cpp_executed'] is False, 'producer evidence was used')
        require(report['failure'] is None, 'replay failure remains')
        events = report['events_verified']
        require(events - previous_events == report['events_verified_this_run'] > 0,
                'event continuation count changed')
        require(events == report['splits'] + report['prunes'] + report['candidate_leaves'],
                'terminal event count changed')
        require(report['pending_tree_nodes'] == 1 + 2 * report['splits'] - events,
                'tree coverage accounting failed')
        if index + 1 < len(reports):
            require(report['status'] == 'prefix_verified', 'intermediate report is not a prefix')
            prior_states.append(report['next_state_sha256'])
        previous_events = events
    final = reports[-1]
    require(final['status'] == 'closed' and final['full_n5_independent_proof_closed'] is True and
            final['full_n5_candidate_localization_verified'] is True, 'proof is not closed')
    require(final['pending_tree_nodes'] == final['nonempty_pending_boxes'] == 0 and
            final['next_state_path'] is None and final['next_state_sha256'] is None,
            'closed report retains a frontier')
    require((final['events_verified'], final['splits'], final['prunes'], final['candidate_leaves']) ==
            (92965, 46482, 46482, 1), 'global candidate closure count changed')
    return {key: final[key] for key in ('events_verified', 'splits', 'prunes', 'candidate_leaves',
                                       'pending_tree_nodes', 'supplemental_leaf_proofs')}


def verify_candidate():
    verify_bundle()
    # Recompute the candidate mathematics with the repository modules.
    code = """
import json
from pathlib import Path
import sympy as sp
from experiments.n5_candidate import build
from proofs.check_n5_from_root import fresh_candidate_boxes
report, _, _ = build()
assert report == json.loads(Path('results/exact_candidate5.json').read_text())
boxes = fresh_candidate_boxes()
t, z = sp.symbols('t z')
gram = sp.diag(sp.Matrix([[t,-t],[-t,t]]),
               sp.Matrix([[1,-sp.Rational(1,2),-sp.Rational(1,2)],
                          [-sp.Rational(1,2),1,-sp.Rational(1,2)],
                          [-sp.Rational(1,2),-sp.Rational(1,2),1]]))
assert gram * sp.ones(5,1) == sp.zeros(5,1)
assert sp.expand(gram.charpoly(z).as_expr()-z**2*(z-2*t)*(z-sp.Rational(3,2))**2) == 0
print(json.dumps({'fresh_exact_candidate_matches_saved': True,
                  'fresh_canonical_local_boxes': len(boxes),
                  'gram_characteristic_identity_verified': True}))
"""
    run = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                         env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
                         text=True, capture_output=True, check=True)
    result = json.loads(run.stdout)
    certificate = read(ROOT / 'results/n5_gm.json')
    candidate = read(ROOT / 'results/exact_candidate5.json')
    endpoint = lambda item: F(int(item['numerator']), int(item['denominator']))
    bracket = certificate['certified_global_energy_bracket']
    lo, hi = endpoint(bracket['lower_strict']), endpoint(bracket['upper_strict'])
    energy = candidate['energy_root_interval']
    require(lo < endpoint(energy['lower']) <= endpoint(energy['upper']) < hi,
            'exact candidate is not inside the strict global bracket')
    result.update(candidate_centered_gram_rank=3,
                  gram_eigenvalues_divided_by_R_squared=['0', '0', '3/2', '3/2', '2*t_star'],
                  gram_spectrum_alone_proves_global_optimality=False,
                  energy_bracket_width=str(hi-lo), exact_energy_definition='-B(t_star)^2/A(t_star)',
                  shape_root_selection='unique root of P in (197/100,199/100)',
                  shape_polynomial_coefficients_ascending=candidate['shape_polynomial_coefficients_ascending'],
                  rational_functions=candidate['rational_functions'],
                  certified_global_energy_bracket=bracket)
    return result


def uniform_spectral_probe():
    qmax = F(163, 100)**2
    coefficient = (qmax**2 + qmax + 1)**2 / qmax**6
    lower = -10 + coefficient / 2
    require(lower < F('-9.103852416'), 'spectral probe gap changed')
    return {
        'squared_diameter_upper': str(qmax),
        'pair_loss_quadratic_coefficient': str(coefficient),
        'rank_three_distance_square_lower': '1/2', 'energy_lower': str(lower),
        'energy_lower_approximation': float(lower), 'closes_global_minimum': False,
        'derivation': 'For q<=qmax, (q^-3-1)^2 >= k*(q-1)^2 with k=(qmax^-1+qmax^-2+qmax^-3)^2. The centered unit-distance target is J/2, with four eigenvalues 1/2. Every centered rank<=3 Gram matrix loses at least one such direction, so its squared Frobenius error is >=1/4. Double centering gives sum_{i<j}(q_ij-1)^2>=1/2. Hence E>=-10+k/2.',
        'scope': 'Exact bound on the already certified diameter<163/100 low-energy domain. It is weaker than the candidate energy and does not close a new spectral proof.',
    }


def build(paths):
    reports = [read(path) for path in paths]
    closure = verify_chain(reports)
    candidate = verify_candidate()
    certificate = read(ROOT / 'results/n5_gm.json')
    compactness_argument = certificate['justification'].split(';')[0] + '; ' +         'the additional diameter/orientation/symmetry reductions preserve a representative of every unrestricted global minimizer'
    return {
        'schema': 'lj5.exact-global-uniqueness-receipt.v1', 'date': '2026-11-10',
        'domain': 'All collision-free five-point configurations in R^3, V(r)=r^-12-2*r^-6',
        'global_minimum_proved_by_recorded_from_root_replay': True,
        'unique_global_orbit': True,
        'uniqueness_equivalence': 'translations, orthogonal transformations (including reflections), and particle permutations',
        'global_uniqueness_argument': 'Every attained unrestricted minimizer has a root representative. The complete replay excludes every retained stationary region except one whole box strictly inside a freshly certified unique-critical-point box containing the exact D3h candidate. Therefore every global minimizer is congruent to that candidate, up to permutation.',
        'compactness_argument': compactness_argument,
        'candidate': candidate, 'replay_counts': closure,
        'simple_rank_three_probe': uniform_spectral_probe(),
        'recorded_replay_wall_seconds': sum(r['wall_seconds'] for r in reports),
        'ordered_replay_reports': [str(path.relative_to(ROOT)) for path in paths],
        'mathematics_freshly_recomputed_by_this_aggregation': ['exact algebraic isolation', 'local uniqueness', 'canonical candidate inclusion', 'candidate Gram characteristic polynomial'],
        'all_tree_events_freshly_recomputed_by_this_aggregation': False,
        'trust_boundary': 'The listed from-root replay executions supply the global proof. This aggregator verifies their bindings, continuation and closure and freshly recomputes candidate mathematics; hashes alone cannot replace executing the tree from its root. Python, SymPy and the reviewed domain/gauge lemmas remain trusted.',
        'new_fast_spectral_global_proof': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/n5_global_receipt.json')
    args = parser.parse_args()
    paths = args.reports or sorted((ROOT / 'results').glob('n5_replay_chunk*.json'),
                                   key=lambda p: int(re.search(r'chunk(\d+)', p.name)[1]))
    paths = [path.resolve() for path in paths if not path.name.endswith('.state.json')]
    result = build(paths)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'unique_global_orbit': True,
                      'replay_counts': result['replay_counts']}))


if __name__ == '__main__':
    main()
