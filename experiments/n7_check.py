"""Assembly of the exact LJ7 theorem from recorded branch-tree closure receipts.

The joining lemmas and the algebraic candidate are re-executed here. The
large branch trees themselves are not rerun; their closure is read from the
compact receipts under proofs/.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import importlib.util
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MATH = 'experiments/n7_assembly.py'
REDUCTION = 'experiments/n7_reduction.py'
CANDIDATE_BUNDLE = 'proofs/n7_candidate'
CANDIDATE_RELATIVE = '.'
AUX_BUNDLE = 'proofs/n7_weak_c5'
STRONG_BUNDLE = 'proofs/n7_strong_c5'
UPPER = F(-2063173021, 125000000)
B, C = F(5329, 2500), F(7829, 5000)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def module(root, relative, expected=None):
    path = root/relative
    spec = importlib.util.spec_from_file_location('lj7_assembly_'+path.stem, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def bundle(root, relative, manifest_name, schema):
    directory = root/relative
    manifest_path = directory/manifest_name
    manifest = json.loads(manifest_path.read_text())
    require(manifest['schema'] == schema, 'unexpected proof-bundle schema: '+relative)
    return directory, manifest


def validate_strong(manifest, short_lower, short_upper):
    require(manifest['execution_complete'] is True and manifest['status'] == 'closed' and
            manifest['strong_c5_domain_covered_under_local_chart'] is True,
            'the strong-C5 proof did not actually close')
    require(F(manifest['target_exact']) == UPPER and manifest['mask'] == 733184 and
            manifest['requires_apex_chart_z3_nonnegative'] is True, 'strong proof target or chart differs')
    expected = [(B, 4*short_upper**2-short_lower**2)]*5
    require([tuple(map(F, pair)) for pair in manifest['root_intervals']] == expected,
            'strong distance root does not cover the domain lemma')
    require(manifest['events'] == 153333 and manifest['splits'] == 76666 and manifest['leaves'] == 76667,
            'strong complete-tree counts differ')
    require(manifest['events'] == manifest['splits']+manifest['leaves'] and
            manifest['leaves'] == manifest['splits']+1 and
            not manifest['pending'] and not manifest['unproved_or_error_leaves'],
            'strong proof has an uncovered or unproved branch')
    require(manifest['leaf_status_counts'] == {'pruned': 72898, 'localized': 3769},
            'strong leaves do not all exclude or localize')
    require(manifest['shared_proof_statuses'] == {'candidate_hull': 'localized'},
            'strong localizations do not identify the certified candidate')


def verify(root=ROOT):
    root, started = Path(root), time.monotonic()
    math = module(root, MATH)
    arithmetic = math.verify(root)
    require(F(arithmetic['final_target']) == UPPER and
            F(arithmetic['minimum_dissociation_lower']) == F(-129, 10), 'wrong attainment thresholds')

    candidate_directory, candidate_manifest = bundle(root, CANDIDATE_BUNDLE, 'closure.json',
        'lj7.c5-apex-axis-diagnostic-producer-bundle.v1')
    candidate_source = candidate_directory/CANDIDATE_RELATIVE
    candidate_path = candidate_source/'candidate.json'
    candidate = json.loads(candidate_path.read_text())
    require(F(candidate['candidate_energy_interval']['upper']) < UPPER,
            'there is no certified feasible energy below the search target')
    command = [sys.executable, '-I', '-B', '-O', str(candidate_source/'candidate.py'),
               '--check', str(candidate_path)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    require(completed.returncode == 0 and json.loads(completed.stdout)['verified'] is True,
            'fresh algebraic and local candidate verification failed: '+completed.stderr)

    domain_source = candidate_source/'audit_domain.py'
    domain = module(candidate_directory, CANDIDATE_RELATIVE+'/audit_domain.py').verify()
    require(domain == json.loads((candidate_source/'audit_domain.json').read_text()) and
            F(domain['upper']) >= UPPER and F(domain['long_q_lower']) == B and
            domain['long_mask_combinations_lex_order'] == 733184,
            'the complete strong apex chart has not been established')

    reduction_module = module(root, REDUCTION)
    reduction = reduction_module.audit(root)
    require(reduction['stationary_sublevel_aux_C5_reduction_proved'] is True and
            reduction['requires_stationarity'] is True and
            F(reduction['upper_exact']) >= UPPER and
            F(reduction['auxiliary_squared_cut_exact']) == C and
            F(reduction['fresh_rational_inequalities']['L6']) == F(-129, 10),
            'the full configuration space has not been reduced to auxiliary C5')

    aux_directory, aux = bundle(root, AUX_BUNDLE, 'closure.json',
        'lj7.aux-nonstrong-independent-proof-seal.v1')
    aux_receipt = json.loads((aux_directory/'receipt.json').read_text())
    require(aux['status'] == aux_receipt['status'] == 'closed' and
            aux['aux_non_strong_stationary_exclusion_proved'] is True and
            aux_receipt['aux_non_strong_stationary_exclusion_proved'] is True and
            aux['pending_nodes'] == aux_receipt['pending_nodes'] == 0 and aux_receipt['failure'] is None,
            'auxiliary C5 with a non-strong edge remains unproved')
    require(F(aux['target_exact']) == F(aux_receipt['target_exact']) == UPPER and
            F(aux['minimum_verified_lower']) > UPPER and
            aux['events_verified'] == aux_receipt['events_verified'] == 1767 and
            aux['spectral_leaves'] == aux_receipt['spectral_leaves'] == 884 and
            aux_receipt['splits'] == 883, 'auxiliary complement coverage or threshold differs')
    expected_aux = [(C, B)]+[(C, 4*C-F(217, 250)**2)]*4
    require([tuple(map(F, pair)) for pair in aux_receipt['root_intervals']] == expected_aux and
            aux_receipt['minimum_edge_position'] == 0 and aux_receipt['mask'] == 733184 and
            aux_receipt['requires_stationarity'] is True and
            aux_receipt['no_cartesian_or_maximum_radius_anchor_imposed'] is True,
            'the minimum-edge complement does not match the full auxiliary family')

    _, strong = bundle(root, STRONG_BUNDLE, 'closure.json',
        'lj7.strong-c5-independent-proof-bundle.v1')
    validate_strong(strong, F(domain['rmin']), F(domain['short_rmax']))
    exact = json.loads((root/'results/exact_candidate.json').read_text())
    require(exact['shape_positive_root_count'] == 1 and exact['all_exact_checks_passed'] is True,
            'the exact positive shape root is not uniquely identified')
    require(candidate_path.is_file() and domain_source.is_file(), 'candidate inputs are missing')
    return {'schema': 'lj7.exact-global-minimum-assembly.v1', 'status': 'proved',
        'global_LJ7_optimality_proved': True, 'particles': 7, 'ambient_dimension': 3,
        'potential': 'sum_{i<j}(r_ij^-12-2*r_ij^-6)',
        'minimum_energy_display': arithmetic['candidate_energy_display'],
        'exact_minimum': '-B(t)^2/A(t)', 'A': '18/25+10/(1+t)^6+1/(4096*t^6)',
        'B': '2+10/(1+t)^3+1/(64*t^3)', 't': 'the unique positive root of the recorded polynomial',
        'shape_polynomial_coefficients_ascending': exact['shape_polynomial_coefficients_ascending'],
        'shape_root_interval': exact['shape_root_interval'],
        'minimizing_configuration': 'pentagonal bipyramid',
        'unique_up_to_euclidean_isometries_and_particle_permutations': True,
        'final_search_target': str(UPPER),
        'proof_join': [
            'The certified feasible candidate lies below U; separation and all proper dissociation bounds give attainment of a global minimum.',
            'An attained minimum is stationary. The independent N6 bound and complete auxiliary-graph proofs force auxiliary graph C5 plus two isolates.',
            'A minimum auxiliary edge at most B is excluded by the complete exact non-strong tree after a graph rotation, without imposing a Cartesian anchor.',
            'Otherwise all five auxiliary edges exceed B. The complete apex chart and complete strong distance tree exclude every branch or localize it to the same exact candidate.',
            'Thus every global minimizer is congruent, up to particle relabeling, to that exact candidate and has its energy; candidate feasibility gives equality.'
        ],
        'candidate_fresh_verification_command': [str(part) for part in command],
        'fresh_attainment_and_strong_domain_checks': True,
        'fresh_graph_reduction_and_component_inventory_checks': True,
        'every_prior_tree_replayed_in_this_command': False,
        'verification_mode': 'Assembly of recorded branch-tree closure receipts with fresh joining lemmas and algebraic/local candidate verification. The full branch trees are distributed separately as a data archive.',
        'component_event_counts': {'non_C5_colored': 129506, 'aux_non_strong': 1767, 'strong_C5': 153333},
        'wall_seconds': time.monotonic()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'refusing to overwrite a global proof receipt')
    result = verify()
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('status', 'global_LJ7_optimality_proved',
        'minimum_energy_display', 'unique_up_to_euclidean_isometries_and_particle_permutations', 'wall_seconds')}))


if __name__ == '__main__':
    main()
