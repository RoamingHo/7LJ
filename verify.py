#!/usr/bin/env python3
"""Single public entry point: recompute every theorem-level result of the paper.

Run:  python verify.py
The script re-executes the exact rational/interval arithmetic behind each
result. Large branch-and-bound enumeration trees are not rerun here; their
closure is consumed through compact recorded receipts.
"""
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import json
import re
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments import propagation as base
from experiments import propagation_compare as sorted_base


def read(name):
    return json.loads((ROOT / name).read_text())


def check_witness(n, witness):
    points = witness['normalized_integer_coordinates']
    grid = witness['coordinate_integer_grid']
    base.require(grid == 10**12 and len(points) == n, 'wrong coordinate grid/count')
    base.require(all(len(p) == 3 and all(type(x) is int for x in p) for p in points),
                 'coordinates must be integer triples')
    lo = hi = 0
    for a, b in combinations(points, 2):
        q = F(sum((x-y)**2 for x, y in zip(a, b)), grid**2)
        lower, upper = base.pair_energy_interval(q)
        lo += lower
        hi += upper
    base.require(witness['pair_sum_lower'] == str(F(lo, base.SCALE)) and
                 witness['pair_sum_upper'] == str(F(hi, base.SCALE)),
                 f'N={n}: rational witness energy mismatch')
    upper = F(-((-hi*10**9)//base.SCALE), 10**9)
    base.require(witness['feasible_upper'] == str(upper), 'witness upper cutoff differs')
    return upper


def check_propagation():
    saved = read('results/lj_bounds_8_150_v3.json')
    base.require([r['n'] for r in saved['rows']] == list(range(8, 151)),
                 'the table must cover exactly N=8..150')
    base.require(set(saved['rational_witnesses']) == {str(n) for n in range(8, 151)},
                 'the witness inventory differs')
    for row in saved['rows']:
        n = row['n']
        upper = check_witness(n, saved['rational_witnesses'][str(n)])
        original = base.layer_threshold(n, upper, 'pair')
        seven = base.layer_threshold(n, upper, 'seven')
        six = base.layer_threshold(n, upper, 'literature_six')
        endpoint = base.endpoint_certificate(n, upper)
        final = F(seven['diameter_strict_upper'])
        if endpoint:
            final = min(final, F(endpoint['diameter_strict_upper']))
        subset = base.cover_lower(n, 7, base.L7)
        ks_subset = base.cover_lower(n, 6, base.KS6)
        stability = n*base.PUBLISHED_STABILITY
        expected = {
            'n': n, 'L7_subset_lower': str(subset),
            'KS6_subset_comparison': str(ks_subset),
            'subset_improvement': str(subset-ks_subset),
            'published_stability_lower': str(stability),
            'combined_lower': str(max(subset, stability)),
            'KS6_combined_comparison': str(max(ks_subset, stability)),
            'combined_improvement': str(max(subset, stability)-max(ks_subset, stability)),
            'upper_feasible': str(upper), 'diameter_strict_upper': str(final),
            'original_pair_layer': original, 'L7_layer': seven,
            'literature_exact_six_layer_comparison': six,
            'endpoint': endpoint, 'literature_six_is_comparison_only': True,
        }
        base.require(expected == row, f'N={n}: published propagation row differs')
    summary = {
        'row_count': len(saved['rows']),
        'combined_energy_improved_sizes': [r['n'] for r in saved['rows'] if F(r['combined_improvement']) > 0],
        'L7_layer_improvements_over_pair': sum(r['L7_layer']['diameter_strict_upper'] < r['original_pair_layer']['diameter_strict_upper'] for r in saved['rows']),
        'L7_layer_improvements_over_literature_six': sum(r['L7_layer']['diameter_strict_upper'] < r['literature_exact_six_layer_comparison']['diameter_strict_upper'] for r in saved['rows']),
        'endpoint_improvements': sum(r['endpoint'] is not None and F(r['endpoint']['diameter_strict_upper']) < r['L7_layer']['diameter_strict_upper'] for r in saved['rows']),
    }
    base.require(summary == saved['summary'], 'propagation summary differs')

    comparison = read('results/lj_sorted_background_8_150_v1.json')
    base.require([r['n'] for r in comparison['rows']] == list(range(8, 151)),
                 'sorted comparison inventory differs')
    for row, old in zip(comparison['rows'], saved['rows']):
        n, upper = old['n'], F(old['upper_feasible'])
        seven = sorted_base.sorted_layer_threshold(n, upper, 'seven')
        six = sorted_base.sorted_layer_threshold(n, upper, 'literature_six')
        old_seven = old['L7_layer']['diameter_strict_upper']
        old_six = old['literature_exact_six_layer_comparison']['diameter_strict_upper']
        boundary = next(r for r in seven['checked_layers'] if r['layers'] == old_seven)
        final = F(seven['diameter_strict_upper'])
        if old['endpoint']:
            final = min(final, F(old['endpoint']['diameter_strict_upper']))
        expected = {
            'n': n, 'upper_feasible': str(upper),
            'pair_layer_upper': old['original_pair_layer']['diameter_strict_upper'],
            'old_L7_layer_upper': old_seven, 'sorted_L7_layer': seven,
            'old_L6_layer_upper': old_six, 'sorted_L6_layer': six,
            'L7_layers_saved': old_seven-seven['diameter_strict_upper'],
            'L6_layers_saved': old_six-six['diameter_strict_upper'],
            'old_final_upper': old['diameter_strict_upper'], 'sorted_final_upper': str(final),
            'old_L7_boundary_gain': boundary['gain'],
            'old_L7_boundary_margin_after': boundary['margin_above_upper'],
        }
        base.require(row == expected, f'N={n}: sorted-background row differs')
    base.require(comparison['summary']['row_count'] == 143, 'sorted summary differs')
    return 143


def main():
    start = time.monotonic()
    results = []

    from experiments import n5_global_receipt as n5
    paths = sorted((ROOT/'results').glob('n5_replay_chunk*.json'),
                   key=lambda p: int(re.search(r'chunk(\d+)', p.name)[1]))
    paths = [p for p in paths if not p.name.endswith('.state.json')]
    n5_result = n5.build(paths)
    base.require(n5_result['unique_global_orbit'] is True and
                 n5_result['global_minimum_proved_by_recorded_from_root_replay'] is True,
                 'LJ5 receipt/candidate check failed')
    results.append(('LJ5 exact global optimum and uniqueness', True))

    from experiments import n6_global
    n6_result = n6_global.verify()
    base.require(n6_result['global_N6_optimality_proved'] and
                 n6_result['global_N6_uniqueness_proved'] and
                 n6_result['minimum_energy_exact'] == '-3267/257', 'LJ6 theorem check failed')
    results.append(('LJ6 exact global optimum -3267/257 and uniqueness', True))

    from experiments import n6_competitor as c2v
    base.require(c2v.generate()['not_global_minimum_proved'] is True,
                 'LJ6 competing local minimum check failed')
    results.append(('LJ6 competing C2v local minimum excluded from global optimum', True))

    from experiments import n7_check
    n7_result = n7_check.verify()
    base.require(n7_result['global_LJ7_optimality_proved'] is True and
                 n7_result['unique_up_to_euclidean_isometries_and_particle_permutations'] is True,
                 'LJ7 theorem check failed')
    results.append(('LJ7 exact global optimum and uniqueness (pentagonal bipyramid)', True))

    rows = check_propagation()
    results.append((f'Propagation N=8..150: {rows} certified bound rows recomputed', True))

    from experiments import n8_candidate, n8_graph, n8_spectral
    cs = n8_candidate.certify(read('results/n8_cs_local_certificate.json')['center'])
    d2d = n8_candidate.certify(read('results/n8_d2d_local_certificate.json')['center'])
    base.require(cs['Cs_symmetry_and_no_additional_distance_symmetry_proved'] is True and
                 F(cs['local_energy_interval'][1]) < n8_candidate.UPPER and
                 not d2d['feasible_energy_below_upper'], 'LJ8 local check failed')
    spectral = n8_spectral.verify(read('results/n8_spectral_probe.json'))
    base.require(F(spectral['zero_long']['lower']) > n8_candidate.UPPER,
                 'LJ8 zero-long check failed')
    graph = n8_graph.run()
    base.require(graph['global_minimum_attained'] is True, 'LJ8 graph check failed')
    results.append(('LJ8 frontier: Cs/D2d local certificates, spectral and graph probes', True))

    width = max(len(name) for name, _ in results)
    for name, ok in results:
        print(f'[PASS] {name.ljust(width)}')
    print(f'\nAll {len(results)} theorem-level checks passed in {time.monotonic()-start:.1f} s')


if __name__ == '__main__':
    main()
