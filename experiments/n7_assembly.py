"""Exact arithmetic premises for assembling the final LJ7 global proof.

This audits compactness thresholds and the feasible energy upper bound. It
does not replace the candidate identity or either final covering proof.
"""
from decimal import Decimal, localcontext
from fractions import Fraction as F
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = 'proofs/n7_candidate/candidate.json'
L6 = F(-129, 10)
OLD_UPPER = F(-33010767, 2000000)
UPPER = F(-2063173021, 125000000)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def partitions(total, largest):
    if total == 0:
        yield ()
    for first in range(min(total, largest), 0, -1):
        for tail in partitions(total-first, first):
            yield (first,)+tail


def verify(root=ROOT):
    path = Path(root)/CANDIDATE
    candidate = json.loads(path.read_text())
    energy_lo, energy_hi = (F(candidate['candidate_energy_interval'][key]) for key in ('lower', 'upper'))
    require(energy_lo <= energy_hi < UPPER < OLD_UPPER < L6, 'nested proof thresholds do not hold')
    lower = {size: F(-size*(size-1), 2) for size in range(1, 6)}
    lower[6] = L6
    proper = [p for p in partitions(7, 6)]
    require(len(proper) == len(set(proper)) == 14 and all(sum(p) == 7 for p in proper),
            'proper dissociation partition inventory is incomplete')
    rows = [{'parts': list(p), 'lower_bound': str(sum(lower[s] for s in p))} for p in proper]
    dissociation_lower = min(F(row['lower_bound']) for row in rows)
    require(dissociation_lower == L6 > UPPER, 'a dissociation channel reaches the target')
    # phi(r)=(r^-6-1)^2-1>=-1. An E<=UPPER configuration has
    # (r^-6-1)^2<=UPPER+21 for every pair, hence r>1/2.
    separation = F(1, 2)
    separation_margin = (separation**-6-1)**2-(UPPER+21)
    require(separation_margin > 0, 'elementary uniform separation failed')
    with localcontext() as context:
        context.prec = 65
        midpoint = (energy_lo+energy_hi)/2
        display = str(Decimal(midpoint.numerator)/Decimal(midpoint.denominator))
    return {'schema': 'lj7.global-assembly-arithmetic.v1',
        'candidate_artifact': CANDIDATE,
        'candidate_energy_display': display, 'candidate_energy_upper_below_final_target': True,
        'candidate_identity_proof_is_a_separate_required_dependency': True,
        'coarse_target': str(OLD_UPPER), 'final_target': str(UPPER), 'N6_lower': str(L6),
        'N6_independent_proof_is_a_separate_required_dependency': True,
        'proper_dissociation_partitions': rows, 'minimum_dissociation_lower': str(dissociation_lower),
        'uniform_separation_lower': str(separation), 'separation_margin': str(separation_margin),
        'compactness_argument': 'A sequence below the final target cannot collide, since each pair loss is bounded. If its diameter diverged modulo translation, a subsequence would split into a proper partition with cross distances diverging. All cross interactions tend to zero, giving liminf energy at least the recorded dissociation bound, a contradiction. Bounded separated sequences have convergent subsequences; a global minimizer at or below the certified feasible energy is therefore attained and stationary.',
        'requires_global_coverage_and_candidate_localization': True,
        'global_LJ7_optimality_proved': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args()
    result = verify()
    if args.check:
        require(json.loads(args.check.read_text()) == result, 'saved assembly arithmetic differs')
    if args.write:
        require(not args.write.exists(), 'refusing to overwrite an arithmetic audit')
        args.write.write_text(json.dumps(result, sort_keys=True, indent=2)+'\n')
    print(json.dumps({'verified': True, 'proper_partitions': 14,
        'minimum_dissociation_lower': str(L6), 'final_target': str(UPPER),
        'candidate_energy_display': result['candidate_energy_display'],
        'global_LJ7_optimality_proved': False}))


if __name__ == '__main__':
    main()
