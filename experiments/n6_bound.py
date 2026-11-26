"""Read the recorded independent N6 lower-bound from-root closure receipt.

Full mathematical reproduction requires running the from-root checker with the
complete branch tree, which is distributed separately as a data archive.
"""
from fractions import Fraction
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N6_BUNDLE = 'proofs/n6_tree'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verify_n6_execution(root=ROOT):
    bundle = Path(root)/N6_BUNDLE
    manifest = json.loads((bundle/'closure.json').read_text())
    require(manifest['schema'] == 'lj6.independent-lower-proof-bundle.v1', 'wrong N6 bundle schema')
    require(manifest['checker'] == 'proofs/check_n6_from_root.py', 'wrong N6 checker')
    final = json.loads((bundle/manifest['final_receipt']).read_text())
    require(final['schema'] == 'lj6.independent-lower-from-root-checker.v1', 'wrong N6 receipt schema')
    require(final['failure'] is None and final['producer_cpp_executed'] is False and
            final['producer_lower_bounds_and_witnesses_used'] is False,
            'N6 prefix was not independently verified')
    require(final['status'] == 'closed' and final['full_n6_independent_lower_bound_proved'] is True,
            'N6 independent proof is incomplete')
    require(final['pending_tree_nodes'] == final['nonempty_pending_boxes'] == 0 and
            final['prunes'] == final['splits']+1 and final['next_state_sha256'] is None,
            'N6 final tree accounting differs')
    require(final['events_verified'] == manifest['events_verified'] == 586523,
            'N6 final event count differs')
    lower = Fraction(final['target_exact'])
    require(lower == Fraction(final['global_energy_lower_bound']) == Fraction(manifest['lower_bound']),
            'N6 final exact lower bound differs')
    require(final['n7_global_optimality_proved'] is False, 'N6 receipt incorrectly asserts N7 closure')
    return lower


if __name__ == '__main__':
    lower = verify_n6_execution()
    print(json.dumps({'L6': str(lower),
                      'scope': 'Recorded from-root checker chain; rerun for full mathematical reproduction.'}))
