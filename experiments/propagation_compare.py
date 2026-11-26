"""Sorted-background variant of the propagation layer bounds.

This branch
imports it unchanged. It replays its full 8..150 baseline, then replaces e*c_D
inside each partition objective and restricts the number of parts to D.
"""
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import propagation as base

BASELINE = ROOT/'results/lj_bounds_8_150_v3.json'
OUTPUT = ROOT/'results/lj_sorted_background_8_150_v1.json'


@lru_cache(None)
def sorted_background_fields(d):
    """Sorted downward integer enclosures of all D actual background fields."""
    base.require(type(d) is int and d >= 1, 'invalid layer count')
    prefix = [0]
    for j in range(1, d):
        prefix.append(prefix[-1]+base.floor_units(base.kernel(j)))
    fields = tuple(sorted(prefix[m]+prefix[d-1-m] for m in range(d)))
    base.require(fields[0] == base.layer_base(d)[1], 'central-field identity failed')
    return fields


def background_dot(part, d):
    base.require(len(part) <= d and all(type(x) is int and x > 0 for x in part)
                 and all(a >= b for a, b in zip(part, part[1:])),
                 'positive descending partition must fit into D layers')
    return sum(x*b for x, b in zip(part, sorted_background_fields(d)))


@lru_cache(None)
def partition_costs(e, variant):
    base.require(variant in ('seven', 'literature_six'), 'unknown subcluster variant')
    lower = lru_cache(None)(lambda n: base.known_lower(n, include_seven=variant == 'seven',
                                                    literature_six=variant == 'literature_six'))
    return tuple((part, base.surplus_lower(part, lower)) for part in base.partitions(e))


@lru_cache(None)
def minimum_sorted_layer(n, d, variant):
    base.require(1 <= d <= n, 'invalid layer count')
    bg, _ = base.layer_base(d)
    best, witness, count = None, None, 0
    for part, cost in partition_costs(n-d, variant):
        if len(part) > d:
            continue
        value = cost+background_dot(part, d)
        count += 1
        if best is None or value < best:
            best, witness = value, part
    base.require(best is not None, 'no admissible surplus partition')
    return F(bg+best, base.SCALE), witness, count


def sorted_layer_threshold(n, upper, variant):
    checked = []
    for d in range(n, 1, -1):
        value, part, count = minimum_sorted_layer(n, d, variant)
        old_cost, _, _ = base.minimum_surplus(n-d, variant)
        old = base.layer_lower(n, d, old_cost)
        base.require(value >= old, 'sorted background weakened the original bound')
        checked.append({'layers': d, 'lower': str(value), 'margin_above_upper': str(value-upper),
                        'old_lower': str(old), 'gain': str(value-old),
                        'minimizing_partition': list(part), 'admissible_partition_count': count})
        if value <= upper:
            break
    base.require(len(checked) >= 2 and F(checked[-1]['lower']) <= upper,
                 'expected a strict layer exclusion followed by an unexcluded count')
    base.require(all(F(row['lower']) > upper for row in checked[:-1]), 'incomplete exclusion chain')
    return {'diameter_strict_upper': checked[-1]['layers'],
            'first_excluded_layer_count': checked[-2]['layers'],
            'first_unexcluded_layer_count': checked[-1]['layers'],
            'checked_layers': checked}
