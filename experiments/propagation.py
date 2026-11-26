"""Exact LJ7 subcluster propagation and arrangement-free layer certificates.

All mathematical decisions use integers/Fraction. Coordinates are explicit
rational feasible witnesses. The fresh LJ7 assembler verifies the previously
recorded theorem receipts; it does not replay every historical search tree.
The independent layer derivation follows the exact rational arithmetic used in this module.
"""
from fractions import Fraction as F
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCALE = 10**30
L5 = F('-9.103852416')
L6 = F(-3267, 257)
L7 = F('-16.5053841681')
KS6 = F(-3267, 257)
PUBLISHED_STABILITY = F('-14.316')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def floor_units(x):
    return x.numerator*SCALE//x.denominator


def ceil_units(x):
    return -((-x.numerator*SCALE)//x.denominator)


def cover_lower(n, m, lower):
    require(2 <= m <= n, 'invalid subset size')
    return F(n*(n-1), m*(m-1))*lower


def known_lower(n, include_seven=True, literature_six=False):
    if n < 2:
        return F(0)
    candidates = [-F(n*(n-1), 2)]
    for m, value in ((5, L5), (6, KS6 if literature_six else L6)):
        if n >= m:
            candidates.append(cover_lower(n, m, value))
    if include_seven and n >= 7:
        candidates.append(cover_lower(n, 7, L7))
    return max(candidates)


@lru_cache(None)
def kernel(d):
    require(type(d) is int and d >= 1, 'invalid layer index separation')
    if d <= 2:
        return F(-1)
    r = F(d-1)
    return r**-12-2*r**-6


def half_open_layer_count(diameter):
    require(diameter >= 0, 'negative diameter')
    return diameter.numerator//diameter.denominator+1


def partitions(total, ceiling=None):
    require(type(total) is int and total >= 0, 'invalid partition total')
    if total == 0:
        yield ()
        return
    for first in range(min(total, total if ceiling is None else ceiling), 0, -1):
        for rest in partitions(total-first, first):
            yield (first,)+rest


@lru_cache(None)
def sorted_binding_kernel(k):
    return tuple(ceil_units(-kernel(d)) for d in range(1, k)
                 for _ in range(k-d))


def surplus_lower(part, lower_function):
    products = sorted((a*b for a, b in combinations(part, 2)), reverse=True)
    binding = sum(p*w for p, w in zip(products, sorted_binding_kernel(len(part))))
    return sum(floor_units(lower_function(1+x)) for x in part)-binding


@lru_cache(None)
def minimum_surplus(e, variant):
    lower = lambda n: known_lower(n, include_seven=variant == 'seven',
                                 literature_six=variant == 'literature_six')
    best, witness, count = None, None, 0
    for part in partitions(e):
        cost = surplus_lower(part, lower)
        count += 1
        if best is None or cost < best:
            best, witness = cost, part
    return best, witness, count


@lru_cache(None)
def layer_base(d):
    require(d >= 1, 'at least one layer required')
    background = sum((d-j)*floor_units(kernel(j)) for j in range(1, d))
    # Monotonicity of -kernel places the most negative field at the centre.
    left, right = (d-1)//2, d//2
    field = sum(floor_units(kernel(j)) for j in range(1, left+1))
    field += sum(floor_units(kernel(j)) for j in range(1, right+1))
    return background, field


def layer_lower(n, d, surplus_cost):
    require(1 <= d <= n, 'invalid layer count')
    bg, field = layer_base(d)
    return F(bg+(n-d)*field+surplus_cost, SCALE)


def original_ks_layer_lower(n, d):
    e = n-d
    return layer_lower(n, d, -e*(e+1)//2*SCALE)


def layer_threshold(n, upper, variant):
    excluded = []
    for d in range(n, 1, -1):
        if variant == 'pair':
            value = original_ks_layer_lower(n, d)
            part, count = (n-d,), 1
        else:
            cost, part, count = minimum_surplus(n-d, variant)
            value = layer_lower(n, d, cost)
        if value <= upper:
            break
        excluded.append((d, value, part, count))
    require(excluded, 'no layer exclusion established')
    last_d, last_value, part, count = excluded[-1]
    return {'diameter_strict_upper': last_d-1,
            'first_excluded_layer_count': last_d,
            'all_larger_counts_checked_through': n,
            'number_of_layer_counts_excluded': len(excluded),
            'last_exclusion_lower': str(last_value),
            'last_exclusion_margin': str(last_value-upper),
            'next_relaxation_lower': str(value),
            'surplus_partition_count': count,
            'minimizing_relaxation_partition': list(part),
            'scope': 'Global minimizers; half-open layers anchored at a diameter endpoint.'}


def pair_energy_interval(q):
    require(q > 0, 'colliding rational witness')
    a, b = q.numerator**3, q.denominator**3
    numerator, denominator = b*(b-2*a), a*a
    lo = numerator*SCALE//denominator
    hi = -((-numerator*SCALE)//denominator)
    return lo, hi


def rational_witness(n, source_root):
    path = source_root/'coordinates'/str(n)
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    require(len(rows) == n and all(len(row) == 3 for row in rows), 'bad coordinate table')
    # This is a rational configuration, not a claim about rounded exact minima.
    sigma_to_equilibrium = F(8908987181403393, 10**16)
    grid = 10**12
    points = [tuple(round(F(x)*sigma_to_equilibrium*grid) for x in row) for row in rows]
    lo = hi = 0
    for a, b in combinations(points, 2):
        q = F(sum((x-y)**2 for x, y in zip(a, b)), grid**2)
        l, u = pair_energy_interval(q)
        lo += l
        hi += u
    require(hi-lo <= n*(n-1)//2, 'energy interval rounding accounting failed')
    # A compact upper cutoff rounded upward to 9 decimal places.
    upper = F(-((-hi*10**9)//SCALE), 10**9)
    require(upper >= F(hi, SCALE), 'upper cutoff rounded in wrong direction')
    return upper, {'coordinate_integer_grid': grid,
                   'normalized_integer_coordinates': points,
                   'pair_sum_lower': str(F(lo, SCALE)),
                   'pair_sum_upper': str(F(hi, SCALE)),
                   'feasible_upper': str(upper)}


def v(r):
    return r**-12-2*r**-6


def endpoint_h(d, count=1024):
    require(d > 2 and count >= 1, 'invalid two-centre partition')
    width = (d-2)/count
    values = [v(1+k*width)+v(d-1-(k+1)*width) for k in range(count)]
    return min(values)


def endpoint_certificate(n, upper):
    # Endpoint removal and the sum of the two vertex-star budgets.
    lm2, lm1 = known_lower(n-2), known_lower(n-1)
    if max(lm2-(n-2), lm1-F(n-2, 2)) <= upper:
        return None
    def evaluate(ticks):
        d = F(ticks, 10000)
        h = endpoint_h(d)
        first = lm2+v(d)+(n-2)*h
        second = lm1+v(d)+F(n-2, 2)*h
        return max(first, second), h, 'remove_two' if first >= second else 'two_stars'
    lo, hi = 20000, 40000
    while evaluate(hi)[0] <= upper:
        hi *= 2
        require(hi <= 10000000, 'endpoint bracket not obtained')
    while hi-lo > 1:
        mid = (lo+hi)//2
        if evaluate(mid)[0] > upper:
            hi = mid
        else:
            lo = mid
    value, h, method = evaluate(hi)
    require(value > upper, 'invalid endpoint exclusion')
    return {'diameter_strict_upper': str(F(hi, 10000)), 'lower_at_endpoint': str(value),
            'strict_margin': str(value-upper), 'h_lower': str(h), 'partition_count': 1024,
            'method': method, 'scope': 'Every configuration in E <= feasible_upper.'}
