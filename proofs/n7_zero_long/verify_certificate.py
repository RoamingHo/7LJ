"""Standard-library, exact rational checker for a stationary LJ7 diameter bound."""
from fractions import Fraction as F
from itertools import combinations, permutations
from math import comb
from pathlib import Path
import argparse
import json
import time

HERE = Path(__file__).resolve().parent
PAIRS = tuple(combinations(range(7), 2))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def multiply(a, b):
    out = [F(0)]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i+j] += x*y
    return out


def coefficients(data):
    D, z, lam, gamma = (F(data[k]) for k in ('D', 'z', 'lambda', 'gamma'))
    B = D*D
    require(D > 1 and 0 < z < B and lam > -1 and gamma > 0, 'invalid scalar parameters')
    f = lambda q: (1+lam)*q**-6-(2+lam)*q**-3
    derivative = -6*(1+lam)*z**-7+3*(2+lam)*z**-4
    a = (f(B)-f(z)-derivative*(B-z))/(B-z)**2
    b, c = derivative-2*a*z, f(z)-derivative*z+a*z*z
    K = a+gamma
    require(K > 0, 'nonpositive quadratic coefficient')
    return B, z, lam, gamma, a, b, c, K


def contact_factor(values):
    B, z, lam, gamma, a, b, c, K = values
    p = [1+lam, F(0), F(0), -(2+lam), F(0), F(0), -c, -b, -a]
    divisor = multiply([z*z, -2*z, F(1)], [B, F(-1)])
    quotient = [F(0)]*6
    for degree in range(8, 2, -1):
        q = p[degree]/divisor[3]
        quotient[degree-3] = q
        for j, value in enumerate(divisor):
            p[degree-3+j] -= q*value
    require(not any(p), 'contact polynomial has a nonzero remainder')
    return quotient


def bernstein_positive(polynomial, lo, hi):
    """Exact sufficient nonnegativity certificate on a closed interval."""
    n = len(polynomial)-1
    monomial = [sum(polynomial[j]*comb(j, k)*lo**(j-k)*(hi-lo)**k
                    for j in range(k, n+1)) for k in range(n+1)]
    beta = [sum(monomial[k]*F(comb(i, k), comb(n, k)) for k in range(i+1))
            for i in range(n+1)]
    return min(beta) >= 0


def verify_scalar(values, subdivisions):
    B, z, lam, gamma, a, b, c, K = values
    factor = contact_factor(values)
    short = multiply([B, F(-1)], factor)
    short[6] -= gamma
    long = multiply([z*z, -2*z, F(1)], factor)
    long[6] -= gamma*B
    long[7] += gamma
    cut = (z+B)/2
    require(type(subdivisions) is int and 1 <= subdivisions <= 1024, 'invalid scalar subdivision count')
    for poly, lo, hi in ((short, F(0), cut), (long, cut, B)):
        for k in range(subdivisions):
            left = lo+(hi-lo)*F(k, subdivisions)
            right = lo+(hi-lo)*F(k+1, subdivisions)
            require(bernstein_positive(poly, left, right), 'scalar polynomial positivity unproved')


def matrices(mask, dual, values):
    B, z, lam, gamma, a, b, c, K = values
    require(type(mask) is int and 0 <= mask < 1 << 21, 'invalid graph mask')
    require(len(dual) == 8, 'incorrect dual dimension')
    v, t = list(map(F, dual[:7])), F(dual[7])
    T = [[F(0) for _ in range(7)] for _ in range(7)]
    M = [[F(0) for _ in range(7)] for _ in range(7)]
    for index, (i, j) in enumerate(PAIRS):
        T[i][j] = T[j][i] = B if mask >> index & 1 else z
        M[i][j] = M[j][i] = (gamma*T[i][j]-b/2)/K
    row = [sum(r)/7 for r in M]
    mean = sum(row)/7
    G = [[-(M[i][j]-row[i]-row[j]+mean)/2 for j in range(7)] for i in range(7)]
    total_v = sum(v)
    C = [[7*v[i]*(i == j)-v[i]-v[j]+total_v/7+t*(F(i == j)-F(1, 7))
          for j in range(7)] for i in range(7)]
    H = [[G[i][j]-C[i][j]/2 for j in range(7)] for i in range(7)]
    constant = gamma*sum(T[i][j]**2 for i, j in PAIRS)+21*c-K*sum(M[i][j]**2 for i, j in PAIRS)
    base = (2*sum(x*x for r in G for x in r)-2*sum(C[i][j]*G[i][j] for i in range(7) for j in range(7))
            -7*sum(x*x for x in v)-t*t)
    return H, base, constant


def positive_inertia(matrix, upper):
    """Sylvester congruence elimination; no approximate eigenvalues are used."""
    a = [[x-(upper if i == j else 0) for j, x in enumerate(row)]
         for i, row in enumerate(matrix)]
    positive = 0
    for k in range(7):
        pivot = a[k][k]
        require(pivot != 0, 'zero inertia pivot: provide a different rational spectral bound')
        positive += pivot > 0
        for i in range(k+1, 7):
            for j in range(i, 7):
                a[i][j] -= a[i][k]*a[k][j]/pivot
                a[j][i] = a[i][j]
    return positive


def graph_bound(record, values):
    H, base, constant = matrices(record['mask'], record['dual'], values)
    upper = [F(x) for x in record['eigenvalue_upper_bounds']]
    require(len(upper) == 3 and all(x > 0 for x in upper), 'three positive spectral bounds required')
    for k, endpoint in enumerate(upper):
        require(positive_inertia(H, endpoint) <= k, 'eigenvalue upper bound failed exact inertia test')
    return values[-1]*(base-2*sum(x*x for x in upper))+constant


def verify_graph_cover(masks):
    """Exhaust all labelled masks via permutations, independent of atlas generation."""
    require(len(masks) == len(set(masks)) and all(type(m) is int and 0 <= m < 1 << 21 for m in masks),
            'duplicate or invalid representatives')
    index = {pair: k for k, pair in enumerate(PAIRS)}
    labelled = set()
    for p in permutations(range(7)):
        weights = [1 << index[tuple(sorted((p[i], p[j])))] for i, j in PAIRS]
        tables = []
        for group in range(3):
            table = [0]*128
            for bits in range(1, 128):
                low = bits & -bits
                table[bits] = table[bits-low] | weights[7*group+low.bit_length()-1]
            tables.append(table)
        a, b, c = tables
        labelled.update(a[m & 127] | b[(m >> 7) & 127] | c[m >> 14] for m in masks)
    require(len(labelled) == 1 << 21, 'graph representatives do not cover every labelled graph')
    return len(labelled)


def verify_certificate(data):
    require(data['schema'] == 'lj7.exact-spectral-diameter.v1', 'wrong certificate schema')
    require(data['particles'] == 7 and data['ambient_dimension'] == 3, 'wrong dimension')
    require(data['requires_stationarity'] is True and data['global_LJ7_optimality_proved'] is False,
            'wrong theorem scope')
    values = coefficients(data['parameters'])
    require(F(data['upper']) == F('-16.5053835'), 'wrong target')
    verify_scalar(values, data['scalar_subdivisions'])
    count = verify_graph_cover([r['mask'] for r in data['graphs']])
    bounds = [graph_bound(row, values) for row in data['graphs']]
    lower = min(bounds)
    require(lower > F(data['upper']), 'no strict global exclusion margin')
    return {'D': data['parameters']['D'], 'labelled_graphs': count, 'representatives': len(bounds),
            'lower': str(lower), 'lower_display': float(lower),
            'margin_display': float(lower-F(data['upper'])), 'requires_stationarity': True,
            'requires_L5_or_L6': False, 'global_LJ7_optimality_proved': False}


def load_verified_certificate(path=HERE/'atlas.json'):
    return verify_certificate(json.loads(Path(path).read_text()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('certificate', nargs='?', type=Path, default=HERE/'atlas.json')
    args = parser.parse_args()
    start = time.monotonic()
    result = load_verified_certificate(args.certificate)
    print(json.dumps({'verified': True,
                      'seconds': time.monotonic()-start, **result}), flush=True)


if __name__ == '__main__':
    main()
