"""Exact rational/Q(sqrt(5)) audit of the C5 apex-axis domain.

This checks domain constants and candidate identities, not global optimality.
It uses only the standard library, makes no floating-point decisions, performs
no optimization, and never modifies an engine or an existing certificate.
"""
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import argparse
import json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def loss(q):
    return (q**-3-1)**2


def add(x, y):
    return x[0]+y[0], x[1]+y[1]


def scale(x, a):
    return x[0]*a, x[1]*a


def sub(x, y):
    return add(x, scale(y, -1))


def mul(x, y):
    return x[0]*y[0]+5*x[1]*y[1], x[0]*y[1]+x[1]*y[0]


def power(x, n):
    if n < 0:
        norm = x[0]*x[0]-5*x[1]*x[1]
        require(norm != 0, 'zero Q(sqrt(5)) denominator')
        x, n = (x[0]/norm, -x[1]/norm), -n
    result = F(1), F(0)
    for _ in range(n):
        result = mul(result, x)
    return result


def verify():
    upper, radius = F('-16.5053835'), F(73, 50)
    B, cap = radius**2, F(237, 500)
    a, S, h, T = F(229, 250), F(243, 200), F(11, 25), F(563, 500)
    budget = upper+21-5*loss(B)
    margins = {
        'budget_cap': cap-budget,
        'short_lower_loss': loss(a*a)-cap,
        'short_upper_loss': loss(S*S)-cap,
        'height_packing': 36*(a*a-4*h*h)-(2*S-a)**2,
        'transverse_upper': T*T-(S*S-a*a/4),
    }
    require(0 < budget < cap < 1 and 0 < 2*h < a < 1 < T < S < radius,
            'invalid ordered domain constants')
    require(all(value > 0 for value in margins.values()), 'domain margin failed')
    require(margins['height_packing'] == F(1771, 50000), 'packing identity failed')

    ring = tuple(range(2, 7))
    short_cycle = {tuple(sorted((ring[k], ring[(k+1) % 5]))) for k in range(5)}
    long_pairs = set(combinations(ring, 2))-short_cycle
    pairs = tuple(combinations(range(7), 2))
    mask = sum(1 << i for i, pair in enumerate(pairs) if pair in long_pairs)
    require(mask == 733184 and len(long_pairs) == len(short_cycle) == 5,
            'cycle labeling changed')
    for vertex in ring:
        require(sum(vertex in pair for pair in short_cycle) == 2 and
                sum(vertex in pair for pair in long_pairs) == 2,
                'ring graph is not two complementary five-cycles')

    # Q(sqrt(5)): u^2=(5+s)/8, v^2=(5-s)/8, uv=s/4, u,v>0.
    u2, v2, uv = (F(5, 8), F(1, 8)), (F(5, 8), F(-1, 8)), (F(0), F(1, 4))
    require(mul(u2, v2) == mul(uv, uv), 'positive radical product identity failed')
    ys = [(F(1), F(0)), (F(-1, 4), F(1, 4)),
          (F(-1, 4), F(-1, 4)), (F(-1, 4), F(-1, 4)),
          (F(-1, 4), F(1, 4))]
    zs = [(0, 0), (1, 0), (0, 1), (0, -1), (-1, 0)]

    def zsquare(z):
        c, d = z
        return add(add(scale(u2, c*c), scale(v2, d*d)), scale(uv, 2*c*d))

    alpha, beta = (F(5, 2), F(-1, 2)), (F(5, 2), F(1, 2))
    for y, z in zip(ys, zs):
        require(add(mul(y, y), zsquare(z)) == (1, 0), 'candidate transverse radius differs')
    for i, j in combinations(range(5), 2):
        dy, dz = sub(ys[i], ys[j]), (zs[i][0]-zs[j][0], zs[i][1]-zs[j][1])
        expected = alpha if (i+2, j+2) in short_cycle else beta
        require(add(mul(dy, dy), zsquare(dz)) == expected, 'candidate ring distance failed')
    require(scale(add(power(alpha, -3), power(beta, -3)), 5) == (2, 0),
            'candidate attractive pentagon sum failed')
    require(scale(add(power(alpha, -6), power(beta, -6)), 5) == (F(18, 25), 0),
            'candidate repulsive pentagon sum failed')

    # Coarse rational enclosures suffice for domain membership. The existing
    # exact-candidate theorem supplies its positive shape root in this bracket.
    t0, t1, R0, R1 = F(9, 25), F(37, 100), F(21, 25), F(87, 100)
    A = lambda t: F(18, 25)+10/(1+t)**6+1/(4096*t**6)
    C = lambda t: 2+10/(1+t)**3+1/(64*t**3)
    require(R0**6 < A(t1)/C(t0) and A(t0)/C(t1) < R1**6,
            'coarse exact candidate radius bound failed')
    sqrt5_lo, sqrt5_hi = F(223, 100), F(56, 25)
    require(sqrt5_lo**2 < 5 < sqrt5_hi**2, 'sqrt(5) rational enclosure failed')
    short_q_bounds = {
        'apex_apex': (4*t0*R0**2, 4*t1*R1**2),
        'apex_ring': ((1+t0)*R0**2, (1+t1)*R1**2),
        'ring_short': ((5-sqrt5_hi)*R0**2/2, (5-sqrt5_lo)*R1**2/2),
    }
    require(all(a*a < lo <= hi < S*S for lo, hi in short_q_bounds.values()),
            'candidate short pair outside domain')
    require(B < (5+sqrt5_lo)*R0**2/2 and
            (5+sqrt5_hi)*R1**2/2 < 4*S*S,
            'candidate long pair outside domain')
    require(h < R0 < R1 < T, 'candidate anchor outside transverse bounds')

    return {
        'schema': 'lj7.c5-apex-axis-domain-audit.v1',
        'scope': 'All finite-energy E<=upper configurations whose long graph is C5 plus two isolates.',
        'requires_stationarity': False,
        'requires_subcluster_lower_bound': False,
        'global_LJ7_optimality_proved': False,
        'arithmetic': 'Exact Fraction and Q(sqrt(5)); no floating point or numerical optimization.',
        'upper': str(upper), 'long_radius': str(radius), 'long_q_lower': str(B),
        'short_loss_budget': str(budget), 'short_loss_budget_cap': str(cap),
        'rmin': str(a), 'short_rmax': str(S), 'p2_rho_lower': str(h),
        'transverse_rmax': str(T), 'whole_x_span_upper': str(2*S-a),
        'strict_margins': {key: str(value) for key, value in margins.items()},
        'short_cycle': [list(pair) for pair in sorted(short_cycle)],
        'long_pairs': [list(pair) for pair in sorted(long_pairs)],
        'long_mask_combinations_lex_order': mask,
        'static_box': {
            'd': [str(a), str(S)], 'x2': [str(a-S), str(S/2)],
            'y2': [str(h), str(T)], 'x3_x4_x5_x6': [str(a-S), str(S)],
            'y3_y4_y5_y6_z4_z5_z6': [str(-T), str(T)], 'z3': ['0', str(T)],
        },
        'dynamic_constraints': [
            'd-S <= x_i <= S for i=2..6; x2<=d/2; z3>=0',
            'rho_i^2=y_i^2+z_i^2 <= y2^2 for i=2..6',
            'a^2 <= q_ij <= S^2 for every one of the 16 short pairs',
            'B <= q_ij <= 4*S^2 for every one of the 5 long pairs',
            'rho_i^2 <= S^2-max(x_i^2,(d-x_i)^2) for i=2..6',
            'y2^2 <= S^2-d^2/4',
        ],
        'coverage_ties': 'Keep weak max-rho, x2<=d/2 and z3>=0 inequalities; choose any tied ring anchor.',
        'candidate': {
            'parameters': 't is the existing isolated positive shape root; R=(A(t)/C(t))^(1/6); H=R*sqrt(t)>0.',
            'coordinates': [
                '(0,0,0)', '(2*H,0,0)', '(H,R,0)',
                '(H,(sqrt(5)-1)*R/4,u*R)', '(H,-(sqrt(5)+1)*R/4,v*R)',
                '(H,-(sqrt(5)+1)*R/4,-v*R)', '(H,(sqrt(5)-1)*R/4,-u*R)',
            ],
            'radicals': 'u=sqrt((5+sqrt(5))/8)>0; v=sqrt((5-sqrt(5))/8)>0',
            'short_squared_distance_types': {key: list(map(str, value)) for key, value in short_q_bounds.items()},
            'all_ring_radii_equal': True, 'x2_equals_d_over_2': True, 'z3_strictly_positive': True,
            'candidate_gauge_templates': 1,
            'scope': 'Exact distance identities and domain inclusion conditional on the existing isolated shape-root theorem. No new local certificate is claimed.',
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', type=Path, help='Compare an existing saved audit with fresh exact checks.')
    args = parser.parse_args()
    result = verify()
    if args.check is not None:
        require(result == json.loads(args.check.read_text()), 'saved audit differs from fresh exact result')
        print(json.dumps({'status': 'verified', 'global_LJ7_optimality_proved': False}))
    else:
        print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
