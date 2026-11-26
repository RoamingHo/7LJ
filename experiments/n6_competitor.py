"""Exact local receipt for the six-atom C2v capped-triangular-bipyramid competitor.

Numerical root finding proposes rational centres only. Directed rational
intervals prove existence, uniqueness and positivity afterwards.
"""
from fractions import Fraction as F
from itertools import permutations
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.certificate import (
    Interval as I, evaluate, propose_preconditioner, interval_matmul, transpose,
)
from experiments.tools import positive_root


def require(ok, message):
    if not ok:
        raise ValueError(message)


def interval(x):
    return {'lower': str(x.lo), 'upper': str(x.hi)}


# Squared-distance orbits, as sums of squares of linear forms in (a,b,c,d,e).
FORMS = [
    (1, [(2, 0, 0, 0, 0)]),
    (4, [(1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0)]),
    (4, [(1, 0, 0, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1)]),
    (1, [(0, 0, 2, 0, 0)]), (1, [(0, 0, 0, 0, 2)]),
    (2, [(0, 1, 0, -1, 0), (0, 0, 1, 0, -1)]),
    (2, [(0, 1, 0, -1, 0), (0, 0, 1, 0, 1)]),
]


def family_evaluate(x):
    energy, gradient = I(0), [I(0) for _ in x]
    hessian = [[I(0) for _ in x] for _ in x]
    for multiplicity, forms in FORMS:
        matrix = [[sum(f[i]*f[j] for f in forms) for j in range(5)] for i in range(5)]
        mx = [sum((matrix[i][j]*x[j] for j in range(5)), I(0)) for i in range(5)]
        q = sum((x[i]*mx[i] for i in range(5)), I(0))
        require(q.lo > 0, 'family box contains a collision')
        energy += multiplicity*(q**-6-2*q**-3)
        w1, w2 = 6*(q**-4-q**-7), 42*q**-8-24*q**-5
        for i in range(5):
            gradient[i] += multiplicity*w1*2*mx[i]
            for j in range(5):
                hessian[i][j] += multiplicity*(4*w2*mx[i]*mx[j]+2*w1*matrix[i][j])
    return energy, gradient, hessian


def certify(center, radius, evaluator):
    center, radius = [F(x) for x in center], F(radius)
    require(radius > 0 and len(center) in (5, 12), 'invalid local box')
    box = [I(x-radius, x+radius) for x in center]
    if len(center) == 12:
        require(box[0].lo > 0 and box[2].lo > 0, 'singular Cartesian gauge')
    energy, gradient, h0 = evaluator([I(x) for x in center])[:3]
    hessian = evaluator(box)[2]
    p = propose_preconditioner(h0)
    ph = interval_matmul(transpose(p), interval_matmul(hessian, p))
    margin = min(ph[i][i].lo-sum(ph[i][j].abs_upper() for j in range(len(center)) if j != i)
                 for i in range(len(center)))
    require(margin > 0, 'positive Hessian not certified')
    convexity = margin/sum(x*x for row in p for x in row)
    c = [[sum(p[i][k]*p[j][k] for k in range(len(center)))
          for j in range(len(center))] for i in range(len(center))]
    ch = interval_matmul(c, hessian)
    rows = [sum((int(i == j)-ch[i][j]).abs_upper() for j in range(len(center)))
            for i in range(len(center))]
    correction = [r[0].abs_upper() for r in interval_matmul(c, [[g] for g in gradient])]
    contraction = max(rows)
    require(contraction < 1 and max(z+radius*r for z, r in zip(correction, rows)) < radius,
            'strict Banach inclusion not certified')
    lower = energy.lo-sum(g.abs_upper()**2 for g in gradient)/(2*convexity)
    return {'center': [str(x) for x in center], 'radius': str(radius),
            'positive_hessian_margin': str(margin), 'strong_convexity_lower': str(convexity),
            'contraction_upper': str(contraction),
            'critical_radius_upper': str(max(correction)/(1-contraction)),
            'energy_interval': interval(I(lower, energy.hi)),
            'exactly_one_critical_point_in_box': True,
            'critical_point_is_strict_local_minimum': True}, box


def gauge(x):
    a, b, c, d, e = x
    q = b*b+c*c
    rho = I(*positive_root((q.lo, q.hi), 2))
    inv = rho**-1
    return [2*a, a, rho, a, (b*b-c*c)*inv, -2*b*c*inv,
            a, (b*d+c*e)*inv, (-c*d+b*e)*inv,
            a, (b*d-c*e)*inv, (-c*d-b*e)*inv]


def generate():
    import mpmath as mp
    import sympy as sp
    mp.mp.dps = 75
    variables = sp.symbols('a b c d e')
    q = [sum(sum(f[i]*variables[i] for i in range(5))**2 for f in forms)
         for _, forms in FORMS]
    energy = sum(m*(s**-6-2*s**-3) for (m, _), s in zip(FORMS, q))
    gradient = sp.lambdify(variables, [sp.diff(energy, x) for x in variables], 'mpmath')
    proposal = mp.findroot(gradient, ('.504', '.702', '.499', '-.243', '.822'),
                           tol=mp.mpf('1e-70'), maxsteps=30)
    center = [F(str(x)) for x in proposal]
    family, family_box = certify(center, F(1, 10**25), family_evaluate)
    a, b, c, d, e = family_box
    family_points = [(a, I(0), I(0)), (-a, I(0), I(0)),
                     (I(0), b, c), (I(0), b, -c), (I(0), d, e), (I(0), d, -e)]
    edges = set()
    for i in range(6):
        for j in range(i):
            distance = sum(((family_points[i][k]-family_points[j][k])**2 for k in range(3)), I(0))
            require(distance.hi < F(6, 5) or distance.lo > F(6, 5), 'contact graph is not isolated')
            if distance.hi < F(6, 5):
                edges.add(frozenset((i, j)))
    automorphisms = [p for p in permutations(range(6))
                     if {frozenset(p[i] for i in edge) for edge in edges} == edges]
    require(len(edges) == 12 and len(automorphisms) == 4, 'C2v graph symmetry changed')
    full_center_box = gauge([I(x) for x in center])
    full_center = [(x.lo+x.hi)/2 for x in full_center_box]
    local, full_box = certify(full_center, F(1, 10**8), evaluate)
    family_in_gauge = gauge(family_box)
    require(all(y.lo < x.lo <= x.hi < y.hi for x, y in zip(family_in_gauge, full_box)),
            'whole algebraic family box does not fit the Cartesian local box')
    # Independent orbit-count check at a rational point in the symmetric family.
    witness_center = [F(round(x*10**12), 10**12) for x in center]
    a, b, c, d, e = witness_center
    points = [(a, 0, 0), (-a, 0, 0), (0, b, c), (0, b, -c), (0, d, e), (0, d, -e)]
    exact_energy = sum(sum((points[i][k]-points[j][k])**2 for k in range(3))**-6
                       -2*sum((points[i][k]-points[j][k])**2 for k in range(3))**-3
                       for i in range(6) for j in range(i))
    computed = family_evaluate([I(x) for x in witness_center])[0]
    require(computed.lo <= exact_energy <= computed.hi and sum(m for m, _ in FORMS) == 15,
            'six-atom distance orbit energy identity failed')
    lo, hi = F(local['energy_interval']['lower']), F(local['energy_interval']['upper'])
    decimal_lo = F('-12.302927529580728482747833461462')
    decimal_hi = F('-12.302927529580728482747833461461')
    require(decimal_lo < lo <= hi < decimal_hi, 'local energy decimal bracket failed')
    octahedron = F(-3267, 257)
    require(octahedron < lo, 'octahedral witness does not beat the competitor')
    return {'schema': 'lj6.c2v-competing-local-minimum.v1', 'particles': 6,
            'name': 'C2v capped triangular bipyramid (equivalently a tetrahedron capped on two faces)',
            'potential': 'r^-12-2*r^-6',
            'symmetric_coordinates': ['(a,0,0)', '(-a,0,0)', '(0,b,c)', '(0,b,-c)', '(0,d,e)', '(0,d,-e)'],
            'symmetry_generators': ['x -> -x', 'z -> -z'],
            'exact_point_group': 'C2v',
            'point_group_proof': 'The rational family box fixes the 12-edge graph at q<6/5. Its graph automorphism group has exactly four elements. The two geometric reflections generate four actual symmetries, so there are no additional Euclidean symmetries.',
            'short_edge_graph': [sorted(edge) for edge in sorted(edges, key=lambda edge: sorted(edge))],
            'exact_algebraic_definition': 'The unique root of the five family-energy partial derivatives in the rational box certified below; all squared distances stay positive, so clearing denominators gives polynomial equations.',
            'squared_distance_orbits': [{'multiplicity': m, 'q': str(s)} for (m, _), s in zip(FORMS, q)],
            'family_stationarity_argument': 'The two reflections force zero x-force on the four x=0 atoms and zero z-force on the axial pair. The five family derivatives kill the remaining equatorial y/z forces and axial x-force. Translation invariance then kills the common axial y-force. Thus the family critical point is fully Cartesian stationary.',
            'family_certificate': family, 'cartesian_local_certificate': local,
            'whole_family_box_inside_cartesian_box': True,
            'energy_bracket_strict': [str(decimal_lo), str(decimal_hi)],
            'energy_approximation': str(mp.mpf(lo.numerator)/lo.denominator),
            'rational_feasible_coordinates': [[str(x) for x in point] for point in points],
            'rational_feasible_energy': str(exact_energy),
            'octahedron_feasible_energy': str(octahedron),
            'not_global_minimum_proved': True, 'global_minimum_proved': False,
            'trust_boundary': '75-digit numerical proposals are accepted only after 160-bit outward rational interval Hessian and strict Banach checks in both five-parameter and twelve-coordinate boxes. This identifies a strict local competitor and proves it is above an octahedral feasible witness; no N6 global closure is claimed.'}


if __name__ == '__main__':
    result = generate()
    path = ROOT/'results/n6_competing_candidate.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'output': str(path), 'particles': 6, 'energy': result['energy_approximation'],
                      'strict_local_minimum_proved': True, 'not_global_minimum_proved': True}))
