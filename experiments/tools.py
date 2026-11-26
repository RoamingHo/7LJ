#!/usr/bin/env python3
"""Exact algebraic D5h LJ7 candidate, root isolation, and local-certificate link.

Run with the existing Python environment containing SymPy.  Every acceptance
check is exact and remains active under python -O.  This writes only the new
results/exact_candidate.json artifact; no global optimality is claimed.
"""

from decimal import Decimal, localcontext
from fractions import Fraction as F
from functools import reduce
from math import gcd, lcm
from pathlib import Path
import json
import sys

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.certificate import Interval, evaluate


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(value):
    value = F(value)
    with localcontext() as context:
        context.prec = 75
        display = str(Decimal(value.numerator)/Decimal(value.denominator))
    return {"numerator": str(value.numerator), "denominator": str(value.denominator),
            "decimal_approximation": display}


def interval_record(value):
    return {"lower": record(value[0]), "upper": record(value[1])}


def trim(poly):
    poly = list(poly)
    while len(poly) > 1 and poly[-1] == 0:
        poly.pop()
    return poly


def primitive_preserving_sign(poly):
    poly = trim([F(x) for x in poly])
    denominator = lcm(*(x.denominator for x in poly))
    integers = [x.numerator*(denominator//x.denominator) for x in poly]
    divisor = reduce(gcd, (abs(x) for x in integers), 0)
    return [F(x//divisor) for x in integers] if divisor else [F(0)]


def remainder(left, right):
    result = trim([F(x) for x in left])
    right = trim(right)
    require(right != [0], "Polynomial division by zero")
    while result != [0] and len(result) >= len(right):
        shift = len(result)-len(right)
        factor = result[-1]/right[-1]
        for i, coefficient in enumerate(right):
            result[shift+i] -= factor*coefficient
        result = trim(result)
    return result


def sturm_chain(poly):
    first = primitive_preserving_sign(poly)
    second = primitive_preserving_sign([i*first[i] for i in range(1, len(first))])
    chain = [first, second]
    while len(chain[-1]) > 1:
        next_poly = primitive_preserving_sign([-x for x in remainder(chain[-2], chain[-1])])
        require(next_poly != [0], "Polynomial is not square-free")
        chain.append(next_poly)
    return chain


def polynomial_value(poly, x):
    value = F(0)
    for coefficient in reversed(poly):
        value = value*x+coefficient
    return value


def sign(value):
    return int(value > 0)-int(value < 0)


def signs_at(chain, endpoint):
    if endpoint == '+infinity':
        return [sign(poly[-1]) for poly in chain]
    if endpoint == '-infinity':
        return [sign(poly[-1])*(-1)**(len(poly)-1) for poly in chain]
    return [sign(polynomial_value(poly, endpoint)) for poly in chain]


def variations(signs):
    nonzero = [x for x in signs if x]
    return sum(a != b for a, b in zip(nonzero, nonzero[1:]))


def root_count(chain, left, right):
    return variations(signs_at(chain, left))-variations(signs_at(chain, right))


def sturm_record(chain, endpoints):
    return {
        "coefficient_order": "ascending powers",
        "chain_integer_coefficients": [[str(int(x)) for x in p] for p in chain],
        "signs": {name: signs_at(chain, value) for name, value in endpoints.items()},
        "variations": {name: variations(signs_at(chain, value)) for name, value in endpoints.items()},
    }


def modular_remainder(left, right, prime):
    result = trim([int(x) % prime for x in left])
    right = trim([int(x) % prime for x in right])
    require(right != [0], "Modular polynomial division by zero")
    inverse = pow(right[-1], -1, prime)
    while result != [0] and len(result) >= len(right):
        shift = len(result)-len(right)
        factor = result[-1]*inverse % prime
        for i, coefficient in enumerate(right):
            result[shift+i] = (result[shift+i]-factor*coefficient) % prime
        result = trim(result)
    return result


def modular_product(left, right, modulus, prime):
    result = [0]*(len(left)+len(right)-1)
    for i, x in enumerate(left):
        for j, y in enumerate(right):
            result[i+j] = (result[i+j]+x*y) % prime
    return modular_remainder(result, modulus, prime)


def modular_x_power(exponent, modulus, prime):
    result, base = [1], [0, 1]
    while exponent:
        if exponent % 2:
            result = modular_product(result, base, modulus, prime)
        exponent //= 2
        if exponent:
            base = modular_product(base, base, modulus, prime)
    return result


def modular_gcd(left, right, prime):
    while right != [0]:
        left, right = right, modular_remainder(left, right, prime)
    inverse = pow(left[-1], -1, prime)
    return [(x*inverse) % prime for x in left]


def irreducibility_certificate(poly):
    prime, degree = 23, len(poly)-1
    require(degree == 10 and int(poly[-1]) % prime != 0, "Modular degree changed")
    reduced = [int(x) % prime for x in poly]
    final = modular_x_power(prime**degree, reduced, prime)
    require(final == [0, 1], "Frobenius identity failed")
    tests = []
    for power in (2, 5):
        remainder_x = modular_x_power(prime**power, reduced, prime)
        difference = remainder_x + [0]*max(0, 2-len(remainder_x))
        difference[1] = (difference[1]-1) % prime
        divisor = modular_gcd(reduced, trim(difference), prime)
        require(divisor == [1], "Proper Frobenius divisor test failed")
        tests.append({"frobenius_power": power, "x_power_remainder": remainder_x,
                      "gcd_with_reduced_polynomial": divisor})
    return {"prime": prime, "degree": degree,
            "reduced_polynomial_coefficients_ascending": reduced,
            "x_to_prime_power_degree_remainder": final,
            "proper_divisor_tests": tests,
            "irreducible_modulo_23_and_over_Q": True}


def add(a, b):
    return a[0]+b[0], a[1]+b[1]


def neg(a):
    return -a[1], -a[0]


def sub(a, b):
    return add(a, neg(b))


def mul(a, b):
    products = [x*y for x in a for y in b]
    return min(products), max(products)


def point(x):
    return F(x), F(x)


def scale(a, x):
    return mul(a, point(x))


def positive_power(a, exponent):
    require(a[0] > 0, "Power input must be strictly positive")
    if exponent >= 0:
        return a[0]**exponent, a[1]**exponent
    return a[1]**exponent, a[0]**exponent


def divide(a, b):
    require(b[0] > 0, "Division denominator must be positive")
    return mul(a, (1/b[1], 1/b[0]))


def integer_nth_root(number, degree):
    require(number >= 0 and degree >= 1, "Invalid integer root")
    low, high = 0, 1 << ((number.bit_length()+degree-1)//degree)
    while high-low > 1:
        middle = (low+high)//2
        if middle**degree <= number:
            low = middle
        else:
            high = middle
    return high if high**degree == number else low


def positive_root(a, degree, bits=192):
    require(a[0] > 0, "Root input must be strictly positive")
    denominator = 1 << bits
    scaled = denominator**degree
    lower = integer_nth_root((a[0].numerator*scaled)//a[0].denominator, degree)
    upper = integer_nth_root((a[1].numerator*scaled)//a[1].denominator, degree)
    if upper**degree*a[1].denominator != a[1].numerator*scaled:
        upper += 1
    result = F(lower, denominator), F(upper, denominator)
    require(result[0]**degree <= a[0] and result[1]**degree >= a[1], "Root enclosure failed")
    return result


def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def matrix_product(left, right):
    result = []
    for row in left:
        output_row = []
        for column in transpose(right):
            value = point(0)
            for x, y in zip(row, column):
                value = add(value, mul(x, y))
            output_row.append(value)
        result.append(output_row)
    return result


def abs_upper(interval):
    return max(abs(interval[0]), abs(interval[1]))


def verify_local_link(coordinates, energy_interval):
    data = json.loads((ROOT/'results/local_certificate_7.json').read_text())
    center = [F(entry['fraction']) for entry in data['center']]
    radius = F(data['box_uniform_radius']['fraction'])
    dimension = len(center)
    require(data['particles'] == 7 and dimension == 15 and radius > 0, "Unexpected local certificate")
    box = [(x-radius, x+radius) for x in center]
    require(all(lo < qlo <= qhi < hi for (lo, hi), (qlo, qhi) in zip(box, coordinates)),
            "Algebraic candidate enclosure is not strictly inside the local box")
    require(box[0][0] > 0 and box[2][0] > 0, "Local rigid-motion gauge is singular")
    _, gradient_center, _, _ = evaluate([Interval(x) for x in center])
    _, _, hessian, minimum_q = evaluate([Interval(*entry) for entry in box])
    require(minimum_q > 0, "Local box might contain a collision")
    H = [[(x.lo, x.hi) for x in row] for row in hessian]
    P = [[F(x) for x in row] for row in data['rational_preconditioner_P']]
    require(len(P) == dimension and all(len(row) == dimension for row in P), "Invalid preconditioner size")
    require(all(P[i][i] > 0 for i in range(dimension)) and
            all(P[i][j] == 0 for i in range(dimension) for j in range(i)), "Preconditioner is not proved nonsingular")
    PI = [[point(x) for x in row] for row in P]
    transformed = matrix_product(transpose(PI), matrix_product(H, PI))
    margin = min(transformed[i][i][0]-sum(abs_upper(transformed[i][j])
                 for j in range(dimension) if j != i) for i in range(dimension))
    require(margin > 0, "Local Hessian positive definiteness did not reverify")
    C = [[sum(P[i][k]*P[j][k] for k in range(dimension)) for j in range(dimension)]
         for i in range(dimension)]
    CI = [[point(x) for x in row] for row in C]
    CH = matrix_product(CI, H)
    row_norms = [sum(abs_upper(sub(point(int(i == j)), CH[i][j])) for j in range(dimension))
                 for i in range(dimension)]
    contraction = max(row_norms)
    correction = [row[0] for row in matrix_product(CI, [[(x.lo, x.hi)] for x in gradient_center])]
    image_radii = [abs_upper(delta)+radius*k for delta, k in zip(correction, row_norms)]
    require(contraction < 1 and max(image_radii) < radius, "Local Banach conditions did not reverify")
    reported_energy = data['unique_local_minimum_energy_interval']
    old_interval = F(reported_energy['lower']['fraction']), F(reported_energy['upper']['fraction'])
    require(old_interval[0] <= energy_interval[0] <= energy_interval[1] <= old_interval[1],
            "Algebraic energy enclosure is not contained in the reported local energy enclosure")
    return {
        "source": "results/local_certificate_7.json",
        "box_radius": record(radius),
        "maximum_coordinate_distance_from_saved_center": record(max(abs_upper(sub(q, point(c))) for q, c in zip(coordinates, center))),
        "candidate_coordinate_intervals_strictly_inside_box": True,
        "fresh_hessian_gershgorin_margin": record(margin),
        "fresh_contraction_upper_bound": record(contraction),
        "fresh_maximum_image_radius": record(max(image_radii)),
        "local_uniqueness_and_positive_definiteness_recomputed": True,
        "algebraic_energy_interval_inside_previous_local_interval": True,
        "identification": "The algebraic candidate is a full stationary point by the symmetry argument, lies inside this box, and is therefore its unique certified local minimum modulo rigid motions.",
    }


def integer_coefficients(poly):
    return [F(int(poly.nth(i))) for i in range(poly.degree()+1)]


def primitive_positive(poly):
    primitive = poly.clear_denoms()[1].primitive()[1]
    return -primitive if primitive.LC() < 0 else primitive


def main():
    t, e = sp.symbols('t e')
    A = sp.Rational(18, 25)+10/(1+t)**6+1/(4096*t**6)
    B = 2+10/(1+t)**3+1/(64*t**3)
    raw_derivative = sp.cancel(2*sp.diff(B, t)*A-B*sp.diff(A, t))
    shape_num, shape_den = sp.fraction(raw_derivative)
    shape_poly = primitive_positive(sp.Poly(shape_num, t))
    require(shape_poly.degree() == 10, "Shape polynomial was not correctly cancelled")
    require(sp.cancel(raw_derivative+3*shape_poly.as_expr()/(25600*t**7*(t+1)**7)) == 0,
            "Shape derivative identity failed")
    sqrt5 = sp.sqrt(5)
    alpha, beta = (5-sqrt5)/2, (5+sqrt5)/2
    require(sp.simplify(5*(alpha**-3+beta**-3)-2) == 0 and
            sp.simplify(5*(alpha**-6+beta**-6)-sp.Rational(18, 25)) == 0,
            "Pentagon sqrt(5) cancellation failed")

    energy = sp.cancel(-B**2/A)
    numerator, denominator = sp.fraction(energy)
    require(sp.gcd(shape_poly, sp.Poly(denominator, t)).degree() == 0, "Energy denominator has a shape root")
    energy_poly = primitive_positive(sp.Poly(sp.resultant(shape_poly.as_expr(), e*denominator-numerator, t), e))
    require(energy_poly.degree() == 10, "Unexpected energy resultant degree")
    require(sp.gcd(energy_poly, energy_poly.diff()).degree() == 0, "Energy resultant has repeated roots")

    p = integer_coefficients(shape_poly)
    shape_sturm = sturm_chain(p)
    require(root_count(shape_sturm, F(0), '+infinity') == 1, "Shape polynomial does not have exactly one positive root")
    low, high = F(36, 100), F(37, 100)
    require(polynomial_value(p, low) < 0 < polynomial_value(p, high), "Initial shape bracket failed")
    for _ in range(192):
        middle = (low+high)/2
        value = polynomial_value(p, middle)
        require(value != 0, "Unexpected exact rational shape root")
        if value < 0:
            low = middle
        else:
            high = middle
    require(root_count(shape_sturm, low, high) == 1, "Shape isolating interval failed")
    shape_interval = low, high
    one_plus_t = add(point(1), shape_interval)
    AI = add(add(point(F(18, 25)), scale(positive_power(one_plus_t, -6), 10)),
             scale(positive_power(shape_interval, -6), F(1, 4096)))
    BI = add(add(point(2), scale(positive_power(one_plus_t, -3), 10)),
             scale(positive_power(shape_interval, -3), F(1, 64)))
    energy_interval = neg(divide(positive_power(BI, 2), AI))
    ep = integer_coefficients(energy_poly)
    energy_sturm = sturm_chain(ep)
    require(root_count(energy_sturm, *energy_interval) == 1, "Energy interval does not isolate one resultant root")
    shape_irreducibility = irreducibility_certificate(p)
    energy_irreducibility = irreducibility_certificate(ep)

    R = positive_root(divide(AI, BI), 6)
    H = mul(R, positive_root(shape_interval, 2))
    s = positive_root(point(5), 2)
    u = positive_root(scale(add(point(5), s), F(1, 8)), 2)
    v = positive_root(scale(sub(point(5), s), F(1, 8)), 2)
    ru = mul(R, u)
    ring_y = neg(scale(mul(R, s), F(1, 2)))
    pole_y = neg(scale(mul(R, sub(s, point(1))), F(1, 4)))
    coordinates = [scale(ru, 2), ru, scale(mul(R, sub(point(5), s)), F(1, 4)),
                   mul(R, add(u, v)), ring_y, point(0),
                   mul(R, sub(u, v)), ring_y, point(0),
                   ru, pole_y, neg(H), ru, pole_y, H]
    local_link = verify_local_link(coordinates, energy_interval)

    shape_endpoints = {'negative_infinity': '-infinity', 'zero': F(0),
                       'positive_infinity': '+infinity', 'isolating_lower': low, 'isolating_upper': high}
    energy_endpoints = {'negative_infinity': '-infinity', 'positive_infinity': '+infinity',
                        'isolating_lower': energy_interval[0], 'isolating_upper': energy_interval[1]}
    report = {
        "schema": "lj7.exact-d5h-candidate.v1",
        "potential": "sum over unordered pairs of r^-12-2*r^-6",
        "status": "Exactly defined algebraic D5h candidate, rigorously isolated and identified with the certified local minimum; global optimality is not established.",
        "shape_variable": "t=H^2/R^2>0",
        "rational_functions": {"A": str(A), "B": str(B), "scale": "R^6=A(t)/B(t)", "optimized_energy": str(energy)},
        "sqrt5_cancels_from_A_and_B": True,
        "shape_polynomial_coefficients_ascending": [str(int(x)) for x in p],
        "shape_polynomial_degree": 10,
        "shape_irreducibility_certificate": shape_irreducibility,
        "shape_root_interval": interval_record(shape_interval),
        "shape_positive_root_count": root_count(shape_sturm, F(0), '+infinity'),
        "shape_negative_root_count": root_count(shape_sturm, '-infinity', F(0)),
        "shape_sturm_certificate": sturm_record(shape_sturm, shape_endpoints),
        "shape_derivative_identity": "2*B'(t)*A(t)-B(t)*A'(t)=-3*P(t)/(25600*t^7*(t+1)^7)",
        "energy_derivative_sign": "For t>0, derivative(-B(t)^2/A(t)) has the sign of P(t).",
        "unique_global_minimum_within_D5h_R_H_family": True,
        "family_endpoint_energy_limits": {"t_to_zero": record(-1), "t_to_infinity": record(F(-50, 9))},
        "energy_polynomial_coefficients_ascending": [str(int(x)) for x in ep],
        "energy_polynomial_degree": 10,
        "energy_irreducibility_certificate": energy_irreducibility,
        "energy_algebraic_degree_over_Q": 10,
        "energy_polynomial_definition": "Primitive positive-leading resultant in t of P(t) and e*denominator(-B^2/A)-numerator(-B^2/A)",
        "energy_denominator_coprime_to_shape_polynomial": True,
        "energy_root_interval": interval_record(energy_interval),
        "energy_real_root_count": root_count(energy_sturm, '-infinity', '+infinity'),
        "energy_sturm_certificate": sturm_record(energy_sturm, energy_endpoints),
        "equatorial_radius_interval": interval_record(R),
        "pole_height_interval": interval_record(H),
        "gauge_coordinate_intervals": [interval_record(x) for x in coordinates],
        "local_certificate_link": local_link,
        "full_stationarity_argument": [
            "At an equatorial atom, vertical-plane and equatorial-plane reflection symmetries force its gradient to be radial; fivefold rotation makes the radial coefficient common to all five atoms.",
            "At each pole, fivefold symmetry forces its gradient onto the axis; reflection in the equatorial plane makes the two axial coefficients opposite.",
            "The two unrestricted family derivatives with respect to R and H are therefore five times the common radial coefficient and twice the upper axial coefficient.",
            "The scale equation R^6=A/B and isolated shape equation P(t)=0 make both family derivatives vanish; the change of variables (R,t) to (R,H) is nonsingular for R,t>0.",
            "Thus the algebraic configuration has zero full gradient. Its gauged enclosure lies strictly inside the reverified local uniqueness box, identifying it with that unique local minimum."
        ],
        "excluded_spurious_roots": [
            "t<=0 is physically inadmissible because t=H^2/R^2 with noncolliding R,H>0.",
            "The shape derivative is cancelled before polynomial extraction, removing denominator-clearing factors at t=0 and t=-1.",
            "The physical energy is the rational function of the unique positive shape root, and its exact interval isolates one root of the degree-ten energy resultant. Other resultant roots are not selected."
        ],
        "arithmetic": "Exact symbolic rational identities, Fraction Sturm sequences and interval operations, and outward rational nth-root bounds; explicit require checks stay active under python -O.",
        "all_exact_checks_passed": True,
    }
    output = ROOT/'results/exact_candidate.json'
    output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('Shape polynomial degree:', len(p)-1, 'positive roots:', report['shape_positive_root_count'])
    print('t interval:', record(low)['decimal_approximation'], record(high)['decimal_approximation'])
    print('Exact energy interval:', record(energy_interval[0])['decimal_approximation'], record(energy_interval[1])['decimal_approximation'])
    print('R interval:', record(R[0])['decimal_approximation'], record(R[1])['decimal_approximation'])
    print('H interval:', record(H[0])['decimal_approximation'], record(H[1])['decimal_approximation'])
    print('Local-box inclusion and local uniqueness conditions reverified; wrote', output)


if __name__ == '__main__':
    main()
