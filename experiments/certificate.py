#!/usr/bin/env python3
"""Strict local LJ5/LJ7 certificates; no claim outside their respective boxes.

Only the standard library is used. All acceptance comparisons use Fraction.
Decimal Cholesky proposes a rational preconditioner; its accuracy is not trusted.
Every interval operation rounds outwards onto a 160-bit binary grid.
"""

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction as F
import json
from pathlib import Path


BITS = 160
SCALE = 1 << BITS
CENTERS = {5: [F(x, 10**18) for x in (
    1626671568153954140, 813335784076977070, 578189461706699510,
    813335784076977070, -289094730853349755, 500726762038451669,
    813335784076977070, -289094730853349755, -500726762038451669,
)], 7: [F(x, 10**18) for x in (
    1620377709945509094, 810188854972754547, 588636658853673790,
    1310913104652276119, -952434121049420908, 0,
    309464605293232974, -952434121049420908, 0,
    810188854972754547, -263246316649033605, -511260198136543463,
    810188854972754547, -263246316649033605, 511260198136543463,
)]}
RADII = {5: F(1, 10000), 7: F(1, 100000)}


def active_coordinates(n):
    return {pair: index for index, pair in enumerate(
        [(1, 0), (2, 0), (2, 1)] + [(atom, axis) for atom in range(3, n) for axis in range(3)])}


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Interval:
    def __init__(self, lo, hi=None):
        self.lo, self.hi = F(lo), F(lo if hi is None else hi)
        if self.lo > self.hi:
            raise ValueError("reversed interval")

    @staticmethod
    def enclose(lo, hi):
        lower = (lo.numerator * SCALE) // lo.denominator
        upper = -((-hi.numerator * SCALE) // hi.denominator)
        return Interval(F(lower, SCALE), F(upper, SCALE))

    def __add__(self, other):
        other = as_interval(other)
        return self.enclose(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -as_interval(other)

    def __rsub__(self, other):
        return as_interval(other) + -self

    def __mul__(self, other):
        other = as_interval(other)
        products = [x * y for x in (self.lo, self.hi) for y in (other.lo, other.hi)]
        return self.enclose(min(products), max(products))

    __rmul__ = __mul__

    def __pow__(self, exponent):
        if exponent < 0 and self.lo <= 0 <= self.hi:
            raise ValueError("negative power across zero")
        if exponent == 0:
            return Interval(1)
        candidates = [self.lo**exponent, self.hi**exponent]
        if exponent > 0 and exponent % 2 == 0 and self.lo <= 0 <= self.hi:
            candidates.append(F(0))
        return self.enclose(min(candidates), max(candidates))

    def abs_upper(self):
        return max(abs(self.lo), abs(self.hi))


def as_interval(value):
    return value if isinstance(value, Interval) else Interval(value)


def pair_terms(delta):
    q = sum((component**2 for component in delta), Interval(0))
    if q.lo <= 0:
        raise ValueError("box does not exclude a collision")
    energy = q**-6 - 2 * q**-3
    radial = 12 * (q**-4 - q**-7)
    outer = 168 * q**-8 - 96 * q**-5
    gradient = [radial * component for component in delta]
    hessian = [[outer * delta[k] * delta[l] + (radial if k == l else 0)
                for l in range(3)] for k in range(3)]
    return energy, gradient, hessian, q


def evaluate(variables):
    dimension = len(variables)
    n = (dimension + 6) // 3
    active = active_coordinates(n)
    if len(active) != dimension:
        raise ValueError("invalid gauge dimension")
    points = [[Interval(0) for _ in range(3)] for _ in range(n)]
    for (atom, axis), variable in active.items():
        points[atom][axis] = variables[variable]
    energy = Interval(0)
    gradient = [Interval(0) for _ in range(dimension)]
    hessian = [[Interval(0) for _ in range(dimension)] for _ in range(dimension)]
    minimum_q = None
    for i in range(n):
        for j in range(i):
            delta = [points[i][axis] - points[j][axis] for axis in range(3)]
            pair_e, pair_g, block, q = pair_terms(delta)
            energy += pair_e
            minimum_q = q.lo if minimum_q is None else min(minimum_q, q.lo)
            for k in range(3):
                row_terms = [(active[(atom, k)], sign)
                             for atom, sign in ((i, 1), (j, -1)) if (atom, k) in active]
                for row, row_sign in row_terms:
                    gradient[row] += row_sign * pair_g[k]
                    for l in range(3):
                        col_terms = [(active[(atom, l)], sign)
                                     for atom, sign in ((i, 1), (j, -1)) if (atom, l) in active]
                        for col, col_sign in col_terms:
                            hessian[row][col] += row_sign * col_sign * block[k][l]
    return energy, gradient, hessian, minimum_q


def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def interval_matmul(left, right):
    columns = transpose(right)
    return [[sum((as_interval(a) * as_interval(b) for a, b in zip(row, col)), Interval(0))
             for col in columns] for row in left]


def propose_preconditioner(center_hessian):
    """Proposal only; later exact inequalities independently decide success."""
    dimension = len(center_hessian)
    with localcontext() as context:
        context.prec = 70
        def decimal(value):
            return Decimal(value.numerator) / Decimal(value.denominator)
        midpoint = [[decimal((entry.lo + entry.hi) / 2) for entry in row]
                    for row in center_hessian]
        lower = [[Decimal(0) for _ in range(dimension)] for _ in range(dimension)]
        for i in range(dimension):
            for j in range(i + 1):
                value = midpoint[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
                lower[i][j] = value.sqrt() if i == j else value / lower[j][j]
        upper = transpose(lower)
        inverse = [[Decimal(0) for _ in range(dimension)] for _ in range(dimension)]
        for column in range(dimension):
            for row in reversed(range(dimension)):
                rhs = Decimal(int(row == column)) - sum(
                    upper[row][k] * inverse[k][column] for k in range(row + 1, dimension))
                inverse[row][column] = rhs / upper[row][row]
        denominator = 10**20
        result = [[F(int((value * denominator).to_integral_value()), denominator)
                   for value in row] for row in inverse]
    require(all(result[i][i] > 0 for i in range(dimension)), "preconditioner diagonal must be positive")
    require(all(result[i][j] == 0 for i in range(dimension) for j in range(i)),
            "preconditioner must be upper triangular")
    return result


def record(value):
    value = F(value)
    with localcontext() as context:
        context.prec = 45
        display = str(Decimal(value.numerator) / Decimal(value.denominator))
    return {"fraction": str(value), "decimal_approximation": display}


def interval_record(value):
    return {"lower": record(value.lo), "upper": record(value.hi)}


def self_check():
    square = Interval(F(-3, 2), F(2, 3))**2
    require(square.lo == 0 and square.hi == F(9, 4), "interval square self-check failed")
    product = Interval(-2, 3) * Interval(-5, 7)
    require(product.lo == -15 and product.hi == 21, "interval product self-check failed")
    tiny = Interval.enclose(F(1, 3), F(1, 3))
    require(tiny.lo < F(1, 3) < tiny.hi, "outward rounding self-check failed")
    energy, gradient, hessian, _ = pair_terms([Interval(1), Interval(0), Interval(0)])
    require(energy.lo <= -1 <= energy.hi, "unit dimer energy self-check failed")
    require(all(item.lo <= 0 <= item.hi for item in gradient), "unit dimer gradient self-check failed")
    require(hessian[0][0].lo <= 72 <= hessian[0][0].hi, "radial Hessian self-check failed")
    require(hessian[1][1].lo <= 0 <= hessian[1][1].hi, "transverse Hessian self-check failed")


def certify(n, center, radius):
    """Return a checked local certificate without writing files or printing."""
    center, radius = [F(value) for value in center], F(radius)
    require(n in (5, 7) and len(center) == 3*n-6, "invalid particle count or gauge dimension")
    require(radius > 0, "radius must be positive")
    dimension = len(center)
    self_check()
    box = [Interval(value - radius, value + radius) for value in center]
    require(box[0].lo > 0 and box[2].lo > 0, "the rigid-motion gauge is singular in this box")
    energy_center, gradient_center, hessian_center, _ = evaluate([Interval(x) for x in center])
    _, _, hessian_box, minimum_q = evaluate(box)
    preconditioner = propose_preconditioner(hessian_center)
    transformed = interval_matmul(transpose(preconditioner),
                                 interval_matmul(hessian_box, preconditioner))
    margins = [transformed[i][i].lo - sum(transformed[i][j].abs_upper()
                                         for j in range(dimension) if j != i)
               for i in range(dimension)]
    positive_margin = min(margins)
    require(positive_margin > 0, "Hessian positive definiteness was not established")
    # ||P||_2^2 <= ||P||_F^2 and P^T H P >= positive_margin*I.
    frobenius_squared = sum(value * value for row in preconditioner for value in row)
    strong_convexity = positive_margin / frobenius_squared

    inverse = [[sum(preconditioner[i][k] * preconditioner[j][k] for k in range(dimension))
                for j in range(dimension)] for i in range(dimension)]
    inverse_times_hessian = interval_matmul(inverse, hessian_box)
    derivative = [[int(i == j) - inverse_times_hessian[i][j] for j in range(dimension)]
                  for i in range(dimension)]
    row_norms = [sum(entry.abs_upper() for entry in row) for row in derivative]
    contraction = max(row_norms)
    center_correction = [row[0] for row in interval_matmul(inverse, [[x] for x in gradient_center])]
    image_radii = [correction.abs_upper() + radius * row_norm
                   for correction, row_norm in zip(center_correction, row_norms)]
    require(contraction < 1, "Newton-like map contraction was not established")
    require(max(image_radii) < radius, "Newton-like map does not map the box into its interior")
    critical_point_radius = max(entry.abs_upper() for entry in center_correction) / (1 - contraction)
    gradient_squared_upper = sum(entry.abs_upper()**2 for entry in gradient_center)
    energy_lower = energy_center.lo - gradient_squared_upper / (2 * strong_convexity)
    energy_upper = energy_center.hi
    require(energy_lower <= energy_upper, "local energy enclosure is reversed")

    report = {
        "schema": f"lj{n}.strict-local-certificate.v1",
        "status": "verified local existence, uniqueness and strict minimality; no global lower bound",
        "potential": "sum_{i<j}(r_ij^-12-2*r_ij^-6)",
        "particles": n,
        "dimension": dimension,
        "gauge": ["p0=(0,0,0)", "p1=(u0,0,0)", "p2=(u1,u2,0)",
                  "p_i=(u_(3i-6),u_(3i-5),u_(3i-4)) for i>=3"],
        "center": [record(x) for x in center],
        "box_uniform_radius": record(radius),
        "box": [interval_record(x) for x in box],
        "minimum_squared_distance_lower_bound": record(minimum_q),
        "arithmetic": "Fraction endpoints; each operation rounded outwards to denominator 2^160",
        "proposal": "70-digit Decimal Cholesky, then rationalization; proposal accuracy is not assumed",
        "rational_preconditioner_P": [[str(x) for x in row] for row in preconditioner],
        "hessian_box": [[interval_record(x) for x in row] for row in hessian_box],
        "preconditioned_gershgorin_row_margins": [record(x) for x in margins],
        "preconditioned_gershgorin_minimum_margin": record(positive_margin),
        "strong_convexity_euclidean_lower_bound": record(strong_convexity),
        "fixed_point_map": "T(u)=u-C*gradient(E)(u), C=P*P^T; C is exactly nonsingular",
        "contraction_infinity_norm_upper_bound": record(contraction),
        "image_radius_upper_bounds": [record(x) for x in image_radii],
        "unique_critical_point_distance_from_center_infinity_upper_bound": record(critical_point_radius),
        "energy_at_center": interval_record(energy_center),
        "gradient_at_center": [interval_record(x) for x in gradient_center],
        "unique_local_minimum_energy_interval": interval_record(Interval(energy_lower, energy_upper)),
        "proved": {
            "collision_free_box": minimum_q > 0,
            "hessian_positive_definite_everywhere_in_box": positive_margin > 0,
            "strict_contraction": contraction < 1,
            "box_maps_into_its_interior": max(image_radii) < radius,
            "exactly_one_critical_point_in_box": True,
            "critical_point_is_unique_minimum_over_box": True,
            "global_LJ_optimality": False,
            "global_energy_lower_bound": False,
        },
        "self_checks_passed": True,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, choices=(5, 7), default=5)
    parser.add_argument('--radius', type=F, help='override the verified default radius for this N')
    args = parser.parse_args()
    n = args.n
    radius = args.radius if args.radius is not None else RADII[n]
    if radius <= 0:
        parser.error('radius must be positive')
    report = certify(n, CENTERS[n], radius)
    output = Path(__file__).resolve().parents[1] / "results" / f"local_certificate_{n}.json"
    serialized = json.dumps(report, indent=2) + "\n"
    output.write_text(serialized, encoding="utf-8")
    if n == 5:
        output.with_name('local_certificate.json').write_text(serialized, encoding='utf-8')
    print("particles:", n, "box radius:", float(radius))
    print("PD margin:", report['preconditioned_gershgorin_minimum_margin']['decimal_approximation'],
          "strong convexity:", report['strong_convexity_euclidean_lower_bound']['decimal_approximation'])
    print("contraction:", report['contraction_infinity_norm_upper_bound']['decimal_approximation'],
          "max image radius:", float(max(F(entry['fraction']) for entry in report['image_radius_upper_bounds'])))
    print("critical point radius:", report['unique_critical_point_distance_from_center_infinity_upper_bound']['decimal_approximation'])
    print("local energy interval:", report['unique_local_minimum_energy_interval']['lower']['decimal_approximation'],
          report['unique_local_minimum_energy_interval']['upper']['decimal_approximation'])
    print("all exact checks passed; wrote", output)


if __name__ == "__main__":
    main()
