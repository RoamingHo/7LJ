#!/usr/bin/env python3
"""Independent mathematical replay prototype; producer witnesses are never used.

The event stream supplies only tree topology and cut coordinates. This checker
owns every box and re-proves contractions/prunes using 160-bit integer dyadics.
An unproved leaf stops replay and remains unresolved. No C++ is run or imported.
"""
import argparse
from decimal import Decimal, localcontext
from fractions import Fraction as F
import hashlib
import json
from math import isqrt
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'proofs'))
BITS = 160
S = 1 << BITS


def require(ok, message):
    if not ok:
        raise ValueError(message)


def ceil_div(a, b):
    return -((-a) // b)


def integer_root(value, degree):
    require(value >= 0 and degree in (2, 3), 'invalid integer root')
    if degree == 2:
        return isqrt(value)
    if value == 0:
        return 0
    x = 1 << ((value.bit_length() + degree - 1) // degree)
    while True:
        y = ((degree - 1) * x + value // x**(degree - 1)) // degree
        if y >= x:
            require(x**degree <= value < (x+1)**degree, 'integer root verification failed')
            return x
        x = y


class D:
    __slots__ = ('lo', 'hi')

    def __init__(self, lo, hi=None):
        self.lo, self.hi = lo, lo if hi is None else hi
        require(type(self.lo) is int and type(self.hi) is int and self.lo <= self.hi,
                'invalid dyadic interval')

    @classmethod
    def rational(cls, value):
        value = F(value)
        return cls(value.numerator*S // value.denominator,
                   ceil_div(value.numerator*S, value.denominator))

    def __add__(self, other):
        other = as_d(other)
        return D(self.lo+other.lo, self.hi+other.hi)

    __radd__ = __add__

    def __neg__(self):
        return D(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -as_d(other)

    def __rsub__(self, other):
        return as_d(other) + -self

    def __mul__(self, other):
        other = as_d(other)
        values = [a*b for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        return D(min(values)//S, ceil_div(max(values), S))

    __rmul__ = __mul__

    def __pow__(self, exponent):
        if exponent < 0:
            require(self.lo > 0, 'negative power requires a positive interval')
            power = -exponent
            numerator = 1 << (BITS*(power+1))
            return D(numerator // self.hi**power, ceil_div(numerator, self.lo**power))
        if exponent == 0:
            return as_d(1)
        values = [self.lo**exponent, self.hi**exponent]
        if exponent % 2 == 0 and self.lo <= 0 <= self.hi:
            values.append(0)
        denominator = 1 << (BITS*(exponent-1))
        return D(min(values)//denominator, ceil_div(max(values), denominator))

    def root(self, degree):
        require(self.lo >= 0, 'root requires nonnegative interval')
        factor = 1 << (BITS*(degree-1))
        lo, hi = integer_root(self.lo*factor, degree), integer_root(self.hi*factor, degree)
        return D(lo, hi + int(hi**degree < self.hi*factor))


def as_d(value):
    return value if isinstance(value, D) else D.rational(value)


ZERO = D(0)
ONE = D(S)
EDGES = [(i, j) for i in range(5) for j in range(i)]
ACTIVE = {(1, 0): 0, (2, 0): 1, (2, 1): 2}
ACTIVE.update({(i, a): 3*i-6+a for i in range(3, 5) for a in range(3)})
SHARED_DEPENDENCIES = (
    'proofs/run_search.py', 'proofs/preflight.py',
    'experiments/verify_certificates.py', 'experiments/certificate.py',
    'experiments/tools.py', 'experiments/n5_candidate.py',
    'experiments/box.py', 'experiments/n5_candidate_boxes.py',
    'results/five_point_bound.json', 'results/recursive_bounds.json',
    'results/stationary_bounds.json', 'results/local_certificate_5.json',
    'results/exact_candidate5.json', 'results/max_transverse_candidate5_boxes.json',
)


def coord(box, atom, axis):
    index = ACTIVE.get((atom, axis))
    return ZERO if index is None else box[index]


def difference(box, i, j):
    return [coord(box, i, a)-coord(box, j, a) for a in range(3)]


def q_of(box, i, j):
    return sum((d**2 for d in difference(box, i, j)), ZERO)


def restrict(box, atom, axis, lo, hi):
    if lo > hi:
        return False
    index = ACTIVE.get((atom, axis))
    if index is None:
        return lo <= 0 <= hi
    lo, hi = max(box[index].lo, lo), min(box[index].hi, hi)
    if lo > hi:
        return False
    box[index] = D(lo, hi)
    return True


def restrict_difference(box, i, j, axis, lo, hi):
    if lo > hi:
        return False
    a, b = coord(box, i, axis), coord(box, j, axis)
    if not restrict(box, i, axis, b.lo+lo, b.hi+hi):
        return False
    a = coord(box, i, axis)
    return restrict(box, j, axis, a.lo-hi, a.hi-lo)


def squared_distance_contract(box, i, j, q):
    for axis in range(3):
        delta = difference(box, i, j)
        other = sum((delta[k]**2 for k in range(3) if k != axis), ZERO)
        available = q.hi-other.lo
        if available < 0:
            return False
        extent = D(available).root(2).hi
        lo, hi = max(delta[axis].lo, -extent), min(delta[axis].hi, extent)
        needed = q.lo-other.hi
        if needed > 0:
            radius = D(needed).root(2).lo
            # Intersect with (-infinity,-radius] union [radius,infinity).
            # Keep its hull when both signs survive.
            if lo > -radius:
                lo = max(lo, radius)
            elif hi < radius:
                hi = min(hi, -radius)
        if not restrict_difference(box, i, j, axis, lo, hi):
            return False
    return True


def potential(q):
    return (q**-3-1)**2-1


def pair_lower(q):
    if q.lo <= S <= q.hi:
        return -S
    return max(-S, potential(D(q.hi if q.hi < S else q.lo)).lo)


def potential_upper_contract(q, upper):
    if upper < -S:
        return None
    radius = D(upper+S).root(2).hi
    lo = (D(S+radius)**-1).root(3).lo
    hi = q.hi
    if radius < S:
        hi = min(hi, (D(S-radius)**-1).root(3).hi)
    lo = max(lo, q.lo)
    return None if lo > hi else D(lo, hi)


def geometric_contract(box, cfg):
    # Exact canonical constraints: diameter 01, max transverse atom 2,
    # x2<=d/2, x3<=x4, and z3>=0. They apply to the attained minimizers.
    for atom in range(2, 5):
        if not restrict(box, atom, 0, 0, box[0].hi):
            return False
        if not restrict(box, 1, 0, coord(box, atom, 0).lo, box[0].hi):
            return False
    if not restrict(box, 2, 0, 0, ceil_div(box[0].hi, 2)):
        return False
    if not restrict(box, 1, 0, 2*box[1].lo, box[0].hi):
        return False
    for atom in (3, 4):
        rho = coord(box, atom, 1)**2+coord(box, atom, 2)**2
        if not restrict(box, 2, 1, D(rho.lo).root(2).lo, box[2].hi):
            return False
        for axis in (1, 2):
            available = (box[2]**2).hi-(coord(box, atom, 3-axis)**2).lo
            if available < 0:
                return False
            extent = D(available).root(2).hi
            if not restrict(box, atom, axis, -extent, extent):
                return False
    if not restrict(box, 3, 0, box[3].lo, box[6].hi):
        return False
    if not restrict(box, 4, 0, box[3].lo, box[6].hi):
        return False
    for i, j in EDGES:
        raw = q_of(box, i, j)
        lo, hi = max(raw.lo, cfg['qmin']), min(raw.hi, (box[0]**2).hi)
        if lo > hi:
            return False
        if not restrict(box, 1, 0, D(lo).root(2).lo, box[0].hi):
            return False
        if not squared_distance_contract(box, i, j, D(lo, hi)):
            return False
    return True


def contract(box, cfg):
    cache = None
    for _ in range(6):
        before = [(x.lo, x.hi) for x in box]
        if not geometric_contract(box, cfg):
            return None
        distances = []
        for k, (i, j) in enumerate(EDGES):
            q = q_of(box, i, j)
            lo, hi = max(q.lo, cfg['qmin']), min(q.hi, (box[0]**2).hi)
            if cache is not None:
                lo, hi = max(lo, cache[k].lo), min(hi, cache[k].hi)
            if lo > hi:
                return None
            distances.append(D(lo, hi))
        lower = [pair_lower(q) for q in distances]
        total = sum(lower)
        if total >= cfg['target']:
            return None
        stars = [sum(lower[k] for k, edge in enumerate(EDGES) if atom in edge) for atom in range(5)]
        for k, (i, j) in enumerate(EDGES):
            # Arbitrary 4- and 3-atom residual clusters have bounds -6 and -3.
            rest = max(total-lower[k], -6*S+stars[i]-lower[k],
                       -6*S+stars[j]-lower[k], -3*S+stars[i]+stars[j]-2*lower[k])
            distances[k] = potential_upper_contract(distances[k], cfg['target']-rest)
            if distances[k] is None or not squared_distance_contract(box, i, j, distances[k]):
                return None
        if cache is not None and before == [(x.lo, x.hi) for x in box] and all(
                (a.lo, a.hi) == (b.lo, b.hi) for a, b in zip(cache, distances)):
            return distances
        cache = distances
    return cache


CRITICAL_RADIAL = D.rational(F(7, 4)).root(3)
CRITICAL_OUTER = D.rational(F(14, 5)).root(3)
CRITICAL_VIRIAL = D.rational(2).root(3)


def radial(q):
    return 12*(q**-4-q**-7)


def outer(q):
    return 168*q**-8-96*q**-5


def virial(q):
    return q**-6-q**-3


def sharp_range(q, function, critical, maximum=False):
    endpoints = [function(D(q.lo)), function(D(q.hi))]
    lo, hi = min(x.lo for x in endpoints), max(x.hi for x in endpoints)
    if q.lo <= critical.hi and q.hi >= critical.lo:
        value = function(critical)
        if maximum:
            hi = max(hi, value.hi)
        else:
            lo = min(lo, value.lo)
    return D(lo, hi)


def derivative_prune(box, distances):
    gradients, diagonals = [ZERO]*15, [ZERO]*15
    scale = ZERO
    for (i, j), q in zip(EDGES, distances):
        a = sharp_range(q, radial, CRITICAL_RADIAL, True)
        b = sharp_range(q, outer, CRITICAL_OUTER)
        scale += sharp_range(q, virial, CRITICAL_VIRIAL)
        delta = difference(box, i, j)
        for axis in range(3):
            g, h = a*delta[axis], a+b*delta[axis]**2
            gradients[3*i+axis] += g
            gradients[3*j+axis] -= g
            diagonals[3*i+axis] += h
            diagonals[3*j+axis] += h
    if scale.lo > 0 or scale.hi < 0:
        return 'virial'
    if any(g.lo > 0 or g.hi < 0 for g in gradients):
        return 'gradient'
    if any(h.hi < 0 for h in diagonals):
        return 'curvature'
    return None


def raw_gradient_hessian(box):
    """Full-box derivatives: never use stationary-only narrowed distances here."""
    gradient = [ZERO]*9
    hessian = [[ZERO]*9 for _ in range(9)]
    for i, j in EDGES:
        delta = difference(box, i, j)
        q = sum((x**2 for x in delta), ZERO)
        if q.lo <= 0:
            return None
        a, b = sharp_range(q, radial, CRITICAL_RADIAL, True), sharp_range(q, outer, CRITICAL_OUTER)
        for axis in range(3):
            rows = [(ACTIVE[(atom, axis)], sign) for atom, sign in ((i, 1), (j, -1)) if (atom, axis) in ACTIVE]
            for row, sign in rows:
                gradient[row] += sign*a*delta[axis]
            for other in range(3):
                cols = [(ACTIVE[(atom, other)], sign) for atom, sign in ((i, 1), (j, -1)) if (atom, other) in ACTIVE]
                value = b*(delta[axis]**2 if axis == other else delta[axis]*delta[other])
                if axis == other:
                    value += a
                for row, rs in rows:
                    for col, cs in cols:
                        hessian[row][col] += (rs*cs)*value
    return gradient, hessian


def inverse_proposal(hessian):
    """Decimal Gaussian elimination proposes exact 80-bit dyadic coefficients.

    Inverse accuracy is not assumed by Krawczyk zero containment.
    """
    with localcontext() as context:
        context.prec = 60
        matrix = [[Decimal(hessian[i][j].lo+hessian[i][j].hi)/Decimal(2*S) for j in range(9)]
                  + [Decimal(int(i == j)) for j in range(9)] for i in range(9)]
        for col in range(9):
            pivot = max(range(col, 9), key=lambda row: abs(matrix[row][col]))
            if abs(matrix[pivot][col]) < Decimal('1e-35'):
                return None
            matrix[pivot], matrix[col] = matrix[col], matrix[pivot]
            divisor = matrix[col][col]
            matrix[col] = [x/divisor for x in matrix[col]]
            for row in range(9):
                if row != col:
                    factor = matrix[row][col]
                    matrix[row] = [x-factor*y for x, y in zip(matrix[row], matrix[col])]
        if any(abs(x) > Decimal('1e20') for row in matrix for x in row[9:]):
            return None
        return [[D(int(x*Decimal(1 << 80)) << 80) for x in row[9:]] for row in matrix]


def preconditioned_raw_hessian(box, inverse):
    """Enclose C H after combining each pair, preserving exact C_i-C_j.

    For one pair, H_delta=a I+b delta delta^T. In a row of C, put
    v=C_i-C_j (zero for frozen coordinates). Its positive endpoint block is
    a*v+b*(v dot delta)*delta; the other endpoint has the opposite sign.
    All distances are raw full-box ranges, independent of stationary pruning.
    """
    result = [[ZERO]*9 for _ in range(9)]
    for i, j in EDGES:
        delta = difference(box, i, j)
        q = sum((x**2 for x in delta), ZERO)
        if q.lo <= 0:
            return None
        a, b = sharp_range(q, radial, CRITICAL_RADIAL, True), sharp_range(q, outer, CRITICAL_OUTER)
        for row in range(9):
            v = []
            for axis in range(3):
                ia, ja = ACTIVE.get((i, axis)), ACTIVE.get((j, axis))
                v.append((ZERO if ia is None else inverse[row][ia])-
                         (ZERO if ja is None else inverse[row][ja]))
            dot = sum((v[k]*delta[k] for k in range(3)), ZERO)
            for axis in range(3):
                value = a*v[axis]+b*dot*delta[axis]
                for atom, sign in ((i, 1), (j, -1)):
                    col = ACTIVE.get((atom, axis))
                    if col is not None:
                        result[row][col] += sign*value
    return result


def newton_contract(box):
    center = [D((x.lo+x.hi)//2) for x in box]
    point_data = raw_gradient_hessian(center)
    if point_data is None:
        return 0
    gradient, center_hessian = point_data
    inverse = inverse_proposal(center_hessian)
    if inverse is None:
        return 0
    preconditioned = preconditioned_raw_hessian(box, inverse)
    if preconditioned is None:
        return 0
    result = []
    for i in range(9):
        image = center[i]-sum((inverse[i][k]*gradient[k] for k in range(9)), ZERO)
        for j in range(9):
            coefficient = int(i == j)-preconditioned[i][j]
            image += coefficient*(box[j]-center[j])
        lo, hi = max(box[i].lo, image.lo), min(box[i].hi, image.hi)
        if lo > hi:
            return -1
        result.append(D(lo, hi))
    changed = any((x.lo, x.hi) != (y.lo, y.hi) for x, y in zip(box, result))
    box[:] = result
    return int(changed)


def prove_small_leaf(box, cfg, budget=63):
    """A producer leaf may need more subdivision with our different enclosure.

    This proves exclusion only, without accepting candidate-containing pieces.
    Exhausting the local budget is a failure to prove, never a prune.
    """
    if max(x.hi-x.lo for x in box) > S//100:
        return False, 0, 0
    pending = [(box.copy(), 0)]
    visited, maximum_depth = 0, 0
    while pending and visited < budget:
        current, depth = pending.pop()
        visited += 1
        maximum_depth = max(maximum_depth, depth)
        distances = contract(current, cfg)
        if distances is None or derivative_prune(current, distances):
            continue
        excluded = False
        for _ in range(4):
            answer = newton_contract(current)
            if answer < 0:
                excluded = True
                break
            distances = contract(current, cfg)
            if distances is None or derivative_prune(current, distances):
                excluded = True
                break
            if answer == 0:
                break
        if excluded:
            continue
        axis = max(range(9), key=lambda k: current[k].hi-current[k].lo)
        interval = current[axis]
        cut = (interval.lo+interval.hi)//2
        if not interval.lo < cut < interval.hi:
            return False, visited, maximum_depth
        for lo, hi in ((interval.lo, cut), (cut, interval.hi)):
            child = current.copy()
            child[axis] = D(lo, hi)
            pending.append((child, depth+1))
    return not pending, visited, maximum_depth


def root_and_config(path):
    from run_search import validate_config
    domain = validate_config(path)  # exact proof-input checks; never runs C++.
    require(domain['n'] == 5 and domain['gauge'] == 'max_transverse', 'checker currently covers N5 max gauge')
    fields = dict(line.split() for line in path.read_text().splitlines()[1:])
    def endpoint(key):
        value = F(float.fromhex(fields[key]))
        require((value*S).denominator == 1, 'input is not on checker grid')
        return int(value*S)
    diameter, radius, rho = endpoint('diameter_hi'), endpoint('rmin_lo'), endpoint('p2_rho_lo')
    box = [D(radius, diameter), D(0, ceil_div(diameter, 2)), D(rho, diameter),
           D(0, diameter), D(-diameter, diameter), D(0, diameter),
           D(0, diameter), D(-diameter, diameter), D(-diameter, diameter)]
    return box, {'qmin': endpoint('qmin_lo'), 'target': endpoint('target_hi'),
                 'domain_sha256': fields['domain_sha256']}


def fresh_candidate_boxes():
    from experiments.n5_candidate_boxes import exact_candidate_enclosures, enumerate_alternate
    from experiments.certificate import certify
    data = json.loads((ROOT/'results/max_transverse_candidate5_boxes.json').read_text())
    require(data['particles'] == 5 and data['dimension'] == 9 and data['scale_bits'] == 50,
            'candidate particle count, dimension or scale mismatch')
    require(data['coordinate_order'] == [list(pair) for pair in ACTIVE], 'candidate coordinate order mismatch')
    require(data['boxes'] and all(len(item[key]) == 9 for item in data['boxes']
                                 for key in ('center_numerators', 'lower_numerators', 'upper_numerators')),
            'candidate coordinate list must have exactly nine entries')
    points, energy = exact_candidate_enclosures()
    groups, counts = enumerate_alternate(points)
    require(counts == data['enumeration_counts'], 'candidate enumeration changed')
    require(set(groups) == {tuple(b['center_numerators']) for b in data['boxes']}, 'candidate templates changed')
    answer = []
    for item in data['boxes']:
        center = tuple(item['center_numerators'])
        local = certify(5, [F(x, 2**50) for x in center], F(item['radius_numerator'], 2**50))
        require(local == item['local_certificate'], 'local certificate failed fresh verification')
        require(len(local['box']) == len(groups[center]['enclosure']) == 9,
                'fresh candidate or local certificate has the wrong dimension')
        require(all(F(lo, 2**50) == F(entry['lower']['fraction']) and
                    F(hi, 2**50) == F(entry['upper']['fraction'])
                    for lo, hi, entry in zip(item['lower_numerators'], item['upper_numerators'], local['box'])),
                'candidate endpoints differ from the freshly certified box')
        box = [D.rational(F(lo, 2**50)) for lo in item['lower_numerators']]
        box = [D(x.lo, D.rational(F(hi, 2**50)).hi) for x, hi in zip(box, item['upper_numerators'])]
        for x, (lo, hi) in zip(box, groups[center]['enclosure']):
            require(F(x.lo, S) < lo <= hi < F(x.hi, S), 'exact candidate escapes local box')
        answer.append(box)
    return answer


def local_contains(box, candidates):
    return len(box) == 9 and any(len(candidate) == 9 and
        all(c.lo < x.lo <= x.hi < c.hi for x, c in zip(box, candidate)) for candidate in candidates)


def replay(config, events, limit, seconds, resume=None, expected_state_sha=None, state_output=None):
    start = time.monotonic()
    checker_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    input_hashes = {'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
                    'events_sha256': hashlib.sha256(events.read_bytes()).hexdigest(),
                    'checker_sha256': checker_sha,
                    'shared_dependencies_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                                   for name in SHARED_DEPENDENCIES}}
    root, cfg = root_and_config(config)
    frontier = {0: root}
    seen = {0}
    candidates = None
    counters = {'events_verified': 0, 'splits': 0, 'prunes': 0, 'empty_subtree_events': 0,
                'candidate_leaves': 0, 'newton_attempts': 0, 'newton_exclusions': 0,
                'newton_contractions': 0, 'supplemental_leaf_proofs': 0,
                'supplemental_nodes': 0, 'supplemental_max_depth': 0, 'prune_methods': {}}
    failure = None
    status = 'prefix_verified'
    offset, physical_line, prior_states = 0, 0, []
    if resume is not None:
        require(expected_state_sha is not None, 'resume requires the independently recorded state SHA256')
        require(hashlib.sha256(resume.read_bytes()).hexdigest() == expected_state_sha, 'resume state hash mismatch')
        saved = json.loads(resume.read_text())
        require(saved['schema'] == 'lj5.independent-checker-state.v1' and saved['status'] == 'prefix_verified',
                'only a verified incomplete checker prefix may resume')
        require(saved['bindings'] == input_hashes, 'checker source or proof inputs changed across resume')
        frontier = {int(k): None if b is None else [D(int(lo), int(hi)) for lo, hi in b]
                    for k, b in saved['frontier'].items()}
        seen, counters = set(saved['seen_nodes']), saved['counters']
        offset, physical_line = saved['next_offset'], saved['physical_line']
        prior_states = saved['prior_state_sha256']+[expected_state_sha]
    starting_events = counters['events_verified']
    with events.open() as stream:
        stream.seek(offset)
        while True:
            if counters['events_verified'] >= limit or time.monotonic()-start >= seconds:
                break
            line = stream.readline()
            if not line:
                require(not frontier, 'event stream ended before the frontier was covered')
                status = 'closed'
                break
            physical_line += 1
            if line.startswith('#') or not line.strip():
                offset = stream.tell()
                continue
            fields = line.split()
            kind = fields[0]
            require(kind in ('SPLIT', 'PRUNE', 'ACCEPT'), 'unsupported/unresolved producer event')
            node = int(fields[1])
            require(node in frontier, 'missing, repeated or invented parent node')
            box = frontier.pop(node)
            distances = None if box is None else contract(box, cfg)
            if box is not None and distances is None:
                box = None
            method = None
            if box is not None and max(x.hi-x.lo for x in box) <= S//200:
                for _ in range(4):
                    counters['newton_attempts'] += 1
                    step = newton_contract(box)
                    if step < 0:
                        box = None
                        method = 'independent_krawczyk'
                        counters['newton_exclusions'] += 1
                        break
                    if step == 0:
                        break
                    counters['newton_contractions'] += 1
                    distances = contract(box, cfg)
                    if distances is None:
                        box = None
                        method = 'geometry_or_energy_after_independent_krawczyk'
                        break
            if box is not None and kind != 'SPLIT':
                method = derivative_prune(box, distances)
                if method:
                    box = None
            if box is not None and kind == 'PRUNE':
                proved, extra, depth = prove_small_leaf(box, cfg)
                counters['supplemental_nodes'] += extra
                counters['supplemental_max_depth'] = max(counters['supplemental_max_depth'], depth)
                if proved:
                    box = None
                    method = 'independent_local_subdivision'
                    counters['supplemental_leaf_proofs'] += 1
            if kind == 'SPLIT':
                require(len(fields) == 6, 'bad split shape')
                axis, cut, left, right = int(fields[2]), F(float.fromhex(fields[3])), int(fields[4]), int(fields[5])
                require(0 <= axis < 9 and left != right and left not in seen and right not in seen,
                        'invalid split coordinate or reused child')
                require((cut*S).denominator == 1, 'cut is not exact on checker grid')
                cut = int(cut*S)
                for child, lower_side in ((left, True), (right, False)):
                    copied = None if box is None else box.copy()
                    if copied is not None:
                        interval = copied[axis]
                        lo, hi = (interval.lo, min(interval.hi, cut)) if lower_side else (max(interval.lo, cut), interval.hi)
                        copied = None if lo > hi else copied
                        if copied is not None:
                            copied[axis] = D(lo, hi)
                    frontier[child] = copied
                    seen.add(child)
                counters['splits'] += 1
            elif box is None:
                counters['prunes'] += 1
                proof = method or 'independent_geometry_or_energy_or_already_empty'
                counters['prune_methods'][proof] = counters['prune_methods'].get(proof, 0)+1
            elif kind == 'ACCEPT':
                if candidates is None:
                    candidates = fresh_candidate_boxes()
                if local_contains(box, candidates):
                    counters['candidate_leaves'] += 1
                else:
                    failure = 'whole checker box is not inside a freshly verified local candidate box'
            else:
                failure = 'independent contractions and derivative tests did not prove this leaf'
            if failure:
                frontier[node] = box
                status = 'unproved_leaf'
                failure = {'message': failure, 'line': physical_line, 'node': node,
                           'producer_event': kind, 'producer_reason_ignored': fields[2] if len(fields)>2 else None,
                           'box': [[str(F(x.lo,S)), str(F(x.hi,S))] for x in box]}
                break
            counters['events_verified'] += 1
            offset = stream.tell()
            if box is None:
                counters['empty_subtree_events'] += 1
    # Hitting an event limit exactly at EOF may still establish full closure.
    if not frontier and failure is None:
        with events.open() as stream:
            stream.seek(offset)
            require(all(not line.strip() or line.startswith('#') for line in stream), 'events remain after root closure')
        status = 'closed'
    state_sha = None
    if state_output is not None and status == 'prefix_verified':
        saved = {'schema': 'lj5.independent-checker-state.v1', 'status': status, 'bindings': input_hashes,
                 'counters': counters, 'next_offset': offset, 'physical_line': physical_line,
                 'frontier': {str(k): None if b is None else [[str(x.lo), str(x.hi)] for x in b]
                              for k, b in frontier.items()}, 'seen_nodes': sorted(seen),
                 'prior_state_sha256': prior_states,
                 'scope': 'State of this independent checker after a verified prefix, not a producer checkpoint. '
                          'Resume trusts only the externally pinned state hash from that earlier checker invocation.'}
        state_output.parent.mkdir(parents=True, exist_ok=True)
        state_output.write_text(json.dumps(saved, separators=(',', ':'))+'\n')
        state_sha = hashlib.sha256(state_output.read_bytes()).hexdigest()
    return {'schema': 'lj5.independent-from-root-checker.v1', 'status': status,
            'arithmetic': '160-bit fixed dyadic integer intervals; directed integer powers and roots',
            'python_version': sys.version,
            'config': str(config), 'events': str(events), 'domain_sha256': cfg['domain_sha256'],
            **input_hashes,
            'producer_lower_bounds_and_witnesses_used': False, 'producer_cpp_executed': False,
            **counters, 'pending_tree_nodes': len(frontier),
            'events_verified_this_run': counters['events_verified']-starting_events,
            'nonempty_pending_boxes': sum(box is not None for box in frontier.values()),
            'failure': failure, 'wall_seconds': time.monotonic()-start,
            'resume_prior_state_sha256': prior_states,
            'next_state_path': str(state_output) if state_sha is not None else None,
            'next_state_sha256': state_sha,
            'full_n5_independent_proof_closed': status == 'closed',
            'full_n5_candidate_localization_verified': status == 'closed' and counters['candidate_leaves'] > 0,
            'scope': 'Partial replay until status=closed. A failed leaf remains unresolved; no producer pruning claim is trusted.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'proofs/n5_allfeatures_max.cfg')
    parser.add_argument('--events', type=Path, default=ROOT/'proofs/n5_full_independent_repeat.events.tsv')
    parser.add_argument('--max-events', type=int, default=300)
    parser.add_argument('--seconds', type=float, default=45)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-state', type=Path)
    parser.add_argument('--expected-state-sha256')
    args = parser.parse_args()
    require(args.max_events > 0 and 0 < args.seconds <= 60, 'invalid bounded checker budget')
    result = replay(args.config.resolve(), args.events.resolve(), args.max_events, args.seconds,
                    args.resume_state, args.expected_state_sha256, args.output.with_suffix('.state.json'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','events_verified','prunes','pending_tree_nodes',
                                         'nonempty_pending_boxes','wall_seconds','full_n5_independent_proof_closed')}))
    return int(result['status'] == 'unproved_leaf')


if __name__ == '__main__':
    sys.exit(main())
