"""Exact antipodal Gram witnesses on the remaining N6 auxiliary-matching cell.

Bounded runs are explicitly incomplete unless the whole cube is closed. A local
Gram enclosure is not yet an unrestricted global-minimum certificate.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.n6_reduction import B, CUT, PAIRS, TARGET, potential, require
from experiments.n6_single_long import polynomial_positive

CURVATURE = F(2)
CONTACT = F(9910823276741785, 10**16)
QS_LO = F(9910823276741784, 10**16)
QS_HI = F(9910823276741787, 10**16)
MATCHING_INDICES = (0, 9, 14)


def short_support():
    z, K = CONTACT, CURVATURE
    require(QS_LO**3 < F(257, 264) < QS_HI**3, 'octahedron scale interval invalid')
    derivative = -6*z**-7+6*z**-4
    slope = derivative-2*K*z
    offset = potential(z)-K*z*z-slope*z
    poly = [F(1), F(0), F(0), F(-2), F(0), F(0), -offset, -slope, -K]
    divisor = [z*z, -2*z, F(1)]
    quotient = [F(0)]*7
    for degree in range(8, 1, -1):
        quotient[degree-2] = poly[degree]
        for j, value in enumerate(divisor):
            poly[degree-2+j] -= quotient[degree-2]*value
    require(not any(poly), 'short tangent contact factor has a remainder')
    subdivisions = polynomial_positive(tuple(quotient), F(0), CUT)
    return slope, offset, subdivisions


def long_support(lo, hi):
    require(CUT <= lo < hi <= 4*B, 'matching interval outside proved root')
    # f''(q)=q^-8*(42-24*q^3)<0 for q>=3/2. Thus f(q)-2q^2 is concave.
    require(42-24*CUT**3 < 0, 'long secant concavity condition failed')
    slope = (potential(hi)-CURVATURE*hi*hi-potential(lo)+CURVATURE*lo*lo)/(hi-lo)
    return slope, potential(lo)-CURVATURE*lo*lo-slope*lo


def witness(box, short):
    """Explicit top-three truncation, exact lower bound, and certified Gram gap."""
    bs, cs = short[:2]
    long = [long_support(*interval) for interval in box]
    linear = [slope+2*bs for slope, _ in long]
    total = sum(linear)
    y = [-F(2, 5)/CURVATURE*(slope-total/8) for slope in linear]
    mu = [length/2 for length in y]
    if min(mu) <= 0:
        return None
    # y minimizes K[(5/4)sum(y_i^2)+(1/4)(sum y_i)^2]+sum(linear_i*y_i).
    lower = 12*cs+sum(offset for _, offset in long)+sum(a*b for a, b in zip(linear, y))/2
    M = [[F(0) for _ in range(6)] for _ in range(6)]
    for index, (i, j) in enumerate(PAIRS):
        slope = long[MATCHING_INDICES.index(index)][0] if index in MATCHING_INDICES else bs
        M[i][j] = M[j][i] = -slope/(2*CURVATURE)
    row = [sum(values)/6 for values in M]
    mean = sum(row)/6
    G0 = [[-(M[i][j]-row[i]-row[j]+mean)/2 for j in range(6)] for i in range(6)]
    T = [[F(0) for _ in range(6)] for _ in range(6)]
    for pair, length in enumerate(y):
        i, j = 2*pair, 2*pair+1
        T[i][i] = T[j][j] = length/4
        T[i][j] = T[j][i] = -length/4
    v = [T[i][i]-G0[i][i] for i in range(6)]
    t = sum(v)
    C = [[6*v[i]*(i == j)-v[i]-v[j]+sum(v)/6+t*(F(i == j)-F(1, 6))
          for j in range(6)] for i in range(6)]
    H = [[G0[i][j]-C[i][j]/2 for j in range(6)] for i in range(6)]
    residual = [[H[i][j]-T[i][j] for j in range(6)] for i in range(6)]
    # The residual annihilates all three antisymmetric pair vectors. Its action
    # on pair-center vectors is the exact symmetric 3x3 matrix S below.
    require(all(residual[i][2*j] == residual[i][2*j+1] for i in range(6) for j in range(3)),
            'explicit antipodal top-space identity failed')
    S = [[2*residual[2*i][2*j] for j in range(3)] for i in range(3)]
    tail_upper = max(F(0), *(S[i][i]+sum(abs(S[i][j]) for j in range(3) if i != j)
                            for i in range(3)))
    if tail_upper >= min(mu):
        return None
    gap = 1-tail_upper/min(mu)
    # Verify that the closed-form scalar minimum equals the completed-square
    # spectral constant. Neither expression is accepted from a float proposal.
    base = (2*sum(x*x for row in G0 for x in row)-
            2*sum(C[i][j]*G0[i][j] for i in range(6) for j in range(6))-
            6*sum(x*x for x in v)-t*t)
    constant = 12*cs+sum(offset for _, offset in long)-CURVATURE*sum(M[i][j]**2 for i, j in PAIRS)
    require(lower == CURVATURE*(base-2*sum(x*x for x in mu))+constant,
            'closed-form/spectral energy constants differ')
    delta = sum(max(abs(x-QS_LO), abs(x-QS_HI))**2 for x in mu)
    return {'lower': lower, 'gap': gap, 'delta': delta, 'mu': mu, 'tail_upper': tail_upper}


def search(seconds=50, eta=F(1, 100000), max_nodes=100000):
    start = time.monotonic()
    short = short_support()
    root = [(CUT, 4*B)]*3
    stack = [('r', root)]
    events = []
    excluded = localized = invalid_gap = 0
    maximum_eta_squared = F(0)
    minimum_lower = None
    while stack and len(events) < max_nodes and time.monotonic()-start < seconds:
        node, box = stack.pop()
        independent = -12+sum(potential(lo) for lo, _ in box)
        if independent > TARGET:
            excluded += 1
            events.append({'node': node, 'kind': 'pair_loss'})
            continue
        result = witness(box, short)
        if result is not None:
            lower = result['lower']
            minimum_lower = lower if minimum_lower is None else min(minimum_lower, lower)
            if lower > TARGET:
                excluded += 1
                events.append({'node': node, 'kind': 'spectral'})
                continue
            # Triangle inequality squared: ||G-Gc||^2 <= 2||G-T||^2+2||T-Gc||^2.
            eta_squared = (TARGET-lower)/(CURVATURE*result['gap'])+2*result['delta']
            if eta_squared <= eta*eta:
                localized += 1
                maximum_eta_squared = max(maximum_eta_squared, eta_squared)
                events.append({'node': node, 'kind': 'localized'})
                continue
        else:
            invalid_gap += 1
        axis = max(range(3), key=lambda k: box[k][1]-box[k][0])
        lo, hi = box[axis]
        cut = (lo+hi)/2
        left, right = list(box), list(box)
        left[axis], right[axis] = (lo, cut), (cut, hi)
        events.append({'node': node, 'kind': 'split', 'axis': axis, 'cut': str(cut)})
        stack.extend([(node+'1', right), (node+'0', left)])
    return {'schema': 'lj6.exact-auxiliary-matching-gap-search.v1', 'seconds': time.monotonic()-start,
            'nodes': len(events), 'excluded': excluded, 'localized': localized, 'invalid_gap_nodes': invalid_gap,
            'pending': [[node, [[str(lo), str(hi)] for lo, hi in box]] for node, box in stack],
            'events': events, 'eta_exact': str(eta), 'maximum_localized_eta_squared': str(maximum_eta_squared),
            'short_support_bernstein_subintervals': short[2],
            'matching_cube_covered': not stack, 'global_N6_optimality_proved': False,
            'requires_stationarity': False,
            'remaining_obligation': 'Replay the full exact tree and link every localized Gram enclosure to a proved local uniqueness box.'}


def verify_tree(data):
    require(data['schema'] == 'lj6.exact-auxiliary-matching-gap-search.v1' and
            data['matching_cube_covered'] is True and data['pending'] == [] and
            data['global_N6_optimality_proved'] is False and data['requires_stationarity'] is False,
            'matching tree is incomplete or has wrong theorem scope')
    eta = F(data['eta_exact'])
    require(0 < eta <= F(1, 10000), 'invalid matching localization radius')
    short = short_support()
    stack = [('r', [(CUT, 4*B)]*3)]
    splits = excluded = localized = 0
    maximum_eta_squared = F(0)
    for event in data['events']:
        require(bool(stack), 'tree contains an event after root closure')
        node, box = stack.pop()
        require(event['node'] == node, 'matching tree event order/identity differs')
        if event['kind'] == 'split':
            axis, cut = event['axis'], F(event['cut'])
            require(type(axis) is int and 0 <= axis < 3, 'invalid matching split axis')
            lo, hi = box[axis]
            require(lo < cut < hi, 'matching split outside parent interval')
            left, right = list(box), list(box)
            left[axis], right[axis] = (lo, cut), (cut, hi)
            stack.extend([(node+'1', right), (node+'0', left)])
            splits += 1
        elif event['kind'] == 'pair_loss':
            require(-12+sum(potential(lo) for lo, _ in box) > TARGET,
                    'pair loss leaf does not exclude target')
            excluded += 1
        elif event['kind'] in ('spectral', 'localized'):
            result = witness(box, short)
            require(result is not None, 'matching leaf lacks an exact top-three eigengap')
            if event['kind'] == 'spectral':
                require(result['lower'] > TARGET, 'spectral leaf does not exclude target')
                excluded += 1
            else:
                require(result['lower'] <= TARGET, 'localization leaf should be an exclusion')
                eta_squared = ((TARGET-result['lower'])/(CURVATURE*result['gap'])+
                               2*result['delta'])
                require(eta_squared <= eta*eta, 'matching Gram radius is not certified')
                maximum_eta_squared = max(maximum_eta_squared, eta_squared)
                localized += 1
        else:
            raise ValueError('unsupported matching terminal event')
    require(not stack, 'matching tree leaves part of the cube unverified')
    require(len(data['events']) == data['nodes'] == 2*splits+1 and
            excluded == data['excluded'] and localized == data['localized'] and
            excluded+localized == splits+1, 'matching tree closure accounting differs')
    return {'verified': True, 'nodes': len(data['events']), 'splits': splits,
            'exclusion_leaves': excluded, 'localization_leaves': localized,
            'eta_exact': str(eta), 'maximum_eta_squared_display': float(maximum_eta_squared),
            'matching_cube_covered': True, 'requires_stationarity': False,
            'global_N6_optimality_proved': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=50)
    parser.add_argument('--eta', type=F, default=F(1, 100000))
    parser.add_argument('--verify-tree', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'refusing to overwrite gap search')
    result = verify_tree(json.loads(args.verify_tree.read_text())) if args.verify_tree else search(args.seconds, args.eta)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({key: (len(value) if key == 'pending' else value)
                      for key, value in result.items() if key != 'events'}), flush=True)


if __name__ == '__main__':
    main()
