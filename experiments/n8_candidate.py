"""Bounded LJ8 proposals and exact local certificates; never a global proof.

The Cambridge eight-atom candidate has Cs, not D2d, symmetry. Numerical
Newton steps propose rational centres. Only Fraction interval inequalities
accept the local certificate or the feasible global upper bound.
"""
from fractions import Fraction as F
from itertools import combinations, permutations
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import certificate as local

UPPER = F('-19.82148919215')
RAW_CAMBRIDGE_SIGMA = '''0.2730989500 1.1469629952 -0.3319578813
-0.4728837298 -0.6223685080 0.7664213265
-0.9666537615 -0.2393056630 -0.1698094248
0.6209419539 -0.3628130566 0.7094425990
0.8035441992 0.1648033307 -0.2639206823
-0.1784380914 0.2412141513 -0.8077599510
0.0639788373 -0.6647479592 -0.2089132333
-0.1435883576 0.3362547097 0.3064972473'''
SOURCE = 'https://www-wales.ch.cam.ac.uk/~wales/CCD/jon/structures/LJ/points/8'
PAIRS = tuple(combinations(range(8), 2))
ACTIVE = tuple(local.active_coordinates(8))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def points(center):
    xyz = [[F(0)]*3 for _ in range(8)]
    require(len(center) == 18, 'wrong LJ8 gauge dimension')
    for (atom, axis), value in zip(ACTIVE, center):
        xyz[atom][axis] = value
    return xyz


def exact_energy(center):
    xyz = points(center)
    q = [sum((xyz[i][k]-xyz[j][k])**2 for k in range(3)) for i, j in PAIRS]
    require(min(q) > 0, 'candidate collision')
    return sum(s**-6-2*s**-3 for s in q)


def propose(kind='Cs'):
    """Optional numerical producer. Its output is always rechecked exactly."""
    import mpmath as mp
    import numpy as np
    from scipy.optimize import minimize
    from lj import canonicalize, energy_gradient
    if kind == 'Cs':
        seed = np.array([[float(t) for t in line.split()]
                         for line in RAW_CAMBRIDGE_SIGMA.splitlines()])/2**(1/6)
        def fg(x):
            energy, gradient = energy_gradient(x)
            return energy, gradient.ravel()
        fitted = minimize(fg, seed.ravel(), jac=True, method='BFGS', options={'gtol': 1e-10})
        xyz = canonicalize(fitted.x.reshape(-1, 3))[[0, 1, 5, 4, 2, 3, 6, 7]]
        ey = xyz[2].copy(); ey[0] = 0; ey /= np.linalg.norm(ey)
        ex = np.array([1., 0., 0.]); ez = np.cross(ex, ey)
        xyz = xyz@np.column_stack([ex, ey, ez])
    else:
        def shape(t):
            a, b, c, d = t
            return np.array([[a,0,c],[-a,0,c],[0,a,-c],[0,-a,-c],
                             [b,0,-d],[-b,0,-d],[0,b,d],[0,-b,d]])
        fitted = minimize(lambda t: energy_gradient(shape(t))[0], [.634,.500,.206,.783],
                          method='BFGS', options={'gtol': 1e-10})
        xyz = canonicalize(shape(fitted.x))
    mp.mp.dps = 85
    x0 = [mp.mpf(str(xyz[i, a])) for i, a in ACTIVE]
    def evaluate(*variables):
        xx = [[mp.mpf(0)]*3 for _ in range(8)]
        for (i, a), value in zip(ACTIVE, variables):
            xx[i][a] = value
        grad = [[mp.mpf(0)]*3 for _ in range(8)]
        for i, j in PAIRS:
            delta = [xx[i][k]-xx[j][k] for k in range(3)]
            q = sum(v*v for v in delta)
            radial = 12*(q**-4-q**-7)
            for k in range(3):
                grad[i][k] += radial*delta[k]; grad[j][k] -= radial*delta[k]
        return tuple(grad[i][a] for i, a in ACTIVE)
    root = mp.findroot(evaluate, x0, tol=mp.mpf('1e-78'), maxsteps=15)
    denominator = 10**40
    center = [F(int(mp.nint(x*denominator)), denominator) for x in root]
    if kind == 'Cs':
        xyz = points(center)
        xyz[3][2] = F(0)
        for i, j in ((4,5), (6,7)):
            for k in (0, 1):
                xyz[i][k] = xyz[j][k] = (xyz[i][k]+xyz[j][k])/2
            xyz[i][2] = (xyz[i][2]-xyz[j][2])/2
            xyz[j][2] = -xyz[i][2]
        center = [xyz[i][a] for i, a in ACTIVE]
    return center


def certify(center, radius=F(1, 10**6)):
    center, radius = list(map(F, center)), F(radius)
    require(len(center) == 18 and radius > 0, 'invalid local box')
    local.self_check()
    I = local.Interval
    box = [I(x-radius, x+radius) for x in center]
    require(box[0].lo > 0 and box[2].lo > 0, 'singular rigid-motion gauge')
    ec, gc, hc, _ = local.evaluate([I(x) for x in center])
    _, _, hb, qmin = local.evaluate(box)
    P = local.propose_preconditioner(hc)
    product = local.interval_matmul(local.transpose(P), local.interval_matmul(hb, P))
    pd_margin = min(product[i][i].lo-sum(product[i][j].abs_upper()
                    for j in range(18) if i != j) for i in range(18))
    require(pd_margin > 0, 'Hessian positivity failed')
    strong_convexity = pd_margin/sum(x*x for row in P for x in row)
    C = [[sum(P[i][k]*P[j][k] for k in range(18)) for j in range(18)] for i in range(18)]
    CH = local.interval_matmul(C, hb)
    rows = [sum((int(i == j)-CH[i][j]).abs_upper() for j in range(18)) for i in range(18)]
    contraction = max(rows)
    correction = [r[0].abs_upper() for r in local.interval_matmul(C, [[g] for g in gc])]
    image = max(c+radius*r for c, r in zip(correction, rows))
    require(contraction < 1 and image < radius, 'strict contraction/self-mapping failed')
    error = max(correction)/(1-contraction)
    lower = ec.lo-sum(g.abs_upper()**2 for g in gc)/(2*strong_convexity)
    xyz = points(center)
    qbox = []
    refined = [I(x-error, x+error) for x in center]
    refined_xyz = [[I(0)]*3 for _ in range(8)]
    for (i, a), value in zip(ACTIVE, refined):
        refined_xyz[i][a] = value
    full_dimension = any(row[2].lo>0 or row[2].hi<0 for row in refined_xyz)
    require(full_dimension, 'three-dimensional affine span not certified')
    for i, j in PAIRS:
        qbox.append(sum(((refined_xyz[i][k]-refined_xyz[j][k])**2 for k in range(3)), I(0)))
    reflection = (0,1,2,3,5,4,7,6)
    symmetric = all(xyz[reflection[i]][k] == (xyz[i][k] if k < 2 else -xyz[i][k])
                    for i in range(8) for k in range(3))
    # A possible distance symmetry must preserve every certified distance interval.
    index = {pair: k for k, pair in enumerate(PAIRS)}
    compatible = []
    for p in permutations(range(8)):
        if all(qbox[k].lo <= qbox[index[tuple(sorted((p[i],p[j])))]].hi and
               qbox[index[tuple(sorted((p[i],p[j])))]].lo <= qbox[k].hi
               for k, (i,j) in enumerate(PAIRS)):
            compatible.append(p)
    cs_proved = symmetric and set(compatible) == {tuple(range(8)), reflection}
    classes = {}
    for k, (i,j) in enumerate(PAIRS):
        orbit = min((i,j), tuple(sorted((reflection[i],reflection[j])))) if cs_proved else (i,j)
        classes.setdefault(orbit, []).append(k)
    distribution = []
    for orbit in classes.values():
        lo, hi = min(qbox[k].lo for k in orbit), max(qbox[k].hi for k in orbit)
        distribution.append({'pairs': [list(PAIRS[k]) for k in orbit], 'multiplicity':len(orbit),
                             'q_lower':str(lo), 'q_upper':str(hi), 'distance_display':float((lo+hi)/2)**.5})
    distribution.sort(key=lambda row: F(row['q_lower']))
    return {'schema':'lj8.exact-local-candidate.v1', 'particles':8, 'dimension':18,
        'center':[str(x) for x in center], 'radius':str(radius),
        'minimum_q':str(qmin), 'positive_definiteness_margin':str(pd_margin),
        'strong_convexity':str(strong_convexity), 'contraction':str(contraction),
        'image_radius':str(image), 'critical_point_radius':str(error),
        'local_energy_interval':[str(lower), str(ec.hi)],
        'local_energy_display':[float(lower),float(ec.hi)],
        'strict_local_existence_uniqueness_minimality_proved':True,
        'affine_dimension_three_proved':full_dimension,
        'distance_compatible_permutations':[list(p) for p in compatible],
        'Cs_symmetry_and_no_additional_distance_symmetry_proved':cs_proved,
        'symmetry_argument':'Reflection in z=0 with swaps (4 5)(6 7) preserves the exact centre, gauge and cube. Uniqueness of its critical point forces this symmetry. Disjoint distance intervals exclude every other permutation.' if cs_proved else None,
        'distance_distribution':distribution,
        'feasible_energy_below_upper':exact_energy(center) < UPPER,
        'upper':str(UPPER), 'global_LJ8_optimality_proved':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('Cs','D2d'), default='Cs')
    parser.add_argument('--check', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(); start = time.monotonic()
    old = json.loads(args.check.read_text()) if args.check else None
    center = old['center'] if old else propose(args.kind)
    result = certify(center)
    if old:
        require(all(old[k] == v for k,v in result.items()), 'saved local certificate differs')
    elif args.output:
        require(not args.output.exists(), 'refusing to overwrite local receipt')
        result['proposal_source'] = SOURCE if args.kind == 'Cs' else 'four-parameter D2d orbit, explicitly defined in producer'
        result['proposed_point_group'] = args.kind
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
        xyz = points(list(map(F, center)))
        args.output.with_suffix('.xyz').write_text('8\nLJ8 '+args.kind+'; minimum-distance units; exact rational centre in JSON\n'+
            ''.join('Ar '+' '.join(format(float(x),'.16g') for x in row)+'\n' for row in xyz))
    print(json.dumps({'verified':True,'seconds':time.monotonic()-start,
                      'energy':result['local_energy_display'], 'Cs_proved':result['Cs_symmetry_and_no_additional_distance_symmetry_proved'],
                      'global_LJ8_optimality_proved':False}))


if __name__ == '__main__':
    main()
