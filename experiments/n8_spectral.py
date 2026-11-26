"""Exact rank-three zero-long exclusion and bounded one-long feasibility probe.

Floating optimization proposes duals; rational polynomial and inertia checks
verify every reported bound. The one-long family deliberately remains open.
"""
from fractions import Fraction as F
from itertools import combinations
from math import comb
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.n8_candidate import UPPER

R = F(73,50)
PAIRS = tuple(combinations(range(8),2))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def multiply(a,b):
    result = [F(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):
            result[i+j] += x*y
    return result


def coefficients(z, lam):
    z, lam = F(z), F(lam); B = R*R
    require(0<z<B and lam>-1,'invalid scalar profile')
    f = lambda q:(1+lam)*q**-6-(2+lam)*q**-3
    derivative = -6*(1+lam)*z**-7+3*(2+lam)*z**-4
    a = (f(B)-f(z)-derivative*(B-z))/(B-z)**2
    b, c = derivative-2*a*z, f(z)-derivative*z+a*z*z
    require(a>0,'nonpositive quadratic coefficient')
    return z,lam,a,b,c


def verify_scalar(profile):
    z,lam,a,b,c = coefficients(*profile); B = R*R
    residual = [1+lam,F(0),F(0),-(2+lam),F(0),F(0),-c,-b,-a]
    divisor = multiply([z*z,-2*z,F(1)],[B,F(-1)])
    quotient = [F(0)]*6
    for degree in range(8,2,-1):
        value = residual[degree]/divisor[3]; quotient[degree-3] = value
        for j, d in enumerate(divisor):
            residual[degree-3+j] -= value*d
    require(not any(residual),'scalar contact factorization failed')
    n = len(quotient)-1
    minimum = None
    for slab in range(8):
        lo,hi = B*F(slab,8),B*F(slab+1,8)
        mono = [sum(quotient[j]*comb(j,k)*lo**(j-k)*(hi-lo)**k for j in range(k,n+1)) for k in range(n+1)]
        beta = [sum(mono[k]*F(comb(i,k),comb(n,k)) for k in range(i+1)) for i in range(n+1)]
        minimum = min(beta) if minimum is None else min(minimum,*beta)
    require(minimum>=0,'scalar Bernstein positivity failed')
    require(42*(1+lam)-12*(2+lam)*B**3-2*a*B**8<0,'long-interval concavity failed')
    return (z,lam,a,b,c),minimum


def zero_long():
    profile = ('49/50','2')
    (_,_,a,b,c), bernstein = verify_scalar(profile)
    lower = 28*c-6*b*b/a
    require(lower>UPPER,'zero-long exclusion failed')
    return {'profile':list(profile),'radius':str(R),'lower':str(lower),
            'lower_display':float(lower),'margin':str(lower-UPPER),
            'minimum_bernstein_coefficient':str(bernstein),
            'requires_stationarity':True,'zero_long_stationary_sublevel_excluded':True}


def spectral_matrix(profile,lo,hi,slope,dual):
    z,lam,a,b,c = coefficients(*profile)
    f = lambda q:(1+lam)*q**-6-(2+lam)*q**-3
    offset = min(f(lo)-a*lo*lo-slope*lo,f(hi)-a*hi*hi-slope*hi)
    M = [[F(0) if i==j else -b/(2*a) for j in range(8)] for i in range(8)]
    M[0][1] = M[1][0] = -slope/(2*a)
    constant = 27*(c-b*b/(4*a))+offset-slope*slope/(4*a)
    rows = [sum(row)/8 for row in M]; mean = sum(rows)/8
    G = [[-(M[i][j]-rows[i]-rows[j]+mean)/2 for j in range(8)] for i in range(8)]
    require(len(dual)==3,'one-long invariant dual needs three parameters')
    v = [F(dual[0])]*2+[F(dual[1])]*6; t = F(dual[2])
    C = [[8*v[i]*(i==j)-v[i]-v[j]+sum(v)/8+t*(F(i==j)-F(1,8)) for j in range(8)] for i in range(8)]
    H = [[G[i][j]-C[i][j]/2 for j in range(8)] for i in range(8)]
    scalar = 2*sum(x*x for row in G for x in row)-2*sum(C[i][j]*G[i][j] for i in range(8) for j in range(8))-8*sum(x*x for x in v)-t*t
    return H,scalar,constant,a


def positive_inertia(matrix,upper):
    work = [[x-(upper if i==j else 0) for j,x in enumerate(row)] for i,row in enumerate(matrix)]
    positive = 0
    for k in range(8):
        pivot = work[k][k]
        require(pivot!=0,'zero exact inertia pivot')
        positive += pivot>0
        for i in range(k+1,8):
            for j in range(i,8):
                work[i][j] -= work[i][k]*work[k][j]/pivot
                work[j][i] = work[i][j]
    return positive


def verify_record(row, profiles, partitions):
    require(type(row['slab']) is int and 0<=row['slab']<len(partitions)-1,'invalid long slab')
    require(type(row['profile']) is int and 0<=row['profile']<len(profiles),'invalid profile index')
    lo,hi = partitions[row['slab']:row['slab']+2]
    H,scalar,constant,a = spectral_matrix(profiles[row['profile']],lo,hi,F(row['slope']),row['dual'])
    upper = list(map(F,row['eigenvalue_upper_bounds']))
    require(len(upper)==3 and all(x>0 for x in upper),'three positive spectral bounds required')
    for k,u in enumerate(upper):
        require(positive_inertia(H,u)<=k,'exact spectral bound failed')
    return a*(scalar-2*sum(u*u for u in upper))+constant


def propose_one_long():
    import numpy as np
    from scipy.optimize import minimize
    profiles = [('1','0'),('1','1'),('49/50','2'),('24/25','3'),('47/50','5')]
    partitions = [R*R*(1+F(k,16)) for k in range(49)]
    J = np.eye(8)-np.ones((8,8))/8
    records = []
    for slab,(lo_q,hi_q) in enumerate(zip(partitions,partitions[1:])):
        best = None
        for index,profile in enumerate(profiles):
            _,lam,a,b,c = map(float,coefficients(*profile));lo,hi = float(lo_q),float(hi_q)
            f = lambda q:(1+lam)*q**-6-(2+lam)*q**-3
            def objective(x):
                v = np.array([x[0]]*2+[x[1]]*6);t,slope = x[2:]
                offset = min(f(lo)-a*lo*lo-slope*lo,f(hi)-a*hi*hi-slope*hi)
                M = np.full((8,8),-b/(2*a));np.fill_diagonal(M,0);M[0,1]=M[1,0]=-slope/(2*a)
                constant = 27*(c-b*b/(4*a))+offset-slope*slope/(4*a)
                G = -J@M@J/2;C = J@(8*np.diag(v)+t*np.eye(8))@J;H = G-C/2
                eigen = np.maximum(np.linalg.eigvalsh(H)[-3:],0)
                scalar = 2*np.sum(G*G)-2*np.sum(C*G)-8*np.dot(v,v)-t*t
                return -(a*(scalar-2*np.dot(eigen,eigen))+constant)
            result = minimize(objective,[0.,0.,0.,b],method='BFGS',options={'maxiter':200,'gtol':1e-8})
            if best is None or result.fun<best[0]:
                best = (result.fun,index,result.x)
        _,index,x = best
        rational = [F(round(float(value)*10**8),10**8) for value in x]
        H,_,_,_ = spectral_matrix(profiles[index],lo_q,hi_q,rational[3],rational[:3])
        eigen = np.linalg.eigvalsh(np.array([[float(v) for v in row] for row in H]))[::-1][:3]
        upper = [F(max(1,int(np.ceil(v*10**6))+2),10**6) for v in eigen]
        row = {'slab':slab,'profile':index,'slope':str(rational[3]),
               'dual':[str(v) for v in rational[:3]],'eigenvalue_upper_bounds':[str(v) for v in upper]}
        verify_record(row,profiles,partitions)
        records.append(row)
    return {'schema':'lj8.exact-bounded-one-long-probe.v1','upper':str(UPPER),
            'radius':str(R),'profiles':profiles,'partitions':[str(q) for q in partitions], 'records':records}


def verify(data):
    require(data['schema']=='lj8.exact-bounded-one-long-probe.v1' and F(data['upper'])==UPPER and F(data['radius'])==R,
            'wrong one-long scope')
    profiles = data['profiles'];partitions = list(map(F,data['partitions']))
    require(len(partitions)==49 and partitions[0]==R*R and partitions[-1]==4*R*R and
            all(a<b for a,b in zip(partitions,partitions[1:])), 'incomplete long-distance cover')
    for profile in profiles:
        verify_scalar(profile)
    require(sorted(row['slab'] for row in data['records'])==list(range(48)),'missing or repeated slab')
    bounds = [verify_record(row,profiles,partitions) for row in data['records']]
    closed = [row['slab'] for row,lower in zip(data['records'],bounds) if lower>UPPER]
    return {'zero_long':zero_long(),'one_long':{'slabs':48,'slabs_excluded':len(closed),
        'excluded_slab_indices':closed,'minimum_bound':str(min(bounds)),
        'minimum_bound_display':float(min(bounds)),'all_lower_bounds':[str(x) for x in bounds],
        'complete_one_long_exclusion':len(closed)==48,
        'q_domain':[str(R*R),str(4*R*R)],'requires_stationarity':True,
        'remaining_scope':'Every unexcluded slab remains an unresolved continuous domain; a low relaxation bound is not evidence for a realizable configuration.'},
        'global_LJ8_optimality_proved':False}


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);p.add_argument('--check',type=Path)
    args = p.parse_args();start = time.monotonic()
    data = json.loads(args.check.read_text()) if args.check else propose_one_long()
    result = verify(data)
    if args.check:
        require(data['verification']==result,'saved spectral verification differs')
    if args.output:
        require(not args.output.exists(),'refusing to overwrite spectral receipt')
        data['verification']=result;args.output.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'verified':True,'seconds':time.monotonic()-start,'zero_long_lower':result['zero_long']['lower_display'],
        'one_long_slabs_excluded':result['one_long']['slabs_excluded'],
        'one_long_minimum_bound':result['one_long']['minimum_bound_display'],'global_LJ8_optimality_proved':False}))


if __name__ == '__main__':
    main()
