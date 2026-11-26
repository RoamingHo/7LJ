"""Exact D3h LJ5 candidate, selecting the root linked to the certified minimum."""
from fractions import Fraction as F
import json
from pathlib import Path
import sys
import sympy as sp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.tools import (
    add,mul,neg,point,scale,divide,positive_power,positive_root,
    integer_coefficients,primitive_positive,polynomial_value,sturm_chain,
    root_count,sturm_record,record,interval_record,require,
)
from experiments.certificate import certify

EXPECTED_POLYNOMIAL=[-1485,-4563,-4941,90973,-497,777,2051,-636763,75072,74176,24640]


def build():
    t=sp.symbols('t')
    A=sp.Rational(1,243)+6/(1+t)**6+1/(4096*t**6)
    B=sp.Rational(1,9)+6/(1+t)**3+1/(64*t**3)
    derivative=sp.cancel(2*sp.diff(B,t)*A-B*sp.diff(A,t))
    polynomial=primitive_positive(sp.Poly(sp.fraction(derivative)[0],t))
    coefficients=integer_coefficients(polynomial)
    require(coefficients==EXPECTED_POLYNOMIAL,'unexpected D3h shape polynomial')
    require(sp.cancel(derivative+polynomial.as_expr()/(165888*t**7*(t+1)**7))==0,
            'D3h shape derivative identity failed')
    chain=sturm_chain(coefficients)
    require(root_count(chain,F(0),'+infinity')==3,'D3h positive-root count changed')
    low,high=F(197,100),F(199,100)
    require(root_count(chain,low,high)==1 and
            polynomial_value(coefficients,low)<0<polynomial_value(coefficients,high),
            'selected prolate root not isolated')
    for _ in range(192):
        middle=(low+high)/2;value=polynomial_value(coefficients,middle)
        require(value!=0,'unexpected rational D3h root')
        if value<0:low=middle
        else:high=middle
    shape=(low,high)
    require(root_count(chain,*shape)==1,'refined shape interval lost its root')
    one_plus_t=add(point(1),shape)
    AI=add(add(point(F(1,243)),scale(positive_power(one_plus_t,-6),6)),
           scale(positive_power(shape,-6),F(1,4096)))
    BI=add(add(point(F(1,9)),scale(positive_power(one_plus_t,-3),6)),
           scale(positive_power(shape,-3),F(1,64)))
    energy=neg(divide(positive_power(BI,2),AI))
    R=positive_root(divide(AI,BI),6);H=mul(R,positive_root(shape,2))
    side_z=scale(mul(R,positive_root(point(3),2)),F(1,2))
    coordinates=[scale(H,2),H,R,H,scale(R,F(-1,2)),side_z,
                 H,scale(R,F(-1,2)),neg(side_z)]
    points=[[point(0),point(0),point(0)],[coordinates[0],point(0),point(0)],
            [H,R,point(0)],coordinates[3:6],coordinates[6:9]]
    path=ROOT/'results/local_certificate_5.json';saved=json.loads(path.read_text())
    center=[F(item['fraction']) for item in saved['center']]
    radius=F(saved['box_uniform_radius']['fraction'])
    fresh=certify(5,center,radius)
    require(fresh==saved,'saved N5 local uniqueness certificate did not reproduce')
    require(all(c-radius<lo<=hi<c+radius for c,(lo,hi) in zip(center,coordinates)),
            'exact D3h root escapes the certified local box')
    old_energy=saved['unique_local_minimum_energy_interval']
    require(F(old_energy['lower']['fraction'])<=energy[0]<=energy[1]<=F(old_energy['upper']['fraction']),
            'algebraic energy not enclosed by local certificate')
    report={
        'schema':'lj5.exact-d3h-candidate.v1','particles':5,'dimension':9,
        'shape_variable':'t=H^2/R^2','rational_functions':{'A':str(A),'B':str(B),
            'scale':'R^6=A/B','energy':'-B^2/A'},
        'shape_polynomial_coefficients_ascending':EXPECTED_POLYNOMIAL,
        'shape_derivative_identity':'2*Bprime*A-B*Aprime=-P(t)/(165888*t^7*(t+1)^7)',
        'shape_positive_root_count':3,'selected_shape_root_count':1,
        'selection':'the unique real root in (197/100,199/100); other positive roots are not silently discarded',
        'shape_root_interval':interval_record(shape),
        'shape_sturm_certificate':sturm_record(chain,{'zero':F(0),'positive_infinity':'+infinity',
            'selected_lower':low,'selected_upper':high}),
        'equatorial_radius_interval':interval_record(R),'pole_height_interval':interval_record(H),
        'energy_root_interval':interval_record(energy),
        'exact_energy_definition':'-B(t_star)^2/A(t_star), with t_star selected by the rational isolating interval',
        'gauge_coordinate_intervals':[interval_record(x) for x in coordinates],
        'full_stationarity_argument':[
            'Threefold rotation and vertical/equatorial reflections make the three equatorial gradients common radial vectors and the two polar gradients opposite axial vectors.',
            'The R and H family derivatives are three times the equatorial radial force and twice the upper axial force.',
            'R^6=A/B and P(t)=0 annul the two derivatives in nonsingular coordinates (R,t), hence also in (R,H), so all fifteen Cartesian gradient components vanish.',
            'The exact stationary candidate lies strictly inside the freshly reproduced unique local-critical-point box, identifying it with that certified minimum.'],
        'local_certificate_link':{'source':'results/local_certificate_5.json',
            'fresh_local_certificate_matches_saved':True,'strict_coordinate_inclusion':True,
            'exact_energy_enclosed_by_local_certificate':True},
        'all_exact_checks_passed':True,'global_optimality_proved':False,
        'claim':'Exactly defined D3h local minimum; this artifact alone does not exclude any other configuration.'}
    return report,points,energy


def main():
    report,_,energy=build();path=ROOT/'results/exact_candidate5.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(path),'positive_shape_roots':3,
                      'selected_root_interval':report['shape_root_interval'],
                      'energy_lower':record(energy[0])['decimal_approximation'],
                      'energy_upper':record(energy[1])['decimal_approximation']}))


if __name__=='__main__':main()
