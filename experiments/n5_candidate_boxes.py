"""All diameter/sorted-x/z3>=0 canonical copies of the exact LJ5 minimum."""
from fractions import Fraction as F
from itertools import permutations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.n5_candidate import build
from experiments.tools import add,sub,mul,scale,point,divide,positive_root,interval_record,record,require
from experiments.box import square,dot,cross,stable_center
from experiments.certificate import active_coordinates,certify

SCALE_BITS=50
DENOMINATOR=1<<SCALE_BITS
RADIUS_NUMERATOR=1<<(SCALE_BITS-13)
RADIUS=F(RADIUS_NUMERATOR,DENOMINATOR)


def exact_candidate_enclosures():
    report,points,energy=build()
    require(report==json.loads((ROOT/'results/exact_candidate5.json').read_text()),
            'exact N5 candidate artifact does not reproduce')
    return points,energy


def pair_distances(points):
    distances={}
    for i in range(5):
        distances[i,i]=point(0)
        for j in range(i):
            q=point(0)
            for x,y in zip(points[i],points[j]):q=add(q,square(sub(x,y)))
            require(q[0]>0,'candidate collision not excluded')
            distances[i,j]=distances[j,i]=q
    return distances


def map_gauge(points,distances,anchors):
    i0,i1,i2=anchors;d=positive_root(distances[i0,i1],2)
    x={i:divide(sub(add(distances[i0,i],distances[i0,i1]),distances[i1,i]),scale(d,2))
       for i in range(5)}
    height=sub(distances[i0,i2],square(x[i2]))
    require(height[0]>0,'candidate gauge may be singular; do not discard it')
    y2=positive_root(height,2)
    v1=[sub(a,b) for a,b in zip(points[i1],points[i0])]
    v2=[sub(a,b) for a,b in zip(points[i2],points[i0])];normal=cross(v1,v2)
    mapped={}
    for i in range(5):
        y=divide(sub(sub(add(distances[i0,i],distances[i0,i2]),distances[i2,i]),
                     scale(mul(x[i],x[i2]),2)),scale(y2,2))
        vi=[sub(a,b) for a,b in zip(points[i],points[i0])]
        mapped[i]=[x[i],y,divide(dot(normal,vi),mul(d,y2))]
    mapped[i0]=[point(0),point(0),point(0)]
    mapped[i1]=[d,point(0),point(0)];mapped[i2]=[x[i2],y2,point(0)]
    return mapped


def enumerate_canonical(points,gauge='sorted'):
    require(len(points)==5,'N5 enumeration requires five points')
    require(gauge in ('sorted','max_transverse'),'unsupported N5 gauge')
    distances=pair_distances(points);gauges={};groups={}
    counts={key:0 for key in ('total_permutations','total_orientations',
        'provably_non_diameter_orientations','provably_unsorted_x_orientations',
        'provably_negative_z3_orientations','retained_orientations','retained_z3_zero_or_undecided')}
    if gauge=='max_transverse':
        counts.update(provably_non_max_transverse_orientations=0,provably_wrong_diameter_orientation=0)
    active=list(active_coordinates(5))
    for permutation in permutations(range(5)):
        counts['total_permutations']+=1;counts['total_orientations']+=2
        chosen=distances[permutation[0],permutation[1]]
        if any(q[0]>chosen[1] for q in distances.values()):
            counts['provably_non_diameter_orientations']+=2;continue
        anchors=permutation[:3]
        if anchors not in gauges:gauges[anchors]=map_gauge(points,distances,anchors)
        mapped=gauges[anchors]
        if gauge=='max_transverse':
            rho={i:add(square(mapped[i][1]),square(mapped[i][2])) for i in range(5)}
            if any(value[0]>rho[permutation[2]][1] for value in rho.values()):
                counts['provably_non_max_transverse_orientations']+=2;continue
            if mapped[permutation[2]][0][0]>mapped[permutation[1]][0][1]/2:
                counts['provably_wrong_diameter_orientation']+=2;continue
        order=(2,3) if gauge=='sorted' else (3,)
        if any(mapped[permutation[i]][0][0]>mapped[permutation[i+1]][0][1] for i in order):
            counts['provably_unsorted_x_orientations']+=2;continue
        for reflection in (1,-1):
            z3=scale(mapped[permutation[3]][2],reflection)
            if z3[1]<0:
                counts['provably_negative_z3_orientations']+=1;continue
            coordinates=[scale(mapped[permutation[atom]][axis],reflection if axis==2 else 1)
                         for atom,axis in active]
            key=stable_center(coordinates)
            if key not in groups:groups[key]={'enclosure':coordinates,'representations':[]}
            else:groups[key]['enclosure']=[(min(a[0],b[0]),max(a[1],b[1]))
                                           for a,b in zip(groups[key]['enclosure'],coordinates)]
            groups[key]['representations'].append({'permutation':list(permutation),'z_reflection':reflection})
            counts['retained_orientations']+=1
            if z3[0]<=0:counts['retained_z3_zero_or_undecided']+=1
    require(counts['total_permutations']==120 and counts['total_orientations']==240,
            'incomplete N5 permutation enumeration')
    rejected=('provably_non_diameter_orientations','provably_unsorted_x_orientations',
              'provably_negative_z3_orientations','provably_non_max_transverse_orientations',
              'provably_wrong_diameter_orientation')
    require(sum(counts.get(key,0) for key in rejected)+counts['retained_orientations']==240,
            'N5 enumeration accounting failed')
    require(counts['retained_orientations']==12 and len(groups)==1 and len(gauges)==6,
            'N5 candidate symmetry count changed; no representation may be dropped')
    return active,groups,counts,len(gauges)


def enumerate_alternate(points):
    _,groups,counts,_=enumerate_canonical(points,'max_transverse')
    return groups,counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gauge',choices=('sorted','max_transverse'),default='sorted')
    args=parser.parse_args()
    points,energy=exact_candidate_enclosures()
    active,groups,counts,gauge_count=enumerate_canonical(points,args.gauge);boxes=[]
    for index,(numerators,group) in enumerate(sorted(groups.items())):
        center=[F(x,DENOMINATOR) for x in numerators]
        lower=[x-RADIUS_NUMERATOR for x in numerators];upper=[x+RADIUS_NUMERATOR for x in numerators]
        require(all(F(lo,DENOMINATOR)<x[0]<=x[1]<F(hi,DENOMINATOR)
                    for lo,hi,x in zip(lower,upper,group['enclosure'])),'exact N5 candidate escapes its box')
        require(all(abs(x)<1<<53 and F(float(F(x,DENOMINATOR)))==F(x,DENOMINATOR) for x in lower+upper),
                'N5 endpoint is not exactly binary64')
        certificate=certify(5,center,RADIUS);e=certificate['unique_local_minimum_energy_interval']
        require(F(e['lower']['fraction'])<=energy[0]<=energy[1]<=F(e['upper']['fraction']),
                'exact candidate energy escapes local box certificate')
        boxes.append({'id':index,'center_numerators':list(numerators),'lower_numerators':lower,
            'upper_numerators':upper,'radius_numerator':RADIUS_NUMERATOR,
            'representation_count':len(group['representations']),'representations':group['representations'],
            'exact_candidate_coordinate_hull':[interval_record(x) for x in group['enclosure']],
            'candidate_energy_interval':interval_record(energy),'local_certificate':certificate,
            'exact_identity_argument':'Every retained copy is an exact orthogonal transform and permutation of the isolated D3h stationary candidate; strict inclusion identifies it with this box unique critical point.'})
    result={'schema':'lj5.canonical-candidate-boxes.v1','particles':5,'dimension':9,
        'coordinate_order':[list(x) for x in active],'scale_bits':50,
        'endpoint_rule':'integer_numerator * 2^(-50)','common_radius':record(RADIUS),
        'canonical_rules':['p0=0','p1=(d,0,0) realizes the diameter','p2 has y>0 and z=0',
                           'x2<=x3<=x4','z3>=0'],
        'tie_policy':'Reject only strict separated violations; retain all exact or undecided x ties and both z reflections when zero is possible.',
        'enumeration_counts':counts,'distinct_anchor_gauges_evaluated':gauge_count,
        'template_count':len(boxes),'boxes':boxes,'exact_candidate_source':'results/exact_candidate5.json',
        'candidate_root_isolation_reverified':True,'candidate_coordinates_recomputed_from_algebraic_formulas':True,
        'all_exact_checks_passed':True,'global_optimality_proved':False,
        'claim':'Covers every canonical representation of this N5 candidate, with local uniqueness; global completeness requires the separate root search.'}
    filename='canonical_candidate5_boxes.json'
    if args.gauge=='max_transverse':
        result['schema']='lj5.max-transverse-candidate-boxes.v1'
        result['canonical_rules']=['p0=0','p1=(d,0,0) realizes the diameter',
            'p2 maximizes perpendicular radius and has y>0,z=0','x2<=d/2','x3<=x4','z3>=0']
        result['gauge_profile']='max_transverse'
        filename='max_transverse_candidate5_boxes.json'
    path=ROOT/'results'/filename;path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':str(path),'templates':len(boxes),'enumeration':counts,
                      'radius':str(RADIUS),'contraction':certificate['contraction_infinity_norm_upper_bound']}))


if __name__=='__main__':main()
