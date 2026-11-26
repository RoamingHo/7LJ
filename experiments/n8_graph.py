"""Exact LJ8 loss budgets and finite long-graph inventory, not geometry coverage."""
from collections import Counter
from fractions import Fraction as F
from functools import lru_cache
from itertools import combinations
from math import factorial
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.n8_candidate import UPPER, certify

L7 = F('-16.5053841681')
N7_RECEIPT = 'results/n7_extension_preflight_20261110.json'
N7_CANDIDATE = 'proofs/n7_candidate/candidate.json'
N6_RECEIPT = 'results/n6_global_parent_replay.json'
ATLAS = 'proofs/n7_zero_long/atlas.json'
PAIRS = tuple(combinations(range(8), 2))
RADII = (F(7,5), F(73,50), F(3,2), F(8,5), F(9,5))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def bind_n7(root=ROOT):
    """Read the recorded LJ7 global assembly receipt and check its theorem scope."""
    root = Path(root)
    receipt = json.loads((root/N7_RECEIPT).read_text())
    require(receipt['global_LJ7_optimality_proved'] is True and receipt['particles'] == 7 and
            receipt['ambient_dimension'] == 3 and receipt['fresh_attainment_and_strong_domain_checks'] is True,
            'N7 receipt does not prove the required theorem')
    candidate = json.loads((root/N7_CANDIDATE).read_text())
    lo = F(candidate['candidate_energy_interval']['lower'])
    require(L7 < lo, 'rounded N7 lower bound is not below certified minimum')
    return {'lower': str(L7), 'candidate_interval_lower': str(lo),
            'execution_receipt': N7_RECEIPT,
            'scope': 'Recorded LJ7 global assembly receipt; this probe does not replay N7 trees.'}


def bind_n6(root=ROOT):
    root = Path(root)
    receipt = json.loads((root/N6_RECEIPT).read_text())
    require(receipt['global_N6_optimality_proved'] is True and receipt['global_N6_uniqueness_proved'] is True and
            receipt['verified'] is True and F(receipt['minimum_energy_exact']) == F(-3267, 257),
            'N6 receipt does not establish exact minimum')
    return {'lower': '-3267/257', 'execution_receipt': N6_RECEIPT,
            'scope': 'Recorded exact N6 global proof receipt; no new N6 proof-tree replay in this probe.'}


def adjacency(n, mask):
    adj = [0]*n
    for k, (i,j) in enumerate(combinations(range(n), 2)):
        if mask >> k & 1:
            adj[i] |= 1 << j; adj[j] |= 1 << i
    return adj


def canonical_graph(adj):
    """Exact individualization/refinement; twin cells contribute factorials."""
    n = len(adj)
    def refine(partition):
        while True:
            cell_masks = [sum(1 << v for v in cell) for cell in partition]
            refined = []
            for cell in partition:
                groups = {}
                for v in cell:
                    signature = tuple((adj[v]&mask).bit_count() for mask in cell_masks)
                    groups.setdefault(signature, []).append(v)
                refined.extend(tuple(groups[s]) for s in sorted(groups))
            refined = tuple(refined)
            if refined == partition:
                return partition
            partition = refined
    def recurse(partition):
        partition = refine(partition)
        nontrivial = next((i for i,c in enumerate(partition) if len(c)>1), None)
        if nontrivial is None:
            order = [c[0] for c in partition]
            code = sum(1 << k for k,(i,j) in enumerate(combinations(range(n),2))
                       if adj[order[i]] >> order[j] & 1)
            return code, 1
        cell = partition[nontrivial]
        twin = all((adj[cell[0]] & ~(1 << v)) == (adj[v] & ~(1 << cell[0])) for v in cell[1:])
        if twin:
            result, count = recurse(partition[:nontrivial]+tuple((v,) for v in cell)+partition[nontrivial+1:])
            return result, count*factorial(len(cell))
        best, count = None, 0
        for v in cell:
            new = partition[:nontrivial]+((v,),tuple(w for w in cell if w != v))+partition[nontrivial+1:]
            code, copies = recurse(new)
            if best is None or code < best:
                best, count = code, copies
            elif code == best:
                count += copies
        return best, count
    return recurse((tuple(range(n)),))


def labelled_counts(n, max_degree, max_edges):
    """Independent labelled enumeration by residual degree-capacity recursion."""
    @lru_cache(None)
    def visit(capacities):
        if not capacities:
            return (1,)+(0,)*max_edges
        cap, rest = capacities[0], capacities[1:]
        output = [0]*(max_edges+1)
        available = [i for i,c in enumerate(rest) if c > 0]
        for degree in range(min(cap,len(available),max_edges)+1):
            for chosen in combinations(available,degree):
                new = list(rest)
                for i in chosen:
                    new[i] -= 1
                counts = visit(tuple(sorted(new)))
                for edges in range(max_edges-degree+1):
                    output[edges+degree] += counts[edges]
        return tuple(output)
    return visit((max_degree,)*n)


def graph_inventory(root=ROOT, max_degree=4, max_edges=10):
    atlas = json.loads((Path(root)/ATLAS).read_text())
    representatives = {}
    extensions = 0
    for row in atlas['graphs']:
        adj7 = adjacency(7, row['mask'])
        if row['mask'].bit_count()>max_edges or max(map(int.bit_count,adj7))>max_degree:
            continue
        for neighbors in range(128):
            edges = row['mask'].bit_count()+neighbors.bit_count()
            if neighbors.bit_count()>max_degree or edges>max_edges:
                continue
            adj = [bits | (((neighbors >> v)&1) << 7) for v,bits in enumerate(adj7)]+[neighbors]
            if max(map(int.bit_count,adj))>max_degree:
                continue
            extensions += 1
            code, aut = canonical_graph(adj)
            previous = representatives.setdefault(code, aut)
            require(previous == aut, 'inconsistent automorphism count')
    rows = [{'mask':mask,'edges':mask.bit_count(),
             'max_degree':max(map(int.bit_count,adjacency(8,mask))),
             'automorphisms':aut,'labelled_count':factorial(8)//aut}
            for mask,aut in sorted(representatives.items())]
    for degree in {min(3,max_degree),max_degree}:
        reference = labelled_counts(8,degree,max_edges)
        counted = Counter()
        for row in rows:
            if row['max_degree']<=degree:
                counted[row['edges']] += row['labelled_count']
        require(all(counted[k] == reference[k] for k in range(max_edges+1)),
                'graph orbit inventory disagrees with independent labelled degree recursion')
    return rows, extensions


def run(root=ROOT):
    start = time.monotonic()
    dependency = bind_n7(root)
    n6_dependency = bind_n6(root)
    candidate = json.loads((Path(root)/'results/n8_cs_local_certificate.json').read_text())
    # Fresh local replay also confirms exact centre feasibility below the shared U.
    checked = certify(candidate['center'])
    require(checked['feasible_energy_below_upper'], 'N8 threshold lacks feasible configuration')
    from experiments import propagation as propagation
    from experiments import n8_spectral as spectral
    zero_certificate = spectral.zero_long()
    require(propagation.known_lower(6)==F(n6_dependency['lower']) and propagation.known_lower(7)==L7,
            'endpoint helper changed its subcluster premises')
    endpoint = propagation.endpoint_certificate(8,UPPER)
    require(endpoint is not None and F(endpoint['strict_margin'])>0,'endpoint exclusion failed')
    total, vertex = 28+UPPER, 7+UPPER-L7
    two_vertex = 13+UPPER-F(n6_dependency['lower'])
    rows, extensions = graph_inventory(root)
    cut = (1+F(73,50)**2)/2
    auxiliary_loss = (1-cut**-3)**2
    auxiliary_degree,auxiliary_edges = int(vertex//auxiliary_loss),int(total//auxiliary_loss)
    aux_rows,aux_extensions = graph_inventory(root,auxiliary_degree,auxiliary_edges)
    long_loss = (1-F(73,50)**-6)**2
    thresholds = []
    for radius in RADII:
        loss = (1-radius**-6)**2
        max_degree = min(7,int(vertex//loss)); max_edges = min(28,int(total//loss))
        require(max_degree<=4 and max_edges<=10, 'inventory scope is too small')
        allowed = [r for r in rows if r['max_degree']<=max_degree and r['edges']<=max_edges]
        union_limit = int(two_vertex//loss)
        pair_filtered = []
        for row in allowed:
            adj = adjacency(8,row['mask']);degrees = list(map(int.bit_count,adj))
            if all(degrees[i]+degrees[j]-((adj[i]>>j)&1)<=union_limit for i,j in PAIRS):
                pair_filtered.append(row)
        by_edges = []
        for edges in range(max_edges+1):
            chosen = [r for r in allowed if r['edges']==edges]
            by_edges.append({'edges':edges,'unlabelled':len(chosen),
                             'labelled':sum(r['labelled_count'] for r in chosen)})
        zero_excluded = radius<=F(73,50)
        thresholds.append({'radius':str(radius),'loss':str(loss), 'max_degree':max_degree,
            'max_edges':max_edges, 'forbidden_degree_margin':str((max_degree+1)*loss-vertex),
            'forbidden_edge_count_margin':str((max_edges+1)*loss-total),
            'unlabelled':len(allowed), 'labelled':sum(r['labelled_count'] for r in allowed),
            'by_edges':by_edges, 'zero_long_excluded_by_separate_stationary_certificate':zero_excluded,
            'unlabelled_after_zero_long':len(allowed)-int(zero_excluded),
            'labelled_after_zero_long':sum(r['labelled_count'] for r in allowed)-int(zero_excluded),
            'two_vertex_incident_long_edge_limit':union_limit,
            'unlabelled_after_two_vertex_loss':len(pair_filtered),
            'labelled_after_two_vertex_loss':sum(r['labelled_count'] for r in pair_filtered),
            'unlabelled_after_two_vertex_loss_and_zero_long':len(pair_filtered)-int(zero_excluded),
            'labelled_after_two_vertex_loss_and_zero_long':sum(r['labelled_count'] for r in pair_filtered)-int(zero_excluded)})
    # A collision pair consumes at least this much of either endpoint's budget.
    separation = F(5,6)
    require((separation**-6-1)**2 > vertex, 'minimum-separation inequality failed')
    diameter_radius = F(9,5)
    require(4*(1-diameter_radius**-6)**2 > vertex, 'degree-three diameter premise failed')
    candidate_edges = []
    for item in checked['distance_distribution']:
        require(F(item['q_upper'])<F(73,50)**2 or F(item['q_lower'])>F(73,50)**2,
                'candidate long graph not decided by certified intervals')
        if F(item['q_lower'])>F(73,50)**2:
            candidate_edges.extend(tuple(p) for p in item['pairs'])
    candidate_mask = sum(1<<k for k,p in enumerate(PAIRS) if p in candidate_edges)
    graph_code,graph_aut = canonical_graph(adjacency(8,candidate_mask))
    return {'schema':'lj8.exact-feasibility-graph-probe.v1','n7_dependency':dependency,'n6_dependency':n6_dependency,
        'upper':str(UPPER),'total_loss_budget':str(total),'vertex_loss_budget':str(vertex),
        'two_vertex_incident_loss_budget':str(two_vertex),
        'two_vertex_argument':'Deleting vertices i,j leaves E6>=L6. The 13 pairs incident to their union have total loss<=13+U8-L6, so (deg(i)+deg(j)-1_ij)*loss(R) cannot exceed that budget.',
        'loss_identity':'E8=-28+sum_{i<j}(1-r_ij^-6)^2',
        'vertex_argument':'Deleting vertex i leaves E7>=L7, so sum_j v(r_ij)<=U8-L7 and sum_j loss(r_ij)<=7+U8-L7.',
        'uniform_separation_strict_lower':str(separation),
        'uniform_diameter_strict_upper':endpoint['diameter_strict_upper'],
        'endpoint_certificate':endpoint,
        'coarser_vertex_only_diameter_strict_upper':str(2*diameter_radius),
        'coarser_diameter_argument':'At R=9/5 each vertex has at most three long neighbours. For a long pair each endpoint has at most two other long neighbours among the six third vertices, leaving at least two common short neighbours. Triangle inequality gives r_ij<2R=18/5.',
        'global_minimum_attained':True,
        'attainment_argument':'The rational candidate makes E<=U nonempty. Uniform separation and diameter bounds make this closed sublevel compact modulo translation. Its minimum is attained, finite, collision-free and stationary.',
        'certified_Cs_candidate_long_graph':{'radius':'73/50','edges':[list(p) for p in sorted(candidate_edges)],
            'edge_count':len(candidate_edges),'degree_sequence':sorted(map(int.bit_count,adjacency(8,candidate_mask))),
            'canonical_mask':graph_code,'automorphism_order':graph_aut},
        'thresholds':thresholds,'representatives':rows,'extension_candidates':extensions,
        'auxiliary_graph_probe':{'squared_cut':str(cut),'loss':str(auxiliary_loss),
            'max_degree':auxiliary_degree,'max_edges':auxiliary_edges,
            'unlabelled':len(aux_rows),'labelled':sum(r['labelled_count'] for r in aux_rows),
            'extension_candidates':aux_extensions,
            'max_auxiliary_degree_by_long_degree':[min(7,l+int((vertex-l*long_loss)//auxiliary_loss)) for l in range(5)],
            'max_auxiliary_edges_by_long_count':[min(28,k+int((total-k*long_loss)//auxiliary_loss)) for k in range(11)],
            'scope':'Auxiliary graph counts use only their own degree and loss gates. Classifying and excluding compatible true-long colourings is additional work.'},
        'zero_long_certificate':zero_certificate,
        'graph_cover_argument':'Every valid eight-vertex graph is an extension of a seven-vertex atlas representative. Exact canonicalization computes automorphism orders; independent labelled residual-degree recursion agrees at every edge count for degree bounds three and four.',
        'unlabelled_graphs_in_max_degree4_edge10_inventory':len(rows),
        'labelled_graphs_in_inventory':sum(r['labelled_count'] for r in rows),
        'fresh_exact_candidate_local_replay':True,
        'remaining_scope':'Graphs are necessary combinatorial conditions, not realizability or energy exclusions. Each surviving graph still has a continuous rank-three distance domain.',
        'global_LJ8_optimality_proved':False,'wall_seconds':time.monotonic()-start}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    args = p.parse_args(); require(not args.output.exists(),'refusing to overwrite graph receipt')
    result = run(); args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'verified','seconds':result['wall_seconds'],
                     'thresholds':[{k:r[k] for k in ('radius','max_degree','max_edges','unlabelled','labelled')} for r in result['thresholds']],
                     'global_LJ8_optimality_proved':False}))


if __name__ == '__main__':
    main()
