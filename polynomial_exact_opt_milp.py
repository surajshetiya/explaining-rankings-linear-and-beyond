#!/usr/bin/env python3
"""Try to certify polynomial Kendall inversion OPT by SciPy HiGHS mixed integer programming.

Headerless input: rows in target ranking, first row best.
Compares all pairs, includes nonconstant monomials up to requested degree.
Signed coefficients permitted, L1 normalization modeled by positive and negative parts.
Strict inequalities are approximated using a normalized positive margin. Only
report 'exact' if a certified zero MIP gap is returned AND the obtained
score order has no ties and the actual inversion count matches the MIP objective.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.sparse import csc_matrix, hstack, vstack
from scipy.optimize import milp, linprog, Bounds, LinearConstraint
from closest_polynomial_ranking_scipy import polynomial_features

def run(filename, degree, limit=30.0, margin=2e-6, upper_bound=None):
    X=np.loadtxt(filename,delimiter=',',ndmin=2)
    F,meta=polynomial_features(X,degree=degree)
    n,d=F.shape
    i,j=np.triu_indices(n,1)
    m=len(i)
    delta=F[i]-F[j]
    norms=np.max(np.abs(delta),axis=1)
    assert np.all(norms>0), 'duplicate features: special handling required'
    A=delta/norms[:,None]
    c=np.r_[np.zeros(2*d),np.ones(m)]
    lhs=hstack([csc_matrix(A),csc_matrix(-A),csc_matrix(np.diag(np.ones(m)*(1+margin)))],format='csc')
    # constraints: (F[i]-F[j]) / max(abs(diff)) . w + (1+margin) E >= margin
    # If E=1 constraint is redundant because ||w||_1 <= 1
    cons=[LinearConstraint(lhs,margin,np.inf)]
    eqrow=csc_matrix(np.ones((1,2*d)))
    cons.append(LinearConstraint(hstack([eqrow,csc_matrix((1,m))],format='csc'),1,1))
    if upper_bound is not None:
        cons.append(LinearConstraint(csc_matrix(np.r_[np.zeros(2*d),np.ones(m)].reshape(1,-1)), -np.inf, float(upper_bound)))
    print(f'Start MILP: n={n}, degree={degree}, d={d}, comparisons={m}, max time={limit}s, strict margin={margin}',flush=True)
    started=time.perf_counter()
    res=milp(c,integrality=np.r_[np.zeros(2*d),np.ones(m)],bounds=Bounds(np.zeros(2*d+m),np.ones(2*d+m)),constraints=cons,options={'time_limit':limit,'mip_rel_gap':0,'disp':True})
    seconds=time.perf_counter()-started
    output={'degree': degree,'points':n, 'polynomial_features':d,'pairwise_comparisons':m,'margin':margin, 'runtime_seconds':seconds, 'solver_status':int(res.status),'solver_message':res.message,'mip_lower_bound':getattr(res,'mip_dual_bound',None),'mip_gap':getattr(res,'mip_gap',None),'nodes':getattr(res,'mip_node_count',None)}
    if res.x is not None:
        w=res.x[:d]-res.x[d:2*d]
        score=F@w
        differences=delta@w
        inversions=int(np.sum(differences<0))
        ties=int(np.sum(differences==0))
        score_intended=np.sum(res.x[2*d:]>0.5)
        output.update({'milp_objective':float(res.fun),'achieved_inversions':inversions,'tied_pairs':ties,'witness_coefficients':w.tolist(),'milp_binary_count':int(score_intended),'minimum_positive_pairwise_margin':float(np.min(differences[differences>0])) if np.any(differences>0) else None})
    output['certified_exact_opt']=(res.status==0 and output.get('tied_pairs')==0 and output.get('achieved_inversions')==round(res.fun) if res.x is not None else False)
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset')
    p.add_argument('--degree',type=int,required=True)
    p.add_argument('--time-limit',type=float,default=30)
    p.add_argument('--margin',type=float,default=2e-6)
    p.add_argument('--upper-bound',type=int)
    p.add_argument('--output')
    args=p.parse_args()
    o=run(args.dataset,args.degree,args.time_limit,args.margin,args.upper_bound)
    print(json.dumps(o,indent=2))
    if args.output:Path(args.output).write_text(json.dumps(o,indent=2)+'\n')
