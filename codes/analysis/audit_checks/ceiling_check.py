"""Independent audit: compares `compute_exact_qaoa_ceiling` against a dense-grid / multistart global minimum.
Run from anywhere:  python codes/analysis/audit_checks/ceiling_check.py
"""
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent.parent.parent
import sys, numpy as np, time
sys.path.insert(0, str(BASE / "codes" / "src"))
from graph_library import generate_deduplicated_graph_suite
from qaoa_core import QAOACircuitEngine, RetrospectiveEvaluator, get_bounds
from scipy.optimize import minimize
rng=np.random.default_rng(1)
for fam,n,p in [("random_3_regular",8,1),("random_3_regular",12,1),("weighted_3_regular_bimodal",12,2)]:
    g=generate_deduplicated_graph_suite(fam,n,target_count=2,start_seed=300+n)[0]
    eng=QAOACircuitEngine(n,g.edges); ev=RetrospectiveEvaluator(eng,p=p)
    cr,_=ev.compute_exact_qaoa_ceiling(); ceil_loss=1-cr*eng.opt_cut/eng.total_weight
    b=get_bounds(p); t0=time.time(); best=1e9
    if p==1:
        G=np.linspace(0,np.pi,241); Bt=np.linspace(0,np.pi/2,121)
        for gg in G:
            for bb in Bt:
                best=min(best,ev.true_loss(np.array([gg,bb])))
        # polish
        r=minimize(lambda x: ev.true_loss(x), [0.6,0.4], method="L-BFGS-B", bounds=list(map(tuple,b)))
        best=min(best,r.fun)
    else:
        for _ in range(120):
            x0=rng.uniform(b[:,0],b[:,1])
            r=minimize(lambda x: ev.true_loss(x), x0, method="L-BFGS-B", bounds=list(map(tuple,b)),options={"maxiter":200})
            best=min(best,r.fun)
    print(f"{fam:28s} n={n} p={p}: ceiling-loss(used)={ceil_loss:.5f}  independent global min loss={best:.5f}  "
          f"gap(used - global)={ceil_loss-best:+.5f}   tau=ceil-0.004={ceil_loss-0.004:.5f}  -> tau below global min? {ceil_loss-0.004<best}  [{time.time()-t0:.0f}s]",flush=True)
