"""Independent audit: (1) brute-force search for counterexamples to the per-step supermartingale inequality behind the
time-uniform EB confidence sequence, (2) empirical coverage of the running-minimum UCB on hard streams.
Run from anywhere:  python codes/analysis/audit_checks/cs_check.py
"""
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent.parent.parent
import sys, math, numpy as np
sys.path.insert(0, str(BASE / "codes" / "src"))
from stopping_rules import time_uniform_eb_cs_valid

# ---- (1) per-step supermartingale inequality, UPPER-bound process ----
# E[ exp( lam*(m - X) - 4*psi(lam)*(X-muhat)^2 ) ] <= 1 for all X in [0,1] with E X = m, muhat in [0,1], lam in (0,.5]
rng = np.random.default_rng(0)
worst = -1; arg=None
psi = lambda l: (-math.log(1-l)-l)/4
for _ in range(400000):
    k = rng.integers(2,5)
    xs = rng.random(k); 
    if rng.random()<0.5: xs[0]=0.0
    if rng.random()<0.5: xs[1]=1.0
    ps = rng.dirichlet(np.ones(k)*rng.choice([0.3,1,3]))
    m = float(ps@xs); mh = rng.random() if rng.random()<.7 else rng.choice([0.,1.])
    lam = rng.choice([0.5, rng.random()*0.5])
    val = float(ps@np.exp(lam*(m-xs) - 4*psi(lam)*(xs-mh)**2))
    if val>worst: worst=val; arg=(xs,ps,mh,lam)
print("(1) max E[factor] over 400k random cases:", worst, " (must be <= 1)")
print("    argmax:", [np.round(a,3) for a in arg[:2]], round(arg[2],3), arg[3])

# ---- (2) coverage of running-min UCB: P(min_t U_t < mu) <= alpha ----
def sim(name, sampler, mu, T, alpha, reps, n_horizon):
    viol=0
    for r in range(reps):
        y = sampler(T)
        u,_,_,_ = time_uniform_eb_cs_valid(y, alpha, n_horizon=n_horizon)
        if u < mu - 1e-12: viol+=1
    return viol/reps

alpha=0.025
cases = {
 "Bern(0.05)": (lambda T: (rng.random(T)<0.05).astype(float), 0.05),
 "Bern(0.5)":  (lambda T: (rng.random(T)<0.5).astype(float), 0.5),
 "Bern(0.95)": (lambda T: (rng.random(T)<0.95).astype(float), 0.95),
 "bimodal .1/.9 mix .5": (lambda T: np.where(rng.random(T)<0.5,0.1,0.9), 0.5),
 "Beta(0.3,0.3)": (lambda T: rng.beta(.3,.3,T), 0.5),
 "point mass .3": (lambda T: np.full(T,0.3), 0.3),
 "QAOA-like peaked: .22 w.p. .9, .5 w.p. .1": (lambda T: np.where(rng.random(T)<0.9,0.22,0.5), 0.9*0.22+0.1*0.5),
}
print("\n(2) empirical violation rate of running-min UCB vs target alpha=%.3f  (T=150, 4000 reps)"%alpha)
for name,(s,mu) in cases.items():
    a = sim(name,s,mu,150,alpha,4000,None)
    b = sim(name,s,mu,150,alpha,4000,150)
    print(f"  {name:45s} horizon-free: {a:.4f}   fixed-H=150: {b:.4f}")
