AMFS + QAOA finite-shot benchmark
================================

Files
-----
- amfs_qaoa_shot_benchmark.py : benchmark driver
- MAD_1.py                   : user's existing AMFS implementation used by driver
- amfs_qaoa_shot_results.csv : results from the smoke/initial benchmark run

What it tests
-------------
A p=1 QAOA statevector simulator for 8-node, 3-regular MaxCut graphs. Circuit
measurement outcomes are sampled with finite shots. Methods: AMFS coordinate
search, SPSA, a small prototype sequential-shot SPSA, COBYLA, BOBYQA, Gaussian
process Bayesian optimisation, random search, and fixed initial parameters.
The code computes exact statevector expectation only for offline evaluation;
optimizers see shot-sampled objective estimates. The reported first-target-shot
metric is post-hoc: it measures the first queried parameter setting whose exact
expected approximation ratio is >= 0.78. It is NOT a hardware-available stopping
rule, because exact expectation is unavailable on hardware.

Requirements
------------
Python 3.10+
numpy, scipy, scikit-learn, networkx, nlopt

Install:
  python -m pip install numpy scipy scikit-learn networkx nlopt

Keep MAD_1.py in the same directory as amfs_qaoa_shot_benchmark.py.

Run
---
  python amfs_qaoa_shot_benchmark.py --graphs 8 --budget 1800 --shots-per-eval 60 --out results.csv

Limitations
-----------
This is a small simulator proof-of-concept, not a real-device experiment or a
paper-ready benchmark. It uses p=1 QAOA and small graphs. AMFS is adapted to two
parameters via coordinate-wise calls to its existing 1D noisy_amfs routine; it
is not a native multidimensional AMFS implementation. The adaptive SPSA baseline
is a simple prototype, NOT a faithful iCANS/weCANS/DDS implementation. The
results do not establish novelty or superiority. A proper study should add
published shot-frugal baselines, larger graph families, multiple p values,
strong initialisation controls, more seeds, confidence intervals, and ideally
hardware execution or validated noise models.
