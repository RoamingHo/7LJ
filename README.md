# 7LJ: Exact Verification of Six- and Seven-Particle Lennard-Jones Clusters

This repository contains the theorem-level verification code for
*Exact Global Optimality and Uniqueness of Six- and Seven-Particle
Lennard-Jones Clusters via Spectral Gram Relaxations* (Xinze Li), under the
potential normalization $V(r) = r^{-12} - 2r^{-6}$.

All proof steps rely on exact rational arithmetic (`fractions.Fraction`) and
outward-rounded interval bounds; numerical root-finding is used only to propose
rational centers or preconditioners, never to decide a proof step.

## Running the verification

Tested with Python 3.12, SymPy 1.14, and mpmath 1.3:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python verify.py
```

Running `verify.py` takes about 30 seconds and checks six groups of results:

1. **LJ5**: exact candidate isolation, local uniqueness, and global closure receipt.
2. **LJ6 (global)**: exact global minimum `-3267/257`, spectral exclusions, matching-cube localization, and octahedral uniqueness.
3. **LJ6 (competitor)**: local certification of the $C_{2v}$ capped triangular bipyramid and strict separation from `-3267/257`.
4. **LJ7**: algebraic isolation of the $D_{5h}$ pentagonal bipyramid, local uniqueness, and assembly of the global exclusion regions.
5. **Propagation ($8 \le N \le 150$)**: recomputation of all 143 certified energy and diameter bound rows.
6. **LJ8 frontier**: $C_s$ and $D_{2d}$ local certificates, stationary spectral exclusion, and distance-graph inventory.

## What is recomputed here vs. in the Zenodo archive

`verify.py` recomputes the algebraic root isolations, local Krawczyk/Newton
certificates, LJ6 spectral trees (including the 6,469-node matching cube),
LJ7 zero-long spectral exclusion, all 143 propagation rows, and the LJ8 probes
directly from scratch.

The larger branch-and-bound trees (the Cartesian trees for $N=5$ and $N=6$ and
the $N=7$ single-long, colored, weak-$C_5$, and strong-$C_5$ trees, which take
from a minute to roughly half an hour each to replay) are not rerun by
`verify.py`. Instead, their closure is checked here via compact receipts under
`proofs/`, while the complete schedules, inputs, and independent from-root
checkers are distributed in the companion Zenodo archive
([10.5281/zenodo.24560370](https://doi.org/10.5281/zenodo.24560370)).

A note on paths inside `proofs/`: in `proofs/n6_tree/closure.json`, the
`checker`, `config`, `events`, and chunk report/state paths correspond to the
Zenodo layout (`n6_tree/proof_search/` and `n6_tree/results/`), whereas
`final_receipt` is relative to the local directory containing `closure.json`.
Likewise, `proofs/n7_zero_long/closure.json` records a fresh execution of the
bundled checker alongside `historical_receipt.json`, while the accompanying
derivation `n7_zero_long/theorem.md` lives in the Zenodo archive. Full replay
instructions for all eight tree bundles are given in the Zenodo README.

## Repository layout

- `verify.py`: top-level entry point running all six theorem-level checks.
- `lj.py`: Lennard-Jones potential, gradient, and coordinate helpers.
- `experiments/`: verification modules organized by cluster size and theorem.
- `proofs/`: compact closure receipts and the self-contained zero-long checker.
- `results/`: certified candidate records, local boxes, and propagation tables.

## License

MIT; see [LICENSE](LICENSE).
