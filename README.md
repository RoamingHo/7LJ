# Exact LJ5/LJ6/LJ7 global-optimality verification

Recomputes the theorem-level results of
*Exact Global Optimality and Uniqueness of Six- and Seven-Particle
Lennard-Jones Clusters via Spectral Gram Relaxations* for the potential
`V(r) = r^-12 - 2 r^-6`.

Everything here runs on exact rational arithmetic and outward-directed
interval bounds. No floating-point comparison decides a proof step.

## Run

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python verify.py
```

Python 3.12 with SymPy 1.14 and mpmath 1.3. The script finishes in about
30 seconds and prints six checks:

- LJ5 global optimum and uniqueness
- LJ6 global optimum `-3267/257` and uniqueness
- the competing C2v configuration is only a local minimum
- LJ7 global optimum and uniqueness (pentagonal bipyramid)
- all 143 rows of the N=8..150 propagation table
- the LJ8 frontier: local certificates, spectral exclusion, graph inventory

## What is recomputed and what is recorded

`verify.py` re-executes the joining lemmas, the algebraic candidate
isolations, the local Newton/Krawczyk certificates, the spectral graph
exclusions, and every row of the propagation table.

The large branch-and-bound enumeration trees themselves are not rerun.
Their closure is read from compact receipts under `proofs/` (event counts,
leaf statistics, target thresholds, and the recorded closure status). Each
receipt is checked for internal consistency before it is accepted. The
complete trees, which let anyone replay every split leaf by leaf, are
distributed separately as a data archive.

In `proofs/n6_tree/closure.json`, the recorded `checker`, `config`, `events`,
and replay report/state paths refer to the separate Zenodo archive. The
checker, config and event log are in Zenodo's `n6_tree/proof_search/`, and the
chunk reports and states are in `n6_tree/results/`. The local `final_receipt`
path is relative to the directory containing `closure.json`. The historical
proof snapshots retain their original run bindings. The zero-long receipt is
from a fresh run of the bundled checker; its derivation is in Zenodo’s
`n7_zero_long/theorem.md`. The Zenodo README gives the full-tree replay commands.

## Layout

- `verify.py` — single entry point
- `experiments/` — verifiers, one module group per theorem
- `results/` — exact candidates, local certificates, propagation tables
- `proofs/` — closure receipts and the self-contained spectral checker
- `lj.py` — potential, gradient and coordinate helpers

## License

MIT, see [LICENSE](LICENSE).
