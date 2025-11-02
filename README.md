# Matrix Product State (MPS) SWAP Network QUBO Solver

This repository contains the code used to produce the results of the paper "Enhancing Quantum-Inspired Tensor Network Optimization using SWAP Networks and Problem-Aware Qubit Layout".

## Project Description
The solver implements tensor-network-based imaginary time evolution to optimise QUBO instances using matrix product states. Rectangular and triangualr SWAP network layouts are combined with problem-aware qubit orderings (e.g. Fiedler heuristics) to improve convergence on dense graphs. The accompanying utilities cover data loading, sampling, entanglement analysis, and result persistence to support reproducible studies and parameter sweeps.

## Contents
The repository is organized as follows:

- 'src/': Contains the source code for the MPS-TEBD QUBO solver.
- 'benchmarks/': Contains benchmark results and scripts to generate them.
- 'generate_instances/': Scripts to generate problem instances in QUBO format.
- 'figures/': Contains figures and plots used in the paper, and scripts to generate them.

The 'src' directory includes the main implementation of the proposed algorithm, along with utilities for handling QUBO problems, SWAP networks, and qubit layouts. The main implementation of the rectangular and triangular SWAP networks can be found in 'src/tebd.jl'.

## Requirements
The code is written in Julia and was tested with Julia version 1.8.1. Tensor networks simulations were performed using the ITensors.jl 0.6.19 and ITensorMPS 0.2.5 packages.

## How to Run
- Install the Julia version specified above and clone this repository.
- Activate the project and instantiate dependencies: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`.
- Launch the primary TEBD workflow: `julia --project=. src/main.jl`.
  - Adjust `run_TEBD` parameters in `src/main.jl` to target different datasets or sweeps.
- Optional diagnostics (e.g. entanglement entropy) can be recomputed via `julia --project=. src/entanglement_entropy.jl`.

## Contact
For any questions or issues, please contact the authors of the paper.

## Citation
If you build upon this code in academic or industrial work, please cite:

TBD

Feel free to adapt the entry to match the published venue details.
