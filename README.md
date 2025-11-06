# Matrix Product State (MPS) SWAP Network QUBO Solver

<p align="center">
  <img src="figures/paper_figures/Algorithm_Schematic.png" alt="Algorithm schematic overview" width="720">
</p>

This repository contains the code used to produce the results of the paper "Enhancing Quantum-Inspired Tensor Network Optimization using SWAP Networks and Problem-Aware Qubit Layout", available as a [pre-print on arXiv](https://arxiv.org/abs/2511.02980).

## Project Description
The solver implements tensor-network-based imaginary time evolution to optimise QUBO instances using matrix product states. Rectangular and triangular SWAP network layouts are combined with problem-aware qubit orderings (e.g. Fiedler heuristics) to improve convergence on dense graphs. The accompanying utilities cover data loading, sampling, entanglement analysis, and data analysis.

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
If you build upon this code in academic or industrial work, please cite the pre-print as follows:

```bibtex
@misc{åsgrim2025swapnetworkroutingspectralqubit,
      title={SWAP-Network Routing and Spectral Qubit Ordering for MPS Imaginary-Time Optimization}, 
      author={Erik M. Åsgrim and Stefano Markidis},
      year={2025},
      eprint={2511.02980},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2511.02980}, 
}
```

Feel free to adapt the entry to match the published venue details.
