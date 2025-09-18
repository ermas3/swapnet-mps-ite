using ITensors, ITensorMPS, Graphs, Plots, Statistics, JSON, LinearAlgebra, ProgressBars, Random

include("utilities.jl")
include("tebd.jl")
include("qrr.jl")

let
	# MPS parameters
	chi = 32
	cutoff = 1E-9

	# TEBD parameters
	tau = 1E-1 # Timestep scalar
    tau_min = 1E-2 # Minimum timestep
    tau_max = 1E-1 # Maximum timestep
	Nsamples = 1000  # Number of samples per iteration
	Nsteps = 50  # Number of TEBD steps between samples

	# Hardware parameters
	n_threads = 8
	BLAS.set_num_threads(n_threads)

	# Problem instance
	project_dir = joinpath(@__DIR__, "..")
	path = joinpath(project_dir, "data", "MIS", "Ising", "ising_insecta-ant-colony1-day38.json")
	J, h, offset = load_ising(path)
	N = maximum(collect(keys(h)))  # Number of qubits

	println("Optimizing Ising model with $N qubits and $(length(collect(keys(J)))) couplings")

	sites = siteinds("S=1/2", N; conserve_qns = false)
    psi = initialize_superposition(sites)

    # Initial correlations and expectations
    z_expvals, zz_expvals = get_expvals(psi)
    energy = state_energy(z_expvals, zz_expvals, J, h, offset)

    
    init_logical_qubit_order = collect(1:N)
    #shuffle!(init_logical_qubit_order)
    logical_qubit_order = copy(init_logical_qubit_order)

    samples = get_samples(psi, Nsamples, logical_qubit_order)
    energy_samples = [sample_energy(s, J, h, offset) for s in samples]
    energy_std = Statistics.std(energy_samples)
    best_sample_energy = minimum(energy_samples)

    progress_bar = ProgressBar(1:Nsteps)
    for step in progress_bar
        # TEBD evolution
        tau_effective = tau / energy_std
        tau_effective = clamp(tau_effective, tau_min, tau_max)
        TEBD_gates, logical_qubit_order = SWAP_network_block(sites, J, h, tau_effective, logical_qubit_order, zz_expvals, z_expvals)
        psi = apply(TEBD_gates, psi; cutoff=cutoff, maxdim=chi)
        normalize!(psi)

        # State energy
        z_expvals, zz_expvals = get_expvals(psi, logical_qubit_order)
        energy = state_energy(z_expvals, zz_expvals, J, h, offset)
        
        samples = get_samples(psi, Nsamples, logical_qubit_order)
        energy_samples = [sample_energy(s, J, h, offset) for s in samples]
        energy_std = Statistics.std(energy_samples)
        best_sample = samples[argmin(energy_samples)]
        best_sample_energy = sample_energy(best_sample, J, h, offset)
        is_indep = is_independent_set(best_sample, collect(keys(J)))

        set_multiline_postfix(
            progress_bar,
            """
            State energy: $energy
            Best sample energy: $best_sample_energy
            Energy std of samples: $energy_std
            Is independent set: $is_indep
            Number of 1:s in best sample: $(sum(best_sample .== 1))
            Effective time step: $tau_effective"""
        )
    end
end
