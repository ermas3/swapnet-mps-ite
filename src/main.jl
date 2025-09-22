using ITensors, ITensorMPS, Graphs, Plots, Statistics, JSON, LinearAlgebra, ProgressBars, Random, HDF5, SparseArrays, Arpack

include("utilities.jl")
include("tebd.jl")
include("qrr.jl")

function run_TEBD(
	data_path;
	chi = 64,
	cutoff = 1E-9,
	tau = 1.0,
	tau_min = 1E-4,
	tau_max = 1E-1,
	Nsamples = 1000,
	Nsteps = 20,
	n_threads = 8,
	qubit_ordering = "default",
	network_architecture = "triangular",
	save_dir = missing,
	save_prefix = "results",
)
	# Set BLAS threads
	BLAS.set_num_threads(n_threads)

	# Load problem instance
	J, h, offset = load_ising(data_path)
	N = maximum(collect(keys(h)))  # Number of qubits

	# Prepare save directory
	if save_dir === missing
		println("No save directory provided.")
	end

	println("Optimizing Ising model with $N qubits and $(length(collect(keys(J)))) couplings")

	# Initialize MPS
	sites = siteinds("S=1/2", N; conserve_qns = false)
	psi = initialize_superposition(sites)

	# Initial logical qubit order
	if qubit_ordering === "default"
		init_logical_qubit_order = collect(1:N)
	elseif qubit_ordering === "fiedler"
		init_logical_qubit_order = fiedler_ordering(J)
	elseif qubit_ordering === "shuffle"
		# Set random seed
		Random.seed!(42)
		init_logical_qubit_order = collect(1:N)
		shuffle!(init_logical_qubit_order)
	else
		error("Invalid qubit ordering: $qubit_ordering")
	end

	println("Initial logical qubit order: $init_logical_qubit_order")
	logical_qubit_order = copy(init_logical_qubit_order)

	# Initial correlations and expectations
	z_expvals, zz_expvals = get_expvals(psi, logical_qubit_order)
	energy = state_energy(z_expvals, zz_expvals, J, h, offset)

	# Initial samples and energies
	samples = get_samples(psi, Nsamples, logical_qubit_order)
	energy_samples = [sample_energy(samples[:, i], J, h, offset) for i in 1:size(samples, 2)]
	energy_std = Statistics.std(energy_samples)
	best_sample_energy = minimum(energy_samples)

	# Arrays to store data
	state_list = Vector{MPS}([psi])
	state_energies_list = Vector{Float64}([energy])
	z_expvals_list = Vector{Vector{Float64}}([z_expvals])
	zz_expvals_list = Vector{Matrix{Float64}}([zz_expvals])
	samples_list = Vector{Matrix{Int}}([samples])
	energy_samples_list = Vector{Vector{Float64}}([energy_samples])
	time_list = Vector{Float64}([0.0])

	params = Dict(
		"chi" => chi,
		"cutoff" => cutoff,
		"tau" => tau,
		"tau_min" => tau_min,
		"tau_max" => tau_max,
		"Nsamples" => Nsamples,
		"Nsteps" => Nsteps,
	)::Dict{String, Real}

	progress_bar = ProgressBar(1:Nsteps)
	for step in progress_bar
		# TEBD evolution
		tau_effective = tau / energy_std
		tau_effective = clamp(tau_effective, tau_min, tau_max)

		if network_architecture === "triangular"
			bottom_up = true  # Always do bottom-up for now
			bottom_up = step % 2 == 1
			TEBD_gates, logical_qubit_order = triangular_SWAP_network_block(sites, J, h, tau_effective, logical_qubit_order, zz_expvals, z_expvals, bottom_up)
		elseif network_architecture === "quadratic"
			TEBD_gates, logical_qubit_order = SWAP_network_block(sites, J, h, tau_effective, logical_qubit_order, zz_expvals, z_expvals)
		else
			error("Invalid network architecture: $network_architecture")
		end

		psi = apply(TEBD_gates, psi; cutoff = cutoff, maxdim = chi)
		normalize!(psi)

		# State energy
		z_expvals, zz_expvals = get_expvals(psi, logical_qubit_order)
		energy = state_energy(z_expvals, zz_expvals, J, h, offset)

		samples = get_samples(psi, Nsamples, logical_qubit_order)
		num_unique_samples = length(unique([Tuple(col) for col in eachcol(samples)]))

		energy_samples = [sample_energy(samples[:, i], J, h, offset) for i in 1:size(samples, 2)]
		energy_std = Statistics.std(energy_samples)
		energy_mean = mean(energy_samples)
		best_sample_energy = minimum(energy_samples)

		# Store data
		push!(state_list, psi)
		push!(state_energies_list, energy)
		push!(z_expvals_list, z_expvals)
		push!(zz_expvals_list, zz_expvals)
		push!(samples_list, samples)
		push!(energy_samples_list, energy_samples)
		push!(time_list, time_list[end] + tau_effective)

		# Update progress bar
		set_multiline_postfix(
			progress_bar,
			"""
			State energy: $energy
			Sample energy: $energy_mean ± $energy_std
			Best sample energy: $best_sample_energy
			Effective time step: $tau_effective
			Number of unique samples: $num_unique_samples out of $Nsamples """,
		)

		if energy_std < 1E-3
			println("Energy std below threshold, stopping TEBD.")
			break
		end
	end

	# Save results, if save directory provided
	if !ismissing(save_dir)
		isdir(save_dir) || mkpath(save_dir)
		save_results(
			state_list,
			state_energies_list,
			z_expvals_list,
			zz_expvals_list,
			samples_list,
			energy_samples_list,
			time_list,
			params,
			save_dir,
		)
		println("Results saved to $save_dir")
	end

	println("Best bitstring found: ", samples[:, argmin(energy_samples)])
end

let
	# Parameters for running TEBD on multiple instances
	chi = 64
	cutoff = 1E-9
	tau = 5.0
	tau_min = 1E-4
	tau_max = 1E0
	Nsamples = 1000
	Nsteps = 20
	n_threads = 8
	qubit_ordering = "fiedler"  # "default", "fiedler", or "shuffle"
	network_architecture = "quadratic"  # "triangular" or "quadratic"

	Nv = 50

	for idx in 1:1
		#data/MIS/Ising/ising_insecta-ant-colony1-day38.json
#= 		data_path = joinpath(@__DIR__, "..", "data", "MIS","Ising", "ising_insecta-ant-colony1-day38.json")
		save_dir = joinpath(@__DIR__, "..", "results", "MIS", "MPS", "ising_insecta-ant-colony1-day38") =#

		graph_path = joinpath(@__DIR__, "..", "data", "MaxCut", "3Reg", "200v", "graphs", "graph$(idx).json")
		data_path = joinpath(@__DIR__, "..", "data", "MaxCut", "3Reg", "200v", "Ising", "ising_graph$(idx).json")
		save_dir = joinpath(@__DIR__, "..", "results", "MaxCut", "3Reg", "200v", "MPS", "ising_graph$(idx)")

		run_TEBD(
			data_path,
			chi = chi,
			cutoff = cutoff,
			tau = tau,
			tau_min = tau_min,
			tau_max = tau_max,
			Nsamples = Nsamples,
			Nsteps = Nsteps,
			n_threads = n_threads,
			qubit_ordering = qubit_ordering,
			network_architecture = network_architecture,
			save_dir = save_dir,
		)
	end
end