using ITensors, ITensorMPS, Graphs, Plots, Statistics, JSON, LinearAlgebra, ProgressBars, Random, HDF5, SparseArrays, Arpack

include("utilities.jl")
include("tebd.jl")
include("qrr.jl")

function callback(psi)
	return nothing
end

function fiedler_ordering(J::Dict{Tuple{Int,Int}, Float64})
    # Determine number of qubits
    nodes = unique(vcat([i for (i,j) in keys(J)], [j for (i,j) in keys(J)]))
    N = maximum(nodes)

    # Build adjacency matrix using absolute weights
    rows, cols, vals = Int[], Int[], Float64[]
    for ((i,j), Jij) in J
        w = abs(Jij)  # use absolute value
        push!(rows, i); push!(cols, j); push!(vals, w)
        push!(rows, j); push!(cols, i); push!(vals, w)  # symmetric
    end

    A = sparse(rows, cols, vals, N, N)
    D = spdiagm(0 => vec(sum(A, dims=2)))
    L = D - A

    # Compute Fiedler vector (second smallest eigenvector)
    λ, V = Arpack.eigs(L; nev=2, which=:SM)  # smallest magnitude eigenvalues
    V = real(V)
    fiedler_vector = V[:,2]

    # Return permutation of nodes sorted by Fiedler vector
    ordering = sortperm(fiedler_vector)  # 1-based indices
	print("Fiedler ordering: $ordering\n")
    return ordering
end

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
		shuffle!(init_logical_qubit_order)
	elseif qubit_ordering === "fiedler"
		init_logical_qubit_order = fiedler_ordering(J)
	elseif qubit_ordering === "shuffle"
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
		TEBD_gates, logical_qubit_order = SWAP_network_block(
			sites, J, h, tau_effective, logical_qubit_order, zz_expvals, z_expvals,
		)
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
			Number of unique samples: $num_unique_samples""",
		)

		if energy_std < 1E-6
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
end

let
	# Parameters for running TEBD on multiple instances
	chi = 64
	cutoff = 1E-9
	tau = 5.0
	tau_min = 1E-4
	tau_max = 1E0
	Nsamples = 1000
	Nsteps = 10
	n_threads = 8

	for idx in 0:0
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
			qubit_ordering = "fiedler",
			save_dir = save_dir,
		)
	end
end

# let
# 	# MPS parameters
# 	chi = 64::Int
# 	cutoff = 1E-9::Float64

# 	# TEBD parameters
# 	tau = 1.0::Float64          # Timestep scalar
# 	tau_min = 1E-4::Float64      # Minimum timestep
# 	tau_max = 1E-1::Float64      # Maximum timestep
# 	Nsamples = 1000::Int     # Number of samples per iteration
# 	Nsteps = 20::Int         # Number of TEBD steps

# 	# Make parameter dict for saving
# 	params = Dict(
# 		"chi" => chi,
# 		"cutoff" => cutoff,
# 		"tau" => tau,
# 		"tau_min" => tau_min,
# 		"tau_max" => tau_max,
# 		"Nsamples" => Nsamples,
# 		"Nsteps" => Nsteps,
# 	)::Dict{String, Real}

# 	# Hardware parameters
# 	n_threads = 8::Int
# 	BLAS.set_num_threads(n_threads)

# 	# Problem instance
# 	project_dir = joinpath(@__DIR__, "..")
# 	path = joinpath(project_dir, "data", "MIS", "Ising", "ising_insecta-ant-colony1-day38.json")
# 	#path = joinpath(project_dir, "data", "MIS", "Ising", "ising_C125-9.json")
# 	# problem_file = "ising_Ns10_Nt14_Nq2_K10_gamma1_zeta1_rho1.json"
# 	# path = joinpath(project_dir, "data", "portfolio", "Ising", problem_file)
# 	path = "data/MaxCut/Ising/ising_graph1.json"
# 	# path = joinpath(project_dir, path)
# 	J, h, offset = load_ising(path)
# 	N = maximum(collect(keys(h)))  # Number of qubits

# 	# Save directory
# 	save_dir = joinpath(project_dir, "results", "MaxCut", "MPS", "ising_graph0")
# 	isdir(save_dir) || mkpath(save_dir)

# 	println("Optimizing Ising model with $N qubits and $(length(collect(keys(J)))) couplings")

# 	# Initialize MPS
# 	sites = siteinds("S=1/2", N; conserve_qns = false)
# 	psi = initialize_superposition(sites)

# 	# Initial logical qubit order
# 	init_logical_qubit_order = collect(1:N)
# 	#shuffle!(init_logical_qubit_order)
# 	logical_qubit_order = copy(init_logical_qubit_order)

# 	# Initial correlations and expectations
# 	z_expvals, zz_expvals = get_expvals(psi)
# 	energy = state_energy(z_expvals, zz_expvals, J, h, offset)

# 	# Initial samples and energies
# 	samples = get_samples(psi, Nsamples, logical_qubit_order)
# 	energy_samples = [sample_energy(samples[:, i], J, h, offset) for i in 1:size(samples, 2)]
# 	energy_std = Statistics.std(energy_samples)
# 	best_sample_energy = minimum(energy_samples)

# 	# Arrays to store data
# 	state_list = Vector{MPS}()
# 	push!(state_list, psi)

# 	state_energies_list = Vector{Float64}()
# 	push!(state_energies_list, energy)

# 	z_expvals_list = Vector{Vector{Float64}}()
# 	push!(z_expvals_list, z_expvals)

# 	zz_expvals_list = Vector{Matrix{Float64}}()
# 	push!(zz_expvals_list, zz_expvals)

# 	samples_list = Vector{Matrix{Int}}()
# 	push!(samples_list, samples)

# 	energy_samples_list = Vector{Vector{Float64}}()
# 	push!(energy_samples_list, energy_samples)

# 	time_list = Vector{Float64}()
# 	push!(time_list, 0.0)

# 	progress_bar = ProgressBar(1:Nsteps)
# 	for step in progress_bar
# 		# TEBD evolution
# 		tau_effective = tau / energy_std
# 		tau_effective = clamp(tau_effective, tau_min, tau_max)
# 		TEBD_gates, logical_qubit_order = SWAP_network_block(sites, J, h, tau_effective, logical_qubit_order, zz_expvals, z_expvals)
# 		psi = apply(TEBD_gates, psi; cutoff = cutoff, maxdim = chi)
# 		normalize!(psi)

# 		# State energy
# 		z_expvals, zz_expvals = get_expvals(psi, logical_qubit_order)
# 		energy = state_energy(z_expvals, zz_expvals, J, h, offset)

# 		samples = get_samples(psi, Nsamples, logical_qubit_order)
# 		num_unique_samples = length(unique(eachcol(samples)))

# 		energy_samples = [sample_energy(samples[:, i], J, h, offset) for i in 1:size(samples, 2)]
# 		energy_std = Statistics.std(energy_samples)
# 		best_sample_energy = minimum(energy_samples)

# 		# Store data
# 		push!(state_list, psi)
# 		push!(state_energies_list, energy)
# 		push!(z_expvals_list, z_expvals)
# 		push!(zz_expvals_list, zz_expvals)
# 		push!(samples_list, samples)
# 		push!(energy_samples_list, energy_samples)
# 		push!(time_list, time_list[end] + tau_effective)

# 		# Update progress bar
# 		set_multiline_postfix(
# 			progress_bar,
# 			"""
# 			State energy: $energy
# 			Best sample energy: $best_sample_energy
# 			Energy std of samples: $energy_std
# 			Effective time step: $tau_effective
# 			Number of unique samples: $num_unique_samples""",
# 		)
# 	end

# 	save_results(
# 		state_list,
# 		state_energies_list,
# 		z_expvals_list,
# 		zz_expvals_list,
# 		samples_list,
# 		energy_samples_list,
# 		time_list,
# 		params,
# 		save_dir,
# 	)
# 	println("Results saved to $save_dir")
# end
