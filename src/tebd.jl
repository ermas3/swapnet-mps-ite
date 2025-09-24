function SWAP_network_block(s,
	J::Dict{Tuple{Int, Int},
		Float64},
	h::Dict{Int, Float64},
	tau::Float64,
	logical_qubit_order::Vector{Int} = collect(1:length(s)),
	zzcorr::Matrix{Float64} = zeros(length(s), length(s)),
	zvals::Vector{Float64} = zeros(length(s)),
)
	Z_matrix = [1 0; 0 -1]
	ZZ_matrix = [1 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 1]
	SWAP_matrix = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
	N = length(s)

	gates = ITensor[]

	# Apply two-qubit coupling via SWAP network
	for layer in 1:N
		if layer % 2 == 1
			pairs = [(i, i + 1) for i in 1:2:N-1 if i + 1 <= N]
		else
			layer % 2 == 0
			pairs = [(i, i + 1) for i in 2:2:N-1 if i + 1 <= N]
		end

		for (a, b) in pairs
			qa = logical_qubit_order[a]
			qb = logical_qubit_order[b]

			coupling_weight = get(J, (min(qa, qb), max(qa, qb)), 0.0)
			renormalization_matrix = UniformScaling(zzcorr[qa, qb])
			gate_matrix = exp(-tau * coupling_weight * (ZZ_matrix - renormalization_matrix))

			gate = op(SWAP_matrix * gate_matrix, s[a], s[b])
			push!(gates, gate)

			logical_qubit_order[a], logical_qubit_order[b] = logical_qubit_order[b], logical_qubit_order[a]
		end
	end

	# Apply single-qubit fields
	for i in 1:N
		qi = logical_qubit_order[i]
		field_strength = get(h, qi, 0.0)

		renormalization_matrix = UniformScaling(zvals[qi])
		gate_matrix = exp(-tau * field_strength * (Z_matrix - renormalization_matrix))

		gate = op(gate_matrix, s[i])
		push!(gates, gate)
	end
	return gates, logical_qubit_order
end

function triangular_SWAP_network_block(s,
	J::Dict{Tuple{Int, Int},
		Float64},
	h::Dict{Int, Float64},
	tau::Float64,
	logical_qubit_order::Vector{Int} = collect(1:length(s)),
	zzcorr::Matrix{Float64} = zeros(length(s), length(s)),
	zvals::Vector{Float64} = zeros(length(s)),
	bottom_up::Bool = true,  # If true, do the triangular SWAP network from bottom to top
)
	Z_matrix = [1 0; 0 -1]
	ZZ_matrix = [1 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 1]
	SWAP_matrix = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
	N = length(s)

	gates = ITensor[]

	# Apply two-qubit coupling via triangular SWAP network: layer 1 couples 1 and 2, layer 2 couples 2 and 3, etc.
	for layer in 1:(2*N - 3)
		if bottom_up
			max_pair = min(layer, 2*N - 2 - layer)
			if layer % 2 == 1
				pairs = [(i, i + 1) for i in 1:2:max_pair]
			else
				pairs = [(i, i + 1) for i in 2:2:max_pair]
			end
		elseif !bottom_up
			min_pair = max(N-layer, layer - (N - 2))
			if layer % 2 == 1
				pairs = [(i, i + 1) for i in (N-1):-2:min_pair]
			else
				pairs = [(i, i + 1) for i in (N-2):-2:min_pair]
			end
		end

		for (a, b) in pairs
			qa = logical_qubit_order[a]
			qb = logical_qubit_order[b]

			coupling_weight = get(J, (min(qa, qb), max(qa, qb)), 0.0)

			renormalization_matrix = UniformScaling(zzcorr[qa, qb])
			gate_matrix = exp(-tau * coupling_weight * (ZZ_matrix - renormalization_matrix))

			gate = op(SWAP_matrix * gate_matrix, s[a], s[b])
			push!(gates, gate)

			logical_qubit_order[a], logical_qubit_order[b] = logical_qubit_order[b], logical_qubit_order[a]
		end
	end

	# Apply single-qubit fields
	for i in 1:N
		qi = logical_qubit_order[i]
		field_strength = get(h, qi, 0.0)

		renormalization_matrix = UniformScaling(zvals[qi])
		gate_matrix = exp(-tau * field_strength * (Z_matrix - renormalization_matrix))
		gate = op(gate_matrix, s[i])
		push!(gates, gate)
	end
	return gates, logical_qubit_order
end

function Hadamard_block(s)
	H = (1 / sqrt(2)) * [1 1; 1 -1]
	N = length(s)
	gates = ITensor[]
	for i in 1:N
		gate = op(H, s[i])
		push!(gates, gate)
	end
	return gates
end

function initialize_superposition(sites)
	psi = productMPS(sites, "Up")
	Hadamard_gates = Hadamard_block(sites)
	psi = apply(Hadamard_gates, psi; cutoff = 1E-9, maxdim = 32)
	return psi
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
	network_architecture = "triangular",
	save_dir = missing,
	save_prefix = "results",
	seed = 42,
)
	# Set BLAS threads
	BLAS.set_num_threads(n_threads)

	# Load problem instance
	J, h, offset = load_ising(data_path)
	N = maximum(collect(Iterators.flatten(collect(keys(J)))))  # Number of qubits

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
		Random.seed!(seed)
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
	energy_std_t0 = energy_std
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
		"qubit_ordering" => qubit_ordering,
		"network_architecture" => network_architecture,
		"seed" => seed,
	)::Dict{String, Any}

	progress_bar = ProgressBar(1:Nsteps)
	for step in progress_bar
		# TEBD evolution
		tau_effective = tau * energy_std_t0 / energy_std^2
		#tau_effective = tau / energy_std
		tau_effective = clamp(tau_effective, tau_min, tau_max)

		if network_architecture === "triangular"
			#bottom_up = true  # Always do bottom-up for now
			bottom_up = step % 2 == 1
			TEBD_gates, logical_qubit_order = triangular_SWAP_network_block(sites, J, h, tau_effective, logical_qubit_order, zz_expvals, z_expvals, bottom_up)
		elseif network_architecture === "quadratic"
			TEBD_gates, logical_qubit_order = SWAP_network_block(sites, J, h, tau_effective, logical_qubit_order, zz_expvals, z_expvals)
		else
			error("Invalid network architecture: $network_architecture")
		end

		try
			psi = apply(TEBD_gates, psi; cutoff = cutoff, maxdim = chi)
		catch error
			println("Error during TEBD application: $error")
			break
		end
		#println("Norm after TEBD: ", norm(psi))
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