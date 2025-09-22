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
