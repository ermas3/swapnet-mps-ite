function load_ising(path::String)
	data = JSON.parsefile(path)

	Jdict = data["J"]
	hdict = data["h"]
	offset = get(data, "c", 0.0)  # Optional constant term

	J = Dict{Tuple{Int, Int}, Float64}()
	for (k, v) in Jdict
		i, j = split(k, ",")             # ["0","1"]
		i = parse(Int, i) + 1            # shift to 1-based
		j = parse(Int, j) + 1
		J[(i, j)] = v
	end

	h = Dict{Int, Float64}()
	for (k, v) in hdict
		i = parse(Int, k) + 1            # shift to 1-based
		h[i] = v
	end

	return J, h, offset
end

function state_energy(
	zvals::Vector{Float64},
	zzcorr::Matrix{Float64},
	J::Dict{Tuple{Int, Int}, Float64},
	h::Dict{Int, Float64},
	offset::Float64,
)::Float64
	energy = offset
	N = length(zvals)
	for i in 1:N
		si = zvals[i]
		energy += get(h, i, 0.0) * si
		for j in i+1:N
			sj = zvals[j]
			energy += get(J, (i, j), 0.0) * zzcorr[i, j]
		end
	end
	return energy
end

function sample_energy(
	sample::Vector{Int},
	J::Dict{Tuple{Int, Int}, Float64},
	h::Dict{Int, Float64},
	offset::Float64,
)::Float64
	energy = offset
	N = length(sample)
	for i in 1:N
		si = sample[i] == 0 ? 1 : -1
		energy += get(h, i, 0.0) * si
		for j in i+1:N
			sj = sample[j] == 0 ? 1 : -1
			energy += get(J, (i, j), 0.0) * si * sj
		end
	end
	return energy
end

function is_independent_set(
	bitstring::Vector{Int},
	edges::Vector{Tuple{Int, Int}},
)::Bool
	for (i, j) in edges
		if bitstring[i] == 1 && bitstring[j] == 1
			return false
		end
	end
	return true
end

function get_expvals(psi, logical_qubit_order = collect(1:length(psi)))
	z_expvals = 2 * expect(psi, "Sz")
	zz_expvals = 4 * correlation_matrix(psi, "Sz", "Sz")
	# Reorder zvals and zzcorr
	logical_to_site = invperm(logical_qubit_order)
	z_expvals = z_expvals[logical_to_site]
	zz_expvals = zz_expvals[logical_to_site, logical_to_site]
	return z_expvals, zz_expvals
end

function get_samples(psi, N_s, logical_qubit_order)
	psi = orthogonalize(psi, 1)
	samples = [sample(psi) .- 1 for _ in 1:N_s]
	# If logical order is permuted, reorder bits in samples
	logical_to_site = invperm(logical_qubit_order)
	for s in samples
		s[:] = s[logical_to_site]
	end
	return hcat(samples...)
end

function get_entanglement_entropy(psi, b)
	psi = orthogonalize(psi, b)
	U, S, V = svd(psi[b], (linkinds(psi, b - 1)..., siteinds(psi, b)...))
	SvN = 0.0
	for n ∈ 1:dim(S, 1)
		p = S[n, n]^2
		SvN -= p * log(p)
	end
	return SvN
end

function get_all_entanglement_entropies(psi)
	N = length(psi)
	entropies = Float64[]
	for b in 1:N-1
		SvN = get_entanglement_entropy(psi, b)
		push!(entropies, SvN)
	end
	return entropies
end

function save_results(
	state_list::Vector{MPS},
	state_energies_list::Vector{Float64},
	z_expvals_list::Vector{Vector{Float64}},
	zz_expvals_list::Vector{Matrix{Float64}},
	samples_list::Vector{Matrix{Int}},
	energy_samples_list::Vector{Vector{Float64}},
	time_list::Vector{Float64},
	params::Dict{String, Real},
	save_dir::String,
	file_prefix::String = "results"
)
	z_expvals_array = reduce(hcat, z_expvals_list)
	zz_expvals_array = cat(zz_expvals_list...; dims = 3)
	energy_samples_array = reduce(hcat, energy_samples_list)
	samples_array = cat(samples_list...; dims = 3)
	# Save results to HDF5 file
	h5file = joinpath(save_dir, "$(file_prefix).h5")
	println("Saving results to $h5file")
	h5open(h5file, "w") do file
		write(file, "state_energies", state_energies_list)
		write(file, "z_expvals", z_expvals_array)
		write(file, "zz_expvals", zz_expvals_array)
		write(file, "samples", samples_array)
		write(file, "energy_samples", energy_samples_array)
		write(file, "time", time_list)

		for (s, psi) in enumerate(state_list)
			g = create_group(file, "psi_$(s-1)")
			write(g, "MPS", psi)
		end
	end

	# Save parameters to JSON file
	paramfile = joinpath(save_dir, "params.json")
	open(paramfile, "w") do file
		JSON.print(file, params)


	end
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
    return ordering
end