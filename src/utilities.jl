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

function get_expvals(psi, logical_qubit_order=collect(1:length(psi)))
    z_expvals = 2 * expect(psi, "Sz")
    zz_expvals = 4 * correlation_matrix(psi, "Sz", "Sz")
    # Reorder zvals and zzcorr
    z_expvals = z_expvals[logical_qubit_order]
    zz_expvals = zz_expvals[logical_qubit_order, logical_qubit_order]
    return z_expvals, zz_expvals
end

function get_samples(psi, N_s, logical_qubit_order)
    psi = orthogonalize(psi, 1)
    samples = [sample(psi).-1 for _ in 1:N_s]
    # If logical order is permuted, reorder bits in samples
    for s in samples
        s[:] = s[logical_qubit_order]
    end
    return samples
end

function get_entanglement_entropy(psi, b)
    psi = orthogonalize(psi, b)
    U,S,V = svd(psi[b], (linkinds(psi, b-1)..., siteinds(psi, b)...))
    SvN = 0.0
    for n=1:dim(S, 1)
        p = S[n,n]^2
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