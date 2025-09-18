function get_QRR_matrix(zzcorr::Matrix{Float64})
	# Set diagonal to zero
	for i in 1:size(zzcorr, 1)
		zzcorr[i, i] = 0.0
	end
	return -zzcorr
end

function get_QRR(
	zzcorr::Matrix{Float64},
	J::Dict{Tuple{Int, Int}, Float64},
	h::Dict{Int, Float64},
	offset::Float64,
)
	QRR_matrix = get_QRR_matrix(zzcorr)
	eigenvectors = eigen(QRR_matrix).vectors
	# Apply sign function to every element
	sign_eigenvectors = map(x -> x >= 0 ? 1 : -1, eigenvectors)
	# Iterate over the eigenvectors, calculate the energy of each bitstring, and return the one with the lowest energy
	min_energy = Inf
	best_bitstring = nothing
	for i in 1:size(sign_eigenvectors, 2)
		bitstring = sign_eigenvectors[:, i]
		bitstring = map(x -> x == 1 ? 0 : 1, bitstring)  # Convert from +1/-1 to 0/1
		energy = sample_energy(bitstring, J, h, offset)
		if energy < min_energy
			min_energy = energy
			best_bitstring = bitstring
		end
	end
	return best_bitstring, min_energy
end
