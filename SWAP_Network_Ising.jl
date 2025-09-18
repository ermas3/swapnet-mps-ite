using ITensors, ITensorMPS, Graphs, Plots, Statistics, JSON, LinearAlgebra, Random
  
# function expZZ(tau, J)
#   ZZ = [1 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 1]
#   return exp(-tau * J * ZZ)
# end

# function expZ(tau, h)
#   Z = [1 0; 0 -1]
#   return exp(-tau * h * Z)
# end

function load_ising(path::String)
    data = JSON.parsefile(path)

    Jdict = data["J"]
    hdict = data["h"]
    offset = get(data, "c", 0.0)  # Optional constant term

    J = Dict{Tuple{Int,Int}, Float64}()
    for (k, v) in Jdict
        i, j = split(k, ",")             # ["0","1"]
        i = parse(Int, i) + 1            # shift to 1-based
        j = parse(Int, j) + 1
        J[(i,j)] = v
    end

    h = Dict{Int, Float64}()
    for (k, v) in hdict
        i = parse(Int, k) + 1            # shift to 1-based
        h[i] = v
    end

    return J, h, offset
end

function sample_energy(
  sample::Vector{Int}, 
  J::Dict{Tuple{Int,Int}, Float64}, 
  h::Dict{Int, Float64}, 
  offset::Float64
  )::Float64
  energy = offset
  N = length(sample)
  for i in 1:N
    si = sample[i] == 0 ? 1 : -1
    energy += get(h, i, 0.0) * si
    for j in i+1:N
      sj = sample[j] == 0 ? 1 : -1
      energy += get(J, (i,j), 0.0) * si * sj
    end
  end
  return energy
end

function state_energy(
  zvals::Vector{Float64}, 
  zzcorr::Matrix{Float64}, 
  J::Dict{Tuple{Int,Int}, Float64}, 
  h::Dict{Int, Float64}, 
  offset::Float64
  )::Float64
  energy = offset
  N = length(zvals)
  for i in 1:N
    si = zvals[i]
    energy += get(h, i, 0.0) * si
    for j in i+1:N
      sj = zvals[j]
      energy += get(J, (i,j), 0.0) * zzcorr[i,j]
    end
  end
  return energy
end

function SWAP_network_block(s, 
  J::Dict{Tuple{Int,Int}, 
  Float64}, 
  h::Dict{Int, Float64}, 
  tau::Float64, 
  logical_qubit_order::Vector{Int}=collect(1:length(s)), 
  zzcorr::Matrix{Float64}=zeros(length(s), length(s)), 
  zvals::Vector{Float64}=zeros(length(s))
  )

  SWAP_matrix = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
  N = length(s)

  gates = ITensor[]
  # Apply two-qubit coupling via SWAP network
  for layer in 1:N
    if layer % 2 == 1
      pairs = [(i, i+1) for i in 1:2:N-1 if i+1 <= N]
    else layer % 2 == 0
      pairs = [(i, i+1) for i in 2:2:N-1 if i+1 <= N]
    end

    for (a, b) in pairs
      qa = logical_qubit_order[a]
      qb = logical_qubit_order[b]
      
      coupling_weight = get(J, (min(qa,qb), max(qa,qb)), 0.0)

      # if coupling_weight != 0.0
      #   gate = op(expZZ(tau, coupling_weight), s[a], s[b])
      #   push!(gates, gate)
      # end

      # SWAP_gate = ITensorMPS.op(SWAP, s[a], s[b])
      # push!(gates, SWAP_gate)

      # TODO: Is it faster to combine all three operations into a single gate?
      # gate_matrix = expZZ(tau, coupling_weight)
      # renormalization_matrix = UniformScaling(exp(zzcorr[qa, qb] * tau * coupling_weight))
      ZZ_matrix = [1 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 1]
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

    # if field_strength != 0.0
    #   gate = op(expZ(tau, field_strength), s[i])
    #   push!(gates, gate)
    # end

    # gate_matrix = expZ(tau, field_strength)
    # renormalization_matrix = UniformScaling(exp(zvals[qi] * tau * field_strength))
    Z_matrix = [1 0; 0 -1]
    renormalization_matrix = UniformScaling(zvals[qi])
    gate_matrix = exp(-tau * field_strength * (Z_matrix - renormalization_matrix))
    gate = op(gate_matrix, s[i])
    push!(gates, gate)
  end

  return gates, logical_qubit_order
end

function Hadamard_block(s)
  H = (1/sqrt(2)) * [1 1; 1 -1]
  N = length(s)
  gates = ITensor[]
  for i in 1:N
    gate = op(H, s[i])
    push!(gates, gate)
  end
  return gates
end

function get_QRR_matrix(zzcorr::Matrix{Float64})
  # Set diagonal to zero
  for i in 1:size(zzcorr, 1)
    zzcorr[i,i] = 0.0
  end
  return -zzcorr
end

function get_QRR(
  zzcorr::Matrix{Float64}, 
  J::Dict{Tuple{Int,Int}, Float64}, 
  h::Dict{Int, Float64}, 
  offset::Float64
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

function is_independent_set(
  bitstring::Vector{Int}, 
  edges::Vector{Tuple{Int,Int}}
  )::Bool
  for (i, j) in edges
    if bitstring[i] == 1 && bitstring[j] == 1
      return false
    end
  end
  return true
end

let
  BLAS.set_num_threads(8)
  # TEBD parameters
  cutoff = 1E-9
  tau = 5E-2 # Initial timestep
  ttotal = 30.0
  chi = 8
  #dE_target = 100

  # Load JSON file 
  #path = "data/MIS/Ising/ising_C125-9.json"
  path = "data/MIS/Ising/ising_insecta-ant-colony1-day38.json"
  #path = "data/portfolio/Ising/ising_Ns10_Nt9_Nq2_K15_gamma0.5_zeta0.1_rho0.1.json"
  J, h, offset = load_ising(path)
  N = maximum(collect(keys(h)))  # Number of qubits

  println("Optimizing Ising model with $N qubits and $(length(collect(keys(J)))) couplings")

  # Make an array of 'site' indices
  sites = siteinds("S=1/2", N; conserve_qns=false)
  println("Number of sites: $(length(sites))")
  Hadamard_gates = Hadamard_block(sites)

  # Initial state |+++...+>
  psi = productMPS(sites, "Up")
  psi = apply(Hadamard_gates, psi; cutoff=cutoff, maxdim=chi)

  # Initial correlations and expectations
  zzcorr = 4 * correlation_matrix(psi,"Sz","Sz")
  zvals = 2 * expect(psi,"Sz")
  energy = state_energy(zvals, zzcorr, J, h, offset)
  energy_slope = nothing

  # Optimization loop
  init_logical_qubit_order = collect(1:N)
  #init_logical_qubit_order = randperm(N)
  logical_qubit_order = copy(init_logical_qubit_order)
  for t in 0.0:tau:ttotal
    println("Time: $t")

    # Generate N_s samples
    N_s = 1000
    psi = orthogonalize(psi, 1)
    samples = [sample(psi).-1 for _ in 1:N_s]
    # If logical order is permuted, reorder bits in samples
    if logical_qubit_order != init_logical_qubit_order
      for s in samples
        s[:] = s[logical_qubit_order]
      end
    end
    energy_samples = [sample_energy(s, J, h, offset) for s in samples]
    energy_std = Statistics.std(energy_samples)
    println("Energy std of samples: $energy_std.")

    # Apply SWAP Network block
    println("Effective time step: ", tau/energy_std)
    tau_effective = tau / energy_std
    tau_effective = clamp(tau_effective, 1E-2, 1E3)
    TEBD_gates, logical_qubit_order = SWAP_network_block(sites, J, h, tau_effective, logical_qubit_order, zzcorr, zvals)
    psi = apply(TEBD_gates, psi; cutoff=cutoff, maxdim=chi)

    println("Norm before normalization: ", norm(psi))
    normalize!(psi)

    zvals = 2 * expect(psi,"Sz")
    zzcorr = 4 * correlation_matrix(psi,"Sz","Sz")
    # Reorder zvals and zzcorr if logical order is permuted
    if logical_qubit_order != init_logical_qubit_order
      zvals = zvals[logical_qubit_order]
      zzcorr = zzcorr[logical_qubit_order, logical_qubit_order]
    end

    # Recalculate timestep
    new_energy = state_energy(zvals, zzcorr, J, h, offset)
    energy = new_energy
    # dE = new_energy - energy
    # energy_slope = - dE / tau
    # tau = tau * dE_target / energy_slope
    # Clip tau to reasonable values
    # tau = clamp(tau, 1E-2, 1E-1)
    # println("New timestep: $tau")


    # Calculate QRR solution
    #best_bitstring, best_energy = get_QRR(zzcorr, J, h, offset)

    # Print number of 1s in best sample
    best_sample = samples[argmin(energy_samples)]
    is_independent = is_independent_set(best_sample, collect(keys(J)))
    println("Best sample has $(sum(best_sample)) ones and is independent: $is_independent")

    # Compare energies
    println("State energy: $energy", " Best sample energy: $(minimum(energy_samples))")
  end

  # Generate N_s samples
  N_s = 1000
  psi = orthogonalize(psi, 1)
  samples = [sample(psi).-1 for _ in 1:N_s]

  # If logical order is permuted, reorder bits in samples
  if logical_qubit_order != init_logical_qubit_order
    for s in samples
      s[:] = s[logical_qubit_order]
    end
  end

  energy_samples = [sample_energy(s, J, h, offset) for s in samples]

  avg_energy = Statistics.mean(energy_samples)
  std_energy = Statistics.std(energy_samples)
  println("Average energy: $avg_energy ± $std_energy")

  #println("Best sample has $(sum(best_sample)) ones")
  #println("Best sample: $(samples[argmin(energy_samples)])")

  # Generate random bitstrings and compute their energies
  random_samples = [rand(0:1, N) for _ in 1:N_s]
  random_energies = [sample_energy(s, J, h, offset) for s in random_samples]

  avg_random_energy = Statistics.mean(random_energies)
  std_random_energy = Statistics.std(random_energies)
  println("Random bitstrings average energy: $avg_random_energy ± $std_random_energy")
  println("Random best energy: $(minimum(random_energies))")
end