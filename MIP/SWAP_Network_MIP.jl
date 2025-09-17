using ITensors, ITensorMPS, Graphs, Plots, Statistics, JSON, LinearAlgebra

function expN(tau, h, d)
  # N operator with diagonal entries [0, 1, 2, ..., d-1]
  N = diagm(0:d-1)
  return exp(-tau * h * N)
end

function expN_squared(tau, h2, d)
  # NN operator with diagonal entries [0, 1^2, 2^2, ..., (d-1)^2]
  N = diagm(0:(d-1))
  N_squared = N * N
  return exp(-tau * h2 * N_squared)
end

function expNN(tau, J, d)
  # NN operator on two d-level systems with diagonal entries [(i*j) for i in 0:d-1, j in 0:d-1]
  NN = zeros(d^2, d^2)
  for i in 0:d-1
    for j in 0:d-1
      NN[i*d + j + 1, i*d + j + 1] = i * j
    end
  end
  return exp(-tau * J * NN)
end

function load_ising(path::String)
    data = JSON.parsefile(path)

    Jdict = data["J"]
    hdict = data["h"]
    h2dict = data["h2"] 
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

    h2 = Dict{Int, Float64}()
    for (k, v) in h2dict
        i = parse(Int, k) + 1            # shift to 1-based
        h2[i] = v
    end

    return J, h, h2, offset
end

function dSWAP(d)
  # Swap gate on d-level system
  SWAP = zeros(d^2, d^2)
  for i in 1:d
    for j in 1:d
      SWAP[(i-1)*d + j, (j-1)*d + i] = 1.0
    end
  end
  return SWAP
end

function SWAP_network_block(s, J::Dict{Tuple{Int,Int}, Float64}, h::Dict{Int, Float64}, h2::Dict{Int, Float64}, tau::Float64, logical_qubit_order::Vector{Int}=collect(1:length(s)))
  N = length(s)
  d = dim(s[1])
  SWAP = dSWAP(d)
  gates = ITensor[]

  # Apply two-qubit coupling via SWAP network
  for layer in 1:N
    if layer % 2 == 1
      pairs = [(i, i+1) for i in 1:2:N-1 if i+1 <= N]
    else layer % 2 == 0 # TODO Possibly this should be reversed to reduce movement of orthogonal center
      pairs = [(i, i+1) for i in 2:2:N-1 if i+1 <= N]
    end

    for (a, b) in pairs
      qa = logical_qubit_order[a]
      qb = logical_qubit_order[b]
      
      coupling_weight = get(J, (min(qa,qb), max(qa,qb)), 0.0)
      if coupling_weight != 0.0
        gate = op(expNN(tau, coupling_weight, d), s[a], s[b])
        push!(gates, gate)
      end

      SWAP_gate = ITensorMPS.op(SWAP, s[a], s[b])
      push!(gates, SWAP_gate)

      logical_qubit_order[a], logical_qubit_order[b] = logical_qubit_order[b], logical_qubit_order[a]
    end
  end

  # Apply single-qubit fields
  for i in 1:N
    qi = logical_qubit_order[i]
    field_strength = get(h, qi, 0.0)
    if field_strength != 0.0
      gate = op(expN(tau, field_strength, d), s[i])
      push!(gates, gate)
    end
  end

  # Apply single qubit quadratic fields
  for i in 1:N
    qi = logical_qubit_order[i]
    field_strength2 = get(h2, qi, 0.0)
    if field_strength2 != 0.0
      gate = op(expN_squared(tau, field_strength2, d), s[i])
      push!(gates, gate)
    end
  end

  return gates, logical_qubit_order
end

# function SWAP_network_block(s, J::Dict{Tuple{Int,Int}, Float64}, h::Dict{Int, Float64}, h2::Dict{Int, Float64}, tau::Float64)
#   N = length(s)
#   d = dim(s[1])
#   SWAP = dSWAP(d)

#   logical_qubit_order = collect(Int, 1:N)
#   gates = ITensor[]

#   for _ in 1:2 # Two full sweeps to get logical qubits back to original order

#     # Apply two-qubit coupling via SWAP network
#     for layer in 1:N
#       if layer % 2 == 1
#         pairs = [(i, i+1) for i in 1:2:N-1 if i+1 <= N]
#       else layer % 2 == 0 # TODO Possibly this should be reversed to reduce movement of orthogonal center
#         pairs = [(i, i+1) for i in 2:2:N-1 if i+1 <= N]
#       end

#       for (a, b) in pairs
#         qa = logical_qubit_order[a]
#         qb = logical_qubit_order[b]
        
#         coupling_weight = get(J, (min(qa,qb), max(qa,qb)), 0.0)
#         if coupling_weight != 0.0
#           gate = op(expNN(tau, coupling_weight, d), s[a], s[b])
#           push!(gates, gate)
#         end

#         SWAP_gate = ITensorMPS.op(SWAP, s[a], s[b])
#         push!(gates, SWAP_gate)

#         logical_qubit_order[a], logical_qubit_order[b] = logical_qubit_order[b], logical_qubit_order[a]
#       end
#     end

#     # Apply single-qubit fields
#     for i in 1:N
#       qi = logical_qubit_order[i]
#       field_strength = get(h, qi, 0.0)
#       if field_strength != 0.0
#         gate = op(expN(tau, field_strength, d), s[i])
#         push!(gates, gate)
#       end
#     end

#     # Apply single qubit quadratic fields
#     for i in 1:N
#       qi = logical_qubit_order[i]
#       field_strength2 = get(h2, qi, 0.0)
#       if field_strength2 != 0.0
#         gate = op(expN_squared(tau, field_strength2, d), s[i])
#         push!(gates, gate)
#       end
#     end

#   end

#   return gates
# end

function sample_energy(sample::Vector{Int}, J::Dict{Tuple{Int,Int}, Float64}, h::Dict{Int, Float64}, h2::Dict{Int, Float64})
  energy = 0
  N = length(sample)
  for i in 1:N
    energy += get(h, i, 0.0) * sample[i]
    energy += get(h2, i, 0.0) * sample[i]^2
    for j in i+1:N
      energy += get(J, (i,j), 0.0) * sample[i] * sample[j]
    end
  end
  return energy
end


function superpos(site::Index)
    A = ITensor(site)
    for s in 1:dim(site)
        A[site => s] = 1/sqrt(dim(site))
    end
    return A
end

let
  # TEBD parameters
  cutoff = 1E-9
  tau = 4.0
  ttotal = 15*tau
  chi = 64
  d = 2  # local dimension NOTE! MUST MATCH Nq such that d = 2^N_q

  # Load JSON file 
  path = "data/MIP/MIP_Ns10_Nt9_Nq1_K15_gamma0.5_zeta0.1_rho0.1.json"
  println("Loading MIP model from $path")
  J, h, h2 = load_ising(path)
  N = maximum(collect(keys(h)))  # Number of integer variables

  println("Optimizing MIP model with $N integer variables and $(length(collect(keys(J)))) couplings")

  # Make an array of 'site' indices
  sites = [Index(d, "site=$n") for n in 1:N]
  println("Number of sites: $(length(sites))")
  psi = MPS([superpos(s) for s in sites])
  
  init_logical_qubit_order = collect(1:N)
  logical_qubit_order = copy(init_logical_qubit_order)
  for t in 0.0:tau:ttotal
    println("Time: $t")
    TEBD_gates, logical_qubit_order = SWAP_network_block(sites, J, h, h2, tau, logical_qubit_order)
    psi = apply(TEBD_gates, psi; cutoff=cutoff, maxdim=chi)
    println("Norm before normalization: ", norm(psi))
    normalize!(psi)
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

  energy_samples = [sample_energy(s, J, h, h2) for s in samples]

  avg_energy = Statistics.mean(energy_samples)
  std_energy = Statistics.std(energy_samples)
  println("Average energy: $avg_energy ± $std_energy")
  println("Best energy: $(minimum(energy_samples))")

  # Generate random bitstrings and compute their energies
  random_samples = [rand(0:d-1, N) for _ in 1:N_s]
  random_energies = [sample_energy(s, J, h, h2) for s in random_samples]

  avg_random_energy = Statistics.mean(random_energies)
  std_random_energy = Statistics.std(random_energies)
  println("Random bitstrings average energy: $avg_random_energy ± $std_random_energy")
  println("Best energy: $(minimum(random_energies))")
end