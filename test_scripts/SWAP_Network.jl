using ITensors, ITensorMPS, Graphs, Plots, Statistics

# Get matrix for exp(-tau w ZZ) where w is some parameter
function expZZ(tau, w)
  ZZ = [1 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 1]
  return exp(-tau * w * ZZ)
end

# Define function that take a sample and a graph and returns the cut
function cut(g::Graphs.SimpleGraph, sample::Vector{Int})::Int
  cut_size = 0
  for e in edges(g)
    u, v = src(e), dst(e)
    if sample[u] != sample[v]
      cut_size += 1
    end
  end
  return cut_size
end

function ising_energy(g::Graphs.SimpleGraph, sample::Vector{Int})
  energy = 0.0
  for e in edges(g)
    u, v = src(e), dst(e)
    su = sample[u] == 0 ? 1 : -1
    sv = sample[v] == 0 ? 1 : -1
    energy += su * sv
  end
  return energy
end

SWAP = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
H = (1/sqrt(2)) * [1 1; 1 -1]

let
  N = 30
  cutoff = 1E-4
  tau = 0.1
  ttotal = 2.0
  chi = 32

  # Generate a complete graph with N vertices
  #g = Graphs.complete_graph(N)
  g = Graphs.random_regular_graph(N, 3)  # 3-regular graph
  println("Generated graph with $(nv(g)) vertices and $(ne(g)) edges.")

  # Make an array of 'site' indices
  s = siteinds("S=1/2", N; conserve_qns=false)

  Hadamard_gates = ITensor[]
  for i in 1:N
    push!(Hadamard_gates, op(H, s[i]))
  end

  # Order of logical qubits starts as 1, 2, ..., N
  logical_qubit_order = collect(Int, 1:N)
  TEBD_gates = ITensor[]
  for _ in 1:2
    for layer in 1:N
      if layer % 2 == 1
        pairs = [(i, i+1) for i in 1:2:N-1 if i+1 <= N]
      else layer % 2 == 0
        pairs = [(i, i+1) for i in 2:2:N-1 if i+1 <= N]
      end

      for (a, b) in pairs
        qa = logical_qubit_order[a]
        qb = logical_qubit_order[b]

        if has_edge(g, qa, qb)
          TEBD_gate = op(expZZ(tau, 1.0), s[a], s[b])
          push!(TEBD_gates, TEBD_gate)
        end

        SWAP_gate = ITensorMPS.op(SWAP, s[a], s[b])
        push!(TEBD_gates, SWAP_gate)

        logical_qubit_order[a], logical_qubit_order[b] = logical_qubit_order[b], logical_qubit_order[a]
      end
    end
  end

  # Initialize MPS in |+++...+> state
  psi = productMPS(s, "Up")
  psi = apply(Hadamard_gates, psi; cutoff=cutoff, maxdim=chi)

  for t in 0.0:tau:ttotal
    println("Time: $t")
    psi = apply(TEBD_gates, psi; cutoff=cutoff, maxdim=chi)
    # Print norm
    println("Norm before normalization: ", norm(psi))
    normalize!(psi)
  end

  # Generate N_s samples
  N_s = 1000
  psi = orthogonalize(psi, 1)
  samples = [sample(psi) for _ in 1:N_s]
  cuts = [cut(g, s) for s in samples]
  energies = [ising_energy(s) for s in samples]

  avg_cut = Statistics.mean(cuts)
  std_cut = Statistics.std(cuts)
  println("Average cut: $avg_cut ± $std_cut")
  avg_energy = Statistics.mean(energies)
  std_energy = Statistics.std(energies)
  println("Average energy: $avg_energy ± $std_energy")

  # Generate random bitstrings and compute their cuts
  random_cuts = [cut(g, rand(0:1, N)) for _ in 1:N_s]
  avg_random_cut = Statistics.mean(random_cuts)
  std_random_cut = Statistics.std(random_cuts)
  println("Random bitstrings average cut: $avg_random_cut ± $std_random_cut")

end