using ITensors, ITensorMPS, Graphs, Plots

# Get matrix for exp(-tau w ZZ) where w is some parameter
function expZZ(tau, w)
  ZZ = [1 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 1]
  return exp(-tau * w * ZZ)
end

SWAP_matrix = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
H_matrix = (1/sqrt(2)) * [1 1; 1 -1]
PauliZ = [1 0; 0 -1]

let
  N = 100
  cutoff = 1E-8
  tau = 0.1
  ttotal = 5.0
  chi = 32

  # Generate a 3Reg graph with 100 vertices
  g = Graphs.random_regular_graph(N, 3)
  println("Generated graph with $(nv(g)) vertices and $(ne(g)) edges.")

  # Make an array of 'site' indices
  s = siteinds("S=1/2", N; conserve_qns=false)

  # Hadamard ladder
  Hadamard_gates = ITensor[]
  for i in 1:N
    H_gate = op(H_matrix, s[i])
    push!(Hadamard_gates, H_gate)
  end

  # Create the gates in SWAP-network
  TEBD_gates = ITensor[]

  # Iterate over edges in the graph
  for e in edges(g)
    u, v = src(e), dst(e)
    println("Processing edge: ($u, $v)")

    expZZ_gate = op(expZZ(tau, 1.0), s[u], s[v])
    push!(TEBD_gates, expZZ_gate)
  end
  println("Total number of gates created: ", length(TEBD_gates))

  c = div(N, 2)
  # Intialize MPS in all |0> states
  psi = productMPS(s, "Up")
  Sz = expect(psi, PauliZ, sites=c)

  # Apply Hadamard gates to all sites
  psi = apply(Hadamard_gates, psi; cutoff=cutoff, maxdim=chi)
  Sz = expect(psi, PauliZ, sites=c)

  for t in 0.0:tau:ttotal
    println("Time: $t")
    # Apply TEBD gates
    psi = apply(TEBD_gates, psi; cutoff=cutoff, maxdim=chi)
    Sz = expect(psi, PauliZ, sites=c)
    println("Sz at site $c: ", Sz)
  end
  # sites = siteinds("S=1/2", N; conserve_qns=true)
  # psi0 = productMPS(sites, "Up")

  # # Plot the MPS
  # SWAP_matrix = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
  # SWAP = op(SWAP_matrix, sites[1], sites[2])

  # # Apply SWAP to the first two sites
  # psi1 = apply(SWAP, psi0; cutoff=cutoff, maxdim=chi)
end