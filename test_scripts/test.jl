using ITensors, ITensorMPS, LinearAlgebra

N = 4
d = 3

s = [Index(d, "site=$n") for n in 1:N]

psi = productMPS(s, 3)

psi

# Sample 10
psi = orthogonalize(psi, 1)
sample(psi)

# Check norm
sites = [Index(d, "site=$n") for n in 1:N]

function superpos(site::Index)
    A = ITensor(site)
    for s in 1:dim(site)
        A[site => s] = 1/sqrt(dim(site))
    end
    return A
end

psi = MPS([superpos(s) for s in sites])
norm(psi)

N_s = 3
psi = orthogonalize(psi, 1)
samples = [sample(psi).-1 for _ in 1:N_s]

logical_qubit_order = collect(1:N)
# reversed
logical_qubit_order = collect(N:-1:1)
println(logical_qubit_order)
println(samples)
# If logical order is permuted, reorder bits in samples

for s in samples
  s[:] = s[logical_qubit_order]
end

println(samples)