using HDF5, Plots, ITensorMPS, ITensors, Statistics, LaTeXStrings, CSV, Tables

include("../src/utilities.jl")

graph_types = ["3Reg", "ER", "SK"]
chi_values = [8, 16, 32, 64, 128]
architectures = ["quadratic", "triangular"]
ordering = ["shuffle"]

# List subdirectories in dir
for chi in chi_values
    for graph_type in graph_types
        for architecture in architectures
            for order in ordering
                for idx in 0:0
                    if graph_type == "SK" && order == "fiedler"
                        continue
                    end

                    #path = "results/MaxCut/$(graph_type)/100v/MPS_fix_dt/ising_graph$(idx)_$(architecture)_chi$(chi)_$(order)/results.h5"
                    path = "results/portfolio/MPS_dtau10/ising_Ns10_Nt9_Nq2_K10_gamma1_zeta0.042_rho1.0_$(architecture)_chi$(chi)_$(order)/results.h5"

                    should_skip = h5open(path, "r") do file
                        haskey(file, "entanglement_entropy")
                    end

                    if should_skip
                        println("Entanglement entropy already exists for $(path); skipping.")
                        continue
                    end

                    # Otherwise, compute and save it
                    z_expvals = h5read(path, "z_expvals")
                    Nsites = size(z_expvals, 1)
                    Nsteps = size(z_expvals, 2)

                    entanglement_entropy_matrix = zeros(Nsteps, Nsites - 1)

                    for i in 0:Nsteps-1
                        mps = h5open(path, "r") do file
                            read(file["psi_$i"], "MPS", MPS)
                        end
                        entropies = get_all_entanglement_entropies(mps)
                        entanglement_entropy_matrix[i + 1, 1:Nsites - 1] = entropies
                    end

                    h5open(path, "r+") do file
                        write(file, "entanglement_entropy", entanglement_entropy_matrix)
                    end

                    println("Saved entanglement entropy for $(path)")
                end
            end
        end
    end
end

# path = "results/MaxCut/3Reg/50v/MPS/ising_graph1/results.h5"
# z_expvals = h5read(path, "z_expvals")

# Nsites = size(z_expvals, 1)
# Nsteps = size(z_expvals, 2)

# entanglement_entropy_matrix = zeros(Nsteps, Nsites-1)
# # Load all mps file["psi_$i"] for i in 0:length(state_energies)-1
# for i in 0:Nsteps-1
#     mps = h5open(path, "r") do file
#         read(file["psi_$i"], "MPS", MPS)
#     end
#     #print(mps)

#     entropies = get_all_entanglement_entropies(mps)
#     entanglement_entropy_matrix[i+1, 1:Nsites-1] = entropies
# end

# # Save to h5 file
# h5open(path, "r+") do file
#     dname = "entanglement_entropy"

#     if haskey(file, dname)
#         delete_object(file, dname)
#     end

#     write(file, dname, entanglement_entropy_matrix)
# end

# Store in 

# Save entanglement_entropy_matrix in CSV file
# CSV.write("figures/figures_mpl/entanglement_entropy_matrix_$(architecture)_$(graph_type).csv", Tables.table(entanglement_entropy_matrix))

# Heatmap of entanglement entropy vs time and bond
# heatmap(
#     entanglement_entropy_matrix,
#     xlabel="Bond",
#     ylabel="Step",
#     colorbar_title=L"S(\rho)",
#     color=:magma,
# )

