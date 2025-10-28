using ITensors, ITensorMPS, Graphs, Plots, Statistics, JSON, LinearAlgebra, ProgressBars, Random, HDF5, SparseArrays, Arpack

include("utilities.jl")
include("tebd.jl")
include("qrr.jl")

function sweep_parameters()
	chi_values = [8, 16, 32, 64, 128]
	cutoff = 1E-9
	tau_min = 1E-9
	tau_max = 1000.0
	Nsamples = 1000
	Nsteps = 30
	n_threads = 1
	qubit_orderings = ["shuffle", "fiedler"]
	network_architectures = ["triangular", "quadratic"]
	graph_type = ["ER", "SK", "3Reg"]
	Nv = 100

	directory_name = "MPS_run2"

	max_idx = 0

	for graph in graph_type
		for chi in chi_values
			for qubit_ordering in qubit_orderings
				for network_architecture in network_architectures
					for idx in 10:19
						if graph == "SK" && qubit_ordering == "fiedler"
							continue
						end

						if graph == "ER"
							tau = 3/50
							data_path = joinpath(@__DIR__, "..", "data", "MaxCut", "ER", "100v", "Ising", "ising_graph$(idx).json")
							save_dir = joinpath(@__DIR__, "..", "results", "MaxCut", "ER", "100v", directory_name, "ising_graph$(idx)_$(network_architecture)_chi$(chi)_$(qubit_ordering)")
						elseif graph == "3Reg"
							tau = 1.0
							data_path = joinpath(@__DIR__, "..", "data", "MaxCut", "3Reg", "100v", "Ising", "ising_graph$(idx).json")
							save_dir = joinpath(@__DIR__, "..", "results", "MaxCut", "3Reg", "100v", directory_name, "ising_graph$(idx)_$(network_architecture)_chi$(chi)_$(qubit_ordering)")

						elseif graph == "SK"
							tau = 3/100
							data_path = joinpath(@__DIR__, "..", "data", "MaxCut", "SK", "100v", "Ising", "ising_graph$(idx).json")
							save_dir = joinpath(@__DIR__, "..", "results", "MaxCut", "SK", "100v", directory_name, "ising_graph$(idx)_$(network_architecture)_chi$(chi)_$(qubit_ordering)")

						elseif graph == "portfolio"
							tau = 10.0
							data_path = joinpath(@__DIR__, "..", "data", "portfolio", "Ising", "ising_Ns10_Nt9_Nq2_K10_gamma1_zeta0.004_rho1.0.json")
							save_dir = joinpath(@__DIR__, "..", "results", "portfolio", "MPS_nu0_001", "ising_Ns10_Nt9_Nq2_K10_gamma1_zeta0.004_rho1.0_$(network_architecture)_chi$(chi)_$(qubit_ordering)")
						else
							error("Unknown graph type: $graph")
						end

						run_TEBD(
							data_path,
							chi = chi,
							cutoff = cutoff,
							tau = tau,
							tau_min = tau_min,
							tau_max = tau_max,
							Nsamples = Nsamples,
							Nsteps = Nsteps,
							n_threads = n_threads,
							qubit_ordering = qubit_ordering,
							network_architecture = network_architecture,
							save_dir = save_dir,
							seed = idx,
						)
					end
				end
			end
		end
	end
end


let	
	#sweep_parameters() # Uncomment to run parameter sweep

    # Parameters for running MPS-QITE on multiple instances
	chi = 64
	cutoff = 1E-9
	tau = 10.0
	tau_min = 1E-9
	tau_max = 1000
	Nsamples = 1000
	Nsteps = 20
	n_threads = 1
	qubit_ordering = "fiedler"  # "default", "fiedler", or "shuffle"
	network_architecture = "triangular"  # "triangular" or "quadratic"

	Nv = 100

	for idx in 0:1
		data_path = joinpath(@__DIR__, "..", "data", "portfolio", "Ising", "ising_Ns10_Nt9_Nq2_K10_gamma1_zeta0.042_rho1.0.json")
		save_dir = joinpath(@__DIR__, "..", "results", "portfolio", "MPS", "ising_Ns10_Nt9_Nq2_K10_gamma1_zeta0.042_rho1.0_$(network_architecture)_chi$(chi)_$(qubit_ordering)")

		data_path = joinpath(@__DIR__, "..", "data", "MaxCut", "SK", "$(Nv)v", "Ising", "ising_graph$(idx).json")
		save_dir = joinpath(@__DIR__, "..", "results", "MaxCut", "SK", "$(Nv)v", "MPS", "ising_graph$(idx)")

		data_path = joinpath(@__DIR__, "..", "data", "sk_model", "$(Nv)v", "Ising", "isinπg_graph$(idx).json")
		save_dir = joinpath(@__DIR__, "..", "results", "sk_model", "$(Nv)v", "MPS", "ising_graph$(idx)_$network_architecture")

		data_path = joinpath(@__DIR__, "..", "data", "MaxCut", "3Reg", "100v", "Ising", "ising_graph$(idx).json")
		save_dir = joinpath(@__DIR__, "..", "results", "MaxCut", "3Reg", "100v", "MPS", "ising_graph$(idx)")
		
		run_TEBD(
			data_path,
			chi = chi,
			cutoff = cutoff,
			tau = tau,
			tau_min = tau_min,
			tau_max = tau_max,
			Nsamples = Nsamples,
			Nsteps = Nsteps,
			n_threads = n_threads,
			qubit_ordering = qubit_ordering,
			network_architecture = network_architecture,
			save_dir = save_dir,
			seed = idx,
		)
	end
end

