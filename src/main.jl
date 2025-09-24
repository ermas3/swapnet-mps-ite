using ITensors, ITensorMPS, Graphs, Plots, Statistics, JSON, LinearAlgebra, ProgressBars, Random, HDF5, SparseArrays, Arpack

include("utilities.jl")
include("tebd.jl")
include("qrr.jl")

let	
	sweep_parameters()
	# # Parameters for running TEBD on multiple instances
	# chi = 32
	# cutoff = 1E-9
	# tau = 1.0
	# tau_min = 1E-9
	# tau_max = 15/50
	# Nsamples = 1000
	# Nsteps = 20
	# n_threads = 8
	# qubit_ordering = "shuffle"  # "default", "fiedler", or "shuffle"
	# network_architecture = "quadratic"  # "triangular" or "quadratic"

	# Nv = 100

	# for idx in 0:0
	# 	data_path = joinpath(@__DIR__, "..", "data", "MaxCut", "ER", "100v", "Ising", "ising_graph$(idx).json")
	# 	save_dir = joinpath(@__DIR__, "..", "results", "MaxCut", "ER", "100v", "MPS", "ising_graph$(idx)")

	# 	# data_path = joinpath(@__DIR__, "..", "data", "sk_model", "$(Nv)v", "Ising", "ising_graph$(idx).json")
	# 	# save_dir = joinpath(@__DIR__, "..", "results", "sk_model", "$(Nv)v", "MPS", "ising_graph$(idx)_$network_architecture")

	# 	# data_path = joinpath(@__DIR__, "..", "data", "MaxCut", "3Reg", "100v", "Ising", "ising_graph$(idx).json")
	# 	# save_dir = joinpath(@__DIR__, "..", "results", "MaxCut", "3Reg", "100v", "MPS", "ising_graph$(idx)")
		
	# 	run_TEBD(
	# 		data_path,
	# 		chi = chi,
	# 		cutoff = cutoff,
	# 		tau = tau,
	# 		tau_min = tau_min,
	# 		tau_max = tau_max,
	# 		Nsamples = Nsamples,
	# 		Nsteps = Nsteps,
	# 		n_threads = n_threads,
	# 		qubit_ordering = qubit_ordering,
	# 		network_architecture = network_architecture,
	# 		save_dir = save_dir,
	# 		seed = idx,
	# 	)
	# end
end

function sweep_parameters()
	chi_values = [16, 32, 64]
	tau = 1
	tau_min = 1E-9
	tau_max = 1E0
	Nsamples = 1000
	Nsteps = 30
	n_threads = 8
	qubit_ordering = ["fiedler", "shuffle"]
	network_architectures = ["triangular", "quadratic"]
	Nv = 100
	graph_types = ["ER", "3Reg"]

	max_problem_idx = 19


	for qubit_ordering in qubit_orderings
		for chi in chi_values
			for graph_type in graph_types
				for network_architecture in network_architectures
					for idx in 0:max_problem_idx
						if graph_type == "ER"
							tau_max = 15/60
						else
							tau_max = 15/3
						end
						println("Running TEBD for graph type $graph_type, architecture $network_architecture, chi $chi, instance $idx")
						println("-----------------------------------------------------")
						data_path = joinpath(@__DIR__, "..", "data", "MaxCut", graph_type, "$(Nv)v", "Ising", "ising_graph$(idx).json")
						save_dir = joinpath(@__DIR__, "..", "results", "MaxCut", graph_type, "$(Nv)v", "MPS_$(chi)", network_architecture, qubit_ordering, "ising_graph$(idx)")

						run_TEBD(
							data_path,
							chi = chi,
							cutoff = 1E-9,
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