using ITensors, ITensorMPS, Graphs, Plots, Statistics, JSON, LinearAlgebra, ProgressBars, Random, HDF5, LaTeXStrings

include("utilities.jl")
include("tebd.jl")
include("qrr.jl")

function callback(psi)
	return nothing
end

function save_results(
	state_list::Vector{MPS},
	state_energies_list::Vector{Float64},
	z_expvals_list::Vector{Vector{Float64}},
	zz_expvals_list::Vector{Matrix{Float64}},
	samples_list::Vector{Matrix{Int}},
	energy_samples_list::Vector{Vector{Float64}},
	time_list::Vector{Float64},
	params::Dict{String, Real},
	save_dir::String,
)
    z_expvals_array = reduce(hcat, z_expvals_list)
    zz_expvals_array = cat(zz_expvals_list...; dims=3)
    energy_samples_array = reduce(hcat, energy_samples_list)
    samples_array = cat(samples_list...; dims=3)
	# Save results to HDF5 file
	h5file = joinpath(save_dir, "time_sequence.h5")
	println("Saving results to $h5file")
	h5open(h5file, "w") do file
		write(file, "state_energies", state_energies_list)
		write(file, "z_expvals", z_expvals_array)
		write(file, "zz_expvals", zz_expvals_array)
		write(file, "samples", samples_array)
        write(file, "energy_samples", energy_samples_array)
		write(file, "time", time_list)
	end

    # Save parameters to JSON file
    paramfile = joinpath(save_dir, "params.json")
    open(paramfile, "w") do file
        JSON.print(file, params)

    
    end

    h5open("results.h5", "w") do f
        for (s, psi) in enumerate(state_list)
            g = create_group(f, "psi_$s")
            write(g, "MPS", psi)
        end
    end

	return nothing
end

let
	# MPS parameters
	chi = 32::Int
	cutoff = 1E-9::Float64

	# TEBD parameters
	tau = 1.0::Float64          # Timestep scalar
	tau_min = 1E-2::Float64      # Minimum timestep
	tau_max = 1E3::Float64      # Maximum timestep
	Nsamples = 1000::Int     # Number of samples per iteration
	Nsteps = 2::Int         # Number of TEBD steps

	# Make parameter dict for saving
	params = Dict(
		"chi" => chi,
		"cutoff" => cutoff,
		"tau" => tau,
		"tau_min" => tau_min,
		"tau_max" => tau_max,
		"Nsamples" => Nsamples,
		"Nsteps" => Nsteps,
	)::Dict{String, Real}

	# Hardware parameters
	n_threads = 8::Int
	BLAS.set_num_threads(n_threads)

	# Problem instance
	project_dir = joinpath(@__DIR__, "..")
	#path = joinpath(project_dir, "data", "MIS", "Ising", "ising_insecta-ant-colony1-day38.json")
	#path = joinpath(project_dir, "data", "MIS", "Ising", "ising_C125-9.json")
	problem_file = "ising_Ns10_Nt9_Nq1_K15_gamma0.5_zeta0.1_rho0.1.json"
	path = joinpath(project_dir, "data", "portfolio", "Ising", problem_file)
	J, h, offset = load_ising(path)
	N = maximum(collect(keys(h)))  # Number of qubits

	# Save directory

	save_dir = joinpath(project_dir, "results", "portfolio", "MPS", problem_file[1:end-5])
	isdir(save_dir) || mkpath(save_dir)

	println("Optimizing Ising model with $N qubits and $(length(collect(keys(J)))) couplings")

	# Initialize MPS
	sites = siteinds("S=1/2", N; conserve_qns = false)
	psi = initialize_superposition(sites)

	# Initial correlations and expectations
	z_expvals, zz_expvals = get_expvals(psi)
	energy = state_energy(z_expvals, zz_expvals, J, h, offset)

	init_logical_qubit_order = collect(1:N)
	#shuffle!(init_logical_qubit_order)
	logical_qubit_order = copy(init_logical_qubit_order)

	samples = get_samples(psi, Nsamples, logical_qubit_order)
    println("Initial samples shape: ", size(samples))
    println("Initial samples type: ", typeof(samples))
	#energy_samples = [sample_energy(s, J, h, offset) for s in samples]
    energy_samples = [sample_energy(samples[i, :], J, h, offset) for i in 1:size(samples, 1)]
	energy_std = Statistics.std(energy_samples)
	best_sample_energy = minimum(energy_samples)

	# Arrays to store data
	state_list = Vector{MPS}()
	push!(state_list, psi)

	state_energies_list = Vector{Float64}()
	push!(state_energies_list, energy)

	z_expvals_list = Vector{Vector{Float64}}()
	push!(z_expvals_list, z_expvals)

	zz_expvals_list = Vector{Matrix{Float64}}()
	push!(zz_expvals_list, zz_expvals)

	samples_list = Vector{Matrix{Int}}()
	push!(samples_list, samples)

    energy_samples_list = Vector{Vector{Float64}}()
    push!(energy_samples_list, energy_samples)

	time_list = Vector{Float64}()
	push!(time_list, 0.0)

	progress_bar = ProgressBar(1:Nsteps)
	for step in progress_bar
		# TEBD evolution
		tau_effective = tau / energy_std
		tau_effective = clamp(tau_effective, tau_min, tau_max)
		TEBD_gates, logical_qubit_order = SWAP_network_block(sites, J, h, tau_effective, logical_qubit_order, zz_expvals, z_expvals)
		psi = apply(TEBD_gates, psi; cutoff = cutoff, maxdim = chi)
		normalize!(psi)

		# State energy
		z_expvals, zz_expvals = get_expvals(psi, logical_qubit_order)
		energy = state_energy(z_expvals, zz_expvals, J, h, offset)

		samples = get_samples(psi, Nsamples, logical_qubit_order)
		num_unique_samples = length(unique(samples))

        energy_samples = [sample_energy(samples[i, :], J, h, offset) for i in 1:size(samples, 1)]
		energy_std = Statistics.std(energy_samples)
        best_sample_energy = minimum(energy_samples)

		# Store data
		push!(state_list, psi)
		push!(state_energies_list, energy)
		push!(z_expvals_list, z_expvals)
		push!(zz_expvals_list, zz_expvals)
		push!(samples_list, samples)
        push!(energy_samples_list, energy_samples)
		push!(time_list, time_list[end] + tau_effective)

		# Update progress bar
		set_multiline_postfix(
			progress_bar,
			"""
			State energy: $energy
			Best sample energy: $best_sample_energy
			Energy std of samples: $energy_std
			Effective time step: $tau_effective
			Number of unique samples: $num_unique_samples"""
		)
	end
	println("Saving")
	save_results(state_list, state_energies_list, z_expvals_list, zz_expvals_list, samples_list, energy_samples_list, time_list, params, save_dir)
	println("Done")
	# # Plot energies, set label fontsize
	# colwidth_pt = 345.0  # in points
	# inches_per_point = 1 / 72.27
	# fig_width_in = colwidth_pt * inches_per_point  # width in inches
	# golden_ratio = (5^0.5 - 1) / 2
	# fig_height_in = fig_width_in * golden_ratio * 2  # height in inches

	# # Set font size to 12 pt
	# plot_fontsize = 12

	# # When saving, use size in pixels: pixels = inches * dpi
	# dpi = 72  # 1 inch = 72 points in LaTeX
	# fig_width_px = round(Int, fig_width_in * dpi)
	# fig_height_px = round(Int, fig_height_in * dpi)

	# plot_font = "Computer Modern"
	# default(
	#     fontfamily=plot_font,
	#     linewidth=2, 
	#     framestyle=:box, 
	#     label=nothing, 
	#     grid=true,
	#     legendfontsize=12,
	#     tickfontsize=12,
	#     titlefontsize=12,
	# )
	# # Plot both in one figure
	# p1 = plot(1:Nsteps, state_energies_list, marker=2, xlabel="TEBD Steps", ylabel="Energy", title="(a)", titlelocation=:left)
	# p2 = histogram(energy_samples, bins=30, xlabel="Energy", ylabel="Frequency", title="(b)", titlelocation=:left)
	# plot(p1, p2, layout=(2, 1), size=(fig_width_px, fig_height_px))
end
