"""
===============================================================================
 Parameter Estimation Example: RXN CLC - Syngas 133
===============================================================================
 Requirements in the active Julia environment:
     import Pkg
     Pkg.activate(".")
     Pkg.instantiate()
===============================================================================
"""

using FractionalModels
using XLSX, Statistics, SpecialFunctions, CairoMakie

# -----------------------------------------------------------------------------
# Simulation Settings
# -----------------------------------------------------------------------------
T = 50
N = 50
t = range(0, T; length = N + 1)

s = 3
l = 5
n = 6
m = 5

# -----------------------------------------------------------------------------
# Experimental Data
# -----------------------------------------------------------------------------
Data = XLSX.readxlsx(joinpath(@__DIR__, "..", "data", "rxn_clc_syngas_133.xlsx"))

u_exp = Dict{Tuple{Int, Int}, Array{Float64}}()
t_exp = Dict{Tuple{Int, Int}, Array{Float64}}()

for j in 1:s
    for (i, col) in enumerate('B':'G')
        u_exp[j, i] = vec(Data["$j"]["$(col)2:$(col)8"])
        t_exp[j, i] = vec(Data["$j"]["A2:A8"])
    end
end

# -----------------------------------------------------------------------------
# Initial Conditions
# -----------------------------------------------------------------------------
x0 = zeros(s, n + m)
for j in 1:s, i in 1:n
    x0[j, i] = u_exp[j, i][1]
end

# -----------------------------------------------------------------------------
# Model Constants
# -----------------------------------------------------------------------------
Temp = [550, 600, 650] .+ 273.15
Temp_c = 600 + 273.15
Rgas = 0.0012062
Rg = 8.3145e-3
C0_NiO = 390 / 50.7

# -----------------------------------------------------------------------------
# Dynamic Model
# -----------------------------------------------------------------------------
function sys!(dx, x, t, p; u = nothing)
    ᾱ = view(p, 1:m)
    kapp = view(p, m + 1:m + l)
    Ea = -view(p, m + l + 1:m + 2l) ./ Rg

    function Du(i)
        return u.x0[i] * t^(ᾱ[i] - 1) / gamma(ᾱ[i]) + x[n + i]
    end

    temp_factor = 1 / u.T_sys - 1 / Temp_c
    k(j) = kapp[j] * exp(Ea[j] * temp_factor) * u.T_sys * Rgas

    r = zeros(l)
    r[1] = k(1) * (1 - x[6]) * Du(1)
    r[2] = k(2) * (1 - x[6]) * Du(4)
    r[3] = k(3) * (1 - x[6]) * Du(3)
    r[4] = k(4) * x[6] * Du(2)
    r[5] = k(5) * x[6] * Du(5)

    dx[1] = -r[1]
    dx[2] = r[2] - r[4]
    dx[3] = 2 * r[1] - r[3] + r[5]
    dx[4] = r[1] - r[2] + r[4]
    dx[5] = r[3] - r[5]
    dx[6] = (r[1] + r[2] + r[3] - r[4] - r[5]) / C0_NiO

    dx[n + 1:n + m] .= dx[1:m]
end

# -----------------------------------------------------------------------------
# Objective And Fractional Orders
# -----------------------------------------------------------------------------
function obj!(x, x_exp)
    total_distance = zero(eltype(x[1, 1]))
    total_vectors = 0

    for j in axes(x, 2)
        for i in axes(x, 1)
            scale = maximum(x_exp[i, j])
            err = (x_exp[i, j] .- x[i, j]) ./ scale
            total_distance += mean(err .^ 2)
            total_vectors += 1
        end
    end

    return 100 * total_distance / total_vectors
end

function alpha(p)
    α = ones(n + m)
    α[n + 1:n + m] .= p[1:m]
    return α
end

# -----------------------------------------------------------------------------
# Estimation Problem
# -----------------------------------------------------------------------------
u = [(T_sys = Temp[j],) for j in 1:s]

data = [
    (
        t_exp = vec(t_exp[j, 1]),
        x_exp = vcat([reshape(u_exp[j, i], 1, :) for i in 1:n]...),
        x_map = collect(1:n),
    ) for j in 1:s
]

param_setup = (
    p0 = vec([1 1 1 1 1 0.0562 0.105 0.397 0.0154 0.089 4.02e-14 5.13 7.36 36.90 21.23]),
    p_lb = vcat(fill(0.5, m), zeros(2l)),
    p_up = vcat(ones(m), ones(l), fill(50, l)),
)

optim_setup = (
    print_level = 5,
    max_iter = 1000,
    tol = 1e-3,
    acceptable_tol = 1e-3,
    warn_on_unsolved_status = false,
)

p_ord = vec([1 1 1 1 1 0.0562 0.105 0.397 0.0154 0.089 4.02e-14 5.13 7.36 36.90 21.23])
p_guess = param_setup.p0

problem = prepare_estimation_problem(sys!, alpha, x0, u, data, obj!, T, N)

# -----------------------------------------------------------------------------
# Parameter Estimation
# -----------------------------------------------------------------------------
p_est = estimate_params(problem, param_setup; optim_setup = optim_setup)

println("\nParametros iniciales:")
show(IOContext(stdout, :limit => false), round.(p_guess, digits = 6))
println("\n\nParametros estimados:")
show(IOContext(stdout, :limit => false), round.(p_est, digits = 6))
println("\n")

# -----------------------------------------------------------------------------
# Model Evaluation
# -----------------------------------------------------------------------------
obj_ord = evaluate_objective(problem, p_ord)
obj_est = evaluate_objective(problem, p_est)
solution_ord = simulate_scenarios(sys!, alpha, x0, u, p_ord, T, N)
solution_est = simulate_scenarios(sys!, alpha, x0, u, p_est, T, N)

println("Objetivo orden entero: ", obj_ord)
println("Objetivo estimado: ", obj_est)
println("Mejora porcentual: ", 100 * obj_ord / obj_est - 100)
println("Tamano solucion orden entero: ", size(solution_ord[1]))
println("Tamano solucion estimada: ", size(solution_est[1]))

# -----------------------------------------------------------------------------
# Plot Results
# -----------------------------------------------------------------------------
fig = Figure(; size = (2400, 3000))
ax = Dict()

axis_specs = [
    (
        layout = fig[1, 1],
        yticks = 0:0.1:1,
        limits = ((0, 50), (0, 1)),
        ylabel = L"\text{CH_4 partial pressure [psi]}",
    ),
    (
        layout = fig[1, 2],
        yticks = 0:0.2:4,
        limits = ((0, 50), (0, 4)),
        ylabel = L"\text{CO_2 partial pressure [psi]}",
    ),
    (
        layout = fig[2, 1],
        yticks = 0:0.4:5,
        limits = ((0, 50), (0, 4.5)),
        ylabel = L"\text{H_2 partial pressure [psi]}",
    ),
    (
        layout = fig[2, 2],
        yticks = 0:0.3:3,
        limits = ((0, 50), (0, 3)),
        ylabel = L"\text{CO partial pressure [psi]}",
    ),
    (
        layout = fig[3, 1],
        yticks = 0:0.4:5,
        limits = ((0, 50), (0, 5)),
        ylabel = L"\text{H_2O partial pressure [psi]}",
    ),
    (
        layout = fig[3, 2],
        yticks = 0:0.1:1,
        limits = ((0, 50), (0, 0.8)),
        ylabel = L"\text{OC Oxygen Conversion}",
    ),
]

for i in 1:n
    spec = axis_specs[i]
    ax[i] = Axis(
        spec.layout;
        xticks = 0:10:50,
        yticks = spec.yticks,
        limits = spec.limits,
        xminorticksvisible = true,
        yminorticksvisible = true,
        xminorgridvisible = true,
        yminorgridvisible = true,
        xlabel = L"\text{Time [s]}",
        ylabel = spec.ylabel,
        xlabelsize = 40,
        ylabelsize = 40,
        xticklabelsize = 35,
        yticklabelsize = 35,
        titlealign = :left,
        titlesize = 30,
    )
end

Marker = [:circle, :rtriangle, :rect]
Color = [:red, :blue, :black]

SC = Dict{Int, Any}()
L1 = Dict{Int, Any}()
L2 = Dict{Int, Any}()

for j in 1:s, i in 1:n
    sc = scatter!(ax[i], data[j].t_exp, vec(data[j].x_exp[i, :]); marker = Marker[2], strokecolor = Color[j], color = :transparent, markersize = 35, strokewidth = 3)
    l1 = lines!(ax[i], t[:], solution_ord[j][i, :]; linewidth = 3, linestyle = :dash, color = Color[j])
    l2 = lines!(ax[i], t[:], solution_est[j][i, :]; linewidth = 3, linestyle = :solid, color = Color[j])

    if j == 1
        SC[i] = sc
        L1[i] = l1
        L2[i] = l2
    end
end

group_color = [MarkerElement(marker = :circle, color = color, markersize = 40) for color in Color]

for i in 1:n
    axislegend(
        ax[i],
        [SC[i], L1[i], L2[i]],
        [L"\text{Experimental data}", L"\text{Ordinary model}", L"\text{Fractional model}"];
        position = :lt,
        orientation = :vertical,
        labelsize = 40,
        patchsize = (50, 0),
        framevisible = false,
    )

    axislegend(
        ax[i],
        group_color,
        [L"\text{550 °C}", L"\text{600 °C}", L"\text{650 °C}"];
        position = :rt,
        orientation = :vertical,
        labelsize = 40,
        patchsize = (50, 0),
        framevisible = false,
    )
end

# -----------------------------------------------------------------------------
# Save Figure
# -----------------------------------------------------------------------------
fig_dir = joinpath(@__DIR__, "..", "fig")
mkpath(fig_dir)
save(joinpath(fig_dir, "rxn_clc_syngas_133.png"), fig)

println("Figura guardada en: fig/rxn_clc_syngas_133.png")
display(fig)
