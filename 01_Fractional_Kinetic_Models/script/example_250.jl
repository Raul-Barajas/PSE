"""
===============================================================================
 Parameter Estimation Example: RXN CLC - Syngas 250
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
N = 100
t = range(0, T; length = N + 1)

s = 3
l = 5
n = 6
m = 5

# -----------------------------------------------------------------------------
# Experimental Data
# -----------------------------------------------------------------------------
Data = XLSX.readxlsx(joinpath(@__DIR__, "..", "data", "rxn_clc_syngas_250.xlsx"))

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
    Ea = -view(p, m + l + 1:m + 2l) .* 100 ./ Rg

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
            total_distance += mean((err).^2)
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

p_ord = vec([1 1 1 1 1 0.0303 0.125 0.298 0.0294 0.191 3.6e-13 / 100 13.02 / 100 20.10 / 100 40.17 / 100 30.31 / 100])
p_frac = vec([0.999998 0.999901 1.0 0.833271 0.864344 0.0673656 0.169116 0.306081 0.00941967 0.300519 0.18785 0.0125216 0.165233 0.574386 0.369819])

param_setup = (
    p0 = p_ord,
    p_lb = vcat(fill(0.5, m), zeros(2l)),
    p_up = ones(m + 2l).+1e-6,
)

optim_setup = (
    print_level = 5,
    max_iter = 1000,
    tol = 1e-3,
    acceptable_tol = 1e-3,
    warn_on_unsolved_status = false,
)

problem = prepare_estimation_problem(sys!, alpha, x0, u, data, obj!, T, N)

# -----------------------------------------------------------------------------
# Parameter Selection
# -----------------------------------------------------------------------------
# Use `estimate_params(problem, param_setup; optim_setup = optim_setup)` here to
# re-estimate this case. The stored fractional vector below reproduces the
# original case-study comparison quickly.
p_est = p_frac
p_est = estimate_params(problem, param_setup; optim_setup = optim_setup)

# -----------------------------------------------------------------------------
# Model Evaluation
# -----------------------------------------------------------------------------
obj_ord = evaluate_objective(problem, p_ord)
obj_est = evaluate_objective(problem, p_est)
solution_ord = simulate_scenarios(sys!, alpha, x0, u, p_ord, T, N)
solution_est = simulate_scenarios(sys!, alpha, x0, u, p_est, T, N)

println("\nParametros ordinarios:")
show(IOContext(stdout, :limit => false), round.(p_ord, digits = 6))
println("\n\nParametros fraccionales:")
show(IOContext(stdout, :limit => false), round.(p_est, digits = 6))
println("\n")

println("Objetivo orden entero: ", obj_ord)
println("Objetivo fraccional: ", obj_est)
println("Mejora porcentual: ", 100 * obj_ord / obj_est - 100)
println("Tamano solucion orden entero: ", size(solution_ord[1]))
println("Tamano solucion fraccional: ", size(solution_est[1]))

# -----------------------------------------------------------------------------
# Plot Concentrations
# -----------------------------------------------------------------------------
fig = Figure(; size = (2400, 3000))
ax = Dict()

axis_specs = [
    (
        layout = fig[1, 1],
        yticks = 0:0.2:2,
        limits = ((0, 50), (0, 0.6)),
        ylabel = L"\text{CH_4 concentration [mol/m^3]}",
    ),
    (
        layout = fig[1, 2],
        yticks = 0:0.5:4,
        limits = ((0, 50), (0.5, 2)),
        ylabel = L"\text{CO_2 concentration [mol/m^3]}",
    ),
    (
        layout = fig[2, 1],
        yticks = 0:0.5:5,
        limits = ((0, 50), (0, 2.5)),
        ylabel = L"\text{H_2 concentration [mol/m^3]}",
    ),
    (
        layout = fig[2, 2],
        yticks = 0:0.2:2,
        limits = ((0, 50), (0, 1.2)),
        ylabel = L"\text{CO concentration [mol/m^3]}",
    ),
    (
        layout = fig[3, 1],
        yticks = 0:0.5:5,
        limits = ((0, 50), (0, 2.5)),
        ylabel = L"\text{H_2O concentration [mol/m^3]}",
    ),
    (
        layout = fig[3, 2],
        yticks = 0:0.2:0.8,
        limits = ((0, 50), (0, 0.6)),
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
    sc = scatter!(ax[i], t_exp[j, i], u_exp[j, i]; marker = Marker[2], strokecolor = Color[j], color = :transparent, markersize = 35, strokewidth = 3)
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
# Plot CO2 Yield
# -----------------------------------------------------------------------------
yield_CO2_data = Dict{Int, Vector{Float64}}()
yield_CO2_ord = Dict{Int, Vector{Float64}}()
yield_CO2_frac = Dict{Int, Vector{Float64}}()

for j in 1:s
    yield_CO2_data[j] = u_exp[j, 2] ./ (u_exp[j, 1] .+ u_exp[j, 2] .+ u_exp[j, 4])
    yield_CO2_ord[j] = solution_ord[j][2, :] ./ (solution_ord[j][1, :] .+ solution_ord[j][2, :] .+ solution_ord[j][4, :])
    yield_CO2_frac[j] = solution_est[j][2, :] ./ (solution_est[j][1, :] .+ solution_est[j][2, :] .+ solution_est[j][4, :])
end

fig_yield = Figure(; size = (1200, 1000))
ax_yield = Axis(
    fig_yield[1, 1];
    xticks = 0:10:50,
    yticks = 0:0.1:2,
    limits = ((0, 50), (0.4, 1.0)),
    xminorticksvisible = true,
    yminorticksvisible = true,
    xminorgridvisible = true,
    yminorgridvisible = true,
    xlabel = L"\text{Time [s]}",
    ylabel = L"\text{CO_2 yield}",
    xlabelsize = 40,
    ylabelsize = 40,
    xticklabelsize = 35,
    yticklabelsize = 35,
    titlealign = :left,
    titlesize = 30,
)

SC_yield = Dict{Int, Any}()
L1_yield = Dict{Int, Any}()
L2_yield = Dict{Int, Any}()

for j in 1:s
    SC_yield[j] = scatter!(ax_yield, t_exp[j, 2], yield_CO2_data[j]; marker = Marker[2], strokecolor = Color[j], color = :transparent, markersize = 30, strokewidth = 3)
    L1_yield[j] = lines!(ax_yield, t[:], yield_CO2_ord[j]; linewidth = 3, linestyle = :dash, color = Color[j])
    L2_yield[j] = lines!(ax_yield, t[:], yield_CO2_frac[j]; linewidth = 3, linestyle = :solid, color = Color[j])
end

axislegend(
    ax_yield,
    [SC_yield[1], L1_yield[1], L2_yield[1]],
    [L"\text{Experimental data}", L"\text{Ordinary model}", L"\text{Fractional model}"];
    position = :lt,
    orientation = :vertical,
    labelsize = 40,
    patchsize = (50, 0),
    framevisible = false,
)

axislegend(
    ax_yield,
    group_color,
    [L"\text{550 °C}", L"\text{600 °C}", L"\text{650 °C}"];
    position = :rb,
    orientation = :vertical,
    labelsize = 40,
    patchsize = (50, 0),
    framevisible = false,
)

# -----------------------------------------------------------------------------
# Yield Error
# -----------------------------------------------------------------------------
sample_position = map(ti -> argmin(abs.(t .- ti)), t_exp[1, 1])

ord_yield_error = 100 * sum(mean(abs.(yield_CO2_data[j] .- yield_CO2_ord[j][sample_position]) ./ maximum(yield_CO2_data[j])) for j in 1:s) / s
frac_yield_error = 100 * sum(mean(abs.(yield_CO2_data[j] .- yield_CO2_frac[j][sample_position]) ./ maximum(yield_CO2_data[j])) for j in 1:s) / s

println("Error yield CO2 ordinario: ", ord_yield_error)
println("Error yield CO2 fraccional: ", frac_yield_error)

# -----------------------------------------------------------------------------
# Save Figures
# -----------------------------------------------------------------------------
fig_dir = joinpath(@__DIR__, "..", "fig")
mkpath(fig_dir)

save(joinpath(fig_dir, "rxn_clc_syngas_250.png"), fig)
save(joinpath(fig_dir, "rxn_clc_syngas_250_yield.png"), fig_yield)

println("Figura guardada en: fig/rxn_clc_syngas_250.png")
println("Figura guardada en: fig/rxn_clc_syngas_250_yield.png")

display(fig)
display(fig_yield)
