using FractionalModels
using CairoMakie
using SpecialFunctions

# -----------------------------------------------------------------------------
# Simulation Settings
# -----------------------------------------------------------------------------
N = 1000
T = 10
t = range(0, T; length = N + 1)

# -----------------------------------------------------------------------------
# Model Parameters
# -----------------------------------------------------------------------------
α_1 = [0.8; 0.9]
x1_0 = [100.0; 0.0]

α_2 = [1.0; 1.0; α_1[1]; α_1[2]]
x2_0 = [x1_0[1]; x1_0[2]; 0.0; 0.0]

p = [0.75; 1.0]
n = [0.8; 0.8]

# -----------------------------------------------------------------------------
# Dynamic Systems
# -----------------------------------------------------------------------------
function sys!(dx, x, t, p)
    dx[1] = -p[1] * abs(x[1])^n[1]
    dx[2] = p[1] * abs(x[1])^n[1] - p[2] * abs(x[2])^n[2]
end

function sys2!(dx, x, t, p)
    dx[1] = -p[1] * abs(x2_0[1]^n[1] * t^(α_2[3] - 1) / gamma(α_2[3]) + x[3])
    dx[2] = p[1] * abs(x2_0[1]^n[1] * t^(α_2[3] - 1) / gamma(α_2[3]) + x[3]) -
            p[2] * abs(x2_0[2]^n[2] * t^(α_2[4] - 1) / gamma(α_2[4]) + x[4])
    dx[3] = n[1] * abs(x[1])^(n[1] - 1) * dx[1]
    dx[4] = n[2] * abs(x[2])^(n[2] - 1) * dx[2]
end

function sys3!(dx, x, t, p)
    dx[1] = -p[1] * abs(x2_0[1] * t^(α_2[3] - 1) / gamma(α_2[3]) + x[3])^n[1]
    dx[2] = p[1] * abs(x2_0[1] * t^(α_2[3] - 1) / gamma(α_2[3]) + x[3])^n[1] -
            p[2] * abs(x2_0[2] * t^(α_2[4] - 1) / gamma(α_2[4]) + x[4])^n[2]
    dx[3] = dx[1]
    dx[4] = dx[2]
end

# -----------------------------------------------------------------------------
# Solve Systems
# -----------------------------------------------------------------------------
println("Nuevo integrador")
time_sol1 = @elapsed sol1 = solve(sys!, α_1, x1_0, p, T, N)
println("Tiempo solve sys!: ", time_sol1, " s")

time_sol2 = @elapsed sol2 = solve(sys2!, α_2, x2_0, p, T, N)
println("Tiempo solve sys2!: ", time_sol2, " s")

time_sol3 = @elapsed sol3 = solve(sys3!, α_2, x2_0, p, T, N)
println("Tiempo solve sys3!: ", time_sol3, " s")
println()

# -----------------------------------------------------------------------------
# Plot Results
# -----------------------------------------------------------------------------
fig = Figure(size = (1400, 700))
ax1 = Axis(
    fig[1, 1];
    xlabel = "Tiempo [s]",
    ylabel = "Estado 1",
    title = "Comparacion del nuevo integrador",
)
ax2 = Axis(
    fig[1, 2];
    xlabel = "Tiempo [s]",
    ylabel = "Estado 2",
    title = "Comparacion del nuevo integrador",
)

series = [
    (sol1, :solid, :black, "sys!"),
    (sol2, :dash, :blue, "sys2!"),
    (sol3, :dot, :red, "sys3!"),
]

for (sol, linestyle, color, label) in series
    lines!(ax1, t, sol[1, :], linestyle = linestyle, color = color, linewidth = 2, label = label)
    lines!(ax2, t, sol[2, :], linestyle = linestyle, color = color, linewidth = 2, label = label)
end

axislegend(ax1, position = :rt)
axislegend(ax2, position = :rt)

# -----------------------------------------------------------------------------
# Save Figure
# -----------------------------------------------------------------------------
fig_dir = joinpath(@__DIR__, "..", "fig")
mkpath(fig_dir)
save(joinpath(fig_dir, "test.png"), fig)

println("Figura guardada en: fig/test.png")
display(fig)
