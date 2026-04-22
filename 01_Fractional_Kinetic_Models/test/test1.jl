include(joinpath(@__DIR__, "..", "src", "PECE.jl"))
include(joinpath(@__DIR__, "..", "src", "Old_PECE.jl"))
using Plots

# Parametros del experimento
N = 500
T = 50
t = (0:N) .* T ./ N

α_1 = [0.9; 0.7]
u1_0 = [100; 0]

α_2 = [1; 1; α_1[1]; α_1[2]]
u2_0 = [u1_0[1]; u1_0[2]; 0; 0]

u1_0_old = u1_0'
u2_0_old = u2_0'

p = [0.75; 0.5]
n = [0.8; 0.9]

# Sistemas a integrar
function sys!(du, u, t, p)
    du[1] = -p[1] * abs.(u[1]) .^ n[1]
    du[2] = p[1] * abs.(u[1]) .^ n[1] - p[2] * abs.(u[2]) .^ n[2]
end

function sys2!(du, u, t, p)
    du[1] = -p[1] * abs.(u2_0[1] .^ n[1] * t^(α_2[3] - 1) / gamma(α_2[3]) + u[3])
    du[2] = p[1] * abs.(u2_0[1] .^ n[1] * t^(α_2[3] - 1) / gamma(α_2[3]) + u[3]) - p[2] * abs.(u2_0[2] .^ n[2] * t^(α_2[4] - 1) / gamma(α_2[4]) + u[4])
    du[3] = n[1] * abs.(u[1]) .^ (n[1] - 1) * du[1]
    du[4] = n[2] * abs.(u[2]) .^ (n[2] - 1) * du[2]
end

function sys3!(du, u, t, p)
    du[1] = -p[1] * abs.(u2_0[1] * t^(α_2[3] - 1) / gamma(α_2[3]) + u[3]) .^ n[1]
    du[2] = p[1] * abs.(u2_0[1] * t^(α_2[3] - 1) / gamma(α_2[3]) + u[3]) .^ n[1] - p[2] * abs.(u2_0[2] * t^(α_2[4] - 1) / gamma(α_2[4]) + u[4]) .^ n[2]
    du[3] = du[1]
    du[4] = du[2]
end

# Nuevo integrador
println("Nuevo integrador")
time_sol1 = @elapsed sol1 = solve!(sys!, α_1, u1_0, p, T, N)
println("Tiempo solve! sys!: ", time_sol1, " s")

time_sol2 = @elapsed sol2 = solve!(sys2!, α_2, u2_0, p, T, N)
println("Tiempo solve! sys2!: ", time_sol2, " s")

time_sol3 = @elapsed sol3 = solve!(sys3!, α_2, u2_0, p, T, N)
println("Tiempo solve! sys3!: ", time_sol3, " s")
println()

# Viejo integrador
println("Viejo integrador")
time_sol11 = @elapsed sol11 = old_solve!(sys!, α_1, u1_0_old, p, T, N)
println("Tiempo old_solve! sys!: ", time_sol11, " s")

time_sol22 = @elapsed sol22 = old_solve!(sys2!, α_2, u2_0_old, p, T, N)
println("Tiempo old_solve! sys2!: ", time_sol22, " s")

time_sol33 = @elapsed sol33 = old_solve!(sys3!, α_2, u2_0_old, p, T, N)
println("Tiempo old_solve! sys3!: ", time_sol33, " s")
println()

# Resumen de mejora
improve_sys1 = 100 * (time_sol11 - time_sol1) / time_sol11
improve_sys2 = 100 * (time_sol22 - time_sol2) / time_sol22
improve_sys3 = 100 * (time_sol33 - time_sol3) / time_sol33

println("Mejora nuevo vs viejo en sys!: ", improve_sys1, " %")
println("Mejora nuevo vs viejo en sys2!: ", improve_sys2, " %")
println("Mejora nuevo vs viejo en sys3!: ", improve_sys3, " %")
println()

# Visualizacion

withenv("GKSwstype" => "100") do
    fig_test = plot(t, [sol1[1, :], sol1[2, :]], label="Original", lw=1, size=(1200, 800), dpi=300)
    plot!(t, [sol11[1, :], sol11[2, :]], label="Original_Antiguo_Integrador", lw=1)
    plot!(t, [sol2[1, :], sol2[2, :]], label="Nuevo", lw=2, ls=:dash)
    plot!(t, [sol22[1, :], sol22[2, :]], label="Nuevo_Antiguo_Integrador", lw=2, ls=:dash)
    plot!(t, [sol3[1, :], sol3[2, :]], label="Raul", lw=2, ls=:dot)
    plot!(t, [sol33[1, :], sol33[2, :]], label="Raul_Antiguo_Integrador", lw=2, ls=:dot)
    savefig(fig_test, "fig/test1.png")
end

println("Figura guardada en: fig/test1.png")
