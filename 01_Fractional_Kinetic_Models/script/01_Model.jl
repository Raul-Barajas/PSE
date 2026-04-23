"""
===============================================================================
 Algoritmo Para Ajuste de Parametros Cineticos: RXN CLC - Syngas 133
===============================================================================
 Autor:                     Luis Raúl Barajas Villarruel
 Fecha de modificación:     03 de junio de 2025

 Historial de cambios:
 - v2.0 (03-05-2025): Se utilizan concentraciones para calcular los parametros.
===============================================================================
"""

#Dependencias
using FractionalModels
using XLSX, Statistics, Optim, GLMakie, SpecialFunctions

# Leer los datos desde el archivo Excel
Data = XLSX.readxlsx(joinpath(@__DIR__, "..", "data", "01.xlsx"))

# Rango de tiempo y tamaño de paso
T = 50                              # T: segundos
N = 50                              # N: Numero de nodos
t = range(0, T; length=N + 1)

# Conjuntos
s = 3   # s: Número de temperaturas
l = 5   # l: Número de constantes cinéticas
n = 6   # n: Número de variables
m = 5   # m: Número de variables auxiliares

# Crear diccionarios para almacenar los datos experimentales
u_exp = Dict{Tuple{Int,Int},Array{Float64}}()
t_exp = Dict{Tuple{Int,Int},Array{Float64}}()

# Almacenar los datos en los diccionarios
for j in 1:s
    for (i, col) in enumerate('B':'G')
        u_exp[j, i] = Data["$j"]["$(col)2:$(col)8"]
        t_exp[j, i] = Data["$j"]["A2:A8"]
    end
end

# Valores iniciales
x0 = zeros(s, n + m)
for j in 1:s, i in 1:n
    x0[j, i] = u_exp[j, i][1]
end

# Datos del modelo
Temp = [550, 600, 650] .+ 273.15    # Temp: Temperaturas experimentales en K
Temp_c = 600 + 273.15               # Temp_c: Temperatura central en K
Rgas = 0.0012062                    # Rgas: Constante de los gases en psi-m3 / mol-k
Rg = 8.3145e-3                      # Rg: Constante de los gases en kj / mol-k
C0_NiO = 390 / 50.7                 # C0_NiO: Concentracion inicial de NiO en mol/m3

# Modelo dinámico
function sys!(dx, x, t, p; u=nothing)
    # Parámetros a optimizar (vector p)
    ᾱ = view(p, 1:m)                    # ᾱ: Orden de la derivada fraccional
    kapp = view(p, m+1:m+l)             # kapp: Constante cinetica aparente en mol / m3-psi-s
    Ea = -view(p, m+l+1:m+2*l) ./ Rg    # Ea: Energia de activacion en Kj / mol

    # D(u): Función interna para calcular la derivada de R-L
    function Du(i)
        return u.x0[i] * t^(ᾱ[i] - 1) / gamma(ᾱ[i]) + x[n+i]
    end

    # Factor de activación térmica
    temp_factor = 1 / u.T_sys - 1 / Temp_c

    # Función para calcular k
    k(j) = kapp[j] * exp(Ea[j] * temp_factor) * u.T_sys * Rgas    # k: Constante cinetica en 1/s

    # Definir velocidades de reacción
    r = zeros(l)
    r[1] = k(1) * (1 - x[6]) * Du(1)
    r[2] = k(2) * (1 - x[6]) * Du(4)
    r[3] = k(3) * (1 - x[6]) * Du(3)
    r[4] = k(4) * (x[6]) * Du(2)
    r[5] = k(5) * (x[6]) * Du(5)

    # Definir ecuaciones diferenciales
    dx[1] = (-r[1])
    dx[2] = (r[2] - r[4])
    dx[3] = (2 * r[1] - r[3] + r[5])
    dx[4] = (r[1] - r[2] + r[4])
    dx[5] = (r[3] - r[5])
    dx[6] = (r[1] + r[2] + r[3] - r[4] - r[5]) / C0_NiO

    # Relación entre derivadas
    dx[n+1:n+m] .= dx[1:m]
end


function obj!(x, x_exp)
    total_distance = 0.0
    total_vectors = 0

    for j in axes(x, 2)
        for i in axes(x, 1)
            smooth_dist = abs.((x_exp[i, j] .- x[i, j]) ./ maximum(x_exp[i, j]))
            total_distance += mean(smooth_dist)
            total_vectors += 1
        end
    end

    return 100 * total_distance / total_vectors
end

function alpha(p)
    α = ones(n + m)
    α[n+1:n+m] .= p[1:m]
    return α
end

u = [(T_sys=Temp[j],) for j in 1:s]

data = [
    (
        t_exp=vec(t_exp[j, 1]),
        x_exp=vcat([reshape(u_exp[j, i], 1, :) for i in 1:n]...),
        x_map=collect(1:n),
    ) for j in 1:s
]

param_setup = (
    p0=vec([1.0 1 0.768677 1 0.761311 0.0830262 0.105632 0.467741 0.00609389 0.0543712 2.0261 4.43843 3.29446 36.9685 22.0291]),
    p_lb=vcat(zeros(m), zeros(2l)),
    p_up=vcat(ones(m), fill(100.0, 2l)),
)

optim_setup = (
    method=Fminbox(LBFGS()),
    options=Optim.Options(show_trace=true, iterations=1, outer_iterations=1),
)

p_ord = vec([1 1 1 1 1 0.0562 0.105 0.397 0.0154 0.089 4.02e-14 5.13 7.36 36.90 21.23])
p_guess = param_setup.p0

problem = prepare_estimation_problem(sys!, alpha, x0, u, data, obj!, T, N)

p_est = estimate_params(problem, param_setup; optim_setup=optim_setup)
p_est = p_guess

#show(IOContext(stdout, :limit => false), round.(p_est, digits=4))
p_est

obj_ord = evaluate_objective(problem, p_ord)
obj_est = evaluate_objective(problem, p_est)
solution_ord = simulate_scenarios(sys!, alpha, x0, u, p_ord, T, N)
solution_est = simulate_scenarios(sys!, alpha, x0, u, p_est, T, N)

println("")
println(obj_ord)
println(obj_est)
println(100 * obj_ord / obj_est - 100)

fig = Figure(; size=(2000, 1000))
ax = Dict()

ax[1] = Axis(
    fig[1, 1];
    xticks=0:5:50,
    yticks=0:0.1:1,
    limits=((0, 50), (0, 1)),
    xminorticksvisible=true,
    yminorticksvisible=true,
    xminorgridvisible=true,
    yminorgridvisible=true,
    #title="a) Compromise Solution",
    xlabel=L"\text{Tiempo [s]}",
    ylabel=L"\text{Presión parcial de CH_4 [psi]}",
    xlabelsize=20,
    ylabelsize=20,
    titlealign=:left,
    #yticklabelcolor = :blue,
    #ylabelcolor = :blue
)
ax[2] = Axis(
    fig[1, 2];
    xticks=0:5:50,
    yticks=0:0.2:4,
    limits=((0, 50), (0, 4)),
    xminorticksvisible=true,
    yminorticksvisible=true,
    xminorgridvisible=true,
    yminorgridvisible=true,
    #title="a) Compromise Solution",
    xlabel=L"\text{Tiempo [s]}",
    ylabel=L"\text{Presión parcial de CO_2 [psi]}",
    xlabelsize=20,
    ylabelsize=20,
    titlealign=:left,
    #yticklabelcolor = :blue,
    #ylabelcolor = :blue
)
ax[3] = Axis(
    fig[1, 3];
    xticks=0:5:50,
    yticks=0:0.4:5,
    limits=((0, 50), (0, 4.5)),
    xminorticksvisible=true,
    yminorticksvisible=true,
    xminorgridvisible=true,
    yminorgridvisible=true,
    #title="a) Compromise Solution",
    xlabel=L"\text{Tiempo [s]}",
    ylabel=L"\text{Presión parcial de H_2 [psi]}",
    xlabelsize=20,
    ylabelsize=20,
    titlealign=:left,
    #yticklabelcolor = :blue,
    #ylabelcolor = :blue
)
ax[4] = Axis(
    fig[2, 1];
    xticks=0:5:50,
    yticks=0:0.3:3,
    limits=((0, 50), (0, 3)),
    xminorticksvisible=true,
    yminorticksvisible=true,
    xminorgridvisible=true,
    yminorgridvisible=true,
    #title="a) Compromise Solution",
    xlabel=L"\text{Tiempo [s]}",
    ylabel=L"\text{Presión parcial de CO [psi]}",
    xlabelsize=20,
    ylabelsize=20,
    titlealign=:left,
    #yticklabelcolor = :blue,
    #ylabelcolor = :blue
)
ax[5] = Axis(
    fig[2, 2];
    xticks=0:5:50,
    yticks=0:0.4:5,
    limits=((0, 50), (0, 5)),
    xminorticksvisible=true,
    yminorticksvisible=true,
    xminorgridvisible=true,
    yminorgridvisible=true,
    #title="a) Compromise Solution",
    xlabel=L"\text{Tiempo [s]}",
    ylabel=L"\text{Presión parcial de H_2O [psi]}",
    xlabelsize=20,
    ylabelsize=20,
    titlealign=:left,
    #yticklabelcolor = :blue,
    #ylabelcolor = :blue
)
ax[6] = Axis(
    fig[2, 3];
    xticks=0:5:50,
    yticks=0:0.1:1,
    limits=((0, 50), (0, 0.8)),
    xminorticksvisible=true,
    yminorticksvisible=true,
    xminorgridvisible=true,
    yminorgridvisible=true,
    #title="a) Compromise Solution",
    xlabel=L"\text{Tiempo [s]}",
    ylabel=L"\text{Grado de conversión del OC [x_{NiO}]}",
    xlabelsize=20,
    ylabelsize=20,
    titlealign=:left,
    #yticklabelcolor = :blue,
    #ylabelcolor = :blue
)


Marker = [:circle :rtriangle :rect]
Color = [:Black :Blue :Red]

SC = [scatter!(ax[i], data[j].t_exp, vec(data[j].x_exp[i, :]), marker=Marker[j], strokecolor=Color[j], color=:transparent, markersize=10, strokewidth=2) for j in 1:s, i in 1:n]
L1 = [lines!(ax[i], t[:], solution_ord[j][i, :], linewidth=2, linestyle=:dot, color=Color[j]) for j in 1:s, i in 1:n]
L2 = [lines!(ax[i], t[:], solution_est[j][i, :], linewidth=2, linestyle=:solid, color=Color[j]) for j in 1:s, i in 1:n]
#[axislegend(ax[i], [L1, L2, SC], [L"\text{Modelo cinético ordinario}", L"\text{Modelo cinético fraccional}", L"\text{Datos experimentales}"], position=:rt, orientation=:vertical) for i in 1:6]
display(fig)
