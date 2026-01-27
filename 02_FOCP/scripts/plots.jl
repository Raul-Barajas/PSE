"""
===============================================================================
 MODELO DE CONTROL OPTIMO PARA CLC - SYNGAS 250.
===============================================================================
 Autor:                     Luis Raúl Barajas Villarruel
 Fecha de modificación:     10 de junio de 2025

 Historial de cambios:
 - v5.0 (10-06-2025): El archivo se modifica para generar las Figuras.
 - v4.0 (06-05-2025): Se agrega la temperatura como variable de control.
 - v3.0 (06-05-2025): Se cambia la variable de control a la velocidad de alimentacion.
 - v2.0 (05-05-2025): El modelo se actualizo para calcular concentraciones.
===============================================================================
"""

using CairoMakie, Colors
logocolors = Colors.JULIA_LOGO_COLORS

# Rango de tiempo y tamaño de paso
T = 50
N = 100

# Conjuntos
l = 5   # l: Número de constantes cinéticas
n = 6   # n: Número de variables [CH4 CO2 H2 CO H2O OC]
m = 5   # m: Número de variables auxiliares

# Datos del modelo
Temp_c = 600 + 273.15               # Temp_c: Temperatura central en K
Rgas = 0.0012062                    # Rgas: Constante de los gases en psi-m3 / mol-k
Rg = 8.3145e-3                      # Rg: Constante de los gases en kj / mol-k
C0_NiO = 390 / 50.7                 # C0_NiO: Concentracion inicial de NiO en mol/m3
y = [0.1 0.2 0.5 0.2 0]             # y: Fraccion molar del compuesto n

# p: Parametros
p = [0.999998 0.999901 1.0 0.833271 0.864344 0.0673656 0.169116 0.306081 0.00941967 0.300519 0.18785 0.0125216 0.165233 0.574386 0.369819]

# α: Orden de los operadores diferenciales 0 < α < 1
α = ones(n + m)
α[n+1:n+m] = view(p, 1:5)

# x0: Condiciones iniciales del vector de estado.
x0 = zeros(n + m)
x0[1:n-1] = y .* C0_NiO * 0.5

x02 = zeros(n + m)
x02[1:n-1] = y .* C0_NiO * 1

# sys: Sistema dinamico.
function sys!(dx, x, u, t, p)
    # Asignacion de parámetros (vector p)
    ᾱ = view(p, 1:m)                            # ᾱ: Orden de la derivada fraccional
    kapp = view(p, m+1:m+l)                     # kapp: Constante cinetica aparente en mol / m3-psi-s
    Ea = -view(p, m+l+1:m+2*l) .* 100 ./ Rg     # Ea: Energia de activacion en Kj / mol
    T_sys = u[1] .* 1e3                        # Temp: Temperatura de operacion (valiable de control u[1])
    #T_sys = 600 + 273.15                       # Temp: Temperatura de operacion
    FC = u[2]                                   # Flujo molar por unidad de volumen en mol/ m3-s

    # D(x): Función interna para calcular la derivada de R-L
    function Dx(i)
        return x0[i] * t^(ᾱ[i] - 1) / gamma(ᾱ[i]) + x[n+i]
    end

    # temp_factor: Factor de activación térmica
    temp_factor = 1 / T_sys - 1 / Temp_c

    # k: Función para calcular k (constante cinetica en 1/s)
    k(j) = kapp[j] * exp(Ea[j] * temp_factor) * T_sys * Rgas

    # Definir velocidades de reacción
    r = Dict()
    r[1] = k(1) * (1 - x[6]) * Dx(1)
    r[2] = k(2) * (1 - x[6]) * Dx(4)
    r[3] = k(3) * (1 - x[6]) * Dx(3)
    r[4] = k(4) * (x[6]) * Dx(2)
    r[5] = k(5) * (x[6]) * Dx(5)

    # Definir ecuaciones diferenciales
    dx[1] = FC * y[1] + (-r[1])
    dx[2] = FC * y[2] + (r[2] - r[4])
    dx[3] = FC * y[3] + (2 * r[1] - r[3] + r[5])
    dx[4] = FC * y[4] + (r[1] - r[2] + r[4])
    dx[5] = FC * y[5] + (r[3] - r[5])
    dx[6] = (r[1] + r[2] + r[3] - r[4] - r[5]) / C0_NiO

    # Relación entre derivadas
    dx[n+1:n+m] .= dx[1:m]
end

function sys2!(dx, x, u, t, p)
    # Asignacion de parámetros (vector p)
    ᾱ = view(p, 1:m)                            # ᾱ: Orden de la derivada fraccional
    kapp = view(p, m+1:m+l)                     # kapp: Constante cinetica aparente en mol / m3-psi-s
    Ea = -view(p, m+l+1:m+2*l) .* 100 ./ Rg     # Ea: Energia de activacion en Kj / mol
    T_sys = u[1] .* 1e3                        # Temp: Temperatura de operacion (valiable de control u[1])
    #T_sys = 600 + 273.15                       # Temp: Temperatura de operacion
    FC = u[2]                                   # Flujo molar por unidad de volumen en mol/ m3-s

    # D(x): Función interna para calcular la derivada de R-L
    function Dx(i)
        return x02[i] * t^(ᾱ[i] - 1) / gamma(ᾱ[i]) + x[n+i]
    end

    # temp_factor: Factor de activación térmica
    temp_factor = 1 / T_sys - 1 / Temp_c

    # k: Función para calcular k (constante cinetica en 1/s)
    k(j) = kapp[j] * exp(Ea[j] * temp_factor) * T_sys * Rgas

    # Definir velocidades de reacción
    r = Dict()
    r[1] = k(1) * (1 - x[6]) * Dx(1)
    r[2] = k(2) * (1 - x[6]) * Dx(4)
    r[3] = k(3) * (1 - x[6]) * Dx(3)
    r[4] = k(4) * (x[6]) * Dx(2)
    r[5] = k(5) * (x[6]) * Dx(5)

    # Definir ecuaciones diferenciales
    dx[1] = FC * y[1] + (-r[1])
    dx[2] = FC * y[2] + (r[2] - r[4])
    dx[3] = FC * y[3] + (2 * r[1] - r[3] + r[5])
    dx[4] = FC * y[4] + (r[1] - r[2] + r[4])
    dx[5] = FC * y[5] + (r[3] - r[5])
    dx[6] = (r[1] + r[2] + r[3] - r[4] - r[5]) / C0_NiO

    # Relación entre derivadas
    dx[n+1:n+m] .= dx[1:m]
end

time = range(0, T; length=N + 1)

u_opt = FOCP.u
u1 = [0.823; 0]
u2 = [0.923; 0]
u3 = [0.823; 0]
u4 = [0.923; 0]

Sol_opt = Solve_Forward!(sys!, α, x0, u_opt, p, T, N)
Sol_Ymax = Solve_Forward!(sys!, α, x0, u1, p, T, N)
Sol_Xmin = Solve_Forward!(sys!, α, x0, u2, p, T, N)
Sol_Xmax = Solve_Forward!(sys2!, α, x02, u3, p, T, N)
Sol_Ymin = Solve_Forward!(sys2!, α, x02, u4, p, T, N)


# Figuras

# Figura 03 Perfiles dinamicos del control.
Figure_u = Figure(; size=(1400, 1400 * 1.5))
ax_u = Dict()

ax_u[1] = Axis(
    Figure_u[1, 1];
    xticks=0:5:50,
    yticks=550:20:650,
    limits=((0, 50), (540, 660)),
    xminorticksvisible=true,
    yminorticksvisible=true,
    xminorgridvisible=true,
    yminorgridvisible=true,
    #title="a) Compromise Solution",
    xlabel=L"\text{Time [s]}",
    ylabel=L"\text{Temperature [°C]}",
    xlabelsize=40,
    ylabelsize=40,
    xticklabelsize=35,
    yticklabelsize=35,
    titlealign=:left,
    titlesize=30,
    #yticklabelcolor = :blue,
    #ylabelcolor = :blue
)
ax_u[2] = Axis(
    Figure_u[2, 1];
    xticks=0:5:50,
    yticks=0:0.030:0.15,
    limits=((0, 50), (-0.01, 0.16)),
    xminorticksvisible=true,
    yminorticksvisible=true,
    xminorgridvisible=true,
    yminorgridvisible=true,
    #title="a) Compromise Solution",
    xlabel=L"\text{Time [s]}",
    ylabel=L"\text{Syngas 250 feed [mol/s·m^3]}",
    xlabelsize=40,
    ylabelsize=40,
    xticklabelsize=35,
    yticklabelsize=35,
    titlealign=:left,
    titlesize=30,
    #yticklabelcolor = :blue,
    #ylabelcolor = :blue
)

u_Opt = similar(u_opt)
u_Opt[1, :] .= u_opt[1, :] .* 1e3 .- 273
u_Opt[2, :] .= u_opt[2, :]

L_u = [lines!(ax_u[i], time[:], u_Opt[i, :], linewidth=5, linestyle=:dash, color=logocolors.red) for i in 1:2]
[axislegend(ax_u[i], [L_u], [L"\text{Optimal trajectory}"], position=:rt, orientation=:vertical, labelsize=40, patchsize=(50, 0), framevisible=false) for i in 1:2]

save("fig/Figure_03.png", Figure_u, pt_per_unit=1000)

#Figura 04 Perfiles dinamicos del estado x & y.
Figure_XY = Figure(; size=(1400, 1400 * 1.5))
ax_XY = Dict()

ax_XY[1] = Axis(
    Figure_XY[1, 1];
    xticks=0:5:50,
    yticks=0:0.05:1,
    limits=((0, 50), (0.4, 0.9)),
    xminorticksvisible=true,
    yminorticksvisible=true,
    xminorgridvisible=true,
    yminorgridvisible=true,
    #title="a) Compromise Solution",
    xlabel=L"\text{Time [s]}",
    ylabel=L"\text{CO_2 yield}",
    xlabelsize=40,
    ylabelsize=40,
    xticklabelsize=35,
    yticklabelsize=35,
    titlealign=:left,
    titlesize=30,
    #yticklabelcolor = :blue,
    #ylabelcolor = :blue
)
ax_XY[2] = Axis(
    Figure_XY[2, 1];
    xticks=0:5:50,
    yticks=0:0.1:1,
    limits=((0, 50), (0, 0.7)),
    xminorticksvisible=true,
    yminorticksvisible=true,
    xminorgridvisible=true,
    yminorgridvisible=true,
    #title="a) Compromise Solution",
    xlabel=L"\text{Time [s]}",
    ylabel=L"\text{OC Oxygen Conversion}",
    xlabelsize=40,
    ylabelsize=40,
    xticklabelsize=35,
    yticklabelsize=35,
    titlealign=:left,
    titlesize=30,
    #yticklabelcolor = :blue,
    #ylabelcolor = :blue
)

Yco2_opt = Sol_opt[2, :] ./ (Sol_opt[1, :] + Sol_opt[2, :] + Sol_opt[4, :])
Yco2_max = Sol_Ymax[2, :] ./ (Sol_Ymax[1, :] + Sol_Ymax[2, :] + Sol_Ymax[4, :])
Yco2_min = Sol_Ymin[2, :] ./ (Sol_Ymin[1, :] + Sol_Ymin[2, :] + Sol_Ymin[4, :])

Xoc_opt = Sol_opt[6, :]
Xoc_max = Sol_Xmax[6, :]
Xoc_min = Sol_Xmin[6, :]

Yco2 = [Yco2_opt, Yco2_max, Yco2_min]
Xoc = [Xoc_opt, Xoc_max, Xoc_min]

Color = [:black, logocolors.blue, logocolors.red]

L_1 = [lines!(ax_XY[1], time[:], Yco2[i], linewidth=5, linestyle=:dash, color=Color[i]) for i in 1:3]
L_2 = [lines!(ax_XY[2], time[:], Xoc[i], linewidth=5, linestyle=:dash, color=Color[i]) for i in 1:3]
axislegend(ax_XY[1], [L_1[1], L_1[2], L_1[3]], [ L"\text{Optimal trajectory}", L"\text{Best trajectory (SR=0.5, T=550°C)}", L"\text{Worse trajectory (SR=1.0, T=650°C)}",], position=:rb, orientation=:vertical, labelsize=40, patchsize=(50, 0), framevisible=false)
axislegend(ax_XY[2], [L_1[1], L_1[2], L_1[3]], [ L"\text{Optimal trajectory}", L"\text{Best trajectory (SR=1.0, T=550°C)}", L"\text{Worse trajectory (SR=0.5, T=650°C)}",], position=:rb, orientation=:vertical, labelsize=40, patchsize=(50, 0), framevisible=false)

save("fig/Figure_04.png", Figure_XY, pt_per_unit=1000)
