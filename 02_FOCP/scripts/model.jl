"""
===============================================================================
 MODELO DE CONTROL OPTIMO PARA CLC - SYNGAS 250.
===============================================================================
 Autor:                     Luis Raúl Barajas Villarruel
 Fecha de modificación:     08 de junio de 2025

 Historial de cambios:
 - v4.0 (06-05-2025): Se agrega la temperatura como variable de control.
 - v3.0 (06-05-2025): Se cambia la variable de control a la velocidad de alimentacion.
 - v2.0 (05-05-2025): El modelo se actualizo para calcular concentraciones.
===============================================================================
"""

include(joinpath(@__DIR__,"..", "src","10_OPTIMIZER_FOCP.jl"))

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

# sys: Sistema dinamico.
function sys!(dx, x, u, t, p)
    # Asignacion de parámetros (vector p)
    ᾱ = view(p, 1:m)                            # ᾱ: Orden de la derivada fraccional
    kapp = view(p, m+1:m+l)                     # kapp: Constante cinetica aparente en mol / m3-psi-s
    Ea = -view(p, m+l+1:m+2*l) .* 100 ./ Rg     # Ea: Energia de activacion en Kj / mol
    T_sys = u[1] .* 1e3                         # Temp: Temperatura de operacion (valiable de control u[1])
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

# q: Peso para la funcion objetivo de tipo Mayer
q = [1 1]

# r: Peso para la funcion objetivo de tipo Lagrange
r = [0 0]

# Phi: Funcion objetivo de tipo Mayer
function Phi!(x, t, q)
    q[1] * ((0.63 - x[6]) / (0.63 - 0.40))^2 + q[2] * ((0.87 - x[2] / (x[1] + x[2] + x[4])) / (0.87 - 0.74))^2
end

# Laplacian: Función objetivo de tipo Lagrange
function Laplacian!(x, u, t, r)
    0
end

# u0: Estimación inicial del vector de control.
u0 = [0.923; 0.15]

# lb-ub: Rango de busqueda lb: lower_bounds y ub: upper_bounds.
lb = [0.823; 0]
ub = [0.923; 0.15]

# Optimizacion
Optimize!(Phi!, Laplacian!, sys!, α, x0, u0, lb, ub, T, N; p=p, q=q, r=r, tol=1E-4, max_iters=200, step=1.0, linesearch=BackTracking())

############################
# Informacion del problema #
############################

u1 = [0.823; 0]
x1 = Solve_Forward!(sys!, α, x0, u1, p, T, N)

u2 = [0.923; 0]
x2 = Solve_Forward!(sys!, α, x0, u2, p, T, N)

xCI = Solve_Forward!(sys!, α, x0, u0, p, T, N)

x02 = zeros(n + m)
x02[1:n-1] = y .* C0_NiO * 1

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

u3 = [0.823; 0]
x3 = Solve_Forward!(sys2!, α, x02, u3, p, T, N)

u4 = [0.923; 0]
x4 = Solve_Forward!(sys2!, α, x02, u4, p, T, N)

sol1 = x1
sol2 = x2
sol3 = x3
sol4 = x4
sol5 = xCI

println("RESUMEN DE RESULTADOS")

println("Rendimiento: ", FOCP.x[2, end] / (FOCP.x[1, end] + FOCP.x[2, end] + FOCP.x[4, end]))
println("Rendimiento base 0.5-550: ", sol1[2, end] / (sol1[1, end] + sol1[2, end] + sol1[4, end]))
println("Rendimiento base 0.5-650: ", sol2[2, end] / (sol2[1, end] + sol2[2, end] + sol2[4, end]))
println("Rendimiento base 1.0-550: ", sol3[2, end] / (sol3[1, end] + sol3[2, end] + sol3[4, end]))
println("Rendimiento base 1.0-650: ", sol4[2, end] / (sol4[1, end] + sol4[2, end] + sol4[4, end]))
println("Rendimiento base CI: ", sol5[2, end] / (sol5[1, end] + sol5[2, end] + sol5[4, end]))
println("Mejora peor esenario %: ", ((FOCP.x[2, end] / (FOCP.x[1, end] + FOCP.x[2, end] + FOCP.x[4, end])) / (sol4[2, end] / (sol4[1, end] + sol4[2, end] + sol4[4, end])) - 1) * 100)
println("Mejora mejor esenario %: ", ((FOCP.x[2, end] / (FOCP.x[1, end] + FOCP.x[2, end] + FOCP.x[4, end])) / (sol1[2, end] / (sol1[1, end] + sol1[2, end] + sol1[4, end])) - 1) * 100)

println("")

println("Conversion: ", FOCP.x[6, end])
println("Conversion base 0.5-550: ", sol1[6, end])
println("Conversion base 0.5-650: ", sol2[6, end])
println("Conversion base 1.0-550: ", sol3[6, end])
println("Conversion base 1.0-650: ", sol4[6, end])
println("Conversion base CI: ", sol5[6, end])
println("Mejora peor esenario %: ", 100 * (FOCP.x[6, end] / sol2[6, end] - 1))
println("Mejora mejor esenario %: ", 100 * (FOCP.x[6, end] / sol3[6, end] - 1))
