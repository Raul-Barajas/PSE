include(joinpath(@__DIR__,"..", "src","10_OPTIMIZER_FOCP.jl"))

# Rango de tiempo y tanaño de paso
T = 10 #seg
H = 0.05
N = Int(T / H)

# u0: Estimación inicial del vector de control.
u0 = [0]

# lb-ub: Rango de busqueda lb: lower_bounds y ub: upper_bounds.
lb = 0
ub = 5

# x0: Condiciones iniciales del vector de estado.
x0 = [0; 0]

# α: Orden de los operadores diferenciales 0 < α < 1
α = [1; 1]

# p: Parametros de estado
p = nothing

# q: Peso para la funcion objetivo de tipo Mayer
q = 1

# r: Peso para la funcion objetivo de tipo Lagrange
r = 1

# sys: Sistema dinamico.
function sys(dx, x, u, t, p)
    dx[1] = x[2]
    dx[2] = u[1]
end

# Phi: Funcion objetivo de tipo Mayer
function Phi(x, t, q)
    q[1] * (x[1] - 100)^2
end

# Laplacian: Función objetivo de tipo Lagrange
function Laplacian(x, u, t, r)
    r[1] * u[1]^2
end

U_t = Optimize!(Phi, Laplacian, sys, α, x0, u0, lb, ub, T, N; p=p, q=q, r=r, tol=1E-6, max_iters=100, step=1.0, linesearch= StrongWolfe())

using Plots
t = collect(0:H:T) .- T

U_a = (100 / (1 + 3 / 1000) - 100) .* (t)

println(FOCP.x[1, end])

p1 = plot(collect(0:H:T), U_a)
plot!(collect(0:H:T), FOCP.u[1, :])
p2 = plot(collect(0:H:T), FOCP.x[1, :])

p = plot(p1, p2, layout=(1, 2), size=(1000, 400))
