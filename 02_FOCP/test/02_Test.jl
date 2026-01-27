include(joinpath(@__DIR__, "10_OPTIMIZER_FOCP.jl"))

# Rango de tiempo y tanaño de paso
T = 1 #seg
N = 100

# u0: Estimación inicial del vector de control.
u0 = [0]

# lb-ub: Rango de busqueda lb: lower_bounds y ub: upper_bounds.
lb = -10
ub = 10

# x0: Condiciones iniciales del vector de estado.
x0 = [1]

# α: Orden de los operadores diferenciales 0 < α < 1
α = [0.5]

# p: Parametros de estado
p = nothing

# q: Peso para la funcion objetivo de tipo Mayer
q = [0]

# r: Peso para la funcion objetivo de tipo Lagrange
r = [0.5]

# sys: Sistema dinamico.
function sys!(dx, x, u, t, p)
    dx[1] = -x[1] + u[1]
end

# Phi: Funcion objetivo de tipo Mayer
function Phi!(x, t, q)
    q[1]
end

# Laplacian: Función objetivo de tipo Lagrange
function Laplacian!(x, u, t, r)
    r[1] * (x[1]^2 + u[1]^2)
end


Optimize!(Phi!, Laplacian!, sys!, α, x0, u0, lb, ub, T, N; p=p, q=q, r=r, tol=1E-8, max_iters=100, step=1.0)

using Plots

X0 = [1]
U0 = [0]

function sys_x!(dx, x, u, t, p)
    dx[1] = -x[1] + u[1]
end
function sys_u!(du, x, u, t, p)
    t = T - t
    du[1] = -x[1] - u[1]
end

function sol()
    n = 100
    U = zeros(N + 1)
    X = zeros(N + 1)
    for i in 1:n
        X = Solve_Forward!(sys_x!, α, X0, U, p, T, N)
        U = Solve_Forward!(sys_u!, α, U0[:, end:-1:1], X[:, end:-1:1], p, T, N)
        U = U[:, end:-1:1]
    end
    return [X, U]
end

solu = sol()

X = solu[1]
U = solu[2]

plotx = plot(range(0, T; length=N + 1), FOCP.x[1, :])
plot!(range(0, T; length=N + 1), X[1, :])
plotu = plot(range(0, T; length=N + 1), FOCP.u[1, :])
plot!(range(0, T; length=N + 1), U[1, :])
plot!(range(0, T; length=N + 1), -1.0 .* FOCP.λ[1, :])

plotall = plot(plotx, plotu, layout=(1, 2), size=(1000, 500))
display(plotall)
Integral_Laplacian(Laplacian!, X, U, r, T, N)
